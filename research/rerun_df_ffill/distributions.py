"""Frozen calib8/9/10 distribution recipes on a causal-feature-ffill input table.

Distribution algebra is copied from the pinned campaign source archives. The
calib9 first-row Fisher scaling is intentionally preserved; calib10 uses the
corrected all-row scaling. Missing target cells contribute no loss or curvature.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import ndtr

H, BLOCK, K, CYCLES, GROUP_ROUNDS = 72, 24, 2, 2, 450
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
EXPERIMENTS = ("e2000fs_calib8_qxgb", "e2000fs_calib9_lnmix", "e2000fs_calib10_xgboostlss")
EXP_CAP = 700.0
LOG_2PI = float(np.log(2.0 * np.pi))
LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
BANDWIDTH = 0.5
PARAMS = {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
          "min_child_weight": 20, "reg_lambda": 5, "objective": "reg:squarederror", "tree_method": "hist",
          "device": "cpu", "multi_strategy": "multi_output_tree", "eval_metric": "rmse", "seed": 42, "verbosity": 0}
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last", "ace_temperature_k_m24",
            "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area", "suvi_t035_meridian30_equatorial_area",
            "suvi_t035_equatorial_area", "suvi_t055_meridian15_area", "suvi_t055_meridian30_equatorial_area",
            "suvi_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def log_norm_pdf(z, mu, sigma):
    u = (z - mu) / sigma
    return -0.5 * (u * u + LOG_2PI) - np.log(sigma)


class Layout:
    """Column layout of one 24 h block: eta (K-1), m0 (1), delta (K-1), rho (K), each BLOCK wide.

    `width` parameters per horizon, so a block has BLOCK * width outputs (120 for K = 2).
    """

    def __init__(self, horizons: int, k: int):
        self.h, self.k = horizons, k
        self.groups = ([("eta", i) for i in range(k - 1)] + [("m0", 0)]
                       + [("delta", i) for i in range(k - 1)] + [("rho", i) for i in range(k)])

    @property
    def width(self):
        return len(self.groups)

    @property
    def n_out(self):
        return self.h * self.width

    def index(self, name, i=0):
        g = self.groups.index((name, i))
        return slice(g * self.h, (g + 1) * self.h)

    def unpack(self, theta):
        out = {("eta", i): theta[:, self.index("eta", i)] for i in range(self.k - 1)}
        out[("m0", 0)] = theta[:, self.index("m0")]
        out.update({("delta", i): theta[:, self.index("delta", i)] for i in range(self.k - 1)})
        out.update({("rho", i): theta[:, self.index("rho", i)] for i in range(self.k)})
        return out

    def to_offsets(self, pi, mu, sigma):
        """Per-horizon marginal MLE -> the offset vector the booster starts from (unconditional ML initialisation)."""
        pi = np.atleast_2d(np.asarray(pi, dtype=np.float64))
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        off = np.zeros(self.n_out)
        log_pi = np.log(np.maximum(pi.T, 1e-9))
        for i in range(self.k - 1):
            off[self.index("eta", i)] = log_pi[:, i + 1] - log_pi[:, 0]  # component 0 is the softmax reference
        off[self.index("m0")] = mu[0]
        prev = mu[0]
        for i in range(self.k - 1):
            gap = np.maximum(mu[i + 1] - prev, 1e-3)
            off[self.index("delta", i)] = np.log(np.expm1(gap))
            prev = mu[i + 1]
        for i in range(self.k):
            off[self.index("rho", i)] = np.log(np.maximum(sigma[i], 1e-4))
        return off


def components(theta, layout: Layout):
    """(pi (n,h,K), mu (n,h,K), sigma (n,h,K)).

    No data-dependent shift in the softmax: it would be mathematically harmless for pi but would break the
    hand-derived gradient (d eta_shifted / d eta != I); |eta| stays far below the float64 exp limit.
    """
    p = layout.unpack(theta)
    n, h, k = p[("m0", 0)].shape[0], p[("m0", 0)].shape[1], layout.k
    eta = np.zeros((n, h, k))
    for i in range(k - 1):
        eta[:, :, i + 1] = p[("eta", i)]
    pi = np.exp(np.minimum(eta, EXP_CAP))
    pi = pi / pi.sum(axis=2, keepdims=True)
    mu = np.empty((n, h, k))
    mu[:, :, 0] = p[("m0", 0)]
    for i in range(k - 1):
        mu[:, :, i + 1] = mu[:, :, i] + softplus(p[("delta", i)])
    sigma = np.exp(np.stack([p[("rho", i)] for i in range(k)], axis=2))
    return pi, mu, sigma


def nll_and_responsibilities(theta, z, layout: Layout):
    pi, mu, sigma = components(theta, layout)
    lp = np.log(np.maximum(pi, 1e-300)) + log_norm_pdf(z[:, :, None], mu, sigma)
    lse = logsumexp(lp, axis=2)
    r = np.exp(lp - lse[:, :, None])
    return -lse, r, pi, mu, sigma


def mixture_mean(theta, layout: Layout):
    """E[y|x] = sum_k pi_k exp(mu_k + sigma_k^2/2); positive by construction."""
    pi, mu, sigma = components(theta, layout)
    a = mu + 0.5 * sigma ** 2
    m = a.max(axis=2, keepdims=True)
    lm = m[:, :, 0] + np.log(np.exp(np.minimum(np.log(np.maximum(pi, 1e-300)) + (a - m), EXP_CAP)).sum(axis=2))
    return np.exp(np.minimum(lm, EXP_CAP)), (pi, mu, sigma)


def gaussian_mixture_em(z, k, iters=400, tol=1e-10):
    z = np.asarray(z, dtype=np.float64)
    if k == 1:
        return np.array([1.0]), np.array([z.mean()]), np.array([max(z.std(), 1e-6)]), float(log_norm_pdf(z, z.mean(), max(z.std(), 1e-6)).sum())
    q = np.quantile(z, np.linspace(0.5 / k, 1.0 - 0.5 / k, k))
    mu, sd, pi = q.copy(), np.full(k, max(z.std() / k, 1e-3)), np.full(k, 1.0 / k)
    prev, ll = -np.inf, -np.inf
    for _ in range(iters):
        lp = np.log(pi)[None, :] + log_norm_pdf(z[:, None], mu[None, :], sd[None, :])
        lse = logsumexp(lp, axis=1)
        r = np.exp(lp - lse[:, None])
        nk = r.sum(0) + 1e-12
        pi, mu = nk / len(z), (r * z[:, None]).sum(0) / nk
        sd = np.maximum(np.sqrt((r * (z[:, None] - mu[None, :]) ** 2).sum(0) / nk), 1e-4)
        ll = float(lse.sum())
        if np.isfinite(prev) and ll - prev < tol * abs(prev):
            break
        prev = ll
    order = np.argsort(mu)
    return pi[order], mu[order], sd[order], ll


def marginal_init(y, k=K):
    pi, mu, sigma = (np.empty((k, y.shape[1])) for _ in range(3))
    for lead in range(y.shape[1]):
        observed = np.isfinite(y[:, lead])
        values = y[observed, lead]
        if len(values) < k or (values <= 0).any():
            raise ValueError("Every lead requires enough observed positive targets for marginal EM")
        p, m, s, _ = gaussian_mixture_em(np.log(values), k)
        pi[:, lead], mu[:, lead], sigma[:, lead] = p, m, s
    return pi, mu, sigma


def group_scale(theta0, z, layout, legacy=False):
    observed = np.isfinite(z)
    safe_z = np.where(observed, z, 0.)
    pi, mu, sigma = components(np.asarray(theta0)[None, :], layout)
    lp = np.log(np.maximum(pi, 1e-300)) + log_norm_pdf(safe_z[:, :, None], mu, sigma)
    r = np.exp(lp - logsumexp(lp, axis=2)[:, :, None])
    inv_var = 1. / sigma ** 2
    unit = {}
    if legacy:
        # calib9's original implementation indexes r[0], averaging the first row's horizons.
        mask = observed[0]
        if not mask.any():
            raise ValueError("calib9 scaling requires an observed target in the first retained row")
        for i in range(layout.k):
            unit[("rho", i)] = max(float((2 * r[0, mask, i]).mean()), 1e-6)
            unit[("m0", 0) if i == 0 else ("delta", i - 1)] = max(float((r[0, mask, i] * inv_var[0, mask, i]).mean()), 1e-6)
    else:
        for i in range(layout.k):
            unit[("rho", i)] = max(float((2 * r[:, :, i])[observed].mean()), 1e-6)
            unit[("m0", 0) if i == 0 else ("delta", i - 1)] = max(float((r[:, :, i] * inv_var[0, :, i])[observed].mean()), 1e-6)
    for i in range(layout.k - 1):
        p = pi[0, :, i + 1]
        # The original eta curvature is constant over rows and averaged over horizons.
        mask = observed[0] if legacy else observed.any(axis=0)
        unit[("eta", i)] = max(float((p[mask] * (1 - p[mask])).mean()), 1e-6)
    return {g: 1. / unit[g] for g in layout.groups}


def group_objective(layout, group, base, theta_fixed, scale, observed):
    cols = layout.index(*group)

    def objective(pred, matrix):
        z = matrix.get_label().astype(np.float64).reshape(len(pred), layout.h)
        theta = theta_fixed.copy()
        theta[:, cols] = pred.astype(np.float64) + base
        _, r, pi, mu, sigma = nll_and_responsibilities(theta, z, layout)
        inv_var = 1. / sigma ** 2
        g_mu, h_mu = -r * (z[:, :, None] - mu) * inv_var, r * inv_var
        name, i = group
        if name == "rho":
            grad = r[:, :, i] * (1 - (z - mu[:, :, i]) ** 2 * inv_var[:, :, i])
            hess = 2 * r[:, :, i]
        elif name == "eta":
            grad = pi[:, :, i + 1] - r[:, :, i + 1]
            hess = pi[:, :, i + 1] * (1 - pi[:, :, i + 1])
        elif name == "m0":
            grad, hess = g_mu.sum(axis=2), h_mu.sum(axis=2)
        else:
            grad, hess = g_mu.sum(axis=2), h_mu.sum(axis=2)
            for j in range(i + 1):
                grad, hess = grad - g_mu[:, :, j], hess - h_mu[:, :, j]
            derivative = sigmoid(theta[:, cols])
            grad, hess = derivative * grad, derivative ** 2 * hess
        # Mask after the positive-Hessian floor; missing cells must remain exactly zero.
        g = np.where(observed, grad * scale, 0.).astype(np.float32)
        h = np.where(observed, np.maximum(hess, 1e-8), 0.).astype(np.float32) * scale
        return g, h
    return objective


def fit_cyclic(x, xq, y, rounds, threads, out, prefix, legacy=False, cycles=CYCLES):
    layout = Layout(y.shape[1], K)
    observed = np.isfinite(y)
    keep = observed.any(axis=1)
    x, y, observed = x[keep], y[keep], observed[keep]
    if not len(y) or not observed.sum(axis=0).all():
        raise ValueError("Every output requires observed training labels")
    pi, mu, sigma = marginal_init(y)
    offsets = layout.to_offsets(pi, mu, sigma)
    np.savez_compressed(out / f"{prefix}_initialization.npz", offsets=offsets, pi=pi, mu=mu, sigma=sigma)
    z = np.log(np.where(observed, y, 1.))
    theta = np.tile(offsets, (len(x), 1))
    theta_q = np.tile(offsets, (len(xq), 1))
    matrix = xgb.DMatrix(x, label=z.astype(np.float32), nthread=threads)
    history, predict_seconds = [], 0.
    for cycle in range(cycles):
        for group in layout.groups:
            cols = layout.index(*group)
            base, baseq = theta[:, cols].copy(), theta_q[:, cols].copy()
            theta0 = np.empty(layout.n_out)
            for g in layout.groups:
                theta0[layout.index(*g)] = np.where(observed, theta[:, layout.index(*g)], 0.).sum(axis=0) / observed.sum(axis=0)
            scale = group_scale(theta0, np.where(observed, z, np.nan), layout, legacy)[group]
            objective = group_objective(layout, group, base, theta.copy(), scale, observed)
            model = xgb.train(dict(PARAMS, nthread=threads, base_score=0.), matrix,
                              num_boost_round=rounds, obj=objective, verbose_eval=False)
            model_name = f"{prefix}_cycle{cycle}_{group[0]}{group[1]}.ubj"
            model.save_model(out / model_name)
            theta[:, cols] = model.predict(matrix).reshape(len(x), layout.h) + base
            started = time.perf_counter()
            theta_q[:, cols] = model.inplace_predict(xq).astype(np.float64) + baseq
            predict_seconds += time.perf_counter() - started
            loss = nll_and_responsibilities(theta, z, layout)[0][observed].mean()
            row = {"cycle": cycle, "group": list(group), "scale": scale, "train_nll": float(loss),
                   "model": model_name, "rounds": rounds}
            if not np.isfinite(loss):
                raise ValueError("Nonfinite observed training NLL")
            history.append(row)
            print(json.dumps({"prefix": prefix, **row}), flush=True)
    started = time.perf_counter()
    pred = mixture_mean(theta_q, layout)[0]
    predict_seconds += time.perf_counter() - started
    (out / f"{prefix}_loss.json").write_text(json.dumps(history, indent=2) + "\n")
    return pred, {"theta": theta_q}, {"history": history, "predict_seconds": predict_seconds,
                                     "train_rows": len(y), "dropped_no_target": int((~keep).sum()),
                                     "observed_per_lead": observed.sum(axis=0).tolist()}


def quantile_objective(observed, horizons, levels=LEVELS, bandwidth=BANDWIDTH):
    tau = np.repeat(levels, horizons)
    mask = np.tile(observed, (1, len(levels)))

    def objective(pred, matrix):
        target = matrix.get_label().astype(np.float64).reshape(pred.shape)
        z = (target - pred.astype(np.float64)) / bandwidth
        gradient = ndtr(-z) - tau[None, :]
        hessian = np.maximum(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / bandwidth, 1e-6)
        return np.where(mask, gradient, 0.).astype(np.float32), np.where(mask, hessian, 0.).astype(np.float32)
    return objective


def cdf_mean(q, levels=LEVELS):
    ext = np.concatenate([[0.], levels, [1.]])
    values = np.concatenate([q[:, :, :1], q, q[:, :, -1:]], axis=2)
    return np.sum(np.diff(ext)[None, None, :] * .5 * (values[:, :, 1:] + values[:, :, :-1]), axis=2)


def fit_quantiles(x, xq, y, rounds, threads, out, prefix):
    observed = np.isfinite(y)
    keep = observed.any(axis=1)
    x, y, observed = x[keep], y[keep], observed[keep]
    if not observed.sum(axis=0).all():
        raise ValueError("Every quantile output requires observed labels")
    mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Quantile target scale must be positive")
    labels = np.tile(np.where(observed, (y - mu) / sd, 0.), (1, len(LEVELS))).astype(np.float32)
    matrix = xgb.DMatrix(x, label=labels, nthread=threads)
    params = dict(PARAMS, learning_rate=.1, max_delta_step=1., nthread=threads, base_score=0., disable_default_eval_metric=True)
    history = {}
    mask = np.tile(observed, (1, len(LEVELS)))
    tau = np.repeat(LEVELS, y.shape[1])

    def pinball(pred, dmat):
        residual = dmat.get_label().reshape(pred.shape) - pred
        loss = np.maximum(tau * residual, (tau - 1) * residual)
        return "observed_pinball", float(loss[mask].mean())

    model = xgb.train(params, matrix, num_boost_round=rounds,
                      obj=quantile_objective(observed, y.shape[1]), evals=[(matrix, "train")],
                      custom_metric=pinball, evals_result=history, verbose_eval=False)
    model.save_model(out / f"{prefix}_quantiles.ubj")
    (out / f"{prefix}_loss.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / f"{prefix}_scaling.json").write_text(json.dumps({"mu": mu, "sd": sd, "levels": LEVELS.tolist()}) + "\n")
    started = time.perf_counter()
    raw = model.inplace_predict(xq).astype(np.float64)
    quantiles = np.sort(raw.reshape(len(xq), len(LEVELS), y.shape[1]).transpose(0, 2, 1) * sd + mu, axis=2)
    pred = np.clip(cdf_mean(quantiles), 1., 3000.)
    timing = time.perf_counter() - started
    return pred, {"quantiles": quantiles}, {"predict_seconds": timing, "train_rows": len(y),
                                          "dropped_no_target": int((~keep).sum()), "mu": mu, "sd": sd,
                                          "observed_per_lead": observed.sum(axis=0).tolist()}


def table(rows):
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    if not n or len(rows) != n * H or not np.array_equal(rows.h.to_numpy().reshape(n, H), np.tile(np.arange(1, H + 1), (n, 1))):
        raise ValueError("Expected unique complete origin x 72h grid")
    origins = pd.to_datetime(rows.origin, utc=True).array.asi8.reshape(n, H)
    if not (origins == origins[:, :1]).all() or len(np.unique(origins[:, 0])) != n:
        raise ValueError("Duplicate or inconsistent origin grid")
    c = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0].copy()
    e = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return pd.DatetimeIndex(origins[:, 0], tz="UTC"), c, e, rows.y.to_numpy(np.float64).reshape(n, H)


def features(block):
    return CONSTANT + [f"{f}@h{h}" for h in range(block * BLOCK + 1, (block + 1) * BLOCK + 1) for f in VARYING]


def design(c, e, block):
    return np.hstack([c, e[:, block * BLOCK:(block + 1) * BLOCK].reshape(len(c), -1)]).astype(np.float32)


def load(data, period):
    df = pd.read_parquet(data, columns=["set", "origin", "h", "period", f"fit_{period}", "y", *CONSTANT, *VARYING], use_threads=False)
    train = df.loc[df[f"fit_{period}"]]
    cutoff = pd.Timestamp(CUTOFFS[period], tz="UTC")
    if not train["set"].eq("train").all() or not (pd.to_datetime(train.origin, utc=True) + pd.Timedelta(hours=144) <= cutoff).all():
        raise ValueError("Training mask violates period maturity")
    training = table(train)
    query = table(df.loc[df["set"].eq("query") & df.period.eq(period)])
    if len(query[0]) != PERIODS[period]:
        raise ValueError("Unexpected official query grid")
    return training, query


def train_period(data, experiment, period, blocks, rounds, threads, out):
    training, query = load(data, period)
    origins, c, e, y = training
    qo, qc, qe, _ = query
    stat = data.stat()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, check=False)
    for block in blocks:
        t0 = time.perf_counter()
        prefix = f"{period}_b{block}"
        x, xq = design(c, e, block), design(qc, qe, block)
        target = y[:, block * BLOCK:(block + 1) * BLOCK]
        if experiment == "e2000fs_calib8_qxgb":
            pred, extra, info = fit_quantiles(x, xq, target, rounds, threads, out, prefix)
        else:
            pred, extra, info = fit_cyclic(x, xq, target, rounds, threads, out, prefix, legacy=experiment == "e2000fs_calib9_lnmix")
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Invalid distribution forecast")
        np.savez_compressed(out / f"{prefix}.npz", pred=pred, origins=qo.asi8, **extra)
        report = {**info, "experiment": experiment, "period": period, "block": block, "rounds": rounds,
                  "cycles": 1 if experiment == "e2000fs_calib8_qxgb" else CYCLES, "seed": 42,
                  "distribution": ({"family": "quantile", "levels": LEVELS.tolist(), "bandwidth": BANDWIDTH,
                                    "kernel": "gaussian", "point_readout": "CDF trapezoid mean clipped to [1,3000]"}
                                   if experiment == "e2000fs_calib8_qxgb" else
                                   {"family": "lognormal_mixture", "components": K, "groups": Layout(BLOCK, K).groups,
                                    "legacy_first_row_fisher": experiment == "e2000fs_calib9_lnmix",
                                    "point_readout": "mixture posterior mean, no clipping"}),
                  "features": features(block), "feature_count": len(features(block)), "queries": len(qo),
                  "params": dict(PARAMS, nthread=threads, **({"learning_rate": .1, "max_delta_step": 1.} if experiment == "e2000fs_calib8_qxgb" else {})),
                  "train_origin_start": str(origins.min()), "train_origin_end": str(origins.max()),
                  "data": {"path": str(data.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                  "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                  "inference_seconds_per_fold": info["predict_seconds"] / len(qo), "seconds": time.perf_counter() - t0,
                  "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}}
        (out / f"{prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("experiment", "period", "block", "rounds", "train_rows", "seconds")}), flush=True)


def merge(out, experiment):
    frames, seconds, identities = [], 0., set()
    for period, count in PERIODS.items():
        parts = []
        for block in range(3):
            with np.load(out / f"{period}_b{block}.npz") as z:
                parts.append({"pred": z["pred"], "origins": z["origins"]})
            report = json.loads((out / f"{period}_b{block}.json").read_text())
            expected_rounds = 300 if experiment == "e2000fs_calib8_qxgb" else GROUP_ROUNDS
            if (report["experiment"] != experiment or report["period"] != period or report["block"] != block
                    or report["rounds"] != expected_rounds or report["features"] != features(block)):
                raise ValueError("Mismatched or smoke artifacts cannot be assembled")
            identities.add(json.dumps(report["data"], sort_keys=True))
            seconds += report["predict_seconds"]
        if not all(np.array_equal(z["origins"], parts[0]["origins"]) for z in parts):
            raise ValueError("Prediction origins disagree")
        pred = np.hstack([z["pred"] for z in parts])
        if pred.shape != (count, H) or not np.isfinite(pred).all():
            raise ValueError("Invalid assembled forecast")
        origins = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), count), "pred_kms": pred.ravel()}))
    if len(identities) != 1:
        raise ValueError("Input snapshots differ across blocks")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    (out / "merge.json").write_text(json.dumps({"experiment": experiment, "rows": len(frame),
                                                "inference_seconds_per_fold": seconds / sum(PERIODS.values())}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--period", choices=PERIODS)
    parser.add_argument("--block", type=int, choices=range(3))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--phase", choices=("all", "merge"), default="all")
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if a.threads < 1 or (a.block is not None and a.period is None):
        parser.error("threads must be positive; block requires period")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == "merge":
        merge(a.out, a.experiment)
        return
    rounds = 2 if a.smoke else (300 if a.experiment == "e2000fs_calib8_qxgb" else GROUP_ROUNDS)
    periods = [a.period or "F1"] if a.smoke or a.period else list(PERIODS)
    blocks = [a.block or 0] if a.smoke else ([a.block] if a.block is not None else list(range(3)))
    for period in periods:
        train_period(a.data, a.experiment, period, blocks, rounds, a.threads, a.out)
    if not a.smoke and not a.period:
        merge(a.out, a.experiment)


if __name__ == "__main__":
    main()
