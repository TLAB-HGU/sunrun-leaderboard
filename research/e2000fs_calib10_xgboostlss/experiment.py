"""e2000fs_calib10_xgboostlss: the heuristic-free successor of e2000fs_calib2_mixture2comp, fitted and audited
with the probabilistic-gradient-boosting methodology of Chevalier & Côté (2025), Eur. Actuar. J.,
doi:10.1007/s13385-025-00428-5.

Research layer. The input table is built by the DB layer (data-v2/builders/input_df.py); nothing here imports data-v2
or another study. Same 237-column design, same frozen xgboost recipe and the same nine blocks as
`research/e2000fs_calib_point4_144h/experiment.py`, with the loss replaced.

The four heuristics of the 2026-10-31 two-component construction, and what replaces each:
  1. level head with weight 4 on y >= 550 km/s, normalised per horizon -> the unweighted NLL of the same rows.
  2. clip(prediction, 1, 3000) on every head                          -> nothing: the family is supported on (0, inf).
  3. probability head regressing 1{y >= 550} with squared error       -> the latent mixing weight pi_k(x).
  4. gap head trained on rows whose block crosses 550 km/s            -> the component locations and scales,
                                                                         estimated from every row by likelihood.
No threshold is formed anywhere, so P(y >= c) is read off the fitted CDF for any c (`exceedance`).

MODEL
    y | x ~ sum_k pi_k(x) LogNormal(mu_k(x), sigma_k(x)),  pi = softmax(eta),  sigma = exp(rho)
    yhat  = E[y|x] = sum_k pi_k exp(mu_k + sigma_k^2/2)                     (reported forecast)
    loss  = -log p(y|x)                                                     (all rows, no weights)

FIT (the paper's Sect. 3.2.1, Algorithm 4 -- XGBoostLSS)
    Each of the 3K-1 parameter groups of a 24 h block gets its own sequence of GROUP_ROUNDS = 450 Newton boosting
    rounds, and CYCLES = 2 sweeps are made over the groups; while a group is updated the others are held at their
    most recent prediction function, and cycle 0 starts from the x-independent ML initialisation (per-horizon
    marginal EM). 450 x 2 is the frozen recipe's 900-round budget per parameter, so no new tuning enters.
    Identification needs no sorting rule: mu_0 = m0, mu_k = m0 + sum_{i<=k} softplus(delta_i) cannot swap labels.
    K is selected by held-out conditional NLL (`--pilot`), never declared; the two-regime reading is the physical
    slow/fast bimodality of the solar wind at 1 AU (Larrodera & Cid 2020, A&A 638 A72), which the 550 km/s cut only
    approximated. No row is routed to a component and nothing is fitted separately or blended afterwards: this is
    neither a gate nor a stacked ensemble.

ADEQUACY (the paper's Sect. 4.2)
    `merge` also scores the fitted distribution, not only its mean: log score (mean NLL per pair), CRPS, PIT
    (KS distance to uniform plus decile shares), coverage of the 50/75/95 % central intervals, calibration bins of
    the point forecast and portfolio balance. Written to results/adequacy.json. The numeric CRPS is checked against
    the closed-form log-normal score (`crps_check`).

GRADIENTS (checked against central differences by `--self-check`, agreement < 4e-6)
    u_k = (z - mu_k)/sigma_k,  r_k = responsibility,  z = log y
    dNLL/dmu_k  = -r_k (z - mu_k)/sigma_k^2      dNLL/drho_k = r_k (1 - u_k^2)      dNLL/deta_j = pi_j - r_j
    Hessian (diagonal Fisher): r_k/sigma_k^2,  2 r_k,  pi_j(1 - pi_j)

    python experiment.py --jobs 9 --threads 2   # nine blocks, then results/predictions.parquet and results/adequacy.json
    python experiment.py --period F1 --block 0 --out DIR
    python experiment.py --smoke --out DIR      # 2 rounds on F1 block 0
    python experiment.py --pilot --k 1 2 3      # K by held-out NLL on matured 2025-12-01..23
    python experiment.py --self-check           # finite-difference gradient check
    python experiment.py --adequacy             # re-score an existing run from results/

Provenance: the layer plumbing (design/table/read_frame/Layout) and the Algorithm-4 skeleton are shared with
research/e2000fs_calib9_lnmix/experiment.py; this arm is a separate copy so the two runs cannot collide.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA = Path("/home/t-lab01/sunrun/data-v2/input/df.parquet")  # data-v2 layer output (builders/input_df.py)
RESULTS = HERE / "results"
H, BLOCK = 72, 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # name -> query origins
K = 2                 # latent regimes; fixed by `--pilot` (held-out conditional NLL), see results/pilot.json
CYCLES = 2            # XGBoostLSS Alg. 4: q_max + 1 sweeps over the parameter groups
GROUP_ROUNDS = 450    # boosting rounds per parameter group per sweep
ROUNDS = 900
EXP_CAP = 700.0       # float64 exp() overflow point: an arithmetic guard, not a modelling constant
LOG_2PI = float(np.log(2.0 * np.pi))
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
              objective="reg:squarederror", tree_method="hist", device="cpu", multi_strategy="multi_output_tree",
              eval_metric="rmse", seed=42, verbosity=0, nthread=1)
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last", "ace_temperature_k_m24",
            "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "ch_ch_t035_top2_area", "ch_ch_t035_meridian15_equatorial_area", "ch_ch_t035_meridian30_equatorial_area",
            "ch_ch_t035_equatorial_area", "ch_ch_t055_meridian15_area", "ch_ch_t055_meridian30_equatorial_area",
            "ch_ch_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]


# ------------------------------------------------------------------ distribution
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


def exceedance(theta, layout: Layout, threshold: float):
    """P(y >= c|x) from the fitted CDF, for any c: the threshold is a read-out, never a training target."""
    from scipy.special import ndtr

    pi, mu, sigma = components(theta, layout)
    prob = np.zeros(pi.shape[:2])
    for kk in range(layout.k):
        prob += pi[:, :, kk] * ndtr((mu[:, :, kk] - np.log(threshold)) / sigma[:, :, kk])
    return prob


def labels_for(y_block, layout: Layout):
    """Every parameter group carries the same log-target; the tiling matches the column layout."""
    return np.tile(np.log(np.asarray(y_block, dtype=np.float64)), (1, layout.width)).astype(np.float32)


def group_scale(theta0, z, layout: Layout):
    """Per-group 1/mean-Hessian: the frozen lambda/min_child_weight stay in "rows" for every parameter group.

    A leaf is worth -G/(H+lambda) and is allowed when H >= min_child_weight, so multiplying a group's gradient and
    Hessian by 1/mean_h turns both frozen constants into the same penalisation measured in rows. It is a diagonal
    preconditioner of the Newton step: same optimum, no new tuning constant.

    `theta0` is the constant ML offset every row starts from, so the parameters are shared and only the
    responsibilities differ; the mean is taken over the training rows, not over a single row.
    """
    z = np.asarray(z, dtype=np.float64)
    pi, mu, sigma = components(np.asarray(theta0, dtype=np.float64)[None, :], layout)   # each (1, h, k)
    lp = np.log(np.maximum(pi, 1e-300)) + log_norm_pdf(z[:, :, None], mu, sigma)
    r = np.exp(lp - logsumexp(lp, axis=2)[:, :, None])                                 # (n, h, k)
    inv_var = 1.0 / sigma ** 2                                                         # (1, h, k)
    unit = {}
    for i in range(layout.k):
        unit[("rho", i)] = max(float((2.0 * r[:, :, i]).mean()), 1e-6)
        unit[("m0", 0) if i == 0 else ("delta", i - 1)] = max(float((r[:, :, i] * inv_var[0, :, i]).mean()), 1e-6)
    for i in range(layout.k - 1):
        p = pi[0, :, i + 1]
        unit[("eta", i)] = max(float((p * (1.0 - p)).mean()), 1e-6)
    return {g: 1.0 / unit[g] for g in layout.groups}


def make_objective(layout: Layout, offsets, scale=None):
    offsets = np.asarray(offsets, dtype=np.float64)
    scale = scale or {g: 1.0 for g in layout.groups}
    cs = np.empty(layout.n_out)
    for g in layout.groups:
        cs[layout.index(*g)] = scale[g]
    k, h = layout.k, layout.h

    def obj(pred, matrix):
        # labels are already log-target (labels_for); every group carries the same one
        z = matrix.get_label().astype(np.float64).reshape(len(pred), layout.width * h)[:, :h]  # tiled log-target
        theta = pred.astype(np.float64) + offsets
        pi, mu, sigma = components(theta, layout)
        lp = np.log(np.maximum(pi, 1e-300)) + log_norm_pdf(z[:, :, None], mu, sigma)
        lse = logsumexp(lp, axis=2)
        r = np.exp(lp - lse[:, :, None])
        inv_var = 1.0 / sigma ** 2
        g_mu = -r * (z[:, :, None] - mu) * inv_var
        g_rho = r * (1.0 - ((z[:, :, None] - mu) ** 2) * inv_var)
        h_mu, h_rho = r * inv_var, 2.0 * r
        grad = np.zeros_like(theta)
        hess = np.zeros_like(theta)
        tail_g, tail_h = g_mu.sum(axis=2), h_mu.sum(axis=2)
        grad[:, layout.index("m0")] = tail_g
        hess[:, layout.index("m0")] = tail_h
        for i in range(k - 1):
            tail_g = tail_g - g_mu[:, :, i]
            tail_h = tail_h - h_mu[:, :, i]
            dsp = sigmoid(theta[:, layout.index("delta", i)])
            grad[:, layout.index("delta", i)] = dsp * tail_g
            hess[:, layout.index("delta", i)] = dsp ** 2 * tail_h
        for i in range(k - 1):
            grad[:, layout.index("eta", i)] = pi[:, :, i + 1] - r[:, :, i + 1]
            hess[:, layout.index("eta", i)] = pi[:, :, i + 1] * (1.0 - pi[:, :, i + 1])
        for i in range(k):
            grad[:, layout.index("rho", i)] = g_rho[:, :, i]
            hess[:, layout.index("rho", i)] = h_rho[:, :, i]
        grad = grad * cs[None, :]
        hess = np.maximum(hess, 1e-8) * cs[None, :]
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


# ------------------------------------------------------------------ marginal EM (initialisation and BIC)
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


def marginal_init(y, k):
    z = np.log(np.asarray(y, dtype=np.float64))
    pi, mu, sigma = np.empty((k, y.shape[1])), np.empty((k, y.shape[1])), np.empty((k, y.shape[1]))
    for h in range(y.shape[1]):
        p, m, s, _ = gaussian_mixture_em(z[:, h], k)
        pi[:, h], mu[:, h], sigma[:, h] = p, m, s
    return pi, mu, sigma


def bic(z, ks):
    z = np.asarray(z, dtype=np.float64).ravel()
    out = {}
    for k in ks:
        pi, mu, sd, ll = gaussian_mixture_em(z, k)
        out[k] = {"loglik": ll, "bic": -2.0 * ll + (3 * k - 1) * np.log(len(z)),
                  "pi": pi.tolist(), "mu": mu.tolist(), "sigma": sd.tolist()}
    return out


class _Labels:
    """Label-only stand-in for a single-group DMatrix (label = log-target of the 24 h block)."""

    def __init__(self, z):
        self._label = np.asarray(z, dtype=np.float64).ravel()

    def get_label(self):
        return self._label


class _Matrix:
    """Stand-in for xgb.DMatrix in the finite-difference check (only get_label is used)."""

    def __init__(self, z, layout):
        self._label = labels_for(np.exp(z), layout).reshape(-1)

    def get_label(self):
        return self._label


def self_check(k=K, h=2, n=64, seed=0):
    """Largest relative error of the analytic gradient against central differences."""
    rng = np.random.default_rng(seed)
    layout = Layout(h, k)
    theta = rng.normal(scale=0.3, size=(n, layout.n_out))
    theta[:, layout.index("m0")] = rng.normal(6.05, 0.1, size=(n, h))
    z = rng.normal(6.05, 0.15, size=(n, h))

    def loss(t):
        return nll_and_responsibilities(t, z, layout)[0].sum()

    grad, _ = make_objective(layout, np.zeros(layout.n_out))(theta.astype(np.float32), _Matrix(z, layout))
    grad, eps, worst = grad.astype(np.float64), 1e-6, 0.0
    for g in range(layout.width):
        for j in range(h):
            c = g * h + j
            tp, tm = theta.copy(), theta.copy()
            tp[:, c] += eps
            tm[:, c] -= eps
            num = (loss(tp) - loss(tm)) / (2 * eps)
            ana = grad[:, c].sum()
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1.0))
    return worst


def self_check_groups(k=K, h=3, n=48, seed=1) -> float:
    """Every group objective must reproduce the matching columns of the joint objective at the same theta.

    Guards the chain-rule terms of the XGBoostLSS decomposition: the fit never evaluates the joint objective, so
    without this check a wrong delta/eta term would only show up as a quiet loss of fit.
    """
    rng = np.random.default_rng(seed)
    layout = Layout(h, k)
    theta = rng.normal(scale=0.25, size=(n, layout.n_out))
    theta[:, layout.index("m0")] = rng.normal(6.05, 0.1, size=(n, h))
    for i in range(k):
        theta[:, layout.index("rho", i)] = rng.normal(-2.0, 0.1, size=(n, h))
    z = rng.normal(6.05, 0.15, size=(n, h))
    scale = {g: 1.0 for g in layout.groups}
    joint, _ = make_objective(layout, np.zeros(layout.n_out), scale)(theta.astype(np.float32), _Matrix(z, layout))
    worst = 0.0
    for g in layout.groups:
        cols = layout.index(*g)
        gr, _ = group_block_objective(layout, g, theta[:, cols].copy(), theta.copy(), 1.0)(
            np.zeros((n, h), dtype=np.float32), _Labels(z))
        worst = max(worst, float(np.abs(gr.astype(np.float64) - joint[:, cols]).max()))
    return worst


# ------------------------------------------------------------------ data
def table(rows: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Long rows -> (origins, constants[origin, feature], varying[origin, h, feature], y[origin, h])."""
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    assert len(rows) == n * H and (rows["h"].to_numpy().reshape(n, H) == np.arange(1, H + 1)).all(), "rows are not a full origin x 72 h grid"
    origins = pd.DatetimeIndex(rows["origin"].to_numpy()[::H], tz="UTC")
    const = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0, :]
    vary = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return origins, const, vary, rows["y"].to_numpy(np.float64).reshape(n, H)


def design(const: np.ndarray, vary: np.ndarray, start: int) -> np.ndarray:
    return np.hstack([const, vary[:, start:start + BLOCK, :].reshape(len(const), -1)]).astype(np.float32)


def read_frame(columns=None) -> pd.DataFrame:
    cols = ["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y", *CONSTANT, *VARYING]
    return pd.read_parquet(DATA, columns=columns or cols)


def group_block_objective(layout: Layout, group, base, theta_fixed, scale_g):
    """Gradient/Hessian of the full NLL restricted to one parameter group, the others held at `theta_fixed`.

    `base` is the group's current prediction on the training rows; the booster's raw margin is an increment on top
    of it (XGBoostLSS keeps f_{k,q}^{m} = f_{k,q}^{m-1} + eta * tree), so theta_group = pred + base.
    """
    cols = layout.index(*group)
    k, h = layout.k, layout.h

    def obj(pred, matrix):
        z = matrix.get_label().astype(np.float64).reshape(len(pred), h)  # one group -> BLOCK outputs
        theta = theta_fixed.copy()
        theta[:, cols] = pred.astype(np.float64) + base
        pi, mu, sigma = components(theta, layout)
        lp = np.log(np.maximum(pi, 1e-300)) + log_norm_pdf(z[:, :, None], mu, sigma)
        lse = logsumexp(lp, axis=2)
        r = np.exp(lp - lse[:, :, None])
        inv_var = 1.0 / sigma ** 2
        g_mu = -r * (z[:, :, None] - mu) * inv_var
        h_mu = r * inv_var
        grad = np.zeros_like(theta)
        hess = np.zeros_like(theta)
        name, i = group
        if name == "rho":
            grad[:, cols] = r[:, :, i] * (1.0 - ((z[:, :, None] - mu)[:, :, i] ** 2) * inv_var[:, :, i])
            hess[:, cols] = 2.0 * r[:, :, i]
        elif name == "eta":
            grad[:, cols] = pi[:, :, i + 1] - r[:, :, i + 1]
            hess[:, cols] = pi[:, :, i + 1] * (1.0 - pi[:, :, i + 1])
        elif name == "m0":
            grad[:, cols] = g_mu.sum(axis=2)
            hess[:, cols] = h_mu.sum(axis=2)
        elif name == "delta":
            tail_g, tail_h = g_mu.sum(axis=2), h_mu.sum(axis=2)
            for j in range(i + 1):
                tail_g = tail_g - g_mu[:, :, j]
                tail_h = tail_h - h_mu[:, :, j]
            dsp = sigmoid(theta[:, cols])
            grad[:, cols] = dsp * tail_g
            hess[:, cols] = dsp ** 2 * tail_h
        g = grad[:, cols] * scale_g
        return g.astype(np.float32), np.maximum(hess[:, cols], 1e-8).astype(np.float32) * scale_g

    return obj


def fit_cyclic(x_tr, x_qu, y, layout: Layout, rounds: int, threads: int, cycles: int = CYCLES, verbose: bool = True):
    """XGBoostLSS Alg. 4 on one block: cyclic sweeps over the 3K-1 parameter groups.

    Returns (theta_query, history, timing). Each group runs `rounds` Newton boosting rounds with the other groups
    frozen at their most recent prediction function; cycle 0 starts from the x-independent ML initialisation.
    `timing["predict_seconds"]` is the wall time the query prediction of the whole block took, so the submitted
    inference cost belongs to the fitted model rather than to a stand-in.
    """
    t0 = time.perf_counter()
    pi, mu, sigma = marginal_init(y, layout.k)                # Alg. 4 step 1: ML initialisation
    offsets = layout.to_offsets(pi, mu, sigma)
    z = np.log(y)
    theta_tr = np.tile(offsets, (len(x_tr), 1))                # q = 0: x-independent initial values
    theta_qu = np.tile(offsets, (len(x_qu), 1))
    label = np.log(y).astype(np.float32).reshape(-1)   # one group per fit -> BLOCK outputs
    dtr = xgb.DMatrix(x_tr, label=label, nthread=threads)
    history = []
    predict_seconds = 0.0
    for q in range(cycles):
        for group in layout.groups:
            cols = layout.index(*group)
            base_tr, base_qu = theta_tr[:, cols].copy(), theta_qu[:, cols].copy()
            theta0 = np.empty(layout.n_out)
            for g_ in layout.groups:
                theta0[layout.index(*g_)] = theta_tr[:, layout.index(*g_)].mean(axis=0)
            scale_g = group_scale(theta0, z, layout)[group]
            obj = group_block_objective(layout, group, base_tr, theta_tr.copy(), scale_g)
            model = xgb.train(dict(PARAMS, nthread=threads, base_score=0.0), dtr, num_boost_round=rounds,
                              obj=obj, verbose_eval=False)
            theta_tr[:, cols] = model.predict(dtr).reshape(len(x_tr), BLOCK) + base_tr
            t1 = time.perf_counter()
            theta_qu[:, cols] = model.inplace_predict(x_qu).astype(np.float64) + base_qu
            predict_seconds += time.perf_counter() - t1
            nll = nll_and_responsibilities(theta_tr, z, layout)[0].mean()
            history.append({"cycle": q, "group": f"{group[0]}{group[1]}", "train_nll": round(float(nll), 6),
                            "seconds": round(time.perf_counter() - t0, 1)})
            print(json.dumps(history[-1]), flush=True)
    return theta_qu, history, {"predict_seconds": round(predict_seconds, 3), "predict_rows": int(len(x_qu))}


def fit_block(df: pd.DataFrame, period: str, block: int, rounds: int, out: Path, k: int = K, threads: int = 1,
              cycles: int = CYCLES) -> None:
    t0 = time.perf_counter()
    start = block * BLOCK
    _, c_tr, e_tr, y_tr = table(df[df[f"fit_{period}"]])
    assert np.isfinite(y_tr).all(), "training targets must be matured"
    origins, c_qu, e_qu, _ = table(df[(df["set"] == "query") & (df["period"] == period)])
    assert len(origins) == PERIODS[period], (period, len(origins))
    y = y_tr[:, start:start + BLOCK]
    layout = Layout(BLOCK, k)
    theta_qu, history, timing = fit_cyclic(design(c_tr, e_tr, start), design(c_qu, e_qu, start), y, layout, rounds,
                                           threads, cycles)
    pred, (piq, muq, sigq) = mixture_mean(theta_qu, layout)
    assert np.isfinite(pred).all() and (pred > 0).all()
    out.mkdir(parents=True, exist_ok=True)
    theta = theta_qu.reshape(len(origins), layout.width, layout.h).transpose(0, 2, 1)   # (origins, 24 h, groups)
    np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, theta=theta.astype(np.float32), origins=origins.asi8)
    (out / f"{period}_b{block}.json").write_text(json.dumps(
        {"period": period, "block": block, "train_rows": int(len(y)), "queries": int(len(origins)), "rounds": rounds,
         "cycles": cycles, "k": k, "threads": threads, "seconds": round(time.perf_counter() - t0),
         "mean_pi_fast": float(piq[:, :, 1:].sum(axis=2).mean()) if k > 1 else None,
         "mean_sigma": [float(sigq[:, :, i].mean()) for i in range(k)],
         "mean_mu": [float(muq[:, :, i].mean()) for i in range(k)], "history": history,
         "scheme": "xgboostlss-cyclic (Chevalier & Cote 2025, Algorithm 4)", **timing}, indent=1) + "\n")


def run_all(jobs: int, out: Path, k: int, threads: int) -> None:
    parts = [(p, b) for p in PERIODS for b in range(3)]

    def one(part):
        p, b = part
        return subprocess.run([sys.executable, __file__, "--period", p, "--block", str(b), "--out", str(out),
                               "--k", str(k), "--threads", str(threads)]).returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        codes = list(pool.map(one, parts))
    if any(codes):
        raise SystemExit(f"blocks failed: {dict(zip(parts, codes))}")


def merge(out: Path) -> dict:
    frames, stats = [], {}
    for period in PERIODS:
        parts = [np.load(out / f"{period}_b{b}.npz") for b in range(3)]
        pred = np.hstack([np.asarray(z["pred"], dtype=np.float64) for z in parts])
        theta = np.concatenate([np.asarray(z["theta"], dtype=np.float32) for z in parts], axis=1)
        assert pred.shape == (PERIODS[period], H) and np.isfinite(pred).all()
        assert theta.shape[:2] == pred.shape and np.isfinite(theta).all()
        np.savez_compressed(out / f"{period}_predictions.npz", pred=pred, theta=theta, origins=parts[0]["origins"])
        o = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        stats[period] = {"origins": int(len(o)), "min": float(pred.min()), "max": float(pred.max()),
                         "mean": float(pred.mean()), "cells_above_3000": int((pred > 3000).sum())}
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()}))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    (out / "check.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats), flush=True)
    adequacy(out)
    print(f"{len(frame)} rows -> {out / 'predictions.parquet'}", flush=True)
    return stats


def pilot(ks, rounds: int, threads: int, out: Path) -> dict:
    """K by held-out conditional NLL on matured data only: fit before 2025-12-01, score 2025-12-01..23 (F1 fit rows)."""
    df = read_frame()
    valid_first = pd.Timestamp("2025-12-01T00:00Z")
    fit_rows = df[df["fit_F1"]].copy()
    origins, c_tr, e_tr, y_tr = table(fit_rows)
    keep = origins + pd.Timedelta(hours=144) <= valid_first
    val = origins >= valid_first
    results = {}
    for k in ks:
        layout = Layout(BLOCK, k)
        y_fit, y_val = y_tr[keep][:, BLOCK:2 * BLOCK], y_tr[val][:, BLOCK:2 * BLOCK]
        theta, _, _ = fit_cyclic(design(c_tr[keep], e_tr[keep], BLOCK), design(c_tr[val], e_tr[val], BLOCK),
                                 y_fit, layout, rounds, threads)
        nll, _, piq, muq, sigq = nll_and_responsibilities(theta, np.log(y_val), layout)
        mean, _ = mixture_mean(theta, layout)
        results[k] = {"heldout_nll_per_pair": float(nll.mean()), "heldout_mse": float(((mean - y_val) ** 2).mean()),
                      "heldout_mse_ge550": float((((mean - y_val) ** 2)[y_val >= 550]).mean()),
                      "mean_pi_fast": float(piq[:, :, 1:].sum(axis=2).mean()) if k > 1 else None,
                      "mean_mu": [float(muq[:, :, i].mean()) for i in range(k)],
                      "mean_sigma": [float(sigq[:, :, i].mean()) for i in range(k)], "rounds": rounds}
        print(json.dumps({k: results[k]}), flush=True)
    best = min(results, key=lambda c: results[c]["heldout_nll_per_pair"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps({"block": 1, "valid_first": str(valid_first),
                                                "fit_origins": int(keep.sum()), "valid_origins": int(val.sum()),
                                                "results": results, "k_by_heldout_nll": best,
                                                "marginal_bic": bic(np.log(y_tr), ks)}, indent=1) + "\n")
    print(f"K selected by held-out conditional NLL: {best}")
    return results


# ------------------------------------------------------------------ adequacy (paper Sect. 4.2)
def theta_block(theta: np.ndarray, block: int, layout: Layout) -> np.ndarray:
    """(origins, 72 h, groups) -> the (n, n_out) column layout of one 24 h block."""
    sl = slice(block * BLOCK, (block + 1) * BLOCK)
    return theta[:, sl, :].transpose(0, 2, 1).reshape(len(theta), -1).astype(np.float64)


def pit(theta: np.ndarray, layout: Layout, y: np.ndarray) -> np.ndarray:
    """F(y|x): the probability integral transform of the observation under its own predictive CDF."""
    from scipy.special import ndtr

    pi, mu, sigma = components(theta, layout)
    return (pi * ndtr((np.log(y)[:, :, None] - mu) / sigma)).sum(axis=2)


def crps(theta: np.ndarray, layout: Layout, y: np.ndarray, grid: int = 401, span: float = 8.0,
         chunk: int = 2048) -> np.ndarray:
    """CRPS(F, y) = integral (F(z) - 1{y <= z})^2 dz, split at z = y so both pieces are smooth.

    With z = exp(u): CRPS = integral_{-inf}^{ln y} F(e^u)^2 e^u du + integral_{ln y}^{inf} (1 - F(e^u))^2 e^u du.
    Splitting removes the jump of the indicator at z = y, which is what makes a single-grid quadrature only
    first-order accurate; each piece is smooth and the trapezoid rule is second order.
    """
    from scipy.special import ndtr

    pi, mu, sigma = components(theta, layout)
    n, h, k = pi.shape
    out = np.empty((n, h))
    t = np.linspace(0.0, 1.0, grid)
    w = np.ones(grid)
    w[0] = w[-1] = 0.5
    log_y = np.log(y)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        ly = log_y[a:b]
        lo = np.minimum((mu[a:b] - span * sigma[a:b]).min(axis=2), ly)
        hi = np.maximum((mu[a:b] + span * sigma[a:b]).max(axis=2), ly)
        total = np.zeros_like(ly)
        for left, right, lower in ((lo, ly, True), (ly, hi, False)):
            u = left[:, :, None] + (right - left)[:, :, None] * t
            zu = np.exp(np.minimum(u, EXP_CAP))
            f = np.zeros_like(u)
            for j in range(k):
                f += pi[a:b, :, j, None] * ndtr((u - mu[a:b, :, j, None]) / sigma[a:b, :, j, None])
            g = f if lower else (1.0 - f)
            total += (g ** 2 * zu * w).sum(axis=2) * (right - left) / (grid - 1)
        out[a:b] = total
    return out


def crps_lognormal(mu, sigma, y):
    """Closed-form CRPS of a single log-normal: used to validate the numerical integrator only."""
    from scipy.special import ndtr

    u = (np.log(y) - mu) / sigma
    return y * (2.0 * ndtr(u) - 1.0) - 2.0 * np.exp(mu + 0.5 * sigma ** 2) * (
        ndtr(u - sigma) + ndtr(sigma / np.sqrt(2.0)) - 1.0)


def crps_check(seed: int = 0) -> dict:
    """Numerical CRPS vs the closed log-normal form on random (mu, sigma, y)."""
    rng = np.random.default_rng(seed)
    n, h = 400, 4
    layout = Layout(h, 1)
    mu = rng.normal(6.05, 0.15, size=(n, h))
    sigma = np.exp(rng.normal(-2.0, 0.2, size=(n, h)))
    y = np.exp(rng.normal(mu, sigma))
    theta = np.empty((n, layout.n_out))
    theta[:, layout.index("m0")] = mu
    theta[:, layout.index("rho", 0)] = np.log(sigma)
    ana = crps_lognormal(mu, sigma, y)
    num = crps(theta, layout, y)
    return {"max_rel": float((np.abs(num - ana) / np.abs(ana)).max()),
            "mean_rel": float((np.abs(num - ana) / np.abs(ana)).mean())}


def calibration(pred: np.ndarray, y: np.ndarray, bins: int = 10) -> list:
    """Equal-count bins of the point forecast: mean prediction vs mean observation."""
    p, t = pred.ravel(), y.ravel()
    edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] = np.inf
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            rows.append({"bin": i, "n": int(m.sum()), "pred_mean": float(p[m].mean()),
                         "obs_mean": float(t[m].mean()),
                         "rel_bias_pct": float(100.0 * (p[m].mean() - t[m].mean()) / t[m].mean())})
    return rows


def score(pred: np.ndarray, theta: np.ndarray, y: np.ndarray, layout: Layout) -> dict:
    """Log score, CRPS, PIT, CI coverage, calibration and portfolio balance for one set of origins."""
    nll_v, pit_v, crps_v = [], [], []
    for b in range(3):
        th = theta_block(theta, b, layout)
        yb = y[:, b * BLOCK:(b + 1) * BLOCK]
        nll_v.append(nll_and_responsibilities(th, np.log(yb), layout)[0].ravel())
        pit_v.append(pit(th, layout, yb).ravel())
        crps_v.append(crps(th, layout, yb).ravel())
    nll, pit_v, crps_v = np.concatenate(nll_v), np.concatenate(pit_v), np.concatenate(crps_v)
    flat = y.ravel()
    err = pred.ravel() - flat
    hi = flat >= 550.0
    n = len(pit_v)
    ks = float(np.abs(np.sort(pit_v) - (np.arange(1, n + 1) - 0.5) / n).max())
    deciles = np.histogram(pit_v, bins=np.linspace(0, 1, 11))[0] / n
    return {"pairs": int(n), "log_score_nll_per_pair": float(nll.mean()), "crps": float(crps_v.mean()),
            "pit": {"ks_uniform": ks, "decile_share": [round(float(v), 4) for v in deciles],
                    "max_decile_dev": float(np.abs(deciles - 0.1).max()), "mean": float(pit_v.mean())},
            "coverage": {str(int(100 * lvl)): float(((pit_v >= (1 - lvl) / 2) & (pit_v <= (1 + lvl) / 2)).mean())
                         for lvl in (0.50, 0.75, 0.95)},
            "portfolio_balance_pct": float(100.0 * (pred.ravel().sum() - flat.sum()) / flat.sum()),
            "mse": float((err ** 2).mean()), "mse_ge550": float((err[hi] ** 2).mean()) if hi.any() else None,
            "n_ge550": int(hi.sum()), "calibration": calibration(pred, y),
            "forecast": {"min": float(pred.min()), "max": float(pred.max()), "mean": float(pred.mean())}}


def adequacy(out: Path) -> dict:
    """Score the merged run with the paper's adequacy tools and write results/adequacy.json."""
    res = {"crps_check": crps_check(), "periods": {}}
    pool = {"pred": [], "theta": [], "y": []}
    query = read_frame()
    query = query[query["set"] == "query"]
    for period in PERIODS:
        z = np.load(out / f"{period}_predictions.npz")
        theta = np.asarray(z["theta"], dtype=np.float64)
        layout = Layout(BLOCK, (theta.shape[2] + 1) // 3)
        origins, _, _, y = table(query[query["period"] == period])
        pred = np.asarray(z["pred"], dtype=np.float64)
        res["periods"][period] = score(pred, theta, y, layout)
        pool["pred"].append(pred)
        pool["theta"].append(theta)
        pool["y"].append(y)
        print(json.dumps({period: {kk: vv for kk, vv in res["periods"][period].items() if kk != "calibration"}}),
              flush=True)
    res["pooled"] = score(np.concatenate(pool["pred"]), np.concatenate(pool["theta"]),
                          np.concatenate(pool["y"]), layout)
    print(json.dumps({"pooled": {kk: vv for kk, vv in res["pooled"].items() if kk != "calibration"}}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adequacy.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODS))
    ap.add_argument("--block", type=int, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--rounds", type=int, default=GROUP_ROUNDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pilot-rounds", type=int, default=GROUP_ROUNDS)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--adequacy", action="store_true", help="re-score an existing run from --out")
    a = ap.parse_args()
    if a.self_check:
        for kk in (1, 2, 3):
            print(f"K={kk} max relative gradient error {self_check(k=kk):.3e}")
            print(f"K={kk} group-vs-joint max |dgrad| {self_check_groups(k=kk):.3e}")
        return
    if a.adequacy:
        adequacy(a.out)
        return
    if a.pilot:
        pilot(a.pilot_k, a.pilot_rounds, max(a.threads, 2), a.out)
        return
    if a.smoke or a.period:
        df = read_frame()
        fit_block(df, a.period or "F1", a.block or 0, 2 if a.smoke else a.rounds, a.out, a.k, max(a.threads, 1))
        return
    run_all(a.jobs, a.out, a.k, a.threads)
    merge(a.out)


if __name__ == "__main__":
    main()
