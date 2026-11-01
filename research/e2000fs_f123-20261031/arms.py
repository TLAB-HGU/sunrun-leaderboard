"""calib-family arms refit on the F1/F2/F3 quarters.

Faithful ports of research/e2000fs_calib-20261030/tweedie_lane.py (point4 144h control, tweedie15), e2000fs_calib2-20261031/lanes2.py
(mixture), detector_lane.py (exceedance detector + routing) and e2000fs_calib3-20261031/lanes3.py (era, gpd). Mechanics, hyper-parameters,
rounds and pre-registered constants are unchanged; only the stage definition changes from {f1, final} to the three quarters of common.py,
and the training mask stays origin + 144 h <= cutoff (fully matured 72 h vectors) for every arm of this family.

    python arms.py --arm tweedie15 --period F1 --out OUT [--blocks 0 1 2] [--threads 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

H, U, SCALE = K.H, 550.0, 500.0
MASK_144 = pd.Timedelta(hours=144)
RECEIPT = K.ROOT / "research/e2000fs_dirmo_24_feat30_pointweight4/results/source/analysis/e2000fs-visualizations/worktree/docs/visualizations/e2000fs_dirmo_24/feature_importance_receipt.json"
RECIPE = json.loads(K.SOURCE_MANIFEST.read_text())["recipe"]["xgboost"]
LEGACY = {  # reproduction of frozen stages: name -> (frozen query bundle, cutoff)
    "legacy-f1": ("f1", "2026-02-26T00:00Z"), "legacy-final": ("official", "2026-05-29T00:00Z")}


def params_of(threads: int, **over) -> dict:
    p = {k: v for k, v in RECIPE.items() if k not in ("n_estimators", "n_jobs", "random_state", "base_score", "objective")}
    p.update(objective="reg:squarederror", tree_method="hist", device="cpu", multi_strategy="multi_output_tree",
             nthread=threads, seed=42, eval_metric="rmse", verbosity=0)
    p.update(over)
    return p


# ------------------------------------------------------------------ objectives
def point_weights(t, normalizer=None):
    w = np.where(t >= U, 4.0, 1.0)
    norm = w.mean(axis=0) if normalizer is None else np.asarray(normalizer)
    return (w / norm).astype(np.float32)


def obj_point4(pred, matrix):  # tweedie_lane 'control' arm: weights normalised by the batch mean per horizon
    t = matrix.get_label().reshape(pred.shape)
    w = point_weights(t)
    return ((pred - t) * w).astype(np.float32), w.astype(np.float32)


def obj_point4_plain(pred, matrix):  # lanes3 obj_point4: weights 4/1, no normalisation
    t = matrix.get_label().reshape(pred.shape).astype(np.float64)
    w = np.where(t >= U, 4.0, 1.0)
    return ((pred - t) * w).astype(np.float32), w.astype(np.float32)


def obj_mse(pred, matrix):
    t = matrix.get_label().reshape(pred.shape).astype(np.float64)
    return (pred - t).astype(np.float32), np.ones_like(pred, dtype=np.float32)


def make_tweedie(power):
    def f(pred, matrix):
        t = matrix.get_label().reshape(pred.shape).astype(np.float64)
        mu = np.clip(pred.astype(np.float64), 1.0, 3000.0)
        s = SCALE ** power
        grad = s * (mu ** (1.0 - power) - t * mu ** (-power))
        hess = np.clip(s * ((1.0 - power) * mu ** (-power) + power * t * mu ** (-power - 1.0)), 1e-6 * s, None)
        return grad.astype(np.float32), hess.astype(np.float32)
    return f


def make_weighted(w_row):
    w_row = np.asarray(w_row, dtype=np.float64)

    def f(pred, matrix):
        t = matrix.get_label().reshape(pred.shape).astype(np.float64)
        w = np.where(t >= U, 4.0, 1.0) * w_row.reshape(-1, 1)
        return ((pred - t) * w).astype(np.float32), w.astype(np.float32)
    return f


def fit(x, y, params, rounds, obj, threads):
    """xgb.train with the lanes' DMatrix settings (the validation recorder of the originals never changes the model)."""
    dt = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    return xgb.train(dict(params, nthread=threads), dt, num_boost_round=rounds, obj=obj, verbose_eval=False)


# ------------------------------------------------------------------ data per stage
class Stage:
    """Training rows (mask origin + 144 h <= cutoff), design blocks and query blocks of one period."""

    def __init__(self, period: str, mask_hours: int = 144, strict: bool = False):
        nm, sel, series = K.names(), K.selected(), K.speed()
        if period in LEGACY:
            qname, cutoff = LEGACY[period]
            q = K._load(K.FROZEN / f"{qname}.npz")
            train = K.training_rows(pd.Timestamp("2026-05-28T21:00Z"))
            cutoff = pd.Timestamp(cutoff)
        else:
            _, _, cutoff = K.PERIODS[period]
            cutoff = pd.Timestamp(cutoff)
            q = K.period_queries(period)
            train = K.training_rows(cutoff - pd.Timedelta(hours=mask_hours))
        self.period, self.cutoff, self.series = period, cutoff, series
        mask = pd.Timedelta(hours=mask_hours)  # strict: origin + mask < cutoff (the 72 h maturity rule); else <= cutoff
        self.idx = np.flatnonzero(train["origins"] + mask < cutoff if strict else train["origins"] + mask <= cutoff)
        self.origins = train["origins"][self.idx]
        assert (self.origins + K.MATURITY).max() < q["origins"].min(), "training targets must precede the first query origin"
        self.y = K.targets(series, self.origins)
        assert np.isfinite(self.y).all()
        tr = {k: v[self.idx] for k, v in train.items() if k != "origins"}
        self.x = [K.design(tr, nm, sel, s) for s in (0, 24, 48)]
        self.q_origins = q["origins"]
        qb = {k: v for k, v in q.items() if k != "origins"}
        self.xq = [K.design(qb, nm, sel, s) for s in (0, 24, 48)]
        self.truth = K.targets(series, self.q_origins)
        print(f"{period}: train rows={len(self.idx)} (last origin {self.origins.max()}), queries={len(self.q_origins)}, cutoff={cutoff}", flush=True)


def sl(b):
    return slice(b * 24, (b + 1) * 24)


# ------------------------------------------------------------------ arms
def arm_point4_144h(st: Stage, blocks, threads, rounds=900):
    """tweedie_lane 'control' arm."""
    pred = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        y = st.y[:, sl(b)]
        m = fit(st.x[b], y, params_of(threads, base_score=float(np.nanmean(y))), rounds, obj_point4, threads)
        pred[:, sl(b)] = np.clip(m.inplace_predict(st.xq[b]), 1.0, 3000.0)
    return pred


def arm_tweedie15(st: Stage, blocks, threads, rounds=900, power=1.5):
    pred = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        y = st.y[:, sl(b)]
        m = fit(st.x[b], y, params_of(threads, base_score=float(np.nanmean(y))), rounds, make_tweedie(power), threads)
        pred[:, sl(b)] = np.clip(m.inplace_predict(st.xq[b]), 1.0, 3000.0)
    return pred


def arm_mixture(st: Stage, blocks, threads, rounds=900):
    """lanes2.lane_mixture: level + P(y>=550) * (E[y | block has event] - level); the three heads use the 'control' objective."""
    level = np.full((len(st.q_origins), H), np.nan)
    prob, gap = level.copy(), level.copy()
    for b in blocks:
        s_, y, x, xq = sl(b), st.y[:, sl(b)], st.x[b], st.xq[b]
        p = params_of(threads)
        level[:, s_] = fit(x, y, dict(p, base_score=float(np.nanmean(y))), rounds, obj_point4, threads).inplace_predict(xq)
        ind = (y >= U).astype(np.float32)
        prob[:, s_] = np.clip(fit(x, ind, dict(p, base_score=float(np.mean(ind))), rounds, obj_point4, threads).inplace_predict(xq), 0.0, 1.0)
        ev = (y >= U).any(axis=1)
        resid = y[ev] - y[ev].mean(axis=0)
        gap[:, s_] = fit(x[ev], resid, dict(p, base_score=0.0), rounds, obj_point4, threads).inplace_predict(xq) + y[ev].mean(axis=0)
    return np.clip(level + prob * (gap - level), 1.0, 3000.0)


def arm_era(st: Stage, blocks, threads, rounds=900, inner=3, eta=0.5):
    """lanes3.lane_era: month-wise adversarial (Group-DRO style) reweighting on the point4 objective."""
    pred = np.full((len(st.q_origins), H), np.nan)
    months = st.origins.month
    groups = np.unique(months)
    for b in blocks:
        y, x = st.y[:, sl(b)], st.x[b]
        gw, model = np.ones(len(groups)), None
        for _ in range(inner):
            row_w = gw[np.searchsorted(groups, months)]
            row_w = row_w / row_w.mean()
            model = fit(x, y, params_of(threads, base_score=float(np.nanmean(y))), rounds, make_weighted(row_w), threads)
            inpred = model.inplace_predict(x)
            loss = np.array([float(((inpred[months == g] - y[months == g]) ** 2).mean()) for g in groups])
            gw = gw * np.exp(eta * (loss / loss.mean() - 1.0))
            gw = gw / gw.mean()
        pred[:, sl(b)] = model.inplace_predict(st.xq[b])
    return np.clip(pred, 1.0, 3000.0)


def arm_gpd(st: Stage, blocks, threads, rounds=900, tau=0.5):
    """lanes3.lane_gpd: where P(y>=550) >= tau replace the level by 550 + exp(log-excess), else the level head."""
    pred = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        y, x, xq = st.y[:, sl(b)], st.x[b], st.xq[b]
        p = params_of(threads)
        lvl = fit(x, y, dict(p, base_score=float(np.nanmean(y))), rounds, obj_point4, threads)
        ind = (y >= U).astype(np.float32)
        pr = fit(x, ind, dict(p, base_score=float(np.mean(ind))), 400, obj_mse, threads)
        ev = (y >= U).any(axis=1)
        z = np.log(np.maximum(y[ev] - U, 1.0))
        ez = fit(x[ev], z, dict(p, base_score=float(np.mean(z))), 400, obj_mse, threads)
        L = lvl.inplace_predict(xq).astype(np.float64)
        P = np.clip(pr.inplace_predict(xq), 0.0, 1.0)
        Z = np.clip(ez.inplace_predict(xq), 0.0, np.log(3000.0 - U))
        pred[:, sl(b)] = np.where(P >= tau, U + np.exp(Z), L)
    return np.clip(pred, 1.0, 3000.0)


def arm_detector(st: Stage, blocks, threads, rounds=400):
    """detector_lane: binary exceedance classifier on the row-level design (block features + horizon index); returns P(y>=550)."""
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu", max_depth=4, eta=0.05,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5, nthread=threads, seed=42, verbosity=0)
    prob = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        start = b * 24
        n = len(st.x[b])
        x_tr = np.hstack([np.repeat(st.x[b], 24, axis=0), np.tile(np.arange(start + 1, start + 25), n).astype(np.float32)[:, None]])
        lab = (st.y[:, start:start + 24] >= U).astype(np.float32).reshape(-1)
        nq = len(st.xq[b])
        x_ap = np.hstack([np.repeat(st.xq[b], 24, axis=0), np.tile(np.arange(start + 1, start + 25), nq).astype(np.float32)[:, None]])
        model = xgb.train(params, xgb.DMatrix(x_tr, label=lab, nthread=threads), num_boost_round=rounds)
        prob[:, sl(b)] = model.predict(xgb.DMatrix(x_ap, nthread=threads)).reshape(nq, 24)
    return prob


# ---- spectral loss (research/e2000fs_budget_spectral/results/source/run.py): point4 weights + 0.10 * smoothed FFT term
SPEC_COEF, SPEC_SCALE = 0.10, 50.0


def obj_spectral(pred, matrix):
    t = matrix.get_label().reshape(pred.shape)
    w = point_weights(t)
    err = pred - t
    z = np.fft.fft(err, axis=1, norm="ortho")
    rho = 1 / np.sqrt(1 + abs(z) ** 2 / SPEC_SCALE ** 2)
    grad = err * w + SPEC_COEF * np.fft.ifft(z * rho, axis=1, norm="ortho").real
    return grad.astype(np.float32), (w + SPEC_COEF).astype(np.float32)


def arm_spectral(st: Stage, blocks, threads, rounds=900):
    """budget_spectral 'spectral' arm: pointweight4 recipe (num_target 24, base_score = target mean) with the FFT term; 144 h mask."""
    pred = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        y = st.y[:, sl(b)].astype(np.float32)
        p = {k: v for k, v in RECIPE.items() if k not in ("n_jobs", "n_estimators", "random_state")}
        p.update(nthread=threads, seed=42, num_target=24, base_score=float(y.mean()))
        m = xgb.train(p, xgb.DMatrix(st.x[b], label=y, nthread=threads), num_boost_round=rounds, obj=obj_spectral, verbose_eval=False)
        pred[:, sl(b)] = np.clip(m.inplace_predict(st.xq[b]), 1.0, 3000.0)
    return pred


def arm_dirmo24(st: Stage, blocks, threads, rounds=900):
    """e2000fs_dirmo_24_feat30: XGBRegressor (receipt recipe, multi_output_tree), plain squared error; 72 h maturity (origin + 72 h < cutoff)."""
    pred = np.full((len(st.q_origins), H), np.nan)
    recipe = json.loads(RECEIPT.read_text())["recipe"]
    recipe.update(n_jobs=threads, n_estimators=rounds, multi_strategy="multi_output_tree")
    for b in blocks:
        model = xgb.XGBRegressor(**recipe)
        model.fit(st.x[b], st.y[:, sl(b)], verbose=False)
        pred[:, sl(b)] = np.clip(model.predict(st.xq[b]), 1.0, 3000.0)
    return pred


# ---- LightGBM family (scalar model per horizon, shared tuned parameters; 72 h maturity: origin + 72 h < cutoff)
LGB_PARAMS = dict(n_estimators=400, objective="regression", learning_rate=0.011665375590900416, max_depth=6, num_leaves=39,
                  subsample=0.8158360436957313, subsample_freq=1, colsample_bytree=0.796156879318864, min_child_samples=76,
                  min_child_weight=1.7495117760617724, reg_alpha=12.889193670860928, reg_lambda=9.785792859825623,
                  random_state=42, device_type="cpu", verbosity=-1)


def _lgbm(st: Stage, blocks, threads, weight_fn, keep=None):
    import lightgbm as lgb
    idx = np.arange(len(st.idx)) if keep is None else keep(st)
    pred = np.full((len(st.q_origins), H), np.nan)
    for b in blocks:
        for h in range(b * 24, (b + 1) * 24):
            m = lgb.LGBMRegressor(**dict(LGB_PARAMS, n_jobs=threads))
            m.fit(st.x[b][idx], st.y[idx, h], sample_weight=weight_fn(st.y[idx], h, b))
            pred[:, h] = np.clip(m.predict(st.xq[b]), 1.0, 3000.0)
    return pred


def _event_weights(y_all, h, b):
    """feat30_lgbm_extreme4: origins whose 24 h block reaches 550 km/s or ramps 100 km/s in 6 h count four times; mean 1."""
    blk = y_all[:, b * 24:(b + 1) * 24]
    event = (blk.max(axis=1) >= 550) | ((blk[:, 6:] - blk[:, :-6]).max(axis=1) >= 100)
    w = np.where(event, 4.0, 1.0)
    return w / w.mean()


def _point_weights_h(y_all, h, b):
    """feat30_pointweight4 (LightGBM): only this horizon's own target >= 550 gets weight 4; mean 1."""
    w = np.where(y_all[:, h] >= 550.0, 4.0, 1.0)
    return w / w.mean()


def arm_lgbm_extreme4(st: Stage, blocks, threads):
    return _lgbm(st, blocks, threads, _event_weights)


def arm_lgbm_point4(st: Stage, blocks, threads):
    return _lgbm(st, blocks, threads, _point_weights_h)


def arm_lgbm_recent(st: Stage, blocks, threads):
    """feat30_transfer_recent_window: plain L2 on the training rows no older than 365 days before the newest training row."""
    def keep(s):
        age = np.asarray((s.origins.max() - s.origins) / pd.Timedelta(days=1))
        return np.flatnonzero(age <= 365)
    return _lgbm(st, blocks, threads, lambda y, h, b: None, keep)


def arm_qxgb(st: Stage, blocks, threads):
    """QXGB (arXiv:2608.15290): convolution-smoothed dense-quantile multi-output trees; point forecast = CDF mean. See qxgb.py."""
    import qxgb
    return qxgb.arm_qxgb(st, blocks, threads)


STRICT72 = {"dirmo24", "lgbm_extreme4", "lgbm_point4", "lgbm_recent", "qxgb"}
ARMS = {"lgbm_extreme4": arm_lgbm_extreme4, "lgbm_point4": arm_lgbm_point4, "lgbm_recent": arm_lgbm_recent, "spectral": arm_spectral, "dirmo24": arm_dirmo24, "point4_144h": arm_point4_144h, "tweedie15": arm_tweedie15, "mixture": arm_mixture, "era": arm_era, "gpd": arm_gpd,
        "detector": arm_detector, "qxgb": arm_qxgb}


def run(arm: str, period: str, out: Path, blocks, threads: int) -> None:
    """Fit the given 24 h blocks of one arm/period and write them as a part file; merge_parts.py joins the three blocks."""
    st = Stage(period, mask_hours=72, strict=True) if arm in STRICT72 else Stage(period)
    t = time.perf_counter()
    pred = ARMS[arm](st, blocks, threads)
    out.mkdir(parents=True, exist_ok=True)
    tag = "".join(str(b) for b in blocks)
    np.savez_compressed(out / f"{period}_part{tag}.npz", pred=pred, origins=st.q_origins.asi8, blocks=np.array(blocks))
    meta = {"arm": arm, "period": period, "cutoff": str(st.cutoff), "train_rows": int(len(st.idx)), "queries": int(len(st.q_origins)),
            "blocks": list(blocks), "seconds": round(time.perf_counter() - t)}
    (out / f"{period}_part{tag}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--period", required=True, choices=[*K.PERIODS, *LEGACY])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    run(a.arm, a.period, a.out, tuple(a.blocks), a.threads)
