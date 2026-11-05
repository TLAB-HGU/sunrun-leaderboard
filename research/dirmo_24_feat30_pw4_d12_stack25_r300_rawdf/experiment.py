"""DirMO-24 feat30 4x@550 + delta leads 1-12 + level-free residual stack leads 25-72, 300 rounds on leads 25-72, on the raw-only df: leaderboard entry script for the 2026-10-31 F1/F2/F3 split on the team raw-only df (self-contained).

Data: the team df.parquet rebuilt from data-v2/raw (2026-11-02; data-v2/notebooks/raw_to_input_df.ipynb). Origins are
hourly; the newest input of origin O is the hour labelled O-1h; y is SWEPAM speed with status 0 only, never filled.
Gaps in the inputs are filled causally along the hourly origin timeline (jsy301 decision 2026-11-04):
  * solar-wind observations (ace_*, v_*, rec*, ...): last valid value, no limit (gaps are at most 30 h);
  * image-derived columns (suvi_*, strip_*, img_age_h): last valid value for at most 24 h, img_age_h grows by the
    hours carried; longer image gaps stay NaN (XGBoost missing branch);
  * ENLIL columns (NaN means 'no arrival'): not filled.
Targets are never filled: a missing hour gets weight 0 in the loss (masked squared error) and an origin with no
observed target in a block is not a training row. Rows of the query set are read without their y column.

Model: the team DirMO-24 feat30 recipe (three 24-lead vector-leaf XGBoost blocks, Top-30 groups = 21 constant +
24 x 9 lead-varying = 237 columns, 900 rounds, depth 3, lr 0.03), mean of seeds 42, 1, 2; prediction clip(1, 3000).
Loss weight: CONFIG hyperparameters.loss_weight (4x@550: weight 4 on observed y >= 550 km/s, else 1, normalised to
mean 1 over observed rows per lead).
Training rows per period: the df's fit_F1 / fit_F2 / fit_F3 flags (origin + 144 h <= period cutoff).
Optional parts, used only when CONFIG lists them (DirMO lab round RD):
  * extra_inputs: per-lead strips at the observed-speed longitude (v = v_m6; S1 equatorial logmed + dark, S2 + dark
    of the two other latitude bands, S3 + dark change vs the image 24 h earlier and dark one rotation (655 h) earlier),
    'vage' = hours since the last real v_lag0; same arithmetic as the team strips.py, from the df's 12 x 3 SUVI grid;
  * short_leads: a second block-0 model (target y - v_lag0 if 'delta'; weights on the speed itself) whose leads
    1..k replace the main model's;
  * stack: entry C's level-free residual model on leads first_lead..72 (DirMO lab round ND). The main model's
    recipe (first seed only) is cross-fitted on the period's training rows (K contiguous folds, 72 h purge on both
    sides); a LightGBM model learns y - out-of-fold forecast at leads >= first_lead from level-free inputs (forecast
    path shape, recurrence and issue-time constants, see stack_feats/shape_cols; rows without truth left out), and
    the main forecast at those leads moves by kappa x its prediction. The short model's leads replace leads 1..k after;
  * block_rounds: boosting rounds per main-model block (e.g. {"24": 300, "48": 300}; other blocks keep
    xgboost.n_estimators), used by the main model and by the stack's cross-fit alike (DirMO lab round TC2).

    python <this file> --df <team df.parquet | directory with train.parquet + query_inputs.parquet> \
        --out predictions.parquet
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

CONFIG = {'data': {'base': 'e2000fs_dirmo_24_feat30', 'source': 'team data-v2/input/df.parquet (raw-only rebuild 2026-11-02)', 'df_sha256': '33e451e5718722dcd8737c1d0ef56a68f3707df55a3e7af39885efaccfb9f27c', 'split_files_sha256': {'train.parquet': '880c6fa6cfb146e3585c96430b8eb622f4fc6b60ccddee724912f97a63759254', 'query_inputs.parquet': '8f6e3c3c9b729b250049e34aed5e18b54997610a7bd6c38950a474061b5ce763'}, 'split': '2026-10-31 F1/F2/F3', 'training_rows': 'df fit_F1/fit_F2/fit_F3 (origin + 144 h <= cutoff), hourly origins', 'gap_fill': 'inputs causal last value (observations unlimited, images <= 24 h, ENLIL none); y never filled', 'time_convention': 'newest input of origin O is the hour labelled O-1h; lead h target is hour O+h'}, 'hyperparameters': {'strategy': 'DirMO-24-Top30', 'constant_columns': ['v_lag0', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_speed_kms_last', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_by_gsm_nt_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'img_age_h', 'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area', 'suvi_t035_equatorial_area', 'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'enlil_n_pending', 'enlil_kp'], 'varying_columns': ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v650_dark', 'strip_v750_logmed'], 'block_input_columns': 237, 'xgboost': {'n_estimators': 900, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'n_jobs': 8}, 'seeds': [42, 1, 2], 'loss_weight': {'rule': '4 if observed y >= 550 km/s else 1; 0 on missing y; mean 1 per lead over observed rows', 'threshold_kms': 550.0, 'weight': 4.0}, 'base_score': 'block mean of the observed training targets (scalar)', 'short_leads': {'leads': 12, 'target': 'delta', 'extra_inputs': [], 'note': 'block-0 model on y - v_lag0 (same inputs, 4x@550 on the speed); its leads 1-12 replace the main'}, 'stack': {'first_lead': 25, 'folds': 4, 'kappa': 0.25, 'oof_seed': 'first of seeds', 'keep_columns': [1, 4, 10, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35], 'lightgbm': {'objective': 'regression', 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 7, 'min_child_samples': 2000, 'subsample': 0.7, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_lambda': 10.0, 'n_jobs': 8, 'verbosity': -1, 'random_state': 0}}, 'block_rounds': {'24': 300, '48': 300}}}
CONFIG_SHA256 = "9e57fadbfd5fda30b76a678901230f40a445ad926b0931bf48207d5c8c5be3de"
MODEL = "ds300_rawdf"
H = 72
IMAGE_CAP_H = 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # name -> query origins


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fill_rule(name):
    if name.startswith("enlil_"):
        return "none"
    if name.startswith(("suvi_", "strip_")) or name == "img_age_h":
        return "image"
    return "obs"


def ffill(a, limit=None):
    """Causal fill along axis 0 of a 2-D array; returns (filled, hours carried)."""
    n = a.shape[0]
    last = np.maximum.accumulate(np.where(np.isfinite(a), np.arange(n)[:, None], -1), axis=0)
    carried = np.where(last >= 0, np.arange(n)[:, None] - last, np.inf)
    out = np.take_along_axis(a, np.maximum(last, 0), axis=0)
    keep = (last >= 0) & (carried <= (np.inf if limit is None else limit))
    return np.where(keep, out, np.nan), np.where(keep, carried, np.nan)


def read(df_path, constant, varying):
    """(train rows with y, query rows without y) as DataFrames, after checking the input hash."""
    p = Path(df_path)
    data = CONFIG["data"]
    keys = ["origin", "h"]
    if p.is_dir():
        for f, want in data["split_files_sha256"].items():
            assert sha256(p / f) == want, f"{f} sha256 mismatch"
        tr = pd.read_parquet(p / "train.parquet", columns=keys + ["fit_F1", "fit_F2", "fit_F3", "y"] + constant + varying)
        q = pd.read_parquet(p / "query_inputs.parquet", columns=keys + ["period"] + constant + varying)
    else:
        assert sha256(p) == data["df_sha256"], "df.parquet sha256 mismatch"
        tr = pd.read_parquet(p, columns=keys + ["fit_F1", "fit_F2", "fit_F3", "y"] + constant + varying,
                             filters=[("set", "==", "train")])
        q = pd.read_parquet(p, columns=keys + ["period"] + constant + varying, filters=[("set", "==", "query")])
    assert "y" not in q.columns
    return tr, q


def grid(df, cols, origins):
    df = df.sort_values(["origin", "h"], kind="stable")
    o = pd.DatetimeIndex(df["origin"].to_numpy()[::H])
    assert len(df) == len(o) * H and (df["h"].to_numpy().reshape(-1, H) == np.arange(1, H + 1)).all()
    pos = origins.get_indexer(o)
    assert (pos >= 0).all()
    out = np.full((len(origins), H, len(cols)), np.nan, np.float64)
    out[pos] = df[list(cols)].to_numpy(np.float64).reshape(len(o), H, len(cols))
    return out, pos


def masked_objective(observed, weight_fn):
    """Weighted squared error with weight 0 on missing targets; weights mean 1 over observed rows per lead."""
    def obj(y_true, y_pred):
        target = y_true.reshape(y_pred.shape)
        w = weight_fn(target) * observed
        w = w / (w.sum(axis=0) / observed.sum(axis=0))
        return (y_pred - target) * w, w
    return obj


OMEGA, D_KM, CELL_W, HALF, ROT_H = 0.55, 147858620.7, 15.0, 5.0, 655
GRID = [f"suvi_lon{k:02d}_lat{b}_{s}" for b in (0, 1, 2) for s in ("log_median", "dark_fraction") for k in range(12)]


def strip(lm, dk, v, h, age):
    """Team strips.py arithmetic at speed v per origin: lambda = 0.55 (1 AU / v - h - age), +-5 deg over 15 deg cells."""
    edges = -90.0 + CELL_W * np.arange(13)
    tau = D_KM / v / 3600.0
    lam = OMEGA * (tau[:, None] - (np.broadcast_to(h, (len(v), H)) + age[:, None]))
    W = np.clip(np.minimum(lam[..., None] + HALF, edges[1:]) - np.maximum(lam[..., None] - HALF, edges[:-1]),
                0, None) / (2 * HALF)
    ws = W.sum(-1)
    f = lambda g: np.where(ws > 0, (W * g[:, None, :]).sum(-1) / np.maximum(ws, 1e-12), np.nan)  # noqa: E731
    return f(lm), f(dk)


def shift(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def extra_inputs(tokens, col, v_age):
    """Per-origin extra inputs (lab/rawdf_extra.py): 'vage' (n, 1) and strips S1 / S2 / S3 (n, 72, k) at v = v_m6."""
    hh = np.arange(1, H + 1, dtype=float)
    g = {(b, st): np.column_stack([col(f"suvi_lon{k:02d}_lat{b}_{st}") for k in range(12)])
         for b in (0, 1, 2) for st in ("log_median", "dark_fraction")}
    v, age = col("v_m6"), col("img_age_h")
    lm1, dk1 = strip(g[1, "log_median"], g[1, "dark_fraction"], v, hh, age)
    out = {}
    for t in tokens:
        if t == "vage":
            out[t] = v_age[:, None]
        elif t == "S1":
            out[t] = np.stack([lm1, dk1], -1)
        elif t == "S2":
            dk0 = strip(g[0, "log_median"], g[0, "dark_fraction"], v, hh, age)[1]
            dk2 = strip(g[2, "log_median"], g[2, "dark_fraction"], v, hh, age)[1]
            out[t] = np.stack([lm1, dk1, dk0, dk2], -1)
        elif t == "S3":
            d24 = strip(shift(g[1, "log_median"], 24), shift(g[1, "dark_fraction"], 24), v, hh + 24, shift(age, 24))[1]
            drot = strip(shift(g[1, "log_median"], ROT_H), shift(g[1, "dark_fraction"], ROT_H), v, hh,
                         shift(age, ROT_H))[1]
            out[t] = np.stack([lm1, dk1, dk1 - d24, drot], -1)
        else:
            raise ValueError(t)
    return {t: a.astype(np.float32) for t, a in out.items()}


def design(Cm, Em, ex, tokens, s0):
    parts = [Cm, Em[:, s0:s0 + 24, :].reshape(len(Cm), -1)]
    for t in tokens:
        a = ex[t]
        parts.append(a if a.ndim == 2 else a[:, s0:s0 + 24, :].reshape(len(a), -1))
    return np.hstack(parts).astype(np.float32)


def fit_predict(X, Y, Xe, params, seeds, thr, wt, offset_fit=0., offset_ev=0.):
    """Masked weighted fit of one 24-lead block (weights on the speed itself), mean of seeds; returns pred, models."""
    rows = np.flatnonzero(np.isfinite(Y).any(1))
    Y = Y[rows]
    T = Y - (offset_fit[rows] if np.ndim(offset_fit) else offset_fit)
    observed = np.isfinite(T)  # a delta target needs the speed at the origin too (missing only on the first origin)
    w_speed = np.where(observed, Y, -np.inf)
    pred, models = np.zeros((len(Xe), Y.shape[1])), []
    for seed in seeds:
        pp = dict(params, random_state=seed, base_score=float(np.nanmean(T)),
                  objective=masked_objective(observed, lambda t, ws=w_speed: np.where(ws >= thr, wt, 1.)))
        m = xgb.XGBRegressor(**pp).fit(X[rows], np.where(observed, T, 0.), verbose=False)
        pred += np.clip(m.predict(Xe) + offset_ev, 1., 3000.) / len(seeds)
        models.append(m)
    return pred, models, len(rows)


ROW_CONSTANTS = ["v_lag0", "v_m24", "v_m72", "v_sd24", "speed_d24", "ace_density_cm3_last", "ace_temperature_k_last",
                 "ace_bt_nt_last", "suvi_t035_equatorial_area", "suvi_t035_meridian30_equatorial_area",
                 "suvi_t055_equatorial_area", "enlil_kp", "enlil_next_arr_h", "enlil_n_pending"]


def stack_feats(p, idx, st):
    """36 long-format columns per (origin, lead) of forecast p on rows idx (DirMO lab stage2 order): forecast path
    stats, recurrence stats, 24 h range and 6 h change of v_lag0, season, issue-time constants, lead, forecast,
    path differences, recurrence, forecast - its max."""
    rec, o = st["rec"][idx], st["origins"][idx]
    rows = [p.max(1), p.argmax(1), p.min(1), p.mean(1), p.std(1), p[:, 0], p[:, 23], p[:, 47], p[:, 71],
            np.nanmax(rec, 1), np.nanargmax(np.nan_to_num(rec, nan=-1), 1), np.nanmax(rec, 1) - np.nanmin(rec, 1),
            st["range24"][idx], st["d6"][idx], np.sin(2 * np.pi * o.dayofyear / 365.25),
            np.cos(2 * np.pi * o.dayofyear / 365.25)]
    F = np.column_stack(rows + [st["row_constants"][idx, j] for j in range(len(ROW_CONSTANTS))]).astype(np.float32)
    ph = p.ravel()
    prev = np.concatenate([p[:, :1].repeat(6, 1), p[:, :-6]], 1).ravel()
    return np.column_stack([np.repeat(F, H, 0), np.tile(np.arange(1, H + 1), len(p)), ph, ph - prev,
                            ph - np.repeat(p[:, 0], H), rec.ravel(), ph - np.repeat(p.max(1), H)]).astype(np.float32)


def shape_cols(X, keep):
    """Level-free columns plus level differences: recurrence - forecast, speed now - forecast, rec max - fc max."""
    return np.column_stack([X[:, keep], X[:, 34] - X[:, 31], X[:, 16] - X[:, 31], X[:, 9] - X[:, 0]])


def crossfit(Xs, y, fit, origins, pblock, seed, thr, wt, k):
    """Out-of-fold main-model forecasts on rows fit: K contiguous folds, each predicted by a model fit on the other
    rows whose 72 h targets end before the fold or whose origin is more than 72 h after it."""
    o = origins[fit]
    edges = np.linspace(0, len(fit), k + 1).astype(int)
    oof = np.full((len(fit), H), np.nan)
    for f in range(k):
        lo, hi = edges[f], edges[f + 1]
        sub = fit[np.flatnonzero((o + pd.Timedelta(hours=H) < o[lo]) | (o > o[hi - 1] + pd.Timedelta(hours=H)))]
        for s0 in (0, 24, 48):
            oof[lo:hi, s0:s0 + 24] = fit_predict(Xs[s0][sub], y[sub, s0:s0 + 24], Xs[s0][fit[lo:hi]], pblock[s0],
                                                 [seed], thr, wt)[0]
    return oof


def stack_correction(pred, ev, fit, d, pblock, seeds, thr, wt, cfg):
    """kappa x residual-model correction for the rows ev of the main forecast pred (leads >= first_lead only)."""
    import lightgbm as lgb
    st, y, L0 = d["stack"], d["y"], cfg["first_lead"] - 1
    oof = crossfit(d["Xs"], y, fit, d["origins"], pblock, seeds[0], thr, wt, cfg["folds"])
    assert np.isfinite(oof).all(), "out-of-fold forecasts must cover every training row"
    X = shape_cols(stack_feats(oof, fit, st), cfg["keep_columns"])
    R = (y[fit] - oof).ravel()
    rows = np.tile(np.arange(H) >= L0, len(fit)) & np.isfinite(R)
    m = lgb.LGBMRegressor(**cfg["lightgbm"]).fit(X[rows], R[rows])
    Xe = shape_cols(stack_feats(pred, ev, st), cfg["keep_columns"])
    corr = np.zeros((len(ev), H))
    corr[:, L0:] = cfg["kappa"] * m.predict(Xe).reshape(len(ev), H)[:, L0:]
    return corr, m, Xe, int(rows.sum())


def prepare(df_path, hp):
    """Read, check and fill the df; returns the design matrices of every origin of the hourly timeline."""
    constant, varying = hp["constant_columns"], hp["varying_columns"]
    tokens = list(hp.get("extra_inputs", []))  # appended to every block of the main model
    short = hp.get("short_leads")  # optional block-0 model whose leads 1..k replace the main model's
    need = set(tokens) | set(short["extra_inputs"] if short else [])
    stack = hp.get("stack")  # optional residual model on leads >= first_lead
    aux = (GRID + ["v_m6", "img_age_h", "v_lag0"]) if need else []
    aux += ROW_CONSTANTS if stack else []
    read_c = constant + [c for c in dict.fromkeys(aux) if c not in constant]

    tr, q = read(df_path, read_c, varying)
    origins = pd.DatetimeIndex(sorted(set(tr["origin"]) | set(q["origin"])))
    assert (np.diff(origins.asi8) == 3600 * 10 ** 9).all(), "origin timeline must be hourly without gaps"
    cols = read_c + varying
    Xt, pt = grid(tr, cols + ["y"], origins)
    Xq, pq_ = grid(q, cols, origins)
    both = np.intersect1d(pt, pq_)
    same = (Xt[both, :, :-1] == Xq[both]) | (np.isnan(Xt[both, :, :-1]) & np.isnan(Xq[both]))
    assert same.all(), "train and query inputs differ on shared origins"
    X = np.where(np.isnan(Xt[..., :-1]), Xq, Xt[..., :-1])
    y = Xt[..., -1]
    is_train = np.zeros(len(origins), bool)
    is_train[pt] = True
    Call, Em = X[:, 0, :len(read_c)].copy(), X[:, :, len(read_c):].copy()
    v_age = ffill(Call[:, read_c.index("v_lag0"), None])[1][:, 0] if "vage" in need else None
    for j, c in enumerate(read_c):
        if fill_rule(c) != "none":
            f, carried = ffill(Call[:, j:j + 1], IMAGE_CAP_H if fill_rule(c) == "image" else None)
            Call[:, j] = (f + carried if c == "img_age_h" else f)[:, 0]
    for j, c in enumerate(varying):
        if fill_rule(c) != "none":
            Em[:, :, j] = ffill(Em[:, :, j], IMAGE_CAP_H if fill_rule(c) == "image" else None)[0]
    col = lambda c: Call[:, read_c.index(c)].astype(np.float64)  # noqa: E731
    ex = extra_inputs(sorted(need), col, v_age) if need else {}
    Cm = Call[:, [read_c.index(c) for c in constant]]
    Xs = {s0: design(Cm, Em, ex, tokens, s0) for s0 in (0, 24, 48)}
    Xshort = design(Cm, Em, ex, short["extra_inputs"], 0) if short else None
    v0 = col("v_lag0") if short and short["target"] == "delta" else None
    fit_flags = {p: tr.loc[tr["h"] == 1].set_index("origin")[f"fit_{p}"].astype(bool)
                 .reindex(origins, fill_value=False).to_numpy() for p in PERIODS}
    period = q.loc[q["h"] == 1].set_index("origin")["period"].astype(str).reindex(origins).to_numpy()
    st = None
    if stack:
        v = col("v_lag0")
        win = np.lib.stride_tricks.sliding_window_view(np.r_[np.full(23, np.nan), v], 24)
        with np.errstate(all="ignore"):
            st = dict(origins=origins, rec=Em[:, :, varying.index("rec_mean")].astype(np.float64),
                      range24=np.nanmax(win, 1) - np.nanmin(win, 1), d6=v - np.r_[np.full(6, np.nan), v[:-6]],
                      row_constants=np.column_stack([col(c) for c in ROW_CONSTANTS]))
    return dict(origins=origins, Xs=Xs, Xshort=Xshort, v0=v0, y=y, is_train=is_train, fit_flags=fit_flags,
                period=period, stack=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True)
    ap.add_argument("--out", default="predictions.parquet")
    ap.add_argument("--smoke-rounds", type=int, help="testing only: fewer boosting rounds (not for submission)")
    a = ap.parse_args()
    t0 = time.time()
    hp = CONFIG["hyperparameters"]
    seeds, short = hp["seeds"], hp.get("short_leads")
    params = dict(hp["xgboost"])
    rounds = {s0: hp.get("block_rounds", {}).get(str(s0), params["n_estimators"]) for s0 in (0, 24, 48)}
    if a.smoke_rounds:
        params["n_estimators"] = a.smoke_rounds
        rounds = {s0: a.smoke_rounds for s0 in rounds}
    pblock = {s0: dict(params, n_estimators=r) for s0, r in rounds.items()}  # main-model blocks (short: params)
    thr, wt = hp["loss_weight"]["threshold_kms"], hp["loss_weight"]["weight"]
    d = prepare(a.df, hp)
    origins, Xs, Xshort, v0, y, is_train = (d[k] for k in ("origins", "Xs", "Xshort", "v0", "y", "is_train"))
    fit_flags, period = d["fit_flags"], d["period"]

    oo, preds, info, timing = [], [], {}, []
    for p, n_q in PERIODS.items():
        ev = np.flatnonzero(period == p)
        fit = np.flatnonzero(fit_flags[p] & is_train)
        assert len(ev) == n_q and is_train[fit].all(), (p, len(ev))
        last_target = origins[fit].max() + pd.Timedelta(hours=H)
        assert last_target < origins[ev].min(), (p, last_target)  # training targets end before the period's origins
        pred, used, n_rows = np.zeros((len(ev), H)), [], {}
        for s0 in (0, 24, 48):
            pb, ms, n_rows[s0] = fit_predict(Xs[s0][fit], y[fit, s0:s0 + 24], Xs[s0][ev], pblock[s0], seeds, thr, wt)
            pred[:, s0:s0 + 24] = pb
            used.append((Xs[s0], ms))
        stacked = None
        if hp.get("stack"):  # before the short model: the residual inputs are the main model's forecast path
            corr, sm, sXe, n_rows["stack_residual_rows"] = stack_correction(pred, ev, fit, d, pblock, seeds, thr, wt,
                                                                            hp["stack"])
            pred = np.clip(pred + corr, 1., 3000.)
            stacked = (sm, sXe)
        if short:
            k = short["leads"]
            assert 1 <= k <= 24
            off_f, off_e = (v0[fit, None], v0[ev, None]) if v0 is not None else (0., 0.)
            ps, ms, n_rows["short"] = fit_predict(Xshort[fit], y[fit, :24], Xshort[ev], params, seeds, thr, wt,
                                                  off_f, off_e)
            pred[:, :k] = ps[:, :k]
            used.append((Xshort, ms))
        assert np.isfinite(pred).all()
        for i in range(0, len(ev), 29):  # post-fit predict time of one 72 h forecast (all models used)
            ts = time.perf_counter()
            for Xm, ms in used:
                for m in ms:
                    m.predict(Xm[ev[i:i + 1]])
            if stacked is not None:
                stacked[0].predict(stacked[1][i * H:(i + 1) * H])
            timing.append(time.perf_counter() - ts)
        info[p] = dict(n_fit_origins=int(len(fit)), n_rows_per_block=n_rows, last_training_target_utc=str(last_target),
                       observed_target_fraction=round(float(np.isfinite(y[fit]).mean()), 4))
        print(f"[{time.time() - t0:6.0f}s] {p}: {len(ev)} origins, {json.dumps(info[p])}", flush=True)
        preds.append(pred)
        oo.append(origins[ev])
    o = oo[0].append(oo[1]).append(oo[2])
    pred = np.vstack(preds)
    pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                  "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()}).to_parquet(
        a.out, index=False)
    print(json.dumps(dict(rows=len(o) * H, out=a.out, model=MODEL, config_sha256=CONFIG_SHA256, **info,
                          inference_seconds_per_fold=float(np.mean(timing)), elapsed_s=round(time.time() - t0),
                          xgboost=xgb.__version__)))


if __name__ == "__main__":
    main()
