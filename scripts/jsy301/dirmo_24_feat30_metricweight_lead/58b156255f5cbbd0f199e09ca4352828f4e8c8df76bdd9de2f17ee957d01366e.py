"""DirMO-24 feat30 with sample and lead loss weights derived from the board score: leaderboard entry script for the 2026-10-31 F1/F2/F3 split (self-contained).

Model: the team DirMO-24 feat30 recipe (seongeun e2000fs_dirmo_24_feat30: three 24-lead vector-leaf XGBoost blocks,
Top-30 groups, 237 columns per block, 900 rounds, depth 3, lr 0.03) with one change, the training loss weight.
No hand-set threshold or multiplier (the 4x weight above 550 km/s of pointweight4 does not appear).

Loss weight, derived from the board score S = mean_l sqrt(RMSE_l sqrt(TIMSE_l)) with TIMSE_l = sum_k dq_k
tailMSE_l(y >= q_k) / range over the 100 percentile thresholds of lead l:
  TIMSE_l = sum_i e_il^2 c_l(y_il),  c_l(y) = sum over thresholds q_k <= y of dq_k / (N_k range)
  d log S_l / d e_il  ~  e_il w_il,   w_il = a_l + (1 - a_l) n c_l(y_il),   a_l = TIMSE_l / (MSE_l + TIMSE_l)
so a weighted squared error with w (mean 1 per lead) follows the gradient of each lead's S.
The _lead variant also scales lead l's weights by S_l (1/MSE_l + 1/TIMSE_l), normalised to mean 1 within each
24-lead block: how much one unit of weighted squared error at lead l moves the mean S.
Percentiles c_l come from the period's training targets. MSE_l / TIMSE_l come from 4-fold out-of-fold forecasts of
the unweighted recipe on the same training rows (contiguous folds, 72 h purge; seed 42), so nothing after the
period's cutoff is used. Each period's model is the mean of three seeds (42, 1, 2). Prediction = clip(1, 3000).
Lab check before this entry (DirMO lab round MW, windows 2023-07..2025-12, LB metric on raw truth, 3 seeds):
pooled S vs the 4x-above-550 weight -1.17 % (B1) / -1.28 % (B2), better in 5 of 5 windows.

Periods and training cutoffs (board rule, period start - 72 h): F1 targets 2026-01..03, models fit on training origins
whose 72 h targets end before 2025-12-29; F2 (2026-04..06) before 2026-03-29; F3 (2026-07..09) before 2026-06-29.
Training rows: team train.npz (every third hour to 2026-05-28T21) plus official.npz rows on the same 3-hour grid
(2026-05-31T23 on, needed for F3 only).

Inputs per evaluation origin (6,315 hourly origins), same as the jsy301 dirmo_24_feat30_delta12 entry: the team
bundle row (f1.npz / official.npz hourly, else train.npz) when it has one; origins without a team row
(2025-12-31T23..2026-02-28T22 off the 3-hour grid, 2026-05-29T00..05-31T22) are rebuilt causally from the newest
team row r <= t: image-free columns recomputed with the team formulas, SUVI image columns from row r (delta <= 2 h
unchanged, else aged by delta with the team stale rule).
Sources: team cache reference/pointweight4_cache (tlabtlab/sunrun-repro-data, rev d73e13da), collector
spaceweather/solar.csv (raw ACE), NASA DONKI WSAEnlilSimulations monthly JSON. Hashes in CONFIG.

    python <this file> --cache <dir: train.npz f1.npz official.npz names.json speed.parquet> \
        --solar-csv <solar.csv> --donki <dir with WSAEnlilSimulations_YYYY-MM.json> --out predictions.parquet
"""
import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CONFIG = {'data': {'base': 'e2000fs_dirmo_24_feat30', 'feature_count': 30, 'block_input_columns': 237, 'split': '2026-10-31 F1/F2/F3', 'f1_training_target_end_utc': '2025-12-29T00:00:00Z', 'f2_training_target_end_utc': '2026-03-29T00:00:00Z', 'f3_training_target_end_utc': '2026-06-29T00:00:00Z', 'training_rows': 'train.npz (3-hourly) + official.npz rows on the 3-hour grid', 'cache_sha256': {'train.npz': '5f9c446b1beb7c8fb4290236bfbbde08f32153433cfafadc3c02fd189406a8a9', 'f1.npz': 'f126160c2a8d5bb12df376781a4d5712cdae0708385c717cc82419556629db1d', 'official.npz': '9878a560b0d81ddec35f65184afe6a446f425750cce2074290f320d6613ec6ae', 'names.json': '3b25e892c0093215c90f0f3e579a72a0c6cb314a38c694a4c38e5c58744b14ce', 'speed.parquet': '96f6e22e1643d79c5dd49f04825128dc25c3b8985a5415fa5828fd7db3012877'}, 'inputs_sha256': {'solar.csv': '7d8a4e32e4a0b5abeb17916c11704d1a801da0d8315aa936f300f8d9f43f289f', 'donki_enlil_json_concat': '47312aaa23e756839cce43d96bb05b37ca155982628f29e0a94ebd8472067dbe'}, 'rebuilt_rows': 'origins without a team row: image-free columns recomputed with the team formulas; image columns from the newest team row (delta <= 2 h unchanged, else aged + stale rule)'}, 'hyperparameters': {'strategy': 'DirMO-24-Top30, metric-derived loss weights', 'feature_groups': ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last', 'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26', 'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last', 'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed', 'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark', 'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28'], 'xgboost': {'n_estimators': 900, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 20, 'reg_lambda': 5, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': 'cpu', 'multi_strategy': 'multi_output_tree', 'n_jobs': 6}, 'seeds': [42, 1, 2], 'loss_weight': {'rule': 'w_il = a_l + (1 - a_l) n c_l(y_il), a_l = TIMSE_l / (MSE_l + TIMSE_l)', 'lead_factor': 'S_l (1/MSE_l + 1/TIMSE_l), mean 1 per block', 'oof_folds': 4, 'oof_seed': 42, 'oof_purge_h': 72}, 'base_score': 'block mean of the training targets (scalar)'}}
CONFIG_SHA256 = "58b156255f5cbbd0f199e09ca4352828f4e8c8df76bdd9de2f17ee957d01366e"
MODEL = "metricweight_lead"
WEIGHT_MODE = {"metricweight": "sample", "metricweight_lead": "sample+lead"}
H = 72
STALE_H = 6.0
HOLD_H = 2
PERIODS = {"f1": ("2025-12-31T23:00:00Z", "2026-03-28T23:00:00Z"),
           "f2": ("2026-03-31T23:00:00Z", "2026-06-27T23:00:00Z"),
           "f3": ("2026-06-30T23:00:00Z", "2026-09-26T23:00:00Z")}
ACE_COLS = ["ace_speed_kms", "ace_density_cm3", "ace_temperature_k", "ace_bt_nt", "ace_bz_gsm_nt", "ace_by_gsm_nt"]
ACE_LOGS = ["ace_epam_e_channel1", "ace_epam_e_channel2", "ace_epam_p_channel1", "ace_epam_p_channel3",
            "ace_epam_p_channel5", "ace_sis_gt10_flux"]
ENLIL = ["enlil_next_arr_h", "enlil_since_arr_h", "enlil_n_pending", "enlil_kp", "enlil_glancing"]


# ---------------------------------------------------------------- inputs
def utc(x):
    return pd.DatetimeIndex(pd.to_datetime(x, utc=True))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_raw_ace(path):
    raw = pd.read_csv(path, usecols=["timestamp_utc"] + ACE_COLS + ACE_LOGS)
    raw.index = utc(raw.pop("timestamp_utc")).floor("h")
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def load_enlil(donki):
    files = sorted(glob.glob(str(Path(donki) / "WSAEnlilSimulations_*.json")))
    rows = []
    for f in files:
        for s in json.load(open(f)):
            if s.get("estimatedShockArrivalTime"):
                rows.append((pd.Timestamp(s["modelCompletionTime"]), pd.Timestamp(s["estimatedShockArrivalTime"]),
                             max(s.get("kp_90") or 0, s.get("kp_135") or 0, s.get("kp_180") or 0),
                             bool(s.get("isEarthGB"))))
    ev = pd.DataFrame(rows, columns=["done", "arrival", "kp", "glancing"]).drop_duplicates().sort_values("done")
    for c in ("done", "arrival"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    return ev.reset_index(drop=True), files


def speed_features(origins, s):
    f = {f"v_lag{l}": s.shift(l) for l in (0, 1, 3, 6, 12, 24, 48, 72, 120)}
    for w in (6, 24, 72):
        f[f"v_m{w}"] = s.rolling(w, min_periods=1).mean()
        f[f"v_sd{w}"] = s.rolling(w, min_periods=2).std()
    return pd.DataFrame(f).reindex(origins)


def ace_features(origins, raw):
    x = raw[ACE_COLS].astype(float)
    x[ACE_LOGS] = np.log10(raw[ACE_LOGS].astype(float).clip(lower=1e-3))
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="h", tz="UTC"))
    f = {}
    for c in x.columns:  # rolling windows end at the origin hour: causal
        f[f"{c}_last"] = x[c].ffill(limit=6)
        f[f"{c}_m6"] = x[c].rolling(6, min_periods=1).mean()
        f[f"{c}_m24"] = x[c].rolling(24, min_periods=1).mean()
    f["speed_d24"] = f["ace_speed_kms_m6"] - f["ace_speed_kms_m6"].shift(24)
    f["epam_p1_d6"] = f["ace_epam_p_channel1_m6"] - f["ace_epam_p_channel1_m6"].shift(6)
    return pd.DataFrame(f).reindex(origins)


def enlil_features(origins, ev):
    out = np.full((len(origins), 5), np.nan)
    for i, o in enumerate(origins):  # runs completed <= origin, within 7 days
        e = ev.iloc[: ev["done"].searchsorted(o, side="right")]
        e = e[e["done"] >= o - pd.Timedelta("7D")]
        dt = (e["arrival"] - o).dt.total_seconds() / 3600
        fut, past = dt[dt >= -6], dt[(dt < -6) & (dt >= -96)]
        out[i] = [fut.min() if len(fut) else np.nan, -past.max() if len(past) else np.nan,
                  len(fut), e.loc[fut.index, "kp"].max() if len(fut) else 0,
                  float(e.loc[fut.index, "glancing"].all()) if len(fut) else np.nan]
    return pd.DataFrame(out, index=origins, columns=ENLIL)


def recurrence(origins, s):
    tt = origins.values[:, None] + np.arange(1, H + 1).astype("timedelta64[h]")[None, :]
    out = {}
    for d in (26, 27, 28):
        t = pd.DatetimeIndex((tt - np.timedelta64(d, "D")).ravel(), tz="UTC")
        out[f"rec{d}"] = s.reindex(t).to_numpy().reshape(len(origins), H)
    stack3 = np.stack([out["rec26"], out["rec27"], out["rec28"]])
    cnt = np.isfinite(stack3).sum(0)
    out["rec_mean"] = np.where(cnt > 0, np.nansum(stack3, 0) / np.maximum(cnt, 1), np.nan)
    return out


def past_speed(origins, s):
    t = origins.values[:, None] - np.arange(23, -1, -1).astype("timedelta64[h]")[None, :]
    return s.reindex(pd.DatetimeIndex(t.ravel(), tz="UTC")).to_numpy().reshape(len(origins), 24)


def rebuild_rows(targets, base_o, base_C, base_E, names, s, raw, ev):
    """Rows for `targets`: the base row if it exists, else rebuilt from the newest base row <= t (see docstring)."""
    j = np.searchsorted(base_o.asi8, targets.asi8, side="right") - 1
    assert (j >= 0).all(), "target before the first team row"
    delta = ((targets.asi8 - base_o.asi8[j]) // 3_600_000_000_000).astype(int)
    C, E = base_C[j].astype(np.float64).copy(), base_E[j].astype(np.float64).copy()
    past = past_speed(targets, s)
    idx = np.flatnonzero(delta > 0)
    if len(idx):
        cn, vn = names["constant"], names["varying"]
        o = targets[idx]
        c = speed_features(o, s).join(ace_features(o, raw)).join(enlil_features(o, ev))
        c = c[[n for n in cn if n in c.columns]]
        order = np.argsort(np.concatenate([base_o.asi8, o.asi8]), kind="stable")
        for col in c.columns:  # forward fill along origins, as the team bundle does
            k = cn.index(col)
            v = np.concatenate([base_C[:, k].astype(np.float64), c[col].to_numpy()])
            filled = np.empty_like(v)
            filled[order] = pd.Series(v[order]).ffill().to_numpy()
            C[idx, k] = filled[len(base_o):]
        e = recurrence(o, s)
        e["enlil_dt_target"] = np.arange(1, H + 1)[None, :] - C[idx, cn.index("enlil_next_arr_h")][:, None]
        for col, v in e.items():
            E[idx, :, vn.index(col)] = v
        ia, st = cn.index("img_age_h"), cn.index("img_stale")
        strips = [vn.index(n) for n in vn if n.startswith("strip_")]
        for i in idx:
            d = delta[i]
            if d <= HOLD_H:
                continue
            C[i, ia] += d
            lead = np.minimum(np.arange(H) + d, H - 1)
            E[i][:, strips] = E[i][lead][:, strips]
            if C[i, ia] > STALE_H:
                C[i, st] = 1.0
                E[i][:, strips] = np.nan
    return dict(C=C.astype(np.float32), E=E.astype(np.float32), past=past.astype(np.float32), origins=targets), delta


# ---------------------------------------------------------------- recipe
def design(b, names, start, groups):
    ci = [i for i, g in enumerate(names["constant"]) if g in groups]
    ei = [i for i, g in enumerate(names["varying"]) if g in groups]
    n = len(b["C"])
    return np.hstack([b["C"][:, ci], b["E"][:, start:start + 24, :][:, :, ei].reshape(n, -1)]).astype(np.float32)


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, H + 1).astype("timedelta64[h]")[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz="UTC")).to_numpy().reshape(-1, H)


def weighted_objective(w):
    """Weighted squared error: grad = w (pred - y), hess = w (w is n x 24, rows in training order)."""
    def obj(y_true, y_pred):
        ww = w.reshape(y_pred.shape)
        return (y_pred - y_true.reshape(y_pred.shape)) * ww, ww
    return obj


def fit_block(X, Y, params, seed, w=None):
    p = dict(params, random_state=seed)
    if w is not None:
        p["objective"] = weighted_objective(w)
        p["base_score"] = float(Y.mean())  # block mean (xgboost 2.1 takes a scalar base_score)
    return xgb.XGBRegressor(**p).fit(X, Y, verbose=False)


def crossfit(Xs, y, fit, origins, params, k):
    """Out-of-fold forecasts of the unweighted recipe on rows `fit`: K contiguous folds, each predicted by a model fit
    on the other rows of `fit` whose targets end before the fold or whose origin is more than 72 h after it."""
    o = origins[fit]
    edges = np.linspace(0, len(fit), k + 1).astype(int)
    oof = np.full((len(fit), H), np.nan)
    for f in range(k):
        lo, hi = edges[f], edges[f + 1]
        sub = np.flatnonzero((o + pd.Timedelta(hours=H) < o[lo]) | (o > o[hi - 1] + pd.Timedelta(hours=H)))
        for s0 in (0, 24, 48):
            m = fit_block(Xs[s0][fit[sub]], y[fit[sub], s0:s0 + 24], params, 42)
            oof[lo:hi, s0:s0 + 24] = np.clip(m.predict(Xs[s0][fit[lo:hi]]), 1., 3000.)
    return oof


def tail_c(y):
    """c(y_i) for one lead: TIMSE = sum_i e_i^2 c(y_i) (tails {y >= q_k} as in the board scorer); sum_i c = 1."""
    q = np.quantile(y, np.arange(101) / 100, method="linear")
    pos = np.searchsorted(np.sort(y), q[:-1], side="left")
    per_k = np.diff(q) / ((len(y) - pos) * (q[-1] - q[0]))
    return np.cumsum(per_k)[np.searchsorted(q[:-1], y, side="right") - 1]


def metric_weights(Y, oof, mode):
    """(n, 72) weights a_l + (1 - a_l) n c_l(y); 'sample+lead' also scales lead l by S_l (1/MSE_l + 1/TIMSE_l),
    normalised to mean 1 within each 24-lead block."""
    c = np.column_stack([tail_c(Y[:, t]) for t in range(H)])
    e2 = (oof - Y) ** 2
    mse, timse = e2.mean(0), (e2 * c).sum(0)
    alpha = timse / (mse + timse)
    w = alpha[None] + (1 - alpha[None]) * len(Y) * c
    lead = np.sqrt(np.sqrt(mse) * np.sqrt(timse)) * (1 / mse + 1 / timse)
    lf = lead.reshape(3, 24)
    lf = (lf / lf.mean(1, keepdims=True)).ravel()
    if mode == "sample+lead":
        w = w * lf[None]
    else:
        assert mode == "sample", mode
    return w, dict(alpha_min=float(alpha.min()), alpha_max=float(alpha.max()), w_max=float(w.max()),
                   lead_factor_1=float(lf[0]), lead_factor_24=float(lf[23]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--solar-csv", required=True)
    ap.add_argument("--donki", required=True)
    ap.add_argument("--out", default="predictions.parquet")
    ap.add_argument("--sibling-out", help="also write the sibling model (metricweight <-> metricweight_lead) to this "
                                          "path; shares the out-of-fold step, same predictions as the sibling script")
    ap.add_argument("--smoke-rounds", type=int, help="testing only: fewer boosting rounds (not for submission)")
    a = ap.parse_args()
    t0 = time.time()
    hp, data = CONFIG["hyperparameters"], CONFIG["data"]
    cache = Path(a.cache)
    for f, want in data["cache_sha256"].items():
        assert sha256(cache / f) == want, f"{f} sha256 mismatch"
    assert sha256(a.solar_csv) == data["inputs_sha256"]["solar.csv"], "solar.csv sha256 mismatch"
    ev, files = load_enlil(a.donki)
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    assert h.hexdigest() == data["inputs_sha256"]["donki_enlil_json_concat"], "DONKI files sha256 mismatch"
    names = json.loads((cache / "names.json").read_text())
    groups, seeds, folds = hp["feature_groups"], hp["seeds"], hp["loss_weight"]["oof_folds"]
    params = dict(hp["xgboost"])
    if a.smoke_rounds:
        params["n_estimators"] = a.smoke_rounds
    models = [MODEL] + ([m for m in WEIGHT_MODE if m != MODEL] if a.sibling_out else [])
    outs = {MODEL: a.out, **({models[1]: a.sibling_out} if a.sibling_out else {})}
    raw = load_raw_ace(a.solar_csv)
    s = pd.read_parquet(cache / "speed.parquet")["speed"]
    s.index = utc(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC"))
    b = {k: dict(np.load(cache / f"{k}.npz")) for k in ("train", "f1", "official")}
    for k in b:
        b[k]["origins"] = utc(b[k]["origins"])

    # team rows on one time axis (hourly bundles win on shared origins)
    o_all = b["train"]["origins"].append(b["f1"]["origins"]).append(b["official"]["origins"])
    C_all = np.vstack([b[k]["C"] for k in ("train", "f1", "official")])
    E_all = np.vstack([b[k]["E"] for k in ("train", "f1", "official")])
    keep = ~o_all.duplicated(keep="last")
    order = np.argsort(o_all[keep].asi8)
    base_o, base_C, base_E = o_all[keep][order], C_all[keep][order], E_all[keep][order]

    # training pool: train.npz + official.npz rows on the 3-hour grid
    off = b["official"]
    on3 = (off["origins"].hour % 3 == 0) & (off["origins"] > b["train"]["origins"].max())
    tr = {k: np.concatenate([b["train"][k], off[k][on3]]) for k in ("C", "E", "past")}
    tr["origins"] = b["train"]["origins"].append(off["origins"][on3])
    origins = tr["origins"]
    y = targets(s, origins)
    Xs = {s0: design(tr, names, s0, groups) for s0 in (0, 24, 48)}
    cuts = {p: pd.Timestamp(data[f"{p}_training_target_end_utc"]) for p in PERIODS}

    oo, preds, info, timing = [], {m: [] for m in models}, {}, []
    for p, (lo, hi) in PERIODS.items():
        ev_o = pd.date_range(lo, hi, freq="h", tz="UTC")
        g, delta = rebuild_rows(ev_o, base_o, base_C, base_E, names, s, raw, ev)
        Xt = {s0: design(g, names, s0, groups) for s0 in (0, 24, 48)}
        fit = np.flatnonzero(origins + pd.Timedelta(hours=H) < cuts[p])
        assert np.isfinite(y[fit]).all() and (origins[fit] + pd.Timedelta(hours=H) < cuts[p]).all()
        oof = crossfit(Xs, y, fit, origins, params, folds)
        print(f"[{time.time() - t0:6.0f}s] {p}: {len(ev_o)} origins ({int((delta > 0).sum())} rebuilt, max delta "
              f"{int(delta.max())} h); out-of-fold forecasts on {len(fit)} training origins", flush=True)
        info[p] = dict(n_fit=int(len(fit)), n_rebuilt=int((delta > 0).sum()), max_delta_h=int(delta.max()))
        for m in models:
            w, wi = metric_weights(y[fit], oof, WEIGHT_MODE[m])
            pred, last = np.zeros((len(ev_o), H)), {}
            for s0 in (0, 24, 48):
                for seed in seeds:
                    mdl = fit_block(Xs[s0][fit], y[fit, s0:s0 + 24], params, seed, w[:, s0:s0 + 24])
                    pred[:, s0:s0 + 24] += np.clip(mdl.predict(Xt[s0]), 1., 3000.) / len(seeds)
                    last.setdefault(s0, []).append(mdl)
            assert np.isfinite(pred).all()
            preds[m].append(pred)
            info[p][m] = wi
            print(f"[{time.time() - t0:6.0f}s] {p}: {m} ({len(seeds)} seeds) {json.dumps(wi)}", flush=True)
            if m == MODEL:
                for i in range(0, len(ev_o), 29):  # post-fit predict time of one 72 h forecast (all seeds, blocks)
                    ts = time.perf_counter()
                    for s0, ms in last.items():
                        for mdl in ms:
                            mdl.predict(Xt[s0][i:i + 1])
                    timing.append(time.perf_counter() - ts)
        oo.append(ev_o)
    o = oo[0].append(oo[1]).append(oo[2])
    for m in models:
        pred = np.vstack(preds[m])
        frame = pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                              "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()})
        frame.to_parquet(outs[m], index=False)
    print(json.dumps(dict(rows=len(o) * H, out=outs, config_sha256=CONFIG_SHA256, **info,
                          inference_seconds_per_fold=float(np.mean(timing)), elapsed_s=round(time.time() - t0))))


if __name__ == "__main__":
    main()
