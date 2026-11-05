"""Frozen E0472 mixture plus analog residual runner on new causally filled inputs.

Sources: lb-f1f2/experiments/d143_axisA_analog/a2b/resid_lib_a2b.py,
d147_e0421_lean/lean.py, and f1_resubmit_20261026/hsspeak_f1.py.
No old forecasts or modules are loaded. All 28 scalar models are retrained per
period. HSSPEAK helper kernels require fresh physical trajectories and are not
a supported experiment CLI. Residual outcomes explicitly use
prepared training-only target_series, separate from causal ffilled scenario input speeds.
See special-blockers.md for integration requirements and current limits.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
import xgboost as xgb

HORIZON = 72
WIDTH, SHRINK = 12., .15

LEADS = ("1-6", "7-24", "25-48", "49-72")


LEAD_BOUNDS = ((1, 6), (7, 24), (25, 48), (49, 72))


MEMBERS = ("e0003_line", "d33_hsplit", "d45_lvlmean")


SPLIT_H = 24


def level_seeds(seed_idx):
    """Three short-expert RNG seeds owned by seed file `seed_idx`."""
    s = int(seed_idx)
    return (3 * s, 3 * s + 1, 3 * s + 2)


def event_seed(seed_idx):
    """Long-head RNG seed: the middle of this file level block."""
    return 3 * int(seed_idx) + 1


def fit_params(base, depth, random_state):
    """Recipe base params with exactly two fields set: depth plus RNG seed."""
    p = dict(base)
    p["max_depth"] = int(depth)
    p["random_state"] = int(random_state)
    return p


def check_weights(weights):
    if tuple(weights.keys()) != LEADS:
        raise ValueError(f"need leads {LEADS}")
    for lead in LEADS:
        if tuple(weights[lead].keys()) != MEMBERS:
            raise ValueError(f"lead {lead}: need members {MEMBERS}")
        vals = [float(weights[lead][m]) for m in MEMBERS]
        if any(not np.isfinite(v) or v < 0 for v in vals):
            raise ValueError(f"lead {lead}: weights must be finite non-negative")
        if abs(sum(vals) - 1.0) > 1e-9:
            raise ValueError(f"lead {lead}: weights must sum to 1")


def lead_of(h):
    h = int(h)
    for (lo, hi), name in zip(LEAD_BOUNDS, LEADS):
        if lo <= h <= hi:
            return name
    raise ValueError(f"horizon {h} outside 1..72")


def blend_per_lead(preds_by_member, weights, horizons):
    """Convex per-lead combination of the three member vectors."""
    check_weights(weights)
    if tuple(preds_by_member.keys()) != MEMBERS:
        raise ValueError(f"need members {MEMBERS}")
    arrs = {m: np.asarray(preds_by_member[m], dtype=np.float64) for m in MEMBERS}
    shape = arrs[MEMBERS[0]].shape
    if any(a.shape != shape for a in arrs.values()):
        raise ValueError("member predictions must share one shape")
    if any(not np.isfinite(a).all() for a in arrs.values()):
        raise ValueError("member predictions must be finite")
    h = np.asarray(list(horizons), dtype=int)
    if h.shape != shape:
        raise ValueError("horizons must match member prediction shape")
    leads = np.array([lead_of(x) for x in h])
    out = np.empty(shape, dtype=np.float64)
    for lead in LEADS:
        mask = leads == lead
        w = [float(weights[lead][m]) for m in MEMBERS]
        out[mask] = sum(w[i] * arrs[m][mask] for i, m in enumerate(MEMBERS))
    if not np.isfinite(out).all():
        raise ValueError("blended predictions must be finite")
    return out


def stitch_members(short1, short2, long_all, long_tail, m1, m2, m3, level, event):
    """Stitch the three members from fitted-model predictions."""
    if tuple(sorted(short1)) != tuple(sorted(level)):
        raise ValueError("short1 seeds must equal the level block")
    if tuple(sorted(short2)) != tuple(sorted(level)):
        raise ValueError("short2 seeds must equal the level block")
    if event not in level:
        raise ValueError("event seed must be inside the level block")
    m1, m2, m3 = (np.asarray(m, dtype=bool) for m in (m1, m2, m3))
    n = len(np.asarray(long_all))
    if not ((m1 | m2 | m3).all() and not (m1 & m2).any()
            and not (m1 & m3).any() and not (m2 & m3).any()):
        raise ValueError("masks must partition the rows exactly once")
    for s in level:
        if len(np.asarray(short1[s])) != int(m1.sum()) or len(np.asarray(short2[s])) != int(m2.sum()):
            raise ValueError("short prediction lengths must match their masks")
    if len(np.asarray(long_tail)) != int(m3.sum()) or len(np.asarray(long_all)) != n:
        raise ValueError("long prediction lengths must match their masks")
    mem0 = np.asarray(long_all, dtype=np.float64)
    mem1 = np.empty(n, dtype=np.float64)
    mem2 = np.empty(n, dtype=np.float64)
    mem1[m1] = np.asarray(short1[event], dtype=np.float64)
    mem1[m2] = np.asarray(short2[event], dtype=np.float64)
    mem1[m3] = np.asarray(long_tail, dtype=np.float64)
    mean1 = sum(np.asarray(short1[s], dtype=np.float64) for s in level) / len(level)
    mean2 = sum(np.asarray(short2[s], dtype=np.float64) for s in level) / len(level)
    mem2[m1] = mean1
    mem2[m2] = mean2
    mem2[m3] = np.asarray(long_tail, dtype=np.float64)
    return {"e0003_line": mem0, "d33_hsplit": mem1, "d45_lvlmean": mem2}


def check_blend_grid(grid):
    g = tuple(float(x) for x in grid)
    assert len(g) == 21 and abs(g[0]) < 1e-12 and abs(g[-1] - 1.0) < 1e-12, grid
    assert all(0.0 <= x <= 1.0 for x in g), grid


def scenario_past_matrix(origins, filled):
    """(n,12) past-12h observed speeds; row i = [v(O-11h)..v(O)]. NaN if gap."""
    origins = pd.DatetimeIndex(origins, tz="UTC")
    idx = filled.index
    vals = filled.to_numpy(dtype=float)
    out = np.full((len(origins), 12), np.nan)
    for i, o in enumerate(origins):
        try:
            pos = idx.get_loc(o)
        except KeyError:
            continue
        if isinstance(pos, slice):
            continue
        p = int(pos)
        if p - 11 < 0:
            continue
        out[i] = vals[p - 11: p + 1]
    return out


def scenario_error_matrix(past12, hind12):
    """(n,12) recent base error: past observed minus hind first-12 horizons."""
    p = np.asarray(past12, dtype=float)
    h = np.asarray(hind12, dtype=float)
    assert p.shape == h.shape and p.shape[1] == 12, (p.shape, h.shape)
    return p - h


def candidate_pool(o, hours, n_hours, lookback_h=8760):
    """Hourly candidate positions for query origin o (causal, exclusion out).

    o: Timestamp (UTC). hours: hourly DatetimeIndex of filled. n_hours: len.
    Returns int array of candidate positions with H+72h < O (and caller
    applies the pre-cutoff cap for official origins by restricting hours).
    """
    o_pos = int(hours.searchsorted(o))
    hi = o_pos - 655  # strictly beyond 654.48h
    lo = o_pos - int(lookback_h)  # lookback bound
    if hi < 0:
        return np.empty(0, dtype=int)
    lo = max(lo, 0)
    return np.arange(lo, hi + 1, dtype=int)


def as_path_matrix(flat):
    """Reshape a flat (n*72,) base path prediction to (n,72)."""
    a = np.asarray(flat, dtype=float).ravel()
    assert a.size % 72 == 0 and a.size > 0, a.shape
    return a.reshape(-1, 72)


def fit_blend_weight_dv(base, dv, y, grid):
    """Grid argmin of OOF MSE for final=base+w*dv (finite cells)."""
    check_blend_grid(grid)
    b = np.asarray(base, dtype=float).ravel()
    a = np.asarray(dv, dtype=float).ravel()
    t = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(b) & np.isfinite(t)
    m = mask & np.isfinite(a)
    if int(m.sum()) < 100:
        raise ValueError("too few finite OOF residual cells for blend fit")
    best_w, best_mse = 0.0, float("inf")
    mses = {}
    for w in grid:
        w = float(w)
        f = np.where(np.isfinite(a[mask]), b[mask] + w * a[mask], b[mask])
        mse = float(np.mean((f - t[mask]) ** 2))
        mses[w] = mse
        if mse < best_mse:
            best_mse, best_w = mse, w
    base_mse = float(np.mean((b[mask] - t[mask]) ** 2))
    return best_w, base_mse, best_mse, mses


def blend_dv(base, dv, w):
    """Additive residual blend; NaN dv falls back to base cell."""
    b = np.asarray(base, dtype=float)
    a = np.asarray(dv, dtype=float)
    w = float(w)
    assert 0.0 <= w <= 1.0, w
    return np.where(np.isfinite(a), b + w * a, b)


def residual_multi(origins, filled, q_paths, q_hind, cand_times, cand_paths, ks, dist, scales,
                   cutoff=None, lookback_h=8760, *, target_series):
    """residual_for_origins generalised to several K and a scaled distance; identical to it for raw_equal + one K."""
    origins = pd.DatetimeIndex(origins, tz="UTC")
    cand_times = pd.DatetimeIndex(cand_times, tz="UTC")
    hours = filled.index.sort_values()
    fvals = filled.reindex(hours).to_numpy(dtype=float)
    target_values = target_series.reindex(hours).to_numpy(dtype=float)
    past_q = scenario_past_matrix(origins, filled)
    qh = np.asarray(q_hind, dtype=float)
    qp = np.asarray(q_paths, dtype=float)
    cp = np.asarray(cand_paths, dtype=float)
    err_q_all = scenario_error_matrix(past_q, qh[:, 0:12])
    kmax = max(ks)
    out = {k: np.full((len(origins), 72), np.nan) for k in ks}
    cut = pd.to_datetime(cutoff, utc=True) if cutoff is not None else None
    sp, se, sh = scales if dist == "std_equal" else (1.0, 1.0, 1.0)
    for i, o in enumerate(origins):
        q0, qe, q1 = past_q[i], err_q_all[i], qp[i]
        if not (np.isfinite(q0).all() and np.isfinite(qe).all() and np.isfinite(q1).all()):
            continue
        cands = candidate_pool(o, hours, len(hours), lookback_h=lookback_h)
        if len(cands) == 0:
            continue
        cand_h = hours[cands]
        if cut is not None:
            cands = cands[np.asarray((cand_h + pd.Timedelta("72h")) < cut)]
            cand_h = hours[cands]
        if len(cands) == 0:
            continue
        loc = cand_times.searchsorted(cand_h)
        valid = (loc < len(cand_times)) & (np.asarray(cand_times[np.minimum(loc, len(cand_times) - 1)]) == np.asarray(cand_h))
        cands, loc = cands[valid], loc[valid]
        if len(cands) == 0:
            continue
        pc = np.full((len(cands), 12), np.nan)
        for j, p in enumerate(cands):
            if p - 11 >= 0:
                pc[j] = fvals[p - 11: p + 1]
        hc = np.full((len(cands), 12), np.nan)
        ok_h = loc >= 12
        hc[ok_h] = cp[loc[ok_h] - 12, 0:12]
        ec = pc - hc
        ok = np.isfinite(pc).all(axis=1) & np.isfinite(ec).all(axis=1) & np.isfinite(cp[loc]).all(axis=1)
        cands_f, loc_f = cands[ok], loc[ok]
        if len(cands_f) == 0:
            continue
        dp = np.mean((pc[ok] - q0) ** 2, axis=1) / sp
        de = np.mean((ec[ok] - qe) ** 2, axis=1) / se
        dh = np.mean((cp[loc_f] - q1) ** 2, axis=1) / sh
        order = np.argsort(dp + de + dh, kind="stable")[:kmax]
        curves = []
        for p, li in zip(cands_f[order], loc_f[order]):
            seg = target_values[p + 1: p + 73]
            if len(seg) < 72:
                seg = np.pad(seg, (0, 72 - len(seg)), constant_values=np.nan)
            curves.append(seg - cp[int(li)])
        curves = np.asarray(curves, dtype=float)
        for k in ks:
            with np.errstate(all="ignore"):
                observed = np.isfinite(curves[:k])
                count = observed.sum(axis=0)
                out[k][i] = np.divide(np.nansum(curves[:k], axis=0), count,
                                      out=np.full(72, np.nan), where=count > 0)
    return out


def features(o, path, hx, ace):
    """Same 29 columns as experiments/solar_hsspeak72_20261019/run.py features()."""
    def at(lag):
        return ace.reindex(o - pd.Timedelta(hours=lag)).values.astype(float)

    def roll(win):
        return ace.rolling(win, min_periods=1).mean().reindex(o).values.astype(float)
    rec = np.empty_like(path)
    for h in range(1, HORIZON + 1):
        rec[:, h - 1] = ace.reindex(o + pd.Timedelta(hours=h - 654)).values
    return np.column_stack([
        np.nanmax(rec, 1), np.nanargmax(rec, 1) + 1.0, np.nanmax(rec, 1) - np.nanmin(rec, 1),
        at(0), at(6), at(24), at(168), at(648), roll(6), roll(24), roll(168), roll(648),
        np.nanmax(path, 1), np.nanargmax(path, 1) + 1.0, np.nanmean(path, 1), np.nanmin(path, 1), np.nanstd(path, 1),
        path[:, 0], path[:, 23], path[:, 47], path[:, 71],
        np.nanmax(hx, 1), np.nanargmax(hx, 1) + 1.0, hx[:, 0], hx[:, 23], hx[:, 47], hx[:, 71],
        np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25)])


def bump(path, peak, hour, min_lead):
    hh = np.arange(1, HORIZON + 1, dtype=float)
    b = np.exp(-((hh[None, :] - np.clip(hour, 1, HORIZON)[:, None]) ** 2) / (2 * WIDTH * WIDTH))
    b /= np.nanmax(b, 1, keepdims=True)
    inc = SHRINK * (peak[:, None] - np.nanmin(path, 1, keepdims=True)) * b
    inc[:, : min_lead - 1] = 0.0
    return path + np.nan_to_num(inc)


CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}

QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}

CONSTANT = ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing']

VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']

FEATURES = ['rec26', 'rec27', 'rec28', 'rec_mean', 'v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing', 'enlil_dt_target']

def mapped(name):
    return name.replace("ch_ch_", "suvi_", 1) if name.startswith("ch_ch_") else name

def stale_series(raw):
    """Reconstruct original no-image/age>6 rule from raw table image observation ends."""
    raw = raw.sort_values("origin", kind="stable")
    duplicates = raw[raw.duplicated("origin", keep=False)]
    if len(duplicates) and (duplicates.groupby("origin")["img_age_h"].nunique(dropna=False) > 1).any():
        raise ValueError("train/query image timestamps disagree")
    raw = raw.drop_duplicates("origin").set_index("origin")
    ends = pd.Series(raw.index, index=raw.index) - pd.to_timedelta(raw.img_age_h, unit="h")
    if (ends.dropna() > ends.dropna().index).any():
        raise ValueError("image observation lies after its input origin")
    last_end = ends.ffill()
    age = (pd.Series(raw.index, index=raw.index) - last_end).dt.total_seconds() / 3600
    return (last_end.isna() | age.gt(6.0)).astype(np.float32)

def identity(path):
    stat = path.stat()
    return {'path':str(path.resolve()), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}

WEIGHTS = {
    "1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
            "d45_lvlmean": 0.3946530834632206},
    "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
             "d45_lvlmean": 0.3411415011482895},
    "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
              "d45_lvlmean": 0.3333333333333333},
    "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
              "d45_lvlmean": 0.3333333333333333},
}

EXPERIMENT = 'xgb_d76_lead_moe_resid_dv_enlil'
PARAMS = {'max_depth':3, 'learning_rate':.03, 'subsample':.8, 'colsample_bytree':.7,
          'min_child_weight':20, 'reg_lambda':5, 'tree_method':'hist', 'device':'cpu',
          'objective':'reg:squarederror', 'eval_metric':'rmse'}


def calendar(period):
    """Keep original calendar offsets; truncate OOF to the campaign maturity bound."""
    shift = {'F1':-5, 'F2':-2, 'F3':1}[period]
    cutoff = pd.Timestamp(CUTOFFS[period],tz='UTC')
    windows = []
    for name, end, first, last in (
        ('W0','2026-01-01','2026-01-04','2026-01-31 23:00'),
        ('W1','2026-02-01','2026-02-04','2026-02-28 23:00'),
        ('W2','2026-03-01','2026-03-04','2026-05-28 23:00')):
        end,first,last = [pd.Timestamp(v,tz='UTC')+pd.DateOffset(months=shift) for v in (end,first,last)]
        if name != 'W2':
            last = (end+pd.offsets.MonthBegin(1))-pd.Timedelta(hours=1)
        last = min(last,cutoff-pd.Timedelta(hours=144))
        windows.append((name,end,pd.date_range(first,last,freq='3h')))
    return windows, pd.Timestamp('2026-01-01',tz='UTC')+pd.DateOffset(months=shift)


class Grid:
    """All origin features retained compactly, materializing long rows only for a fit/chunk."""
    def __init__(self,origins,c,e,training_y,train_origin,queries,masks):
        self.origins,self.c,self.e,self.y,self.train_origin,self.queries,self.masks = origins,c,e,training_y,train_origin,queries,masks

    def index(self,origins):
        positions = self.origins.get_indexer(origins)
        if (positions < 0).any():
            raise ValueError(f'Missing required origin feature snapshots: {list(origins[positions < 0][:3])}')
        return positions

    def long(self,origins):
        indices = self.index(origins)
        n = len(indices)
        values = [np.tile(np.arange(1,73,dtype=np.float32),n)]
        for name in FEATURES:
            values.append(self.e[indices,:,VARYING.index(name)].ravel() if name in VARYING else
                          np.repeat(self.c[indices,CONSTANT.index(name)],72))
        return np.column_stack(values).astype(np.float32)

    def labels(self,origins):
        return self.y[self.index(origins)]


def load_grid(data,raw_data):
    columns = ['origin','set','period',*[f'fit_{p}' for p in CUTOFFS],*[mapped(n) for n in CONSTANT if n != 'img_stale']]
    c = pd.read_parquet(data,columns=columns,filters=[('h','==',1)]).sort_values('origin',kind='stable')
    c.origin = pd.to_datetime(c.origin,utc=True)
    query = {p:pd.DatetimeIndex(c.loc[c['set'].eq('query') & c.period.eq(p),'origin']) for p in CUTOFFS}
    training = c.loc[c['set'].eq('train')].set_index('origin')
    origins = pd.DatetimeIndex(c.origin.drop_duplicates()).sort_values()
    masks = {p:training[f'fit_{p}'].reindex(origins,fill_value=False).to_numpy(bool) for p in CUTOFFS}
    train_origin = origins.isin(training.index)
    raw = pd.read_parquet(raw_data,columns=['origin','img_age_h'],filters=[('h','==',1)])
    raw.origin = pd.to_datetime(raw.origin,utc=True)
    stale = stale_series(raw)
    c = c.drop_duplicates('origin').set_index('origin').reindex(origins)
    c['img_stale'] = stale.reindex(origins)
    constants = c[[mapped(n) for n in CONSTANT]].to_numpy(np.float32)
    rows = pd.read_parquet(data,columns=['origin','h','set','y',*VARYING]).sort_values(['origin','h'],kind='stable')
    rows.origin = pd.to_datetime(rows.origin,utc=True)
    targets = rows.loc[rows['set'].eq('train'),['origin','h','y']].set_index(['origin','h']).y
    rows = rows.drop_duplicates(['origin','h'])
    if len(rows) != len(origins)*72 or not np.array_equal(rows.h.to_numpy().reshape(-1,72),np.tile(np.arange(1,73),(len(origins),1))):
        raise ValueError('Incomplete hourly 72h feature grid')
    e = rows[VARYING].to_numpy(np.float32).reshape(-1,72,len(VARYING))
    index = pd.MultiIndex.from_product([origins,range(1,73)],names=['origin','h'])
    y = targets.reindex(index).to_numpy(np.float32).reshape(-1,72)
    # Residual outcomes come exclusively from the prepared training labels, including user-requested target ffill; never query y.
    target_times = pd.DatetimeIndex(targets.index.get_level_values('origin'))+pd.to_timedelta(targets.index.get_level_values('h'),unit='h')
    observed = pd.DataFrame({'time':target_times,'y':targets.to_numpy()}).dropna()
    if (observed.groupby('time').y.nunique()>1).any():
        raise ValueError('Training target values disagree at the same timestamp')
    target_series = observed.drop_duplicates('time').set_index('time').y
    timeline = pd.date_range(origins.min(),origins.max()+pd.Timedelta(hours=72),freq='h')
    inputs = pd.Series(constants[:,CONSTANT.index('v_lag0')],index=origins).reindex(timeline)
    target_series = target_series.reindex(timeline)
    return Grid(origins,constants,e,y,train_origin,query,masks),inputs,target_series


def fit_mixture(grid,train_origins,cutoff,seed,rounds,threads,out,tag):
    target = grid.labels(train_origins).copy()
    times = train_origins.asi8[:,None]+np.arange(1,73)[None,:]*3_600_000_000_000
    target[times >= cutoff.value] = np.nan
    x = grid.long(train_origins)
    flat = target.ravel()
    h = np.tile(np.arange(1,73),len(train_origins))
    models = {}
    for kind,seeds,mask in [('short1',level_seeds(seed),h<=6),('short2',level_seeds(seed),(h>=7)&(h<=24)),('long',(event_seed(seed),),np.ones(len(h),bool))]:
        observed = mask & np.isfinite(flat)
        if not observed.any():
            raise ValueError(f'No observed labels for {tag}/{kind}')
        matrix = xgb.QuantileDMatrix(x[observed],label=flat[observed],nthread=threads,max_bin=256)
        for rngseed in seeds:
            history = {}
            model = xgb.train(dict(PARAMS,seed=rngseed,nthread=threads,max_bin=256),matrix,num_boost_round=rounds,
                              evals=[(matrix,'train')],evals_result=history,verbose_eval=False)
            key = f'{kind}_{rngseed}'
            model.save_model(out/f'{tag}_{key}.ubj')
            (out/f'{tag}_{key}_loss.json').write_text(json.dumps(history)+'\n')
            models[key] = model
    return models


def predict_mixture(models,x,seed=0):
    h = x[:,0]
    m1,m2,m3 = h<=6,(h>=7)&(h<=24),h>24
    level,event = level_seeds(seed),event_seed(seed)
    long = models[f'long_{event}'].inplace_predict(x).astype(float)
    short1 = {s:models[f'short1_{s}'].inplace_predict(x[m1]).astype(float) for s in level}
    short2 = {s:models[f'short2_{s}'].inplace_predict(x[m2]).astype(float) for s in level}
    return blend_per_lead(stitch_members(short1,short2,long,long[m3],m1,m2,m3,level,event),WEIGHTS,h)


def paths(models,grid,origins,seed=0):
    return np.concatenate([predict_mixture(models,grid.long(origins[a:a+128]),seed)
                           for a in range(0,len(origins),128)]).reshape(-1,72)


def pool_scales(filled,cand_times,cand_paths,scale_end):
    past = scenario_past_matrix(cand_times,filled)
    err = np.full_like(past,np.nan)
    err[12:] = past[12:]-cand_paths[:-12,:12]
    early = cand_times < scale_end
    result = []
    for values in (past,err,cand_paths):
        x = values[early]
        x = x[np.isfinite(x).all(axis=1)]
        variance = float(np.var(x)) if len(x)>100 else float(np.nanvar(values))
        if not np.isfinite(variance):
            raise ValueError('Insufficient observed scenario variance')
        result.append(max(variance,1e-6))
    return tuple(result)


def run_period(data,raw_data,out,period,threads=1,smoke=False):
    started = time.perf_counter()
    grid,inputs,targets = load_grid(data,raw_data)
    cutoff = pd.Timestamp(CUTOFFS[period],tz='UTC')
    windows,scale_end = calendar(period)
    official = grid.queries[period]
    if len(official)!=QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query count')
    if smoke:
        official = official[:4]
        windows = [(name,end,origins[:8]) for name,end,origins in windows]
    lookback = 8760
    allqueries = official.append(pd.DatetimeIndex(np.concatenate([o.asi8 for _,_,o in windows]),tz='UTC'))
    lo = allqueries.min()-pd.Timedelta(hours=lookback)
    hi = allqueries.max()-pd.Timedelta(hours=655)
    candidates = pd.date_range(lo,hi,freq='h')
    # Original asks for candidate inputs before the table begins; these carry no available
    # 12h observed scenario and cannot enter a neighbour pool. Drop only that unavailable prefix.
    candidates = candidates[candidates >= grid.origins.min()]
    if smoke:
        start = windows[-1][2].min()-pd.Timedelta(hours=900)
        candidates = candidates[(candidates >= start) & (candidates <= start+pd.Timedelta(hours=899))]
    rounds = 2 if smoke else 900
    trainall = grid.origins[grid.train_origin & ((grid.origins.asi8-grid.origins[0].value)%(3*3_600_000_000_000)==0)]
    wins = {}
    query_seconds = 0.
    for name,train_end,oo in windows:
        tr = trainall[trainall < train_end]
        if smoke:
            tr = tr[-96:]
        model = fit_mixture(grid,tr,train_end,0,rounds,threads,out,f'{period}_{name}')
        base,hind = paths(model,grid,oo),paths(model,grid,oo-pd.Timedelta(hours=12))
        cand = paths(model,grid,candidates)
        scales = pool_scales(inputs,candidates,cand,scale_end)
        dv = residual_multi(oo,inputs,base,hind,candidates,cand,(15,),'std_equal',scales,
                            cutoff=cutoff,lookback_h=lookback,target_series=targets)[15]
        labels = grid.labels(oo).copy()
        labels[oo.asi8[:,None]+np.arange(1,73)[None,:]*3_600_000_000_000 >= cutoff.value] = np.nan
        wins[name] = {'base':base,'dv':dv,'y':labels}
        np.savez_compressed(out/f'{period}_{name}_oof.npz',origins=oo.asi8,base=base,dv=dv,y=labels)
        if name == 'W2':
            qstart = time.perf_counter()
            qpath,qhind = paths(model,grid,official),paths(model,grid,official-pd.Timedelta(hours=12))
            qdv = residual_multi(official,inputs,qpath,qhind,candidates,cand,(15,),'std_equal',scales,
                                 cutoff=cutoff,lookback_h=lookback,target_series=targets)[15]
            query_seconds += time.perf_counter()-qstart
    w,_,_,_ = fit_blend_weight_dv(wins['W2']['base'],wins['W2']['dv'],wins['W2']['y'],np.arange(21)*.05)
    tr = trainall[grid.masks[period][grid.index(trainall)]]
    if smoke:
        tr = tr[-96:]
    model = fit_mixture(grid,tr,cutoff,0,rounds,threads,out,f'{period}_full')
    qstart = time.perf_counter()
    pred = blend_dv(paths(model,grid,official).ravel(),qdv.ravel(),w).reshape(-1,72)
    query_seconds += time.perf_counter()-qstart
    if not np.isfinite(pred).all():
        raise ValueError('Invalid final forecast')
    np.savez_compressed(out/f'{period}.npz',origins=official.asi8,pred=pred)
    report = {'experiment':EXPERIMENT,'period':period,'rounds':rounds,'seed_cell':0,'models':28,'params':PARAMS,
              'features':['h',*[mapped(n) for n in FEATURES]],'data':identity(data),'raw_data':identity(raw_data),
              'cutoff':str(cutoff),'windows':[(n,str(t),str(o.min()),str(o.max())) for n,t,o in windows],
              'scale_end':str(scale_end),'scales':scales,'blend_weight':w,'k':15,'lookback_h':lookback,
              'target_policy':'prepared training-set y with user-requested causal ffill; query y ignored; remaining missing outcomes excluded',
              'availability':'new corrected availability; past12 uses causal ffilled v_lag0',
              'queries':len(official),'training_origins':len(tr),'train_origin_start':str(tr.min()),'train_origin_end':str(tr.max()),
              'inference_seconds':query_seconds,'seconds':time.perf_counter()-started,
              'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip(),
              'libraries':{'numpy':np.__version__,'pandas':pd.__version__,'xgboost':xgb.__version__}}
    (out/f'{period}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'period':period,'seconds':report['seconds'],'weight':w}),flush=True)


def merge(out):
    frames,identities,seconds = [],set(),0.
    for period,count in QUERY_COUNTS.items():
        report = json.loads((out/f'{period}.json').read_text())
        if report['experiment'] != EXPERIMENT or report['rounds']!=900 or report['period']!=period:
            raise ValueError('Smoke/mismatched artifacts')
        identities.add(json.dumps([report['data'],report['raw_data']],sort_keys=True))
        seconds += report['inference_seconds']
        with np.load(out/f'{period}.npz') as z:
            origins = pd.DatetimeIndex(z['origins'],tz='UTC'); pred=z['pred']
        if pred.shape!=(count,72) or not np.isfinite(pred).all():
            raise ValueError('Invalid forecast grid')
        frames.append(pd.DataFrame({'origin_last_input_utc':np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'),72),
            'horizon_hours':np.tile(np.arange(1,73),count),'pred_kms':pred.ravel()}))
    if len(identities)!=1:
        raise ValueError('Input snapshot identities differ')
    pd.concat(frames,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    (out/'merge.json').write_text(json.dumps({'experiment':EXPERIMENT,'inference_seconds_per_fold':seconds/sum(QUERY_COUNTS.values())})+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=[EXPERIMENT],default=EXPERIMENT)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--period',choices=CUTOFFS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.threads<1:
        parser.error('threads must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.merge:
        merge(args.out); return
    for period in ([args.period or 'F1'] if args.period or args.smoke else CUTOFFS):
        run_period(args.data,args.raw_data,args.out,period,args.threads,args.smoke)
    if not args.period and not args.smoke:
        merge(args.out)


if __name__=='__main__':
    main()
