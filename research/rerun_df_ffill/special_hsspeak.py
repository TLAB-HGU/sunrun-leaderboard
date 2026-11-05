"""Faithful HSSPEAK25 with freshly rebuilt L72-COND roll6 and E0472 query paths.

Sources: lb-f1f2/experiments/d143_axisA_analog/a2b/resid_lib_a2b.py,
d147_e0421_lean/lean.py, and f1_resubmit_20261026/hsspeak_f1.py.
No old forecasts or modules are loaded. All control, chronological Ridge, peak/hour, and query models are retrained.
Fresh HUXt summary inputs are supplied by the independent GONG physics pipeline. Residual outcomes explicitly use
prepared training-only target_series, separate from causal ffilled scenario input speeds.
See special-blockers.md for integration requirements and current limits.
"""
import argparse
import json
import os
import pickle
import subprocess
import time
import zipfile
from itertools import pairwise
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

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


class NodeCache:
    """Owned, completed fresh-input fits only; prediction/level caches stay in memory."""
    def __init__(self,out,data,raw_data,physics,threads,smoke):
        self.out=Path(out);self.root=self.out/'hsspeak_checkpoints'
        self.context={'data':identity(Path(data)),'raw_data':identity(Path(raw_data)),
                      'huxt_features':identity(Path(physics)),'threads':threads,'smoke':smoke,
                      'versions':{'numpy':np.__version__,'xgboost':xgb.__version__,'sklearn':sklearn.__version__}}
        source=Path(__file__).read_bytes()
        if self.root.exists():
            try:
                same=json.loads((self.root/'context.json').read_text())==self.context
                same= same and (self.root/'source.py').read_bytes()==source
            except (OSError,ValueError):same=False
            if not same:self.root.rename(self.out/f'hsspeak_invalid_{time.time_ns()}')
        self.root.mkdir(exist_ok=True)
        self.atomic(self.root/'source.py',lambda p:p.write_bytes(source))
        self.atomic(self.root/'context.json',lambda p:p.write_text(json.dumps(self.context)))
        self.reused=0;self.trained=0

    @staticmethod
    def atomic(path,write):
        tmp=path.with_name(path.stem+'.partial'+path.suffix)
        write(tmp)
        os.replace(tmp,path)

    def verify_identity(self):
        for key in ('data','raw_data','huxt_features'):
            if identity(Path(self.context[key]['path']))!=self.context[key]:
                raise ValueError('Input changed during checkpointed run: '+key)
        if Path(__file__).read_bytes()!=(self.root/'source.py').read_bytes():
            raise ValueError('Source changed during checkpointed run')

    def node(self,tag,suffix,recipe,witness,fit,validate):
        self.verify_identity()
        # Tag names are internal deterministic graph keys, never external paths.
        if Path(tag).name!=tag:raise ValueError('Invalid checkpoint tag')
        model_path=self.out/f'{tag}{suffix}';record=self.root/f'{tag}.json';proof=self.root/f'{tag}.npz'
        expected={'recipe':recipe,'kind':suffix}
        try:
            saved=json.loads(record.read_text())
            if saved['identity']!=expected or saved['bytes']!=model_path.stat().st_size:
                raise ValueError('Checkpoint recipe/model size differs')
            with np.load(proof,allow_pickle=False) as stored:
                if set(stored.files)!=set(witness) or any(not np.array_equal(stored[k],v,equal_nan=True) for k,v in witness.items()):
                    raise ValueError('Checkpoint training support differs')
            if suffix=='.ubj':model=xgb.Booster(model_file=model_path)
            else:
                with model_path.open('rb') as f:model=pickle.load(f)
            validate(model)
        except (OSError,EOFError,ValueError,KeyError,TypeError,AttributeError,IndexError,
                pickle.UnpicklingError,zipfile.BadZipFile,xgb.core.XGBoostError):
            # A completion record is written last, so interrupted writes never count.
            record.unlink(missing_ok=True)
            model=fit();validate(model);self.verify_identity()
            if suffix=='.ubj':self.atomic(model_path,model.save_model)
            else:
                def dump(p):
                    with p.open('wb') as f:pickle.dump(model,f)
                self.atomic(model_path,dump)
            self.atomic(proof,lambda p:np.savez_compressed(p,**witness))
            self.atomic(record,lambda p:p.write_text(json.dumps({'identity':expected,'bytes':model_path.stat().st_size})))
            self.trained+=1
        else:self.reused+=1
        return model

    def booster(self,tag,params,rounds,features,witness,matrix):
        loss=self.out/f'{tag}_loss.json'
        def validate(model):
            config=json.loads(model.save_config())['learner']['learner_model_param']
            if model.num_boosted_rounds()!=rounds or model.num_features()!=len(features) or int(config['num_target'])!=1:
                raise ValueError('Incomplete booster shape/rounds')
            values=np.asarray(json.loads(loss.read_text())['train']['rmse'],float)
            if values.shape!=(rounds,) or not np.isfinite(values).all():raise ValueError('Incomplete booster loss')
            model.set_param({'nthread':params['nthread']})
        def fit():
            history={};dmatrix=matrix()
            model=xgb.train(params,dmatrix,num_boost_round=rounds,evals=[(dmatrix,'train')],evals_result=history,verbose_eval=False)
            self.atomic(loss,lambda p:p.write_text(json.dumps(history)+'\n'))
            return model
        return self.node(tag,'.ubj',{'params':params,'rounds':rounds,'features':features},witness,fit,validate)

    def ridge(self,tag,x,y,origins,cutoff):
        def validate(model):
            if (list(model.named_steps)!=['simpleimputer','standardscaler','ridge'] or
                    not isinstance(model[-1],Ridge) or model[-1].alpha!=1. or
                    model[0].strategy!='median' or not model[0].keep_empty_features or
                    model.n_features_in_!=x.shape[1] or model[-1].coef_.shape!=(x.shape[1],) or
                    not np.isfinite(model[-1].coef_).all() or not np.isfinite(model[-1].intercept_)):
                raise ValueError('Invalid Ridge checkpoint')
            if not np.isfinite(model.predict(x[:2])).all():raise ValueError('Nonfinite Ridge checkpoint')
        features=['mean','std','min','max',*[f'h{h}' for h in (6,12,24,36,48,60,72)],*HISTORY]
        return self.node(tag,'.pkl',{'kind':'level_ridge','cutoff':str(cutoff),'features':features,'alpha':1.},
                         {'x':x,'y':y,'origins':origins.asi8},lambda:level_head(x,y),validate)

    def head(self,tag,x,y,origins,params):
        def validate(model):
            if (not isinstance(model,HistGradientBoostingRegressor) or model.n_features_in_!=x.shape[1] or
                    any(model.get_params()[k]!=v for k,v in params.items()) or not 0<model.n_iter_<=params['max_iter']):
                raise ValueError('Invalid peak/hour checkpoint')
            if not np.isfinite(model.predict(x[:2])).all():raise ValueError('Nonfinite peak/hour checkpoint')
        return self.node(tag,'.pkl',{'kind':'peak_hour','params':params,'features':HEAD_FEATURES},
                         {'x':x,'y':y,'origins':origins},lambda:HistGradientBoostingRegressor(**params).fit(x,y),validate)


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
        matrix=None
        def get_matrix():
            nonlocal matrix
            if matrix is None:matrix=xgb.QuantileDMatrix(x[observed],label=flat[observed],nthread=threads,max_bin=256)
            return matrix
        for rngseed in seeds:
            key=f'{kind}_{rngseed}'
            models[key]=grid.checkpoints.booster(f'{tag}_{key}',dict(PARAMS,seed=rngseed,nthread=threads,max_bin=256),
                rounds,['h',*FEATURES],{'origins':train_origins.asi8,'observed':np.packbits(observed),
                                      'cutoff':np.array(cutoff.value)},get_matrix)
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










def trailing(series, origins):
    times = pd.DatetimeIndex(origins)
    return np.column_stack([series.reindex(times - i * HOUR).to_numpy() for i in range(11, -1, -1)])

def residual(origins, ace, labels, query, hind, cand_times, cand_paths, cfg, cutoff, scales=None):
    """Analog replay: delayed past input, unshifted outcomes, mature frozen pool."""
    ct, origins = pd.DatetimeIndex(cand_times), pd.DatetimeIndex(origins)
    past = trailing(ace, ct)
    # Latest observed hour is O-2h, so hindcast must begin O-14h.
    err = np.full_like(past, np.nan)
    err[14:] = past[14:] - cand_paths[:-14, :12]
    target = np.column_stack([labels.reindex(ct + h * HOUR).to_numpy() for h in range(1, 73)])
    mature = np.asarray(ct + 144 * HOUR <= utc(cutoff))
    valid = mature & np.isfinite(past).all(1) & np.isfinite(err).all(1) & np.isfinite(cand_paths).all(1)
    if scales is None:
        if not valid.any():
            raise ValueError('no mature analog scenarios')
        scales = tuple(max(float(np.var(x[valid])), 1e-6) for x in (past, err, cand_paths))
    variance = scales if cfg['standardize'] else (1., 1., 1.)
    qp = trailing(ace, origins)
    qe = qp - hind[:, :12]
    out = np.full_like(query, np.nan)
    for i, origin in enumerate(origins):
        mask = valid & np.asarray(ct <= origin - pd.Timedelta(hours=654.48)) & np.asarray(ct >= origin - pd.Timedelta(days=cfg['lookback_days']))
        ids = np.flatnonzero(mask)
        if not len(ids) or not np.isfinite(qp[i]).all() or not np.isfinite(qe[i]).all():
            continue
        distance = sum(np.mean((x[ids] - q) ** 2, axis=1) / s for x, q, s in zip((past, err, cand_paths), (qp[i], qe[i], query[i]), variance))
        rank = np.argsort(distance, kind='stable')[:cfg['k']]
        curves = target[ids[rank]] - cand_paths[ids[rank]]
        weights = np.ones(len(rank))
        if cfg['analog_weighting'] == 'distance':
            d = np.sqrt(distance[rank])
            weights = (d == 0).astype(float) if np.any(d == 0) else 1 / d
        finite = np.isfinite(curves)
        denom = (finite * weights[:, None]).sum(0)
        out[i] = np.divide((np.where(finite, curves, 0) * weights[:, None]).sum(0), denom,
                           out=np.full(72, np.nan), where=denom > 0)
    return out, scales

def fit_alpha(base, dv, labels):
    finite = np.isfinite(labels)
    if not finite.any():
        raise ValueError('no observed calibration labels')
    dv = np.nan_to_num(dv)
    grid = np.arange(21) * .05
    losses = [float(np.mean((base + w * dv - labels)[finite] ** 2)) for w in grid]
    return float(grid[np.argmin(losses)]), losses

HOUR = pd.Timedelta(hours=1)
EXPERIMENT = 'e1604_hsspeak25'
HUXT_COLUMNS = ['huxt_max','huxt_argmax','huxt_h1','huxt_h24','huxt_h48','huxt_h72']
HEAD_PARAMS = {'max_iter':500,'learning_rate':.04,'max_depth':4,'min_samples_leaf':50,'random_state':0}
CFG = {'standardize':True,'k':15,'lookback_days':365,'analog_weighting':'uniform'}
LEVEL_DATES = pd.to_datetime(['2023-01-01','2023-04-01','2023-07-01','2024-01-01','2024-07-01',
                              '2025-01-01','2025-07-01','2026-01-01','2026-05-29'],utc=True)
ROLL6 = [('2024-06-01','2024-09-30 23:00'),('2024-10-01','2025-01-31 23:00'),
         ('2025-02-01','2025-05-31 23:00'),('2025-06-01','2025-09-30 23:00'),
         ('2025-10-01','2026-01-31 23:00'),('2026-02-01','2026-05-28 23:00')]
HISTORY = ['v_lag0','v_m24','v_m72','v_sd24','v_sd72','enlil_n_pending','enlil_next_arr_h']
HEAD_FEATURES = ['rec_max','rec_argmax','rec_range','v0','v6','v24','v168','v648','m6','m24','m168','m648',
                 'path_max','path_argmax','path_mean','path_min','path_std','path_h1','path_h24','path_h48','path_h72',
                 *HUXT_COLUMNS,'annual_sin','annual_cos']


def utc(value):
    stamp=pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def physics_table(path):
    frame=pd.read_parquet(path,columns=['origin',*HUXT_COLUMNS])
    frame['origin']=pd.to_datetime(frame.origin,utc=True)
    if frame.origin.duplicated().any() or not np.isfinite(frame[HUXT_COLUMNS].to_numpy()).all():
        raise ValueError('HUXt physics requires unique origins and finite six summaries')
    if not frame.huxt_argmax.between(1,72).all():
        raise ValueError('HUXt argmax must be one-based lead1..72')
    return frame.set_index('origin').sort_index()


def head_features(origins,path,physics,ace):
    hx=physics.reindex(origins)[HUXT_COLUMNS].to_numpy(float)
    if not np.isfinite(hx).all():
        raise ValueError('Fresh HUXt physics does not cover requested origins')
    # Reuse the exact 29-column formula with a synthetic path only to locate the
    # six physical summary slots; overwrite those slots with actual summaries.
    fake=np.ones_like(path)
    result=features(origins,path,fake,ace)
    result[:,21:27]=hx
    return result


def labels72(labels,origins):
    a=np.column_stack([labels.reindex(origins+h*HOUR).to_numpy() for h in range(1,73)])
    good=np.isfinite(a).sum(1)>=54
    values=np.full(len(origins),np.nan)
    values[good]=np.nanmean(a[good],axis=1)
    return values


def level_summary(grid,origins,base):
    hist=grid.c[grid.index(origins)][:,[CONSTANT.index(n) for n in HISTORY]]
    x=np.column_stack([base.mean(1),base.std(1),base.min(1),base.max(1),base[:,np.array([6,12,24,36,48,60,72])-1],hist])
    x[~np.isfinite(x)]=np.nan
    return x


def level_head(x,y):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=1.)).fit(x,y)


class L72Fit:
    """Seven original E0472 scalar experts, optionally conditioned on OOF level72."""
    def __init__(self,engine,cutoff,bank=None):
        self.engine,self.cutoff,self.bank=engine,utc(cutoff),bank
        self.level_cache={}
        grid=engine.grid
        origins=grid.origins[grid.train_origin & ((grid.origins.asi8-grid.origins[0].value)%(3*HOUR.value)==0)
                             & (grid.origins+72*HOUR<=self.cutoff)]
        level=None
        if bank is not None:
            level=bank.level.reindex(origins).to_numpy()
            good=np.isfinite(level)
            origins,level=origins[good],level[good]
        if engine.smoke:
            origins=origins[-96:]
            if level is not None:level=level[-96:]
        y=grid.labels(origins).copy()
        times=origins.asi8[:,None]+np.arange(1,73)[None,:]*HOUR.value
        y[times+72*HOUR.value>self.cutoff.value]=np.nan
        x=grid.long(origins)
        if level is not None:x=np.column_stack([x,np.repeat(level,72)]).astype(np.float32)
        h=x[:,0];flat=y.ravel();self.models={}
        tag=f'{engine.tag}_{"aug" if bank else "raw"}_{self.cutoff.strftime("%Y%m%dT%H%M")}'
        self.tag=tag
        for name,seeds,mask in [('short1',level_seeds(engine.seed),h<=6),('short2',level_seeds(engine.seed),(h>6)&(h<=24)),
                                ('long',(event_seed(engine.seed),),np.ones(len(h),bool))]:
            observed=mask & np.isfinite(flat)
            if not observed.any():raise ValueError(f'No mature labels: {tag}/{name}')
            matrix=None
            def get_matrix():
                nonlocal matrix
                if matrix is None:matrix=xgb.QuantileDMatrix(x[observed],label=flat[observed],nthread=engine.threads,max_bin=256)
                return matrix
            for seed in seeds:
                key=f'{name}_{seed}'
                witness={'origins':origins.asi8,'observed':np.packbits(observed),'cutoff':np.array(self.cutoff.value)}
                if level is not None:witness['level']=level
                self.models[key]=grid.checkpoints.booster(f'{tag}_{key}',dict(PARAMS,seed=seed,nthread=engine.threads,max_bin=256),
                    engine.rounds,['h',*FEATURES]+(['level72_hat'] if bank is not None else []),witness,get_matrix)
        self.receipt={'cutoff':str(self.cutoff),'origins':len(origins),'observed_labels':int(np.isfinite(y).sum()),
                      'latest_target':str(pd.to_datetime(times[np.isfinite(y)].max(),utc=True)),
                      'extra_target_maturity_hours':72,'features':111+int(bank is not None),'rounds':engine.rounds}
        engine.fits[tag]=self.receipt

    def paths(self,origins):
        outputs=[]
        for a in range(0,len(origins),128):
            oo=origins[a:a+128];x=self.engine.grid.long(oo)
            if self.bank is not None:
                missing=pd.DatetimeIndex([o for o in oo if o not in self.level_cache])
                if len(missing):
                    self.level_cache.update(zip(missing,self.bank.values(missing,self.cutoff)))
                level=np.array([self.level_cache[o] for o in oo])
                x=np.column_stack([x,np.repeat(level,72)]).astype(np.float32)
            outputs.append(predict_mixture(self.models,x,self.engine.seed))
        return np.concatenate(outputs).reshape(-1,72)


class L72Replay:
    def __init__(self,engine,cutoff,bank=None,official_cal=False,end=None):
        self.engine,self.cutoff=engine,utc(cutoff)
        self.model=engine.fit(self.cutoff,bank)
        cal_fit=max(self.cutoff-pd.DateOffset(months=3),utc('2022-11-15'))
        cal_end=(self.cutoff-pd.Timedelta(days=7)).floor('D')+21*HOUR
        cal_start=cal_fit+pd.Timedelta(days=3) if official_cal else cal_end-pd.Timedelta(days=7)+3*HOUR
        if not official_cal and self.cutoff==utc('2026-05-29'):
            cal_fit,cal_start,cal_end=utc('2026-03-01'),utc('2026-03-04'),utc('2026-05-22 21:00')
        self.cal=engine.fit(cal_fit,bank)
        calibration=pd.date_range(cal_start,cal_end,freq='3h')
        if engine.smoke:calibration=calibration[:8]
        start=max(cal_start-pd.Timedelta(days=365),engine.grid.origins.min())
        last=(utc(end or engine.end)-655*HOUR).floor('h')
        self.ct=pd.date_range(start,last,freq='h')
        if engine.smoke:
            early=max(min(calibration.min()-900*HOUR,cal_fit-400*HOUR),engine.grid.origins.min())
            self.ct=self.ct[(self.ct>=early)&(self.ct<=early+900*HOUR)]
        self.cp=self.cal.paths(self.ct)
        cb=self.cal.paths(calibration)
        cd,self.scales=residual(calibration,engine.inputs,engine.targets,cb,self.cal.paths(calibration-14*HOUR),
                                self.ct,self.cp,CFG,cal_fit)
        yc=np.column_stack([engine.targets.reindex(calibration+h*HOUR).to_numpy() for h in range(1,73)])
        self.alpha,self.losses=fit_alpha(cb,cd,yc)
        engine.replays.append({'cutoff':str(self.cutoff),'cal_fit':str(cal_fit),'cal_start':str(cal_start),'cal_end':str(cal_end),
                               'alpha':self.alpha,'losses':self.losses,'official_cal':official_cal})

    def paths(self,origins):
        b=self.model.paths(origins)
        q,hind=self.cal.paths(origins),self.cal.paths(origins-14*HOUR)
        dv,_=residual(origins,self.engine.inputs,self.engine.targets,q,hind,self.ct,self.cp,CFG,self.cutoff,scales=self.scales)
        return b+self.alpha*np.nan_to_num(dv)


class Engine:
    def __init__(self,grid,inputs,targets,cutoff,seed,out,threads=1,smoke=False,tag='base'):
        self.grid,self.inputs,self.targets=grid,inputs,targets
        self.cutoff,self.seed,self.out,self.threads,self.smoke=utc(cutoff),seed,out,threads,smoke
        self.rounds=2 if smoke else 900;self.tag=tag;self.end=grid.origins.max()
        self.models={};self.fits={};self.replays=[]

    def fit(self,cutoff,bank=None):
        key=(utc(cutoff),bank is not None)
        if key not in self.models:self.models[key]=L72Fit(self,cutoff,bank)
        return self.models[key]


class LevelBank:
    def __init__(self,engine):
        self.engine=engine;self.times=pd.DatetimeIndex([],tz='UTC');self.x=np.empty((0,18))
        self.level=pd.Series(dtype=float,index=self.times);self.replays={};self.heads={}

    def replay(self,cutoff):
        cutoff=utc(cutoff)
        if cutoff not in self.replays:
            if cutoff in LEVEL_DATES[:-1]:
                end=LEVEL_DATES[list(LEVEL_DATES).index(cutoff)+1]-HOUR
            else:
                end=self.engine.end
            self.replays[cutoff]=L72Replay(self.engine,cutoff,end=end)
        return self.replays[cutoff]

    def head(self,cutoff):
        y=labels72(self.engine.targets,self.times)
        valid=(self.times+144*HOUR<=cutoff)&np.isfinite(y)
        if valid.sum()<2:return None
        tag=f'{self.engine.tag}_ridge_{utc(cutoff).strftime("%Y%m%dT%H%M")}'
        return self.engine.grid.checkpoints.ridge(tag,self.x[valid],y[valid],self.times[valid],cutoff)

    def build(self):
        bank_end=min(self.engine.cutoff,LEVEL_DATES[-1])
        dates=list(LEVEL_DATES[LEVEL_DATES<bank_end])+[bank_end]
        for start,end in pairwise(dates):
            origins=pd.date_range(start,end-HOUR,freq='h')
            if self.engine.smoke:origins=origins[:24]
            replay=self.replay(start)
            x=level_summary(self.engine.grid,origins,replay.paths(origins))
            head=self.head(start)
            values=np.full(len(origins),np.nan) if head is None else head.predict(x)
            self.times=self.times.append(origins);self.x=np.vstack([self.x,x])
            self.level=pd.concat([self.level,pd.Series(values,index=origins)])
        pd.DataFrame({'origin':self.times,'level72_hat':self.level.to_numpy()}).to_parquet(self.engine.out/f'{self.engine.tag}_level_oof.parquet',index=False)

    def values(self,origins,cutoff):
        values=self.level.reindex(origins).to_numpy()
        infer=origins>=cutoff
        if infer.any():
            if cutoff not in self.heads:self.heads[cutoff]=self.head(cutoff)
            head=self.heads[cutoff]
            if head is not None:
                x=level_summary(self.engine.grid,origins[infer],self.replay(cutoff).paths(origins[infer]))
                values[infer]=head.predict(x)
        return values


def e0472_final(grid,inputs,targets,period,seed,out,threads,smoke):
    """F1 release's unaugmented E0472 base, independently retrained per seed."""
    cutoff=utc(CUTOFFS[period]);windows,scale_end=calendar(period)
    official=grid.queries[period][:4] if smoke else grid.queries[period]
    if smoke:windows=[(name,end,oo[:8]) for name,end,oo in windows]
    allq=official.append(pd.DatetimeIndex(np.concatenate([o.asi8 for _,_,o in windows]),tz='UTC'))
    candidates=pd.date_range(max(allq.min()-8760*HOUR,grid.origins.min()),allq.max()-655*HOUR,freq='h')
    if smoke:
        start=windows[-1][2].min()-900*HOUR
        candidates=candidates[(candidates>=start)&(candidates<=start+899*HOUR)]
    training=grid.origins[grid.train_origin & ((grid.origins.asi8-grid.origins[0].value)%(3*HOUR.value)==0)]
    rounds=2 if smoke else 900;seconds=0.;record=[]
    for name,end,oo in windows:
        tr=training[training<end]
        if smoke:tr=tr[-96:]
        model=fit_mixture(grid,tr,end,seed,rounds,threads,out,f'{period}_official_s{seed}_{name}')
        base,hind=paths(model,grid,oo,seed),paths(model,grid,oo-12*HOUR,seed)
        candidate=paths(model,grid,candidates,seed)
        scales=pool_scales(inputs,candidates,candidate,scale_end)
        dv=residual_multi(oo,inputs,base,hind,candidates,candidate,(15,),'std_equal',scales,
                          cutoff=cutoff,target_series=targets)[15]
        record.append({'window':name,'fit_cutoff':str(end),'origins':len(tr)})
        if name=='W2':
            w,_,_,_=fit_blend_weight_dv(base,dv,grid.labels(oo),np.arange(21)*.05)
            t0=time.perf_counter()
            qpath,qhind=paths(model,grid,official,seed),paths(model,grid,official-12*HOUR,seed)
            qdv=residual_multi(official,inputs,qpath,qhind,candidates,candidate,(15,),'std_equal',scales,
                               cutoff=cutoff,target_series=targets)[15]
            seconds+=time.perf_counter()-t0
    tr=training[grid.masks[period][grid.index(training)]]
    if smoke:tr=tr[-96:]
    model=fit_mixture(grid,tr,cutoff,seed,rounds,threads,out,f'{period}_official_s{seed}_full')
    t0=time.perf_counter();pred=blend_dv(paths(model,grid,official,seed),qdv,w);seconds+=time.perf_counter()-t0
    return pred,seconds,{'fits':record,'full_fit_cutoff':str(cutoff),'blend_weight':w,'seed_cell':seed}


def fold_origins(grid,period,smoke=False):
    cutoff=utc(CUTOFFS[period]);result=[]
    for number,(first,last) in zip((4,5,6,1,2,3),ROLL6):
        oo=pd.date_range(utc(first),min(utc(last),cutoff-144*HOUR),freq='h')
        if not len(oo):continue
        positions=grid.index(oo)
        oo=oo[grid.masks[period][positions]]
        if len(oo):result.append((number,utc(first)-72*HOUR,oo[:8] if smoke else oo))
    return result


def run_period(data,raw_data,huxt_features,out,period,threads=1,smoke=False):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metadata=out/f'{period}_metadata.json'
    if metadata.exists():os.replace(metadata,out/f'{period}_previous_metadata.json')
    checkpoints=NodeCache(out,data,raw_data,huxt_features,threads,smoke)
    grid,inputs,targets=load_grid(data,raw_data)
    grid.checkpoints=checkpoints
    physics=physics_table(huxt_features)
    official=grid.queries[period]
    if len(official)!=QUERY_COUNTS[period]:raise ValueError('Unexpected official grid')
    if smoke:official=official[:4]
    folds=fold_origins(grid,period,smoke)
    required=official.append(pd.DatetimeIndex(np.concatenate([oo.asi8 for _,_,oo in folds]),tz='UTC'))
    if not np.isfinite(physics.reindex(required)[HUXT_COLUMNS].to_numpy()).all():
        raise ValueError('Fresh HUXt summaries missing for roll6 or official origins')
    per_fold={number:[] for number,_,_ in folds};query_bases=[];inference=0.;receipts=[]
    with threadpool_limits(limits=threads):
        for seed in (0,1,2):
            engine=Engine(grid,inputs,targets,utc(CUTOFFS[period]),seed,out,threads,smoke,f'{period}_roll6_s{seed}')
            bank=LevelBank(engine);bank.build()
            for number,cut,oo in folds:
                replay=L72Replay(engine,cut,bank,official_cal=True,end=oo[-1])
                pred=replay.paths(oo)
                if not np.isfinite(pred).all():raise ValueError('Nonfinite roll6 control')
                per_fold[number].append(pred)
                np.savez_compressed(out/f'{period}_fold{number}_s{seed}.npz',origins=oo.asi8,pred=pred)
            receipts.append({'seed_cell':seed,'fits':engine.fits,'replays':engine.replays})
            del bank,engine
            pred,seconds,receipt=e0472_final(grid,inputs,targets,period,seed,out,threads,smoke)
            query_bases.append(pred);inference+=seconds
            receipts[-1]['official_base']=receipt
        xs,peaks,hours,head_origins=[],[],[],[]
        for number,cut,oo in folds:
            mean=np.mean(per_fold[number],axis=0)
            x=head_features(oo,mean,physics,inputs)
            y=grid.labels(oo)
            complete=np.isfinite(y).all(axis=1)&np.isfinite(x).all(axis=1)
            if not complete.any():raise ValueError('No complete mature peak targets')
            xs.append(x[complete]);peaks.append(np.max(y[complete],axis=1));hours.append(np.argmax(y[complete],axis=1)+1.)
            head_origins.append(oo[complete].asi8)
        x=np.vstack(xs);yp=np.concatenate(peaks);yh=np.concatenate(hours)
        params=dict(HEAD_PARAMS,max_iter=2 if smoke else HEAD_PARAMS['max_iter'])
        heads=[checkpoints.head(f'{period}_{name}',x,y,np.concatenate(head_origins),params)
               for name,y in zip(('peak','hour'),(yp,yh))]
        np.savez_compressed(out/f'{period}_head_training.npz',features=x,peak=yp,hour=yh,origins=np.concatenate(head_origins))
        t0=time.perf_counter()
        per_seed=[]
        for base in query_bases:
            xq=head_features(official,base,physics,inputs)
            per_seed.append(bump(base,heads[0].predict(xq),heads[1].predict(xq),25))
        predictions=np.mean(per_seed,axis=0);inference+=time.perf_counter()-t0
    if predictions.shape!=(len(official),72) or not np.isfinite(predictions).all():raise ValueError('Invalid final forecast')
    checkpoints.verify_identity()
    checkpoints.atomic(out/f'{period}_predictions.npz',lambda p:np.savez_compressed(
        p,origins=official.asi8,pred=predictions,per_seed=np.stack(per_seed)))
    report={'experiment':EXPERIMENT,'period':period,'smoke':smoke,'data':identity(Path(data)),
            'raw_data':identity(Path(raw_data)),'huxt_features':identity(Path(huxt_features)),
            'rounds':2 if smoke else 900,'head_parameters':params,'head_features':HEAD_FEATURES,
            'head_iterations':[int(m.n_iter_) for m in heads],
            'head_losses':[{'train':m.train_score_.tolist(),'validation':m.validation_score_.tolist()} for m in heads],
            'maturity_hours':144,'training_cutoff':CUTOFFS[period],'query_origins':len(official),'head_training_origins':len(x),
            'train_path_source':'fresh E1508/E1529 L72-COND augmented analog roll6; three-seed mean features',
            'query_path_source':'fresh three-seed E0472 release pipeline; per-seed features, corrections then average',
            'target_policy':'prepared causal training-y ffill; query y never read as targets',
            'availability':'corrected new-df inputs; original L72COND14h and E0472 12h hindcast structures retained',
            'checkpoint_nodes':{'reused':checkpoints.reused,'trained':checkpoints.trained},
            'base_receipts':receipts,'inference_seconds_per_fold':inference/len(official),'seconds':time.perf_counter()-started,
            'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip(),
            'versions':{'numpy':np.__version__,'pandas':pd.__version__,'xgboost':xgb.__version__,'sklearn':sklearn.__version__}}
    checkpoints.atomic(metadata,lambda p:p.write_text(json.dumps(report,indent=2)+'\n'))
    print(json.dumps({'period':period,'seconds':report['seconds'],'head_rows':len(x)}),flush=True)
    return report


def merge(out):
    frames=[]
    identities=set()
    for period,count in QUERY_COUNTS.items():
        meta=json.loads((out/f'{period}_metadata.json').read_text())
        if meta['smoke'] or meta['rounds']!=900 or meta['experiment']!=EXPERIMENT:raise ValueError('Smoke/mismatched output')
        identities.add(json.dumps([meta['data'],meta['raw_data'],meta['huxt_features']],sort_keys=True))
        with np.load(out/f'{period}_predictions.npz') as z:pred=z['pred'];origins=pd.DatetimeIndex(z['origins'],tz='UTC')
        if pred.shape!=(count,72) or not np.isfinite(pred).all():raise ValueError('Invalid full query grid')
        frames.append(pd.DataFrame({'origin_last_input_utc':np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'),72),
            'horizon_hours':np.tile(np.arange(1,73),count),'pred_kms':pred.ravel()}))
    if len(identities)!=1:raise ValueError('Training/physics input snapshots differ across periods')
    pd.concat(frames,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=[EXPERIMENT],default=EXPERIMENT)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--raw-data',type=Path,required=True)
    parser.add_argument('--huxt-features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1);parser.add_argument('--period',choices=CUTOFFS)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.merge:merge(args.out);return
    for period in ([args.period or 'F1'] if args.period or args.smoke else CUTOFFS):
        run_period(args.data,args.raw_data,args.huxt_features,args.out,period,args.threads,args.smoke)
    if not args.period and not args.smoke:merge(args.out)


if __name__ == '__main__':
    main()
