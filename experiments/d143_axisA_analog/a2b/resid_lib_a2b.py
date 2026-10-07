"""D143 A2b helpers: mixture base plumbing plus 96-dim residual analog.

Causality note: every input used here ends at or before the query origin.
Past vectors are trailing-only filled speeds; hind vectors use the seed
OOF-fit base path scored at O-12h (parameters predate the cutoff);
candidate outcomes filled[H+h] always predate the query origin (H+72h < O,
and H+72h < cutoff for official origins). Path vectors are the seed own
OOF-fit base mixture predictions. Residual dv(H,h) = filled[H+h] minus
basepath(H,h); the query estimate is the uniform mean of the K nearest
candidates dv curves. Final is base_full plus w times dv with OOF-only w.
"""

import numpy as np
import pandas as pd

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


def scenario_distance3(past_q, err_q, path_q, past_c, err_c, path_c):
    """Squared distance with equal weight per subspace (three means)."""
    dp = np.mean((past_c - past_q) ** 2, axis=1)
    de = np.mean((err_c - err_q) ** 2, axis=1)
    dh = np.mean((path_c - path_q) ** 2, axis=1)
    return dp + de + dh


def residual_for_origins(origins, filled, q_paths, q_hind, cand_times, cand_paths,
                         k, cutoff=None, lookback_h=8760):
    """Uniform-mean K-neighbour residual replay on the 96-dim scenario.

    origins: DatetimeIndex UTC (n queries). filled: hourly Series (UTC index).
    q_paths: (n,72) query predicted paths from the seed OOF fit.
    q_hind: (n,72) hind paths scored at Q-12h from the same OOF fit; the
      error subspace uses q_hind[:, 0:12].
    cand_times: hourly DatetimeIndex (m candidates) aligned with cand_paths.
    cand_paths: (m,72) candidate predicted paths from the same OOF fit.
    k: neighbours. cutoff: if given, candidates additionally require
      H+72h < cutoff (pre-cutoff pool for official origins).
    Returns (dv (n,72), coverage (n,)) with dv = y minus basepath.
    """
    origins = pd.DatetimeIndex(origins, tz="UTC")
    cand_times = pd.DatetimeIndex(cand_times, tz="UTC")
    hours = filled.index.sort_values()
    fvals = filled.reindex(hours).to_numpy(dtype=float)
    n_hours = len(hours)
    past_q = scenario_past_matrix(origins, filled)
    qh = np.asarray(q_hind, dtype=float)
    qp = np.asarray(q_paths, dtype=float)
    cp = np.asarray(cand_paths, dtype=float)
    assert qp.shape == (len(origins), 72), qp.shape
    assert qh.shape == (len(origins), 72), qh.shape
    assert cp.shape == (len(cand_times), 72), cp.shape
    err_q_all = scenario_error_matrix(past_q, qh[:, 0:12])
    n = len(origins)
    out = np.full((n, 72), np.nan)
    cov = np.zeros(n, dtype=int)
    cut = pd.to_datetime(cutoff, utc=True) if cutoff is not None else None
    for i, o in enumerate(origins):
        q0 = past_q[i]
        qe = err_q_all[i]
        q1 = qp[i]
        if not (np.isfinite(q0).all() and np.isfinite(qe).all() and np.isfinite(q1).all()):
            continue
        cands = candidate_pool(o, hours, n_hours, lookback_h=lookback_h)
        if len(cands) == 0:
            continue
        cand_h = hours[cands]
        if cut is not None:
            keep = (cand_h + pd.Timedelta("72h")) < cut
            cands = cands[np.asarray(keep)]
            cand_h = hours[cands]
        if len(cands) == 0:
            continue
        loc = cand_times.searchsorted(cand_h)
        valid = (loc < len(cand_times)) & (np.asarray(cand_times[loc]) == np.asarray(cand_h))
        cands = cands[valid]
        loc = loc[valid]
        if len(cands) == 0:
            continue
        pc = np.empty((len(cands), 12))
        for j, p in enumerate(cands):
            if p - 11 < 0:
                pc[j] = np.nan
            else:
                pc[j] = fvals[p - 11: p + 1]
        # hind for candidates comes from the same cand path array shifted by 12h
        hc = np.full((len(cands), 12), np.nan)
        ok_hind = loc >= 12
        hc[ok_hind] = cp[loc[ok_hind] - 12, 0:12]
        ec = pc - hc
        ok = np.isfinite(pc).all(axis=1) & np.isfinite(ec).all(axis=1) & np.isfinite(cp[loc]).all(axis=1)
        cands_f = cands[ok]
        loc_f = loc[ok]
        if len(cands_f) == 0:
            continue
        d2 = scenario_distance3(q0, qe, q1, pc[ok], ec[ok], cp[loc_f])
        order = np.argsort(d2, kind="stable")[:int(k)]
        picks_c = cands_f[order]
        picks_l = loc_f[order]
        curves = []
        for p, li in zip(picks_c, picks_l):
            seg = fvals[p + 1: p + 73]
            if len(seg) < 72:
                seg = np.pad(seg, (0, 72 - len(seg)), constant_values=np.nan)
            base = cp[int(li)]
            curves.append(seg - base)
        curves = np.asarray(curves, dtype=float)
        with np.errstate(all="ignore"):
            m = np.nanmean(curves, axis=0)
        out[i] = m
        cov[i] = int(len(picks_c))
    return out, cov


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


def frame_predictions(te, pred):
    return pd.DataFrame({
        "origin_last_input_utc": pd.to_datetime(te["origin"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": np.asarray(te["h"], dtype=int),
        "pred_kms": np.asarray(pred, dtype=float),
    })


def scenario_poison(origins, filled, cutoff, perturb=10.0):
    """Real future-poison: perturb post-cutoff filled speeds, check invariance."""
    cut = pd.to_datetime(cutoff, utc=True)
    base = scenario_past_matrix(pd.DatetimeIndex(origins, tz="UTC"), filled)
    pert = filled.copy()
    after = pert.index > cut
    pert.loc[after] = pert.loc[after] + float(perturb)
    mod = scenario_past_matrix(pd.DatetimeIndex(origins, tz="UTC"), pert)
    o = pd.DatetimeIndex(origins, tz="UTC")
    pre = o < cut
    post = o >= cut
    pre_inv = bool(np.array_equal(base[pre], mod[pre], equal_nan=True)) if int(pre.sum()) else True
    moved = bool((~np.isclose(base[post], mod[post], equal_nan=True)).any()) if int(post.sum()) else False
    passed = bool(pre_inv and moved and int(pre.sum()) > 0 and int(post.sum()) > 0)
    return {"passed": passed, "cutoff_utc": cut.isoformat(), "perturbation": float(perturb),
            "n_pre_rows": int(pre.sum()), "pre_invariant": bool(pre_inv),
            "n_post_rows": int(post.sum()), "moved": bool(moved),
            "moved_columns": ["scenario"] if moved else [],
            "note": "perturbed post-cutoff filled speeds; pre-cutoff scenarios bitwise identical"}


def frame_poison(origins, filled, cutoff, frame_fn, perturb=10.0):
    """Future-poison for model-input frames (the path subspace inputs)."""
    cut = pd.to_datetime(cutoff, utc=True)
    o = pd.DatetimeIndex(origins, tz="UTC")
    n = len(o)
    base = np.asarray(frame_fn(filled), dtype=float)
    pert = filled.copy()
    after = pert.index > cut
    pert.loc[after] = pert.loc[after] + float(perturb)
    mod = np.asarray(frame_fn(pert), dtype=float)
    assert base.shape == mod.shape, (base.shape, mod.shape)
    assert base.shape[0] % n == 0, (base.shape, n)
    per = base.shape[0] // n
    pre = np.asarray(o < cut)
    post = ~pre
    pre_inv = True
    for i in np.where(pre)[0]:
        if not np.array_equal(base[i * per:(i + 1) * per], mod[i * per:(i + 1) * per], equal_nan=True):
            pre_inv = False
            break
    moved = False
    for i in np.where(post)[0]:
        if not np.allclose(base[i * per:(i + 1) * per], mod[i * per:(i + 1) * per], equal_nan=True):
            moved = True
            break
    passed = bool(pre_inv and moved and int(pre.sum()) > 0 and int(post.sum()) > 0)
    return {"passed": passed, "cutoff_utc": cut.isoformat(), "perturbation": float(perturb),
            "n_pre_rows": int(pre.sum()), "pre_invariant": bool(pre_inv),
            "n_post_rows": int(post.sum()), "moved": bool(moved),
            "moved_columns": ["frame"] if moved else [],
            "note": "perturbed post-cutoff filled speeds; pre-cutoff frame blocks bitwise identical"}
