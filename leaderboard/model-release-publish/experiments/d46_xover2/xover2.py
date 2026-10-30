"""D46 GEAR crossover helpers: D33 hard-switch level + E0060 pinned-joint event branch.

NOT a D44 follow-up. New GEAR combination (pareto-era crossover quota):
- Level branch, MSE parent E0102 (direction D33, hard horizon-split hybrid,
  seed-1 first bar-passer): 1-24h from two short band experts (1-6, 7-24,
  each fit only on its band rows) + 25-72h from one E0003-line single head
  (all rows), same causal rows/purge/features/params as E0003 (+cuda/threads
  only), fixed split h=24, hard switch by horizon, no weight/blending.
  Source: experiments/d33_hsplit/hsplit.py (SPLIT_H, SHORT/LONG bands,
  SELECT_RULE/FULL_RULE) and runner.py (head layout). E0102 is seed-fragile
  (X0096 first bar-passer 2/3 mean -1.08% F1 0.4138; X0097 3-seed means
  +0.52/-1.08/-1.74, F1 0.418/0.414/0.389): it stands as a submit-candidate
  report only, never a single-pass claim, no freeze/official-score.
- Event branch, F1 parent E0060 (direction D21, F1-first strips-analog joint,
  pooled F1 0.486, dev winner CH6V2/l1): existing_all + 17 strips + 4 analog
  columns for the PINNED key CH6V2/l1 in one joint XGBoost-cuda head per
  period per seed. No re-selection by construction: the analog-key grid is
  the pinned 6-combo from experiments/d21_f1joint/joint.py via
  experiments/d10_analog/analog.py (X0045 analogs carry event signal not
  MSE), the floor 0.41121495 from f1first.py (X0054), and the pin follows
  X0074/X0064 (winner differs every seed, F1 0.421/0.486/0.407; single-seed
  F1 is luck, selection unstable) so re-selecting would be seed-fitting.
- Combination is HARD event branching (this direction), never a blend and
  never the D44 vote+replay gate: per (origin,h) cell, final = joint mean
  where the 3-seed COMMON verdict fires (unanimous 3/3 joint surge >=600),
  else the D33 level mean. D44 used a 5-fresh-seed vote level plus a
  dev-selected raw-replay gate; D45 averages only the level branch. D46
  keeps both branches seed-explicit and fires only on unanimity.

Pareto context: MSE end E0102 (D33 switch) x F1 end E0060 (analog joint),
second crossover of the pareto-era cycle (GEAR quota).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import info_ablation as ia  # noqa: E402  (E0003 recipe source, read-only import)

# --- 3-seed common-verdict seeds: the D33 stability trio (E0100/E0102/E0103) ---
SEEDS = (0, 1, 2)
N_SEEDS = len(SEEDS)
UNANIMOUS = 3
SURGE_THRESHOLD = 600.0
LEVEL_RULE = (
    "MSE parent E0102 (D33 hard-switch, source experiments/d33_hsplit/hsplit.py): "
    "fixed split h=24; 1-6/7-24 short band experts (band rows only) + 25-72 "
    "E0003-line single head (all rows), same per-period expanding train "
    "(72h purge), E0003-identical features/params (+device cuda, thread-only "
    "n_jobs 8->6); hard switch by horizon, no weight/blending/averaging; "
    "seeds (0,1,2) refit the same heads, level mean is their cell mean"
)
EVENT_RULE = (
    "F1 parent E0060 (D21 F1-first joint, source experiments/d21_f1joint/joint.py "
    "grid + f1first.py floor via d10_analog replay): PINNED key CH6V2/l1 "
    "(E0060 dev winner, no re-selection per X0074 instability); "
    "existing_all + 17 strips + 4 analog cols in one joint XGBoost-cuda head "
    "per period per seed (0,1,2); dev-only provenance, no selection/audit "
    "statistic feeds any choice"
)
BRANCH_RULE = (
    "hard event branching, no blending: final = joint seed-mean where all 3 "
    "joint seeds vote surge (>=600.0) else D33 level seed-mean; NaN joint "
    "closes the branch (level mean)"
)

# --- Hard-split constants (hsplit.py verbatim values) ---
SPLIT_H = 24
SHORT_BANDS: tuple[tuple[int, int], ...] = ((1, 6), (7, 24))
LONG_BANDS: tuple[tuple[int, int], ...] = ((25, 48), (49, 72))

# --- Pinned analog-key grid (d21_f1joint GRID verbatim; winner pinned, not selected) ---
GRID = ("CH6/l2", "CH6/l1", "CH6V2/l2", "CH6V2/l1", "CH3V3/l2", "CH3V3/l1")
PINNED_KEY = "CH6V2/l1"  # E0060 dev-OOF F1-first winner (selection.json dev F1 0.533)
CH_KEEP = [
    "ch_t035_meridian30_equatorial_area",
    "ch_t035_equatorial_area",
    "ch_t035_meridian30_area",
    "ch_t045_meridian30_equatorial_area",
    "ch_t035_area",
    "ch_t045_equatorial_area",
]
CH3 = [
    "ch_t035_meridian30_equatorial_area",
    "ch_t035_equatorial_area",
    "ch_t045_meridian30_equatorial_area",
]
KEY_SETS = {
    "CH6": ["s_" + c for c in CH_KEEP],
    "CH6V2": ["s_" + c for c in CH_KEEP] + ["s_v_m24", "s_v_d24"],
    "CH3V3": ["s_" + c for c in CH3] + ["s_v_last", "s_v_m24", "s_v_d24"],
}
METRICS = ("l2", "l1")
ANALOG_COLS = ["analog_best", "analog_mean3", "analog_dist", "analog_spread"]
WIN_LO_H = 20 * 24
WIN_HI_H = 35 * 24
K_TOP = 3
H_ALL = np.arange(1, 73)

F1_FLOOR = 0.41121495  # adoption bar: two-seed mean of E0003 pooled F1 (X0054)
GATE_THRESHOLD = 600.0  # fixed evaluator event threshold; not fitted
FULL_IFF = (
    "FULL (3h, new id; same code, else new register) iff poison passes AND "
    "pilot vs E0001 improves >=2/3 or mean <=-1% AND pilot pooled F1 >=0.33 "
    "(no event collapse vs E0085 0.289/X0077) AND branch fires on >0 test "
    "cells in >=2/3 periods; else D46 revises (key/threshold) or closes. "
    "Adoption bar for a submission-candidate full (vs E0003 2/3 + mean<=-1% "
    "+ pooled F1>=0.41121495 + 3-seed common-verdict stability) decides the "
    "report only; E0102 stays a seed-fragile candidate (X0096/X0097), never "
    "a single-pass claim; no freeze/official-score."
)


def check_seeds(seeds) -> tuple:
    """Require exactly the preregistered D33 trio; anything else aborts."""
    s = tuple(int(x) for x in seeds)
    if s != SEEDS:
        raise ValueError(f"D46 requires exactly seeds {SEEDS}, got {s}")
    return s


def check_split() -> dict:
    """Verify the hard split tiles 1..72 exactly once at h=24 (contract)."""
    short = sorted(h for lo, hi in SHORT_BANDS for h in range(lo, hi + 1))
    long = sorted(h for lo, hi in LONG_BANDS for h in range(lo, hi + 1))
    assert short == list(range(1, SPLIT_H + 1)), "short side must be 1..24"
    assert long == list(range(SPLIT_H + 1, 73)), "long side must be 25..72"
    assert len(set(short) & set(long)) == 0
    return {"split_h": SPLIT_H, "short": ["1-6", "7-24"], "long": ["25-48", "49-72"]}


def level_params(seed: int) -> dict:
    """E0003 recipe params (cuda) with only random_state changed (level heads)."""
    if int(seed) not in SEEDS:
        raise ValueError(f"D46 allows only seeds {SEEDS}, got {seed}")
    p = dict(ia.PARAMS)
    p["random_state"] = int(seed)
    p["device"] = "cuda"
    p["n_jobs"] = 6
    return p


def joint_params(seed: int) -> dict:
    """E0003 recipe params (cuda) with only random_state changed (joint head)."""
    if int(seed) not in SEEDS:
        raise ValueError(f"D46 allows only seeds {SEEDS}, got {seed}")
    p = dict(ia.PARAMS)
    p["random_state"] = int(seed)
    p["device"] = "cuda"
    p["n_jobs"] = 6
    return p


def split_config(name: str) -> tuple:
    """Split a pinned grid name into (keys list, metric); unknown names abort."""
    if name not in GRID:
        raise ValueError(f"D46 grid allows only {GRID}, got {name!r}")
    kname, metric = name.split("/")
    if metric not in METRICS:
        raise ValueError(f"D46 metric allows only {METRICS}, got {metric!r}")
    return list(KEY_SETS[kname]), metric


def analog_columns() -> list:
    return list(ANALOG_COLS)


# --- Causal CH-state past-rotation replay (E0060/D21 semantics, owned copy) ---


def build_state() -> tuple:
    """Hourly causal state plus the hourly filled-speed target series.

    Trailing-only state math (rolling means, shifts, 6h-limited ffill) on CH
    areas plus filled speed; scaler inputs stay strictly pre-dev (see
    scaler_from_predev). Mirrors the D10/D21 builder.
    """
    ace = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    ace["timestamp_utc"] = pd.to_datetime(ace["timestamp_utc"], utc=True)
    s = ace.set_index("timestamp_utc")["filled_speed_kms"].sort_index()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full_a = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    filled = s.reindex(full_a)
    ch = pd.read_parquet(LB / "store/ch-v1/ch-hourly.parquet")
    if not isinstance(ch.index, pd.DatetimeIndex):
        tcol = next(c for c in ch.columns if "time" in c.lower())
        ch = ch.set_index(pd.to_datetime(ch.pop(tcol), utc=True))
    ch = ch.sort_index()
    ch = ch[~ch.index.duplicated(keep="last")].sort_index()
    lo = max(filled.index.min(), ch.index.min())
    hi = min(filled.index.max(), ch.index.max())
    full = pd.date_range(lo, hi, freq="h", tz="UTC")
    filled = filled.reindex(full)
    ch = ch.reindex(full)
    frame = pd.DataFrame({"v": filled})
    for c in CH_KEEP:
        frame[c] = ch[c] if c in ch.columns else np.nan
    v = frame["v"].astype(float)
    chf = frame[[c for c in CH_KEEP if c in frame.columns]].reindex(full).ffill(limit=6)
    m6 = v.rolling(6, min_periods=1).mean()
    m24 = v.rolling(24, min_periods=1).mean()
    state = pd.DataFrame(index=full)
    for c in CH_KEEP:
        state["s_" + c] = chf[c] if c in chf.columns else np.nan
    state["s_v_last"] = v
    state["s_v_m6"] = m6
    state["s_v_m24"] = m24
    state["s_v_d24"] = m6 - m6.shift(24)
    return state, filled


def scaler_from_predev(state: pd.DataFrame, dev_lo: pd.Timestamp) -> dict:
    """Mean/std per state column from hours strictly before the dev window."""
    dev_lo = pd.to_datetime(dev_lo, utc=True)
    pre = state.loc[state.index < dev_lo - pd.Timedelta("72h")]
    mu = pre.mean(skipna=True)
    sd = pre.std(skipna=True).replace(0.0, np.nan).fillna(1.0)
    return {"mean": mu, "std": sd, "cutoff": dev_lo.isoformat()}


def analog_long(origins, state: pd.DataFrame, filled: pd.Series,
                keys: list, metric: str, scaler: dict) -> pd.DataFrame:
    """Per-(origin,h) replay columns for one key set.

    For query origin O the pool is hourly H with O-35d <= H <= O-20d and the
    replayed target filled[H+h] always predates O (H+72h <= O-17d). Origins
    with no finite query state or no finite candidate yield NaN (branch off).
    """
    if metric not in METRICS:
        raise ValueError(f"D46 metric allows only {METRICS}, got {metric!r}")
    unknown = [k for k in keys if k not in state.columns]
    if unknown:
        raise ValueError(f"D46 keys must be state columns, unknown={unknown}")
    origins = pd.DatetimeIndex(origins, tz="UTC")
    hours = state.index
    state_vals = state.to_numpy(float)
    filled_vals = filled.reindex(hours).to_numpy(float)
    all_keys = list(state.columns)
    keys_idx = np.array([all_keys.index(k) for k in keys], dtype=int)
    mu = scaler["mean"].reindex(keys).to_numpy(float)
    sd = scaler["std"].reindex(keys).to_numpy(float)
    n = len(origins)
    best = np.full((n, 72), np.nan)
    mean3 = np.full((n, 72), np.nan)
    dist = np.full(n, np.nan)
    spread = np.full((n, 72), np.nan)
    for i, o in enumerate(origins):
        o_pos = int(hours.searchsorted(o))
        if o_pos >= len(hours) or hours[o_pos] != o:
            continue
        lo = o - pd.Timedelta(hours=int(WIN_HI_H))
        hi = o - pd.Timedelta(hours=int(WIN_LO_H))
        c_lo = int(hours.searchsorted(lo))
        c_hi = int(hours.searchsorted(hi, side="right")) - 1
        c_hi = min(c_hi, o_pos - 1)
        if c_hi < c_lo:
            continue
        q = (state_vals[o_pos, keys_idx] - mu) / sd
        if not np.isfinite(q).all():
            continue
        cand = (state_vals[c_lo: c_hi + 1][:, keys_idx] - mu) / sd
        ok = np.isfinite(cand).all(axis=1)
        if not bool(ok.any()):
            continue
        idx = np.flatnonzero(ok)
        c = cand[idx]
        if metric == "l1":
            d = np.abs(c - q).sum(axis=1)
        else:
            d = ((c - q) ** 2).sum(axis=1)
        order = np.argsort(d, kind="stable")[:K_TOP]
        picks = idx[order] + c_lo
        dsorted = d[order]
        dist[i] = float(dsorted[0])
        curves = []
        for p in picks:
            seg = filled_vals[p + 1: p + 73]
            if len(seg) < 72:
                seg = np.pad(seg, (0, 72 - len(seg)), constant_values=np.nan)
            curves.append(seg)
        curves = np.array(curves, dtype=float)
        best[i] = curves[0]
        with np.errstate(all="ignore"):
            mean3[i] = np.nanmean(curves, axis=0)
            spread[i] = np.nanstd(curves, axis=0) if len(curves) > 1 else np.zeros(72)
    df = pd.DataFrame({
        "origin": np.repeat(origins.to_numpy(), 72),
        "h": np.tile(H_ALL, n),
        "analog_best": best.ravel(),
        "analog_mean3": mean3.ravel(),
        "analog_dist": np.repeat(dist, 72),
        "analog_spread": spread.ravel(),
    })
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    return df


# --- 3-seed common-verdict branching (never a blend) ---


def seed_means(pred_by_seed: dict) -> np.ndarray:
    """Cell-wise mean over exactly the preregistered trio (finite required)."""
    seeds = check_seeds(list(pred_by_seed))
    arrs = [np.asarray(pred_by_seed[s], dtype=np.float64) for s in seeds]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("seed predictions must share one shape")
    stacked = np.stack(arrs, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("seed predictions must be finite for seed means")
    return np.asarray(stacked.mean(axis=0), dtype=np.float64)


def surge_votes(pred_by_seed: dict, threshold: float = SURGE_THRESHOLD) -> np.ndarray:
    """Per-cell surge vote counts over the trio (0..3)."""
    seeds = check_seeds(list(pred_by_seed))
    arrs = [np.asarray(pred_by_seed[s], dtype=np.float64) for s in seeds]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("seed predictions must share one shape")
    stacked = np.stack(arrs, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("seed predictions must be finite for voting")
    return np.sum(stacked >= float(threshold), axis=0).astype(int)


def common_fire_mask(pred_by_seed: dict, threshold: float = SURGE_THRESHOLD) -> np.ndarray:
    """True where the common verdict fires: unanimous 3/3 surge."""
    return surge_votes(pred_by_seed, threshold) >= UNANIMOUS


def apply_branch(level_by_seed: dict, joint_by_seed: dict,
                 threshold: float = SURGE_THRESHOLD) -> tuple:
    """Hard event branching: joint mean where unanimous, else level mean.

    Every output cell equals one branch mean bit-for-bit; no averaging mixes
    the branches. NaN can never appear (seed means require finite inputs).
    Returns (final, fired).
    """
    lm = seed_means(level_by_seed)
    jm = seed_means(joint_by_seed)
    if lm.shape != jm.shape:
        raise ValueError("level and joint means must share one shape")
    fired = common_fire_mask(joint_by_seed, threshold)
    if fired.shape != lm.shape:
        raise ValueError("fire mask must share the prediction shape")
    final = np.where(fired, jm, lm)
    return np.asarray(final, dtype=np.float64), np.asarray(fired, dtype=bool)


def branch_stats(fired: np.ndarray, votes: np.ndarray) -> dict:
    n = int(np.asarray(fired).size)
    v = np.asarray(votes).ravel()
    return {"n": n, "rule": BRANCH_RULE,
            "fire_frac": float(np.mean(fired)) if n else 0.0,
            "n_fired": int(np.sum(fired)) if n else 0,
            "unanimous_frac": float(((v == 0) | (v == N_SEEDS)).mean()) if n else 0.0,
            "mean_votes": float(v.mean()) if n else 0.0}


# --- Hash-keyed scratch cache (code + input hashes in the key) ---

FCACHE_VERSION = 1


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fcache_key(kind: str, payload: dict) -> str:
    code = {f: _file_sha(LB / f) for f in
            ("experiments/d46_xover2/runner.py", "experiments/d46_xover2/xover2.py")}
    data = {}
    for f in ("store/ch-v1/ace.parquet", "store/ch-v1/ch-hourly.parquet",
              "store/suvi-fusion/suvi-hourly-v2.parquet"):
        p = LB / f
        st = p.stat()
        data[f] = [st.st_size, int(st.st_mtime_ns)]
    blob = json.dumps({"v": FCACHE_VERSION, "kind": kind, "payload": payload,
                       "code": code, "data": data}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def fcache_get(scratch: Path, key: str):
    side = scratch / f"fcache_{key[:16]}.json"
    tab = scratch / f"fcache_{key[:16]}.parquet"
    try:
        if not (side.exists() and tab.exists()):
            return None
        if json.loads(side.read_text()).get("key") != key:
            return None
        return pd.read_parquet(tab)
    except Exception:  # never raise on a cache path; caller rebuilds
        return None


def fcache_put(scratch: Path, key: str, df: pd.DataFrame):
    scratch.mkdir(parents=True, exist_ok=True)
    tab, tmp = scratch / f"fcache_{key[:16]}.parquet", scratch / f"fcache_{key[:16]}.tmp.parquet"
    df.to_parquet(tmp, index=False)
    tmp.rename(tab)
    (scratch / f"fcache_{key[:16]}.json").write_text(json.dumps({"key": key}))
    return df


def fcache_table(scratch: Path, kind: str, payload: dict, builder):
    key = fcache_key(kind, payload)
    hit = fcache_get(scratch, key)
    if hit is not None:
        return hit
    return fcache_put(scratch, key, builder())
