"""D71 chronological OOF stacking contract: E0003-line + D33 + D45 members.

Members share one fit set per period (7 XGBoost fits): short 1-6h experts at
level seeds (0, 1, 2), short 7-24h experts at (0, 1, 2), one long E0003-line
head at the fixed event seed 1. Stitched members:

- e0003_line: long head for every h (pure E0003 line, seed 1).
- d33_hsplit: short seed-1 experts for h<=24, long seed-1 head for h>24.
- d45_lvlmean: short uniform 1/3 mean over (0, 1, 2) for h<=24, long seed-1
  head for h>24 (event side never averaged).

Weights are a fixed pre-rule from dev-period OOF only (dev test is OOF
relative to the dev train): inverse dev MSE, clipped to [W_MIN, W_MAX] and
renormalized. No selection/audit statistic feeds any weight, mapping, or
choice. One axis differs from default (ens); features/rows/params are
E0003-identical (+device cuda, thread-only n_jobs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/d33_hsplit"))
import info_ablation as ia  # noqa: E402 (read-only import, never vendored)
import hsplit as HS  # noqa: E402 (read-only switch-rule reference, never copied)

MEMBERS: tuple[str, str, str] = ("e0003_line", "d33_hsplit", "d45_lvlmean")
LEVEL_SEEDS: tuple[int, int, int] = (0, 1, 2)
EVENT_SEED: int = 1
SPLIT_H: int = 24
W_MIN: float = 0.05
W_MAX: float = 0.90

BASE_PARAMS = {**ia.PARAMS, "device": "cuda"}
FIT_NJOBS = 4
FIT_JOBS = 3
SCRATCH_NAME = "worker-d71"

SELECT_RULE = (
    "chronological OOF stacking of e0003_line + d33_hsplit + d45_lvlmean. "
    "One shared fit set per period (short 1-6 x3 seeds + short 7-24 x3 seeds "
    "+ long E0003-line x1 at event seed 1, same expanding train with 72h purge, "
    "E0003-identical features/params +device cuda thread-only n_jobs 8->4). "
    "Members stitched by horizon at h=24 (hsplit rule); event side never "
    "averaged. Global convex weights from a fixed pre-rule on dev-period OOF "
    "only (dev test is OOF vs dev train): inverse dev MSE per member, clipped "
    "to [0.05, 0.90], renormalized to sum 1. Same weights apply to "
    "dev/selection/audit (dev numbers are weight-in-sample, selection/audit "
    "are clean). No selection/audit statistic feeds any weight, mapping, or "
    "choice; no fitted choice beyond the dev-OOF weights."
)

FULL_RULE = (
    "full 3h run (same code without --pilot, new id; code change => new "
    "register) iff pilot rc==0 and poison passed and vs-E0001 expos gate "
    "passes and pilot pooled F1>=0.35 (no event collapse) and pilot "
    "mean-relative vs E0003<=-0.005 (stacking does not hurt the mean leg); "
    "otherwise close D71 with no full run and record budget/curves. Repro "
    "(new id) only after a full passes the adoption bar (vs E0003 2/3 + "
    "mean<=-1% + pooled F1>=0.41121495 + seed-repro). The bar decides the "
    "report only; no freeze and no official-score from any D71 run."
)


def fit_params() -> dict:
    """E0003 recipe params with infra-only changes (device cuda, threads)."""
    p = dict(BASE_PARAMS)
    p["n_jobs"] = FIT_NJOBS
    return p


def seed_params(seed: int) -> dict:
    """E0003 recipe params (cuda) with only random_state changed."""
    if int(seed) not in LEVEL_SEEDS and int(seed) != EVENT_SEED:
        raise ValueError(f"D71 allows only seeds {LEVEL_SEEDS} + event {EVENT_SEED}")
    p = fit_params()
    p["random_state"] = int(seed)
    return p


def check_split() -> dict:
    """Verify the hard split tiles 1..72 exactly once at h=24 (via hsplit)."""
    out = HS.check_split()
    assert out["split_h"] == SPLIT_H == 24
    return out


def check_members(keys) -> tuple:
    """Require exactly the three preregistered members in order."""
    k = tuple(keys)
    if k != MEMBERS:
        raise ValueError(f"D71 requires members {MEMBERS}, got {k}")
    return k


def inverse_mse_weights(dev_mses: dict) -> dict:
    """Fixed pre-rule: inverse dev-MSE weights, clipped + renormalized.

    Input is dev-period MSE per member only (dev test is OOF vs dev train).
    Selection/audit statistics must never enter this function.
    """
    check_members(dev_mses.keys())
    mses = {m: float(dev_mses[m]) for m in MEMBERS}
    if any(not np.isfinite(v) or v <= 0 for v in mses.values()):
        raise ValueError(f"dev MSEs must be finite positive, got {mses}")
    inv = {m: 1.0 / v for m, v in mses.items()}
    tot = sum(inv.values())
    w = {m: inv[m] / tot for m in MEMBERS}
    w = {m: min(W_MAX, max(W_MIN, v)) for m, v in w.items()}
    tot2 = sum(w.values())
    w = {m: float(v / tot2) for m, v in w.items()}
    s = sum(w.values())
    if not np.isfinite(s) or abs(s - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {w}")
    if any(v < 0 for v in w.values()):
        raise ValueError(f"weights must be non-negative, got {w}")
    return w


def stack_predictions(preds_by_member: dict, weights: dict) -> np.ndarray:
    """Convex combination of the three member prediction vectors."""
    check_members(preds_by_member.keys())
    check_members(weights.keys())
    arrs = {m: np.asarray(preds_by_member[m], dtype=np.float64) for m in MEMBERS}
    shape = arrs[MEMBERS[0]].shape
    if any(a.shape != shape for a in arrs.values()):
        raise ValueError("member predictions must share one shape")
    if any(not np.isfinite(a).all() for a in arrs.values()):
        raise ValueError("member predictions must be finite")
    w = [float(weights[m]) for m in MEMBERS]
    if any(not np.isfinite(v) or v < 0 for v in w) or abs(sum(w) - 1.0) > 1e-9:
        raise ValueError(f"weights must be finite non-negative summing to 1, got {weights}")
    out = sum(w[i] * arrs[m] for i, m in enumerate(MEMBERS))
    return np.asarray(out, dtype=np.float64)
