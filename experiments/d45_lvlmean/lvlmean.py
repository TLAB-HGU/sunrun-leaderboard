"""D45 level-branch seed-mean contract: 3-seed mean on the level side, fixed event side.

Hypothesis (E0102/X0096 seed-noisy mean leg, X0097 3-seed spread): the D33
hard-switch mean leg is seed-noisy (E0100 seed 0 mean +0.52pct F1 0.4182 vs
E0102 seed 1 mean -1.08pct F1 0.4138; X0097 third seed 2 mean -1.74pct F1
0.3894, 3-seed spread 2.26pp) while the event leg holds. Averaging ONLY the
level (short, h<=24 band-expert) branch over the same three measured seeds
(0, 1, 2) and keeping the event (long E0003-line, h>=25) branch fixed at the
E0102 passer seed 1 with the D33 hard-switch verdict unchanged isolates the
smoothing effect from any event-branch change. X0075 (level-averaging
destroys events, F1 0.189) is avoided by construction: the event side is
never averaged, never reweighted.

E0102 stays the seed-fragile sole submit candidate (1/3 seeds pass, X0097);
a D45 pass is reported as candidate only alongside that fragility note,
never as sole-pass robustness. No freeze/official-score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/d33_hsplit"))
import info_ablation as ia  # noqa: E402  (read-only import, never vendored)
import hsplit as HS  # noqa: E402  (read-only switch-rule reference, never copied)

LEVEL_SEEDS: tuple[int, int, int] = (0, 1, 2)
EVENT_SEED: int = 1
N_LEVEL = len(LEVEL_SEEDS)
SPLIT_H = HS.SPLIT_H

BASE_PARAMS = {**ia.PARAMS, "device": "cuda"}
FIT_NJOBS = 4
FIT_JOBS = 3

SELECT_RULE = (
    "fixed uniform mean over the level branch only: short band experts "
    "(1-6, 7-24), each fit on its band's rows of the same per-period "
    "expanding train (72h purge) with E0003-identical features and params "
    "(+device cuda, thread-only n_jobs 8->4), refit once per level seed "
    "(0, 1, 2); predictions on h<=24 are the uniform 1/3 mean of the three "
    "seed heads per band. Event branch (25-72 E0003-line single head fit on "
    "all rows) is one fixed fit at the E0102 passer seed 1. Hard switch by "
    "horizon at h=24 (hsplit rules), no weight, no blending, no averaging on "
    "the event side. No fitted choice: no threshold, weight, or config is "
    "fitted on any window; no dev/selection statistic feeds any choice; "
    "selection uses dev/OOF only where a statistic is ever needed."
)

FULL_RULE = (
    "full 3h run (same code without --pilot, new id; same code changed => "
    "new register) iff pilot rc==0 and poison passed and vs-E0001 expos "
    "gate passes and pilot pooled F1 >= 0.35 (no event collapse vs X0075 "
    "0.189/X0082 0.289) and pilot mean-relative vs E0003 <= -0.005 "
    "(smoothing does not hurt the mean leg); otherwise close D45 with no "
    "full run. Repro (seed-repro new id, rotated fresh level seeds "
    "10/11/12 + fixed event seed 1) only after a full passes the adoption "
    "bar (vs E0003 2/3 + mean<=-1% + pooled F1>=0.41121495); E0102 stays "
    "the seed-fragile sole submit candidate (1/3 seeds, X0097) whatever a "
    "single D45 id shows. The bar decides the report only; no "
    "freeze/official-score."
)


def fit_params() -> dict:
    """E0003 recipe params with infra-only changes (device cuda, threads)."""
    p = dict(BASE_PARAMS)
    p["n_jobs"] = FIT_NJOBS
    return p


def seed_params(seed: int) -> dict:
    """E0003 recipe params (cuda) with only random_state changed."""
    if int(seed) not in LEVEL_SEEDS and int(seed) != EVENT_SEED:
        raise ValueError(f"D45 allows only level seeds {LEVEL_SEEDS} + event seed {EVENT_SEED}")
    p = fit_params()
    p["random_state"] = int(seed)
    return p


def check_seeds(seeds) -> tuple:
    """Require exactly the preregistered level seeds; anything else aborts."""
    s = tuple(int(x) for x in seeds)
    if s != LEVEL_SEEDS:
        raise ValueError(f"D45 requires exactly level seeds {LEVEL_SEEDS}, got {s}")
    return s


def check_event_seed(seed: int) -> int:
    """Require exactly the preregistered fixed event seed; else abort."""
    if int(seed) != EVENT_SEED:
        raise ValueError(f"D45 requires exactly event seed {EVENT_SEED}, got {seed}")
    return int(seed)


def mean_predictions(pred_by_seed: dict) -> np.ndarray:
    """Fixed uniform 1/3 mean of the three level-seed prediction vectors."""
    seeds = check_seeds(pred_by_seed)
    arrs = [np.asarray(pred_by_seed[s], dtype=np.float64) for s in seeds]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("level-seed predictions must share one shape")
    if not all(np.isfinite(a).all() for a in arrs):
        raise ValueError("level-seed predictions must be finite for the mean")
    out = sum(arrs) / float(len(arrs))
    return np.asarray(out, dtype=np.float64)


def side_of(h: int) -> str:
    """Hard-switch side for a single horizon 1..72; delegates to hsplit."""
    return HS.side_of(h)


def check_split() -> dict:
    """Verify the hard split tiles 1..72 exactly once at h=24 (via hsplit)."""
    return HS.check_split()
