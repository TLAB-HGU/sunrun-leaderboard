"""D33 hard horizon-split hybrid contract: short band experts + E0003 long line.

Hypothesis (E0082 pilot, experience X0076): four independent band experts
improve the 1-6h band by ~40% in every period (audit/dev/selection) but break
long leads (49-72h dev +10.3%) and collapse pooled F1 to 0.289 vs E0003 0.467.
A hard switch -- 1-24h from short band experts (1-6, 7-24) plus 25-72h from
the E0003 line (single all-row head), no weight, no blending, no averaging --
keeps the short-lead gain while restoring long-lead MSE and F1 to the E0003
level. Fixed preregistered split at h=24; no fitted choice consumes any
selection statistic. selection.json is a rule record (dev-window train-row
counts only for provenance); no other-window number feeds any decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/d32_bandexpert"))
import info_ablation as ia  # noqa: E402  (read-only import, never vendored)
import bands as D32B  # noqa: E402  (read-only band-definition import)

BANDS = D32B.BANDS
BAND_NAMES = D32B.BAND_NAMES
SPLIT_H = 24
SHORT_BANDS: tuple[tuple[int, int], ...] = ((1, 6), (7, 24))
LONG_BANDS: tuple[tuple[int, int], ...] = ((25, 48), (49, 72))

BASE_PARAMS = {**ia.PARAMS, "device": "cuda"}
FIT_NJOBS = 6
FIT_JOBS = 3

SELECT_RULE = (
    "fixed preregistered split at h=24; 1-24h from two short band experts "
    "(1-6, 7-24) fit only on that band's rows, 25-72h from one E0003-line "
    "single head fit on all rows of the same per-period expanding train "
    "(72h purge) with E0003-identical features and params (+device cuda, "
    "thread-only n_jobs 8->6); hard switch by horizon, no weight, no "
    "blending, no averaging. No fitted choice: no threshold, weight, or "
    "config is fitted on any window; no dev/selection statistic feeds any "
    "choice."
)

FULL_RULE = (
    "full 3h run (same code without --pilot) iff pilot rc==0 and poison "
    "passed and vs-E0001 expos gate passes and vs-E0003 short-lead bands "
    "(1-6 and 7-24) both improve in >=2/3 periods or pilot mean-relative "
    "<=-0.01, with pooled F1 >= 0.35 (no event collapse vs E0082 0.289); "
    "otherwise close D33 with no full run. Adoption bar for a "
    "submission-candidate full (vs E0003 2/3 + mean<=-1% + pooled "
    "F1>=0.41121495 + seed-repro new id) decides the report only; no "
    "freeze/official-score."
)


def fit_params() -> dict:
    """E0003 recipe params with infra-only changes (device cuda, threads)."""
    p = dict(BASE_PARAMS)
    p["n_jobs"] = FIT_NJOBS
    return p


def side_of(h: int) -> str:
    """Hard-switch side for a single horizon 1..72; raises outside the grid."""
    h = int(h)
    if 1 <= h <= SPLIT_H:
        return "short"
    if SPLIT_H < h <= 72:
        return "long"
    raise ValueError(f"horizon {h} outside 1..72")


def check_split() -> dict:
    """Verify the hard split tiles 1..72 exactly once at h=24 (contract)."""
    assert BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert tuple(BAND_NAMES) == tuple(D32B.BAND_NAMES)
    short = sorted(h for lo, hi in SHORT_BANDS for h in range(lo, hi + 1))
    long = sorted(h for lo, hi in LONG_BANDS for h in range(lo, hi + 1))
    assert short == list(range(1, SPLIT_H + 1)), "short side must be 1..24"
    assert long == list(range(SPLIT_H + 1, 73)), "long side must be 25..72"
    assert len(set(short) & set(long)) == 0
    return {"split_h": SPLIT_H, "short": ["1-6", "7-24"], "long": ["25-48", "49-72"]}
