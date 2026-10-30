"""D32 band-expert contract: four independent horizon-band XGB heads.

Hypothesis (X0008 driver error structure, X0010 band error profile): a single
all-horizon head compromises across bands -- 49-72h MSE is 4-5x the 1-6h MSE
in every period, while shock/HSS/quiet windows carry structurally different
error. Four independent experts on bands 1-6 / 7-24 / 25-48 / 49-72, each fit
only on its band's rows of the E0003-identical train grid (same rows, same
72h purge, same existing_all + 17 proxy-rule strip features via import only),
test whether long-lead (25-72h, HSS-relevant) MSE improves while short-lead
(1-6h) does not regress.

Fixed preregistered bands: no fitted choice consumes any selection statistic.
selection.json is a rule record (dev-window train-row counts only for
provenance); no other-window number feeds any decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import info_ablation as ia  # noqa: E402  (read-only import, never vendored)

BANDS: tuple[tuple[int, int], ...] = ((1, 6), (7, 24), (25, 48), (49, 72))
BAND_NAMES: tuple[str, ...] = tuple(f"{lo}-{hi}" for lo, hi in BANDS)

# Infra-only threading: model math is ia.PARAMS; n_jobs 8->6 keeps 3 parallel
# period fits within the 20-thread budget (3 x 6 = 18). No GPU params.
FIT_NJOBS = 6
PERIOD_JOBS = 3

SELECT_RULE = (
    "fixed preregistered bands 1-6/7-24/25-48/49-72; one XGB expert per band "
    "fit only on that band's rows of the same per-period expanding train "
    "(72h purge) with E0003-identical features and params; test predictions "
    "are the per-band stitches over the full 72 horizons. No fitted choice: "
    "no threshold, weight, or config is fitted on any window; no "
    "dev/selection statistic feeds any choice."
)

FULL_RULE = (
    "full 3h run (same code without --pilot) iff pilot rc==0 and poison "
    "passed and vs-E0001 expos gate passes and vs-E0003 long-lead bands "
    "(25-48 and 49-72) both improve in >=2/3 periods or pilot mean-relative "
    "<=0, with pooled F1 >= 0.30 (no event collapse); otherwise close D32 "
    "with no full run."
)


def fit_params() -> dict:
    """E0003 recipe params with only the thread count changed (infra-only)."""
    p = dict(ia.PARAMS)
    p["n_jobs"] = FIT_NJOBS
    return p


def band_of(h: int) -> str:
    """Band name for a single horizon 1..72; raises outside the grid."""
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        if lo <= int(h) <= hi:
            return name
    raise ValueError(f"horizon {h} outside 1..72")


def check_partition() -> dict:
    """Verify the four bands tile 1..72 exactly once (contract)."""
    covered = sorted(h for lo, hi in BANDS for h in range(lo, hi + 1))
    assert covered == list(range(1, 73)), "bands must tile 1..72 exactly once"
    assert len(set(BAND_NAMES)) == 4
    return {"bands": list(BAND_NAMES), "n_horizons": len(covered)}
