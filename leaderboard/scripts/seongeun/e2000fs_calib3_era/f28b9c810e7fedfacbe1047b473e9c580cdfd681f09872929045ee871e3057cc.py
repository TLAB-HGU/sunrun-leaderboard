"""e2000fs_calib3_era -- reproducible script for config f28b9c810e7fedfacbe1047b473e9c580cdfd681f09872929045ee871e3057cc.

Lane: month-wise adversarial reweighting (three inner rounds, Group-DRO style era splitting) on the
frozen point4 direct-72h recipe. Reproduce with:

    cd /home/t-lab01/sunrun/research/e2000fs_calib3-20261031
    python lanes3.py --lane era --out runs/era_v1 --rounds 900

Inputs are frozen (sha256 in meta.yaml): the archived e2000fs_budget_spectral forecast and the
shared model-input cache; maturity mask origin + 144h <= cutoff; unconditional F1 + F2, no gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_calib3-20261031')))
import lanes3  # noqa: E402

sys.argv = ['lanes3.py', '--lane', 'era', '--out', '/home/t-lab01/sunrun/research/e2000fs_calib3-20261031/runs/era_v1', '--rounds', '900']
lanes3.main()
