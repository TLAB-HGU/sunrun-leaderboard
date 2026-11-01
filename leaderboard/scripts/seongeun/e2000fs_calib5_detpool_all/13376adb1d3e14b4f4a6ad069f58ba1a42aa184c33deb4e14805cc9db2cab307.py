"""e2000fs_calib5_detpool_all -- reproducible script for config 13376adb1d3e14b4f4a6ad069f58ba1a42aa184c33deb4e14805cc9db2cab307.

Stacks every archived arm on top of research/e2000fs_calib4-20261031/runs/gpd_v1/forecast.npz:
  pred = base + sum_j beta_j * (arm_j - base),  beta >= 0, sum(beta) <= 1
beta is refit every 168 origins from the trailing 2160 matured origins (target time <= refit
origin) by standardised NNLS with a 1e-3 ridge, then shrunk until sum(beta) <= 1 and until the
trailing >=550 MSE does not exceed the base's. Arm set, window, cadence, ridge and shrink ladder
are frozen in research/e2000fs_calib5-20261031/preregistration.md; nothing is tuned on F1/F2.
No retraining; inputs are the archived forecasts (sha256 in meta.yaml).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_calib5-20261031')))
import pool_all  # noqa: E402

sys.argv = ['pool_all.py', '--experiment', 'e2000fs_calib5_detpool_all']
pool_all.main()
