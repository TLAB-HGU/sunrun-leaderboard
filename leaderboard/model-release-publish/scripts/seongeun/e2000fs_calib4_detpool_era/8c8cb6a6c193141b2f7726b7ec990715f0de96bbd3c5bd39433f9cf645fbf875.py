"""e2000fs_calib4_detpool_era -- reproducible script for config 8c8cb6a6c193141b2f7726b7ec990715f0de96bbd3c5bd39433f9cf645fbf875.

Submission: e2000fs_calib4_detpool_era
Lane: research/e2000fs_calib4-20261031/pool_lane.py (frozen design: preregistration.md).

pred[i] = (1-w[i]) * detroute[i] + w[i] * era[i]

w[i] is chosen at origin i from residuals whose target time is at or before that origin
(72h maturity) inside a trailing 2160-origin window: the w on the grid
0..0.75 step 0.025 that minimises the trailing total MSE subject to the trailing >=550 MSE
not exceeding detroute's. Rule, window and grid are fixed in the preregistration and are
never tuned on F1/F2. No retraining; the inputs are the archived forecasts listed in meta.yaml.

Inputs (sha256 recorded in meta.yaml):
  research/e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz
  <arm forecast path from pool_lane.ARMS['era']>
  data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_calib4-20261031')))
import pool_lane  # noqa: E402

sys.argv = ['pool_lane.py', '--arm', 'era', '--experiment', 'e2000fs_calib4_detpool_era']
pool_lane.main()
