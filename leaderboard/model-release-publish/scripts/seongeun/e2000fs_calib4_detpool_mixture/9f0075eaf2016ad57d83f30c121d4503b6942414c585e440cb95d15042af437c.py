"""e2000fs_calib4_detpool_mixture -- reproducible script for config 9f0075eaf2016ad57d83f30c121d4503b6942414c585e440cb95d15042af437c.

Submission: e2000fs_calib4_detpool_mixture
Lane: research/e2000fs_calib4-20261031/pool_lane.py (frozen design: preregistration.md).

pred[i] = (1-w[i]) * detroute[i] + w[i] * mixture[i]

w[i] is chosen at origin i from residuals whose target time is at or before that origin
(72h maturity) inside a trailing 2160-origin window: the w on the grid
0..0.75 step 0.025 that minimises the trailing total MSE subject to the trailing >=550 MSE
not exceeding detroute's. Rule, window and grid are fixed in the preregistration and are
never tuned on F1/F2. No retraining; the inputs are the archived forecasts listed in meta.yaml.

Inputs (sha256 recorded in meta.yaml):
  research/e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz
  <arm forecast path from pool_lane.ARMS['mixture']>
  data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_calib4-20261031')))
import pool_lane  # noqa: E402

sys.argv = ['pool_lane.py', '--arm', 'mixture', '--experiment', 'e2000fs_calib4_detpool_mixture']
pool_lane.main()
