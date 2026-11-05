"""e2000fs_chgeo_control_144h -- replay of the paired control arm (round-7 coronal-hole geometry ablation).

The e2000fs_calib_point4_144h recipe (three 24-output vector trees, frozen Top30 237 columns, 900 rounds,
depth 3, eta 0.03, y>=550 weight 4, mask origin+144h <= cutoff) with NO geometry block, fitted at the
F1/F2/F3 cutoffs. Reproduce:

    cd /home/t-lab01/sunrun/research/e2000fs_chgeo-20261101
    python run_lane.py --only control --out runs/v1

Design record: preregistration.md; literature basis: research/notes/lit-review-round7-20261101.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_chgeo-20261101')))
import run_lane  # noqa: E402

run_lane.main(['--only', 'control', '--out', '/home/t-lab01/sunrun/research/e2000fs_chgeo-20261101/runs/v1'])
