"""e2000fs_chgeo_point4_144h -- replay of the coronal-hole geometry arm (round-7 single axis).

point4_144h recipe + 22 coronal-hole geometry columns from ch-hourly (top1 lat/lon/widths/area/share,
three lat-band area sums, N-S asymmetry at t035 and t055, and the 27-day change of the top1 area share),
value at the newest slot <= origin-1h. Reproduce:

    cd /home/t-lab01/sunrun/research/e2000fs_chgeo-20261101
    python run_lane.py --only chgeo --out runs/v1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_chgeo-20261101')))
import run_lane  # noqa: E402

run_lane.main(['--only', 'chgeo', '--out', '/home/t-lab01/sunrun/research/e2000fs_chgeo-20261101/runs/v1'])
