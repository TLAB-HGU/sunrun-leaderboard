"""Submit every baseline in <store>/baselines as team_member=baseline.  python seed_baselines.py <store>"""
import argparse
from pathlib import Path

from submit import DATASET, SPACE, submit

ap = argparse.ArgumentParser()
ap.add_argument("store")
ap.add_argument("--dataset", default=DATASET)
ap.add_argument("--space", default=SPACE)
a = ap.parse_args()
for f in sorted((Path(a.store) / "baselines").glob("*.parquet")):
    meta = {"team_member": "baseline", "experiment": f.stem, "description": f"reference baseline: {f.stem}", "no_future_leakage": True}
    r = submit(str(f), meta, a.dataset, a.space)
    print(f"{f.stem:20s} mse={r['mse']:.1f} beats_naive={r['vs_naive']['beats_reference']}")
print("leaderboard:", r["_leaderboard"])
