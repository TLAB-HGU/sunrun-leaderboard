"""Fit SES + non-seasonal AutoARIMA (same config as analysis/naive_baselines/run_extended.py) for origins after the
existing study's last origin, using data/hourly_series.csv. Run with analysis/naive_baselines/.venv (needs statsforecast).

    ../analysis/naive_baselines/.venv/bin/python extend_baselines.py [--first 2026-09-23T00:00:00Z --last 2026-09-26T23:00:00Z]
"""
import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis" / "naive_baselines"))
import run_extended as R  # noqa: E402  (fit_origin, ARIMA_CONFIG, WINDOW, HORIZON)

ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2026-09-23T00:00:00Z")
ap.add_argument("--last", default="2026-09-26T23:00:00Z")
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

s = pd.read_csv(HERE / "data" / "hourly_series.csv")
s = s.set_index(pd.to_datetime(s["timestamp_utc"], utc=True))["filled_speed_kms"]
origins = pd.date_range(a.first, a.last, freq="h")
pos = {t: i for i, t in enumerate(s.index)}
vals = s.to_numpy()
jobs = [(pos[o], R.history(vals, pos[o])) for o in origins]  # workers only see past observations
with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
    res = list(pool.map(R.fit_origin, jobs, chunksize=4))
rows = []
for o, (idx, es, ar, meta) in zip(origins, res):
    assert idx == pos[o]
    for h in range(R.HORIZON):
        rows.append((o.strftime("%Y-%m-%dT%H:%M:%SZ"), h + 1, es[h], ar[h]))
out = pd.DataFrame(rows, columns=["origin_last_input_utc", "horizon_hours", "es_kms", "auto_arima_kms"])
assert np.isfinite(out[["es_kms", "auto_arima_kms"]].to_numpy()).all()
out.to_parquet(HERE / "data" / "extra_es_arima.parquet", index=False)
print(f"{len(origins)} origins -> data/extra_es_arima.parquet; arima fit codes:", pd.Series([m['code'] for *_, m in res]).value_counts().to_dict())
