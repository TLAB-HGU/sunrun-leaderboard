"""Rebuild only folds.parquet, truth.parquet and naive.parquet for the labeling/config.py PERIODS (F1, F2, F3).

    python rebuild_eval_sets.py --out DIR

build_store.py also regenerates the baseline submissions, which needs the analysis/naive_baselines study files;
this script needs only data/hourly_series.csv (refresh_data.py). The naive forecast is the last observed value,
exactly as in build_store.py. Existing submissions are rescored against these sets by rescore.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "scorer"), str(HERE / "labeling")]
from config import PERIODS  # noqa: E402
from regimes import build_folds  # noqa: E402
from scoring import HORIZON, build_truth  # noqa: E402

SERIES = HERE / "data" / "hourly_series.csv"


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = raw.set_index(pd.to_datetime(raw["timestamp_utc"], utc=True))["filled_speed_kms"]
    folds = pd.concat([build_folds(raw, s, e).assign(period=p) for p, (s, e) in PERIODS.items()], ignore_index=True)
    truth = build_truth(raw, folds)
    origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    pos = series.index.get_indexer(origins)
    assert (pos >= 0).all()
    naive = pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON),
                          "horizon_hours": np.tile(np.arange(1, HORIZON + 1), len(origins)),
                          "pred_kms": np.repeat(series.to_numpy()[pos], HORIZON)})
    return folds, truth, naive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    folds, truth, naive = build(pd.read_csv(SERIES))
    folds.to_parquet(out / "folds.parquet", index=False)
    truth.to_parquet(out / "truth.parquet", index=False)
    naive.to_parquet(out / "naive.parquet", index=False)
    counts = folds.groupby("period")["origin_last_input_utc"].agg(["count", "min", "max"])
    print(counts.to_string())
    print(f"{len(folds)} folds, {len(truth)} truth rows -> {out}")


if __name__ == "__main__":
    main()
