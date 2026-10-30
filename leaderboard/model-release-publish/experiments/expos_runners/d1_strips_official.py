"""Official-window forecasts for the E0003 recipe (existing_all + D1 SUVI longitude strips), for leaderboard submission.

Same features, params and code path as E0003 (experiments/d1_longitude_strips/runner.py); one fit on all training rows
with target time < 2026-05-29T00Z, then predicts every official origin (2026-05-31T23Z..2026-09-26T23Z hourly) x 72h.
Reads no truth after any origin; writes $EXPOS_OUT/predictions_official.parquet and timing.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
import runner as d1  # noqa: E402  (E0003 code, unchanged)

OUT = Path(os.environ.get("EXPOS_OUT", "."))
CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
ORIGINS = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")


def main():
    df = d1.ia.build()
    meta, block = d1.ST.load_inputs()
    df, _ = d1.add_strips(df, meta, block)
    cols = d1.ia.arms(df.columns)["existing_all"] + d1.ST.strip_columns()
    tr = df[df["target_t"] < CUTOFF]
    model = xgb.XGBRegressor(**d1.ia.PARAMS).fit(tr[cols], tr["y"])
    te, _ = d1.add_strips(d1.base_grid(ORIGINS), meta, block)
    t0 = time.perf_counter()
    pred = model.predict(te[cols])
    per_fold = (time.perf_counter() - t0) / len(ORIGINS)
    d1.write(te, pred, "official")
    (OUT / "timing.json").write_text(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": len(tr),
                                                 "origins": len(ORIGINS), "img_stale_frac": float(te["img_stale"].mean())}))
    print(f"train={len(tr)} origins={len(ORIGINS)} rows={len(te)} per_fold={per_fold:.6f}s", flush=True)


if __name__ == "__main__":
    main()
