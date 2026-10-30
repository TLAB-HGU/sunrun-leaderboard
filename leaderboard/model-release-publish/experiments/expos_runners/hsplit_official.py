"""Official-window machinery for the T1 rocv-adopted hard-switch recipes (E0207, E0212).

Same fit/predict as experiments/d133_t1rocv/{rocv_runner,e0105r_rocv}.py run_fold_seed for ONE seed cell: short 1-6h and 7-24h
experts with the level seed, one long E0003-line head with the event seed, hard switch at h=24, no weight/blend/averaging.
Only the period differs: train on targets < CUTOFF at the 3h grid, predict the given origins x 72h. Reads no truth after any origin.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
ORIGINS = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")


def fit(R, level_params, event_params, cutoff=CUTOFF):
    df = R.build_train("3h", R.ST.strip_columns())
    cols = R.ia.arms(df.columns)["existing_all"] + R.ST.strip_columns()
    tr = df[df["target_t"] < cutoff]
    tr_s1, tr_s2 = tr[tr["h"].between(1, 6)], tr[tr["h"].between(7, R.HS.SPLIT_H)]
    models = (xgb.XGBRegressor(**level_params).fit(tr_s1[cols], tr_s1["y"]),
              xgb.XGBRegressor(**level_params).fit(tr_s2[cols], tr_s2["y"]),
              xgb.XGBRegressor(**event_params).fit(tr[cols], tr["y"]))
    return df, cols, models, len(tr)


def predict(R, models, te, cols):
    h = te["h"].to_numpy()
    pred = np.empty(len(te), dtype=np.float64)
    for mask, m in zip((h <= 6, (h >= 7) & (h <= R.HS.SPLIT_H), h > R.HS.SPLIT_H), models):
        pred[mask] = np.asarray(m.predict(te.loc[mask, cols]), dtype=np.float64)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite predictions; refusing to write")
    return pred


def run(R, level_params, event_params, out=None, origins=ORIGINS, cutoff=CUTOFF, name="official"):
    out = Path(out or os.environ.get("EXPOS_OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    df, cols, models, n_tr = fit(R, level_params, event_params, cutoff)
    meta, block = R.ST.load_inputs()
    te, _ = R.add_strips(R.base_grid(origins), meta, block)
    pred = predict(R, models, te, cols)
    if len(pred) != len(origins) * 72:
        raise SystemExit("incomplete predictions; refusing to write")
    pd.DataFrame({"origin_last_input_utc": te["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "horizon_hours": te["h"].astype(int), "pred_kms": pred.astype(float)}
                 ).to_parquet(out / f"predictions_{name}.parquet", index=False)
    sample = te[te["origin"].isin(pd.DatetimeIndex(origins)[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        predict(R, models, g, cols)
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    (out / "timing.json").write_text(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": n_tr,
                                                 "origins": len(origins), "img_stale_frac": float(te["img_stale"].mean())}))
    print(f"train={n_tr} origins={len(origins)} rows={len(te)} per_fold={per_fold:.6f}s", flush=True)
