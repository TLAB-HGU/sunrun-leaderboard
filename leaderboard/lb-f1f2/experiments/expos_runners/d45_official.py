"""Official-window forecasts for the E0105 recipe (D45: 3-seed level mean + fixed event head, hard switch at h=24).

Same code path as experiments/d45_lvlmean/runner.py run_period (E0105 pilot, 6h train grid): same features, params, seeds,
band split and hard switch. Only the period differs: one fit on train rows with target time < 2026-05-29T00Z (72h purge before
the first official target), predicting every official origin 2026-05-31T23Z..2026-09-26T23Z hourly x 72h. Reads no truth after
any origin. Writes $EXPOS_OUT/predictions_official.parquet and timing.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/d45_lvlmean"))
import runner as d45  # noqa: E402  (E0105 code, unchanged)

OUT = Path(os.environ.get("EXPOS_OUT", "."))
CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
ORIGINS = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
TRAIN_FREQ = "6h"  # the density E0105 was evaluated at


def fit(df, cols, params):
    LM, HS = d45.LM, d45.HS
    tr = df[df["target_t"] < CUTOFF]
    tr_s1 = tr[tr["h"] <= 6]
    tr_s2 = tr[(tr["h"] >= 7) & (tr["h"] <= HS.SPLIT_H)]
    short = {}
    for s in LM.check_seeds(LM.LEVEL_SEEDS):
        p = dict(params)
        p["random_state"] = int(s)
        short[s] = (xgb.XGBRegressor(**p).fit(tr_s1[cols], tr_s1["y"]),
                    xgb.XGBRegressor(**p).fit(tr_s2[cols], tr_s2["y"]))
    p_ev = dict(params)
    p_ev["random_state"] = LM.check_event_seed(LM.EVENT_SEED)
    m_long = xgb.XGBRegressor(**p_ev).fit(tr[cols], tr["y"])
    return short, m_long, len(tr)


def predict(models, te, cols):
    LM, HS = d45.LM, d45.HS
    short, m_long, _ = models
    h = te["h"].to_numpy()
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= HS.SPLIT_H), h > HS.SPLIT_H
    pred = np.empty(len(te), dtype=np.float64)
    pred[m1] = LM.mean_predictions({s: short[s][0].predict(te.loc[m1, cols]) for s in LM.LEVEL_SEEDS})
    pred[m2] = LM.mean_predictions({s: short[s][1].predict(te.loc[m2, cols]) for s in LM.LEVEL_SEEDS})
    pred[m3] = np.asarray(m_long.predict(te.loc[m3, cols]), dtype=np.float64)
    return pred


def main():
    params = d45.LM.fit_params()
    df = d45.build_train(TRAIN_FREQ, d45.ST.strip_columns())
    cols = d45.ia.arms(df.columns)["existing_all"] + d45.ST.strip_columns()
    meta, block = d45.ST.load_inputs()
    models = fit(df, cols, params)
    te, _ = d45.add_strips(d45.base_grid(ORIGINS), meta, block)
    pred = predict(models, te, cols)
    if not np.isfinite(pred).all() or len(pred) != len(ORIGINS) * 72:
        raise SystemExit("non-finite or incomplete official predictions; refusing to write")
    d45.write(te, pred, "official")
    # inference timing: one 72h forecast per call (features prepared beforehand), mean over 200 origins
    sample = te[te["origin"].isin(ORIGINS[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        predict(models, g, cols)
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    (OUT / "timing.json").write_text(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": models[2],
                                                 "origins": len(ORIGINS), "img_stale_frac": float(te["img_stale"].mean())}))
    print(f"train={models[2]} origins={len(ORIGINS)} rows={len(te)} per_fold={per_fold:.6f}s", flush=True)


if __name__ == "__main__":
    main()
