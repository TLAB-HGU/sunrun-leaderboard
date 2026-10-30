"""Official-window forecasts for the E0106 recipe (D46: D33 level seed-mean + pinned CH6V2/l1 joint event head, hard branch).

Same code path as experiments/d46_xover2/runner.py run_period (E0106 pilot, 6h train grid): same features, params, seeds, band
split, pinned analog key, scaler and hard common-verdict branch. Only the period differs: one fit on train rows with target
time < 2026-05-29T00Z, predicting every official origin 2026-05-31T23Z..2026-09-26T23Z hourly x 72h. The analog scaler is the
same pre-dev one. Reads no truth after any origin. Writes $EXPOS_OUT/predictions_official.parquet and timing.json.
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
sys.path.insert(0, str(LB / "experiments/d46_xover2"))
import runner as d46  # noqa: E402  (E0106 code, unchanged)

XO2 = d46.XO2
OUT = Path(os.environ.get("EXPOS_OUT", "."))
CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
ORIGINS = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
TRAIN_FREQ = "6h"  # the density E0106 was evaluated at
WINDOW = "official"


def fit(df, cols_level, cols_joint, state, filled_s, scaler):
    tr = df[df["target_t"] < CUTOFF]
    tr_s1, tr_s2 = tr[tr["h"].between(1, 6)], tr[tr["h"].between(7, XO2.SPLIT_H)]
    tr_rep = d46.analog_table(d46.SCRATCH, "tr", WINDOW, pd.DatetimeIndex(tr["origin"].unique()), state, filled_s, scaler)
    tr_full = tr.merge(tr_rep, on=["origin", "h"], how="left")
    level, joint = {}, {}
    for s in XO2.check_seeds(XO2.SEEDS):
        lp = XO2.level_params(s)
        level[s] = (xgb.XGBRegressor(**lp).fit(tr_s1[cols_level], tr_s1["y"]),
                    xgb.XGBRegressor(**lp).fit(tr_s2[cols_level], tr_s2["y"]),
                    xgb.XGBRegressor(**lp).fit(tr[cols_level], tr["y"]))
        joint[s] = xgb.XGBRegressor(**XO2.joint_params(s)).fit(tr_full[cols_joint], tr_full["y"])
    return level, joint, len(tr)


def predict(models, te, te_full, cols_level, cols_joint):
    level, joint, _ = models
    h = te["h"].to_numpy()
    masks = (h <= 6, (h >= 7) & (h <= XO2.SPLIT_H), h > XO2.SPLIT_H)
    level_by_seed, joint_by_seed = {}, {}
    for s, (m1, m2, m3) in level.items():
        pred = np.empty(len(te), dtype=np.float64)
        for mask, m in zip(masks, (m1, m2, m3)):
            pred[mask] = np.asarray(m.predict(te.loc[mask, cols_level]), dtype=np.float64)
        level_by_seed[s] = pred
        joint_by_seed[s] = np.asarray(joint[s].predict(te_full[cols_joint]), dtype=np.float64)
    final, fired = XO2.apply_branch(level_by_seed, joint_by_seed)
    return final, fired


def main():
    df = d46.build_train(TRAIN_FREQ, d46.ST.strip_columns())
    cols_level = d46.ia.arms(df.columns)["existing_all"] + d46.ST.strip_columns()
    cols_joint = cols_level + XO2.analog_columns()
    meta, block = d46.ST.load_inputs()
    state, filled_s = XO2.build_state()
    scaler = XO2.scaler_from_predev(state, pd.Timestamp(d46.PERIODS["dev"][0], tz="UTC"))
    models = fit(df, cols_level, cols_joint, state, filled_s, scaler)
    te, _ = d46.add_strips(d46.base_grid(ORIGINS), meta, block)
    te_rep = d46.analog_table(d46.SCRATCH, "te", WINDOW, pd.DatetimeIndex(te["origin"].unique()), state, filled_s, scaler)
    te_full = te.merge(te_rep, on=["origin", "h"], how="left")
    final, fired = predict(models, te, te_full, cols_level, cols_joint)
    if not np.isfinite(final).all() or len(final) != len(ORIGINS) * 72:
        raise SystemExit("non-finite or incomplete official predictions; refusing to write")
    d46.write(te, final, "official")
    sample_o = ORIGINS[::14][:200]
    mask = te["origin"].isin(sample_o).to_numpy()
    t0 = time.perf_counter()
    for o in sample_o:
        m = (te["origin"] == o).to_numpy()
        predict(models, te[m], te_full[m], cols_level, cols_joint)
    per_fold = (time.perf_counter() - t0) / len(sample_o)
    (OUT / "timing.json").write_text(json.dumps({
        "inference_seconds_per_fold": per_fold, "train_rows": models[2], "origins": len(ORIGINS),
        "img_stale_frac": float(te["img_stale"].mean()), "branch_fire_frac": float(np.mean(fired))}))
    print(f"train={models[2]} origins={len(ORIGINS)} rows={len(te)} fire_frac={np.mean(fired):.4f} per_fold={per_fold:.6f}s", flush=True)


if __name__ == "__main__":
    main()
