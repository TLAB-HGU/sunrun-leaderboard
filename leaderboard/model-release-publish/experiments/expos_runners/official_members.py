"""Shared official-window machinery for the D71 / D76 ensembles (members e0003_line, d33_hsplit, d45_lvlmean).

One 7-fit set on train rows with target time < 2026-05-29T00Z at the 3h origin grid (the density E0141/E0145 were evaluated at):
short 1-6h and 7-24h experts at level seeds 0,1,2 plus one long E0003-line head at event seed 1, stitched exactly as in
experiments/d71_stack/runner.py run_period. Only the period differs. Reads no truth after any origin.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")
ORIGINS = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
TRAIN_FREQ = "3h"


def fit_set(R, df, cols, params):
    LM, HS = R.LM, R.HS
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
    return short, xgb.XGBRegressor(**p_ev).fit(tr[cols], tr["y"]), len(tr)


def members(R, models, te, cols):
    LM, HS = R.LM, R.HS
    short, m_long, _ = models
    h = te["h"].to_numpy()
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= HS.SPLIT_H), h > HS.SPLIT_H
    long_all = np.asarray(m_long.predict(te[cols]), dtype=np.float64)
    s1 = {s: np.asarray(short[s][0].predict(te.loc[m1, cols]), dtype=np.float64) for s in LM.LEVEL_SEEDS}
    s2 = {s: np.asarray(short[s][1].predict(te.loc[m2, cols]), dtype=np.float64) for s in LM.LEVEL_SEEDS}
    long_m3 = long_all[m3]
    mem1 = np.empty(len(te), dtype=np.float64)
    mem1[m1], mem1[m2], mem1[m3] = s1[LM.EVENT_SEED], s2[LM.EVENT_SEED], long_m3
    mem2 = np.empty(len(te), dtype=np.float64)
    mem2[m1], mem2[m2], mem2[m3] = LM.mean_predictions(s1), LM.mean_predictions(s2), long_m3
    return {"e0003_line": long_all, "d33_hsplit": mem1, "d45_lvlmean": mem2}


def prepare(R):
    params = R.LM.fit_params() if hasattr(R.LM, "fit_params") else None
    df = R.build_train(TRAIN_FREQ, R.ST.strip_columns())
    cols = R.ia.arms(df.columns)["existing_all"] + R.ST.strip_columns()
    meta, block = R.ST.load_inputs()
    return params, df, cols, meta, block
