"""Reference runner for ExpOS: fixed existing_all direct XGBoost (headroom audit recipe).

Writes $EXPOS_OUT/predictions_{dev,selection,audit}.parquet (expanding train, 72h purge, 3h origins) and,
with --official, predictions_official.parquet (fit on targets < 2026-05-29, 2833 hourly official origins).
Copy this file as the template for new experiments: keep the output contract, change features/model only.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import info_ablation as ia  # noqa: E402
from headroom import ace_features, ch_features, hourly  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
OFFICIAL_ORIGINS = ("2026-05-31 23:00", "2026-09-26 23:00")
CUTOFF = pd.Timestamp("2026-05-29", tz="UTC")


def write(df, pred, name):
    pd.DataFrame({"origin_last_input_utc": df["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "horizon_hours": df["h"].astype(int), "pred_kms": pred.astype(float)}
                 ).to_parquet(OUT / f"predictions_{name}.parquet", index=False)


def full_grid(origins):
    """Causal feature grid for every requested (origin, h), with or without observed truth; reads nothing after each origin."""
    ace = hourly(pd.read_parquet(LB / "store/ch-v1/ace.parquet"))["filled_speed_kms"]
    o = pd.DatetimeIndex(origins)
    feats = (ia.speed_features(o, ace).join(ace_features(o), rsuffix="_raw").join(ch_features(o))
             .join(ia.enlil_features(o, ia.enlil_runs())))
    df = pd.DataFrame({"origin": np.repeat(o, 72), "h": np.tile(ia.H, len(o))})
    df["target_t"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    for d in (26, 27, 28):
        df[f"rec{d}"] = ace.reindex(pd.DatetimeIndex(df["target_t"] - pd.Timedelta(days=d))).to_numpy()
    df["rec_mean"] = df[["rec26", "rec27", "rec28"]].mean(1)
    df = df.join(feats, on="origin")
    df["enlil_dt_target"] = df["h"] - df["enlil_next_arr_h"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true")
    a = ap.parse_args()
    df = ia.build()
    cols = ia.arms(df.columns)["existing_all"]
    for name, (lo, hi) in ia.PERIODS.items():
        lo, hi = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
        tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
        te = full_grid(df.loc[df["origin"].between(lo, hi), "origin"].unique())  # every origin x 72h
        write(te, xgb.XGBRegressor(**ia.PARAMS).fit(tr[cols], tr["y"]).predict(te[cols]), name)
        print(name, len(tr), len(te), flush=True)
    if a.official:
        tr = df[df["target_t"] < CUTOFF]
        grid = full_grid(pd.date_range(*OFFICIAL_ORIGINS, freq="h", tz="UTC"))
        write(grid, xgb.XGBRegressor(**ia.PARAMS).fit(tr[cols], tr["y"]).predict(grid[cols]), "official")


if __name__ == "__main__":
    main()
