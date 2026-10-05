"""ExpOS runner: existing_all recipe + D1 SUVI longitude strips, nothing else.

Same rows (ia.build grid), same PARAMS, same per-period expanding train with
72h purge; the design matrix only gains the 17 strip columns. Writes
$EXPOS_OUT/predictions_{dev,selection,audit}.parquet (full origin x 72h),
poison.json (real future-poison test, exact key "passed") and
availability_manifest.json (200 samples/period: local MD5 vs S3 ETag HEAD).

`--pilot` runs the identical pipeline on a 6h test/train grid (evaluate still
needs all three period files, so all periods are predicted at pilot density).
CPU only (tree_method hist, n_jobs 8).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(Path(__file__).parent))
import info_ablation as ia  # noqa: E402
import strips as ST  # noqa: E402
from existing_all import full_grid as base_grid  # noqa: E402
from existing_all import write  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
PERIODS = dict(ia.PERIODS)


def add_strips(df: pd.DataFrame, meta, block) -> pd.DataFrame:
    """Join per-(origin,h) strip features onto a long (origin,h) table."""
    origins = pd.DatetimeIndex(df["origin"].unique())
    sel = ST.select_slots(origins, meta, block)
    by_origin = {o: i for i, o in enumerate(sel["origin"].to_numpy())}
    order = np.array([by_origin[o] for o in df["origin"].to_numpy()])
    feats, _ = ST.strip_features(origins, meta, block, selection=sel)
    nrow = len(df)
    for c in ST.strip_columns():
        v = feats[c].to_numpy().reshape(len(origins), 72)
        df[c] = v[order, (df["h"].to_numpy() - 1)].astype(np.float32)
    assert len(df) == nrow
    return df, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--manifest-n", type=int, default=200)
    a = ap.parse_args()
    freq = "6h" if a.pilot else "3h"
    print(f"d1-strips pilot={a.pilot} grid={freq}", flush=True)

    df = ia.build(freq=freq)
    cols = ia.arms(df.columns)["existing_all"]
    meta, block = ST.load_inputs()
    df, train_sel = add_strips(df, meta, block)
    cols = cols + ST.strip_columns()
    print(f"train rows={len(df)} strip_cols={len(ST.strip_columns())} "
          f"train_img_stale_frac={float(df['img_stale'].mean()):.4f}", flush=True)

    sels = {"train": train_sel}
    for name, (lo, hi) in PERIODS.items():
        lo, hi = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
        tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
        te_origins = df.loc[df["origin"].between(lo, hi), "origin"].unique()
        te = base_grid(te_origins)
        te, sel = add_strips(te, meta, block)
        sels[name] = sel
        model = xgb.XGBRegressor(**ia.PARAMS).fit(tr[cols], tr["y"])
        write(te, model.predict(te[cols]), name)
        print(f"{name} train={len(tr)} test={len(te)} origins={len(te_origins)} "
              f"stale_frac={float(te['img_stale'].mean()):.4f}", flush=True)

    all_origins = pd.DatetimeIndex(
        np.concatenate([sels[k]["origin"].to_numpy() for k in ("train", *PERIODS)]))
    cut = pd.to_datetime(sels["audit"]["slot"]).dropna().median().isoformat()
    poison = ST.poison_test(all_origins.unique(), cut, meta=meta, block=block)
    (OUT / "poison.json").write_text(json.dumps(poison, indent=1, default=ST.manifest_json_default))
    print("poison:", json.dumps({k: poison[k] for k in ("passed", "cutoff_utc", "pre_invariant",
          "n_pre_rows", "n_post_rows", "moved_columns")}, default=str), flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed Refusing to present predictions as causal.")

    manifest = ST.build_manifest({k: PERIODS[k] for k in PERIODS}, n_per=a.manifest_n)
    manifest["created_utc"] = ST.utcnow()
    manifest["proxy_rule"] = ("newest slot S<=O-1h with status ok, s3_modified<=O, "
                              "local bytes present, finite cells; else newest earlier "
                              "qualifying + img_age_h; none -> NaN + img_stale=1")
    (OUT / "availability_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=ST.manifest_json_default))
    print("manifest:", json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    main()
