"""E1604 (HSSPEAK25) and HSSPEAK72 heads re-fit for the leaderboard F1 period with the F1 training cutoff.

Both entries add the same learned HSS peak correction to E0472: two HistGradientBoosting heads (peak speed and peak
hour) on 29 causal features, trained on the six roll6 folds, then a 12 h Gaussian bump of height
0.15 * (peak - path min) from lead 25 (E1604) or lead 41 (HSSPEAK72). The heads, features and bump are unchanged.

F1 change: the original heads used every roll6 fold row (labels through 2026-05-31). Here only rows whose whole 72 h
label window ends before 2026-02-26 00:00 UTC are kept (folds 4, 5, 6, 1, 2 and fold-3 origins up to 2026-02-22 23:00).
The training rows are the frozen E1604 release matrix (releases/e1604/prepared_inputs.npz); its row order is
recovered from the fold calendar and verified against ACE at the origin. The F1 base paths are the E0472-F1 seeds.

usage: .venv-fusion/bin/python experiments/f1_resubmit_20261026/hsspeak_f1.py --release <e1604 dir> --e0472 <E0472-F1 dir> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB))
from expos import evaluate  # noqa: E402

HORIZON = 72
F1_CUTOFF = pd.Timestamp("2026-02-26T00:00:00Z")
F1_ORIGINS = pd.date_range("2026-02-28 23:00", "2026-05-28 23:00", freq="h", tz="UTC")
# roll6 chronology used by both runners (ORDER = [4, 5, 6, 1, 2, 3]); hourly fold origins
FOLDS = [("2024-06-01", "2024-09-30 23:00"), ("2024-10-01", "2025-01-31 23:00"), ("2025-02-01", "2025-05-31 23:00"),
         ("2025-06-01", "2025-09-30 23:00"), ("2025-10-01", "2026-01-31 23:00"), ("2026-02-01", "2026-05-28 23:00")]
HEAD = dict(max_iter=500, learning_rate=0.04, max_depth=4, min_samples_leaf=50, random_state=0)
WIDTH, SHRINK = 12.0, 0.15
HUXT = LB / "experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz"


def huxt(origins):
    z = np.load(HUXT, allow_pickle=True)
    pos = pd.Series(np.arange(len(z["origins_ns"])),
                    index=pd.DatetimeIndex(z["origins_ns"].astype("datetime64[ns]"), tz="UTC")).reindex(origins)
    if pos.isna().any():
        raise SystemExit("HUXt cache does not cover the F1 origins")
    return z["traj"].astype(float)[pos.values.astype(int)]


def features(o, path, hx, ace):
    """Same 29 columns as experiments/solar_hsspeak72_20261019/run.py features()."""
    def at(lag):
        return ace.reindex(o - pd.Timedelta(hours=lag)).values.astype(float)

    def roll(win):
        return ace.rolling(win, min_periods=1).mean().reindex(o).values.astype(float)
    rec = np.empty_like(path)
    for h in range(1, HORIZON + 1):
        rec[:, h - 1] = ace.reindex(o + pd.Timedelta(hours=h - 654)).values
    return np.column_stack([
        np.nanmax(rec, 1), np.nanargmax(rec, 1) + 1.0, np.nanmax(rec, 1) - np.nanmin(rec, 1),
        at(0), at(6), at(24), at(168), at(648), roll(6), roll(24), roll(168), roll(648),
        np.nanmax(path, 1), np.nanargmax(path, 1) + 1.0, np.nanmean(path, 1), np.nanmin(path, 1), np.nanstd(path, 1),
        path[:, 0], path[:, 23], path[:, 47], path[:, 71],
        np.nanmax(hx, 1), np.nanargmax(hx, 1) + 1.0, hx[:, 0], hx[:, 23], hx[:, 47], hx[:, 71],
        np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25)])


def bump(path, peak, hour, min_lead):
    hh = np.arange(1, HORIZON + 1, dtype=float)
    b = np.exp(-((hh[None, :] - np.clip(hour, 1, HORIZON)[:, None]) ** 2) / (2 * WIDTH * WIDTH))
    b /= np.nanmax(b, 1, keepdims=True)
    inc = SHRINK * (peak[:, None] - np.nanmin(path, 1, keepdims=True)) * b
    inc[:, : min_lead - 1] = 0.0
    return path + np.nan_to_num(inc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--e0472", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ace = evaluate._ace()[0].astype(float)
    with np.load(a.release / "prepared_inputs.npz") as z:
        x, yp, yh = z["train_x"], z["train_peak"], z["train_hour"]
    origins = pd.DatetimeIndex(np.concatenate([pd.date_range(s, e, freq="h", tz="UTC") for s, e in FOLDS]))
    if len(origins) != len(x) or not np.allclose(x[:, 3], ace.reindex(origins).to_numpy(), equal_nan=True):
        raise SystemExit("cannot recover the roll6 row order of the release matrix")
    keep = (origins + pd.Timedelta(hours=HORIZON) < F1_CUTOFF) & np.isfinite(x).all(1) & np.isfinite(yp) & np.isfinite(yh)
    with threadpool_limits(limits=4):
        heads = [HistGradientBoostingRegressor(**HEAD).fit(x[keep], y[keep]) for y in (yp, yh)]
    a.out.mkdir(parents=True, exist_ok=True)
    hx = huxt(F1_ORIGINS)
    outputs = {25: [], 41: []}
    f1_x, f1_base = [], []
    for seed in (0, 1, 2):
        d = pd.read_parquet(a.e0472 / f"predictions_official_s{seed}.parquet")
        d["o"] = pd.to_datetime(d["origin_last_input_utc"], utc=True)
        d = d.sort_values(["o", "horizon_hours"])
        if not (pd.DatetimeIndex(d["o"].unique()) == F1_ORIGINS).all() or len(d) != len(F1_ORIGINS) * HORIZON:
            raise SystemExit(f"E0472-F1 seed {seed} grid is not the F1 grid")
        path = d["pred_kms"].to_numpy(float).reshape(len(F1_ORIGINS), HORIZON)
        f = features(F1_ORIGINS, path, hx, ace)
        if not np.isfinite(f).all():
            raise SystemExit("non-finite F1 features")
        peak, hour = heads[0].predict(f), heads[1].predict(f)
        for lead in outputs:
            outputs[lead].append(bump(path, peak, hour, lead))
        f1_x.append(f)
        f1_base.append(path)
    # release inputs for the published scripts (same layout as releases/e1604/prepared_inputs.npz)
    np.savez(a.out / "hsspeak_f1_inputs.npz", train_origins_ns=origins.asi8, f1_x=np.stack(f1_x),
             f1_base=np.stack(f1_base), f1_origins_ns=F1_ORIGINS.asi8)
    keys = {"origin_last_input_utc": np.repeat(F1_ORIGINS.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON),
            "horizon_hours": np.tile(np.arange(1, HORIZON + 1), len(F1_ORIGINS))}
    for lead, name in ((25, "e1604"), (41, "hsspeak72")):
        for seed, q in enumerate(outputs[lead]):
            pd.DataFrame({**keys, "pred_kms": q.ravel()}).to_parquet(a.out / f"{name}_f1_s{seed}.parquet", index=False)
        pd.DataFrame({**keys, "pred_kms": np.mean(outputs[lead], axis=0).ravel()}).to_parquet(a.out / f"{name}_f1.parquet",
                                                                                               index=False)
    receipt = {"train_rows": int(keep.sum()), "train_rows_original": int(len(x)),
               "last_train_origin": str(origins[keep].max()), "f1_cutoff": str(F1_CUTOFF), "head": HEAD,
               "min_lead": {"e1604": 25, "hsspeak72": 41}, "width": WIDTH, "shrink": SHRINK}
    (a.out / "hsspeak_f1_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
