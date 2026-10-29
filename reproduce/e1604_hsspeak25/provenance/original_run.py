"""HSSPEAK72: explicit HSS peak prediction, injected as a Gaussian bump at long lead.

Preregistered in PREREG.md. Causal by construction: the peak model is trained only on
chronologically earlier folds and every feature is available at the origin.

Usage:
    EXPOS_OUT=<dir> python -m experiments.solar_hsspeak72_20261019.run
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB))
from expos import core, evaluate  # noqa: E402

CONTROL = {1: ("E1508", "roll3"), 2: ("E1508", "roll3"), 3: ("E1508", "roll3"),
           4: ("E1529", "roll3x"), 5: ("E1529", "roll3x"), 6: ("E1529", "roll3x")}
SEEDS = (0, 1, 2)
HORIZON = 72
CARR = 654                      # one Carrington rotation in hours
WIDTH = 12.0                    # bump sigma, hours
SHRINK = 0.15                   # bump height fraction
MIN_LEAD = 25                   # bump from the window start (physical: HSS can peak at h=25)
ORDER = [4, 5, 6, 1, 2, 3]      # chronology; folds 4,5 get no out-of-sample model
CACHE = LB / "experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz"


def control_paths(fold: int) -> tuple[pd.DatetimeIndex, np.ndarray, list[np.ndarray]]:
    eid, sub = CONTROL[fold]
    frames = []
    for seed in SEEDS:
        d = pd.read_parquet(core.RUNS / eid / sub / f"f{fold}_s{seed}.parquet")
        frames.append(pd.DataFrame({
            "origin": pd.to_datetime(d["origin_last_input_utc"], utc=True),
            "h": d["horizon_hours"].astype(int),
            "pred": d["pred_kms"].values}))
    f = frames[0]
    for other in frames[1:]:
        if not (other["origin"].to_numpy() == f["origin"].to_numpy()).all():
            raise SystemExit(f"fold {fold}: seed grids differ")
    per_seed = [x["pred"].values.copy() for x in frames]
    f["pred"] = np.mean(per_seed, axis=0)
    f = f.sort_values(["origin", "h"]).reset_index(drop=True)
    origins = pd.DatetimeIndex(sorted(f["origin"].unique()))
    if len(f) != len(origins) * HORIZON:
        raise SystemExit(f"fold {fold}: incomplete grid")
    return (origins, f["pred"].values.reshape(len(origins), HORIZON),
            [a.reshape(len(origins), HORIZON) for a in per_seed])


def companions(origins: pd.DatetimeIndex):
    z = np.load(CACHE, allow_pickle=True)
    ho = pd.DatetimeIndex(z["origins_ns"].astype("datetime64[ns]"), tz="UTC")
    pos = pd.Series(np.arange(len(ho)), index=ho)
    idx = pos.reindex(origins)
    if idx.isna().any():
        raise SystemExit("HUXt cache does not cover these origins")
    traj = z["traj"].astype(float)[idx.values.astype(int)]
    return traj


def features(origins, path, hx, ace):
    """Causal feature block; each column uses only data at or before the origin."""
    o = origins
    def at(lag):
        return ace.reindex(o - pd.Timedelta(hours=lag)).values.astype(float)
    def roll(win):
        return ace.rolling(win, min_periods=1).mean().reindex(o).values.astype(float)
    rec = np.empty_like(path)
    for h in range(1, HORIZON + 1):
        rec[:, h - 1] = ace.reindex(o + pd.Timedelta(hours=h - CARR)).values
    return np.column_stack([
        np.nanmax(rec, 1), np.nanargmax(rec, 1) + 1.0,
        np.nanmax(rec, 1) - np.nanmin(rec, 1),
        at(0), at(6), at(24), at(168), at(648),
        roll(6), roll(24), roll(168), roll(648),
        np.nanmax(path, 1), np.nanargmax(path, 1) + 1.0,
        np.nanmean(path, 1), np.nanmin(path, 1), np.nanstd(path, 1),
        path[:, 0], path[:, 23], path[:, 47], path[:, 71],
        np.nanmax(hx, 1), np.nanargmax(hx, 1) + 1.0,
        hx[:, 0], hx[:, 23], hx[:, 47], hx[:, 71],
        np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25)])


def bump(path, peak, peak_h):
    hh = np.arange(1, HORIZON + 1, dtype=float)
    c = np.clip(peak_h, 1, HORIZON)
    b = np.exp(-((hh[None, :] - c[:, None]) ** 2) / (2 * WIDTH * WIDTH))
    b = b / np.nanmax(b, 1, keepdims=True)
    base = np.nanmin(path, 1, keepdims=True)
    inc = SHRINK * (peak[:, None] - base) * b
    inc[:, : MIN_LEAD - 1] = 0.0
    return path + np.nan_to_num(inc)


def main() -> None:
    out = Path(os.environ["EXPOS_OUT"])
    ace = evaluate._ace()[0].astype(float)
    data = {}
    for k in range(1, 7):
        origins, path, per_seed = control_paths(k)
        hx = companions(origins)
        data[k] = dict(origins=origins, path=path, seeds=per_seed, hx=hx,
                       F=features(origins, path, hx, ace))
    # targets are the control's residual peak, but ONLY for training folds
    for k in range(1, 7):
        d = data[k]
        y = evaluate._ace()[0].astype(float)
        truth = np.empty_like(d["path"])
        for h in range(1, HORIZON + 1):
            truth[:, h - 1] = y.reindex(d["origins"] + pd.Timedelta(hours=h)).values
        d["tgt_peak"] = np.nanmax(truth, 1)
        d["tgt_hour"] = np.nanargmax(truth, 1) + 1.0
    pred = {}
    for tgt, key in (("tgt_peak", "peak"), ("tgt_hour", "hour")):
        pred[key] = {}
        for i, k in enumerate(ORDER):
            train = ORDER[:i]
            if len(train) < 2:
                continue
            Xtr = np.vstack([data[j]["F"] for j in train])
            Ytr = np.concatenate([data[j][tgt] for j in train])
            m = np.isfinite(Xtr).all(1) & np.isfinite(Ytr)
            if m.sum() < 500:
                continue
            mdl = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.04, max_depth=4,
                min_samples_leaf=50, random_state=0).fit(Xtr[m], Ytr[m])
            x = data[k]["F"]
            ok = np.isfinite(x).all(1)
            v = np.full(len(x), np.nan)
            v[ok] = mdl.predict(x[ok])
            pred[key][k] = v
    receipt = {"experiment": "HSSPEAK72", "width": WIDTH, "shrink": SHRINK,
               "min_lead": MIN_LEAD, "control": {str(k): v for k, v in CONTROL.items()},
               "cache_sha256": core.sha256(CACHE), "folds_corrected": sorted(pred["peak"]),
               "files": {}}
    for k in range(1, 7):
        d = data[k]
        base = pd.DataFrame({
            "origin_last_input_utc": np.repeat(np.asarray(d["origins"]), HORIZON),
            "horizon_hours": np.tile(np.arange(1, HORIZON + 1), len(d["origins"]))})
        for seed in SEEDS:
            own = d["seeds"][SEEDS.index(seed)]
            if k in pred["peak"]:
                q = bump(own, pred["peak"][k], pred["hour"][k])
            else:
                q = own.copy()
            if not np.isfinite(q).all():
                raise SystemExit(f"fold {k} seed {seed}: non-finite prediction")
            dd = base.copy()
            dd["pred_kms"] = q.ravel()
            sub = "roll3" if k <= 3 else "roll3x"
            p = out / sub / f"f{k}_s{seed}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            dd.to_parquet(p, index=False)
            receipt["files"][f"{sub}/f{k}_s{seed}.parquet"] = hashlib.sha256(p.read_bytes()).hexdigest()
        print("HSSPEAK72 FOLD " + json.dumps(
            {"fold": k, "corrected": k in pred["peak"],
             "mean_abs_change": float(np.mean(np.abs(q - d["path"])))}), flush=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"HSSPEAK72 DONE files={len(receipt['files'])}", flush=True)


if __name__ == "__main__":
    main()
