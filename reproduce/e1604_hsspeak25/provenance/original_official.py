"""Official-window arm for HSSPEAK72.

Applies the registered peak-injection transform (experiments/solar_hsspeak72_20261019/run.py)
to a champion's frozen official predictions.

Causality:
  * The peak model is trained on all six roll6 folds, whose last origin is 2026-05-28,
    strictly before the official window (2026-05-31..2026-09-26).
  * Every feature is available at its origin: the frozen champion path, the read-only
    HUXt Earth trajectory, ACE history at the origin, and the phase-aligned recurrence
    v(origin + h - 654h), which is always > 582h in the past.
  * No truth, fold or score at any official time is read.

Usage:
    CHAMPION=E0472 EXPOS_OUT=<dir> python -m experiments.solar_hsspeak72_20261019.official
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
from experiments.solar_hsspeak25_20261020.run import (  # noqa: E402
    CONTROL, HORIZON, ORDER, SEEDS, bump, companions, control_paths, features,
)


def champion_frame(champion: str) -> dict[int, pd.DataFrame]:
    out = {}
    for seed in SEEDS:
        p = core.RUNS / champion / f"predictions_official_s{seed}.parquet"
        if not p.exists():
            raise SystemExit(f"missing frozen official predictions: {p}")
        d = pd.read_parquet(p)
        d["_o"] = pd.to_datetime(d["origin_last_input_utc"], utc=True)
        d = d.sort_values(["_o", "horizon_hours"]).reset_index(drop=True)
        if d.duplicated(["_o", "horizon_hours"]).any():
            raise SystemExit(f"{p.name}: duplicate keys")
        if d.groupby("_o").size().ne(HORIZON).any():
            raise SystemExit(f"{p.name}: every origin needs all {HORIZON} horizons")
        out[seed] = d
    for seed in SEEDS[1:]:
        if not (out[seed]["_o"].to_numpy() == out[0]["_o"].to_numpy()).all():
            raise SystemExit("per-seed official grids differ")
    return out


def main() -> None:
    out_dir = Path(os.environ["EXPOS_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    champion = os.environ["CHAMPION"]
    ace = evaluate._ace()[0].astype(float)

    # ---- train the peak model on all six folds (all end before the official window)
    Xtr, ypk, yhr = [], [], []
    for k in ORDER:
        origins, path, _ = control_paths(k)
        hx = companions(origins)
        F = features(origins, path, hx, ace)
        truth = np.empty_like(path)
        for h in range(1, HORIZON + 1):
            truth[:, h - 1] = ace.reindex(origins + pd.Timedelta(hours=h)).values
        Xtr.append(F)
        ypk.append(np.nanmax(truth, 1))
        yhr.append(np.nanargmax(truth, 1) + 1.0)
    Xtr = np.vstack(Xtr)
    ypk = np.concatenate(ypk)
    yhr = np.concatenate(yhr)
    m = np.isfinite(Xtr).all(1) & np.isfinite(ypk) & np.isfinite(yhr)
    if m.sum() < 5000:
        raise SystemExit(f"too few training origins: {int(m.sum())}")
    mdl_pk = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04, max_depth=4,
        min_samples_leaf=50, random_state=0).fit(Xtr[m], ypk[m])
    mdl_hr = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04, max_depth=4,
        min_samples_leaf=50, random_state=0).fit(Xtr[m], yhr[m])
    train_last = max(control_paths(k)[0].max() for k in ORDER)
    print(f"TRAIN rows={int(m.sum())} last_training_origin={train_last}", flush=True)

    # ---- score the official window
    frames = champion_frame(champion)
    base = frames[0][["origin_last_input_utc", "horizon_hours"]]
    origins = pd.DatetimeIndex(sorted(frames[0]["_o"].unique()))
    if origins.min() <= train_last:
        raise SystemExit(f"official window starts {origins.min()} but training ends {train_last}")
    hx = companions(origins)
    receipt = {"arm": "official", "champion": champion, "shrink": 0.15, "width": 12.0,
               "min_lead": 25, "train_rows": int(m.sum()),
               "train_last_origin": str(train_last), "origins": len(origins),
               "files": {}}
    for seed in SEEDS:
        path = frames[seed]["pred_kms"].to_numpy(float).reshape(len(origins), HORIZON)
        F = features(origins, path, hx, ace)
        ok = np.isfinite(F).all(1)
        pk = np.full(len(origins), np.nan)
        hr = np.full(len(origins), np.nan)
        pk[ok] = mdl_pk.predict(F[ok])
        hr[ok] = mdl_hr.predict(F[ok])
        if not ok.all():
            raise SystemExit(f"seed {seed}: {int((~ok).sum())} origins have non-finite features")
        q = bump(path, pk, hr)
        if not np.isfinite(q).all():
            raise SystemExit(f"seed {seed}: non-finite blend")
        d = pd.DataFrame({"origin_last_input_utc": base["origin_last_input_utc"].to_numpy(),
                          "horizon_hours": base["horizon_hours"].astype(int).to_numpy(),
                          "pred_kms": q.ravel()})
        if (d["pred_kms"] <= 0).any() or (d["pred_kms"] > 3000).any():
            raise SystemExit(f"seed {seed}: pred_kms outside (0, 3000]")
        name = f"predictions_official_s{seed}.parquet"
        d.to_parquet(out_dir / name, index=False)
        receipt["files"][name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        print("OFFICIAL SEED DONE " + json.dumps(
            {"seed": seed, "rows": len(d),
             "mean_abs_change": float(np.mean(np.abs(q - path)))}), flush=True)
    receipt["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print("OFFICIAL DONE", flush=True)


if __name__ == "__main__":
    main()
