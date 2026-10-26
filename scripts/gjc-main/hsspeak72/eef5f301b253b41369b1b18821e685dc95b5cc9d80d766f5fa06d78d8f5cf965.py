"""HSSPEAK72 submission script (leaderboard entry point).

Rebuilds the submitted predictions from the frozen E0472 official predictions by applying an
explicit, physically-shaped HSS peak correction.

Peak amplitude and peak hour are predicted from causal features only (the frozen champion
path, the read-only E1466 TD-HUXt Earth trajectory, ACE history at the origin, and the
phase-aligned recurrence v(origin + h - 654h)) with a HistGradientBoostingRegressor trained on
the six roll6 folds, whose last origin (2026-05-28) is strictly before the official window
(2026-05-31..09-26). A 12h Gaussian bump of height 0.15*(predicted_peak - path_min) is added
at lead >= 41; leads 1-40 are returned bit-identical to the base.

Rationale: an asymmetric Gaussian stream profile level + amp*exp(-((h-mu)/tau_side)^2/2) fits
observed truth at MSE 455.6 versus the champion's 6034.1, and the champion's own fitted
amplitude has median -5 (structurally flat) while truth is +70. The recoverable error is the
peak amplitude at long lead.

F1 (2026-03..05, leaderboard rule since 2026-10-26): build_f1() refits the same two heads on the roll6 rows whose
72h label window ends before 2026-02-26 00:00 UTC and bumps the E0472-F1 base (E0472 recipe with every fit before
2026-02-26, experiments/f1_resubmit_20261026/e0472_f1.py). The roll6 feature rows are the frozen E1604 release matrix,
which reproduces this entry's F2 predictions bit-exactly; the F1 base features are in releases/f1-20261026.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LB = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(LB))
from expos import core, evaluate  # noqa: E402
from experiments.solar_hsspeak72_20261019.run import (  # noqa: E402
    ORDER, SEEDS, bump, companions, control_paths, features,
)

BASE = "E0472"
HORIZON = 72
F1_CUTOFF = pd.Timestamp("2026-02-26T00:00:00Z")
F1_INPUTS = {"releases/e1604/prepared_inputs.npz": "f02d5e04095c28d71c34065b18699765ec4beff3018acc0b9efac736259f5ead",
             "releases/f1-20261026/hsspeak_f1_inputs.npz": "795df2238e423b8246f2c972a1360c2c2cda1721a59da6f662a733c6a1501210"}


def build() -> dict[int, pd.DataFrame]:
    ace = evaluate._ace()[0].astype(float)
    Xtr, ypk, yhr = [], [], []
    for k in ORDER:
        origins, path, _ = control_paths(k)
        Xtr.append(features(origins, path, companions(origins), ace))
        truth = np.empty_like(path)
        for h in range(1, HORIZON + 1):
            truth[:, h - 1] = ace.reindex(origins + pd.Timedelta(hours=h)).values
        ypk.append(np.nanmax(truth, 1))
        yhr.append(np.nanargmax(truth, 1) + 1.0)
    Xtr, ypk, yhr = np.vstack(Xtr), np.concatenate(ypk), np.concatenate(yhr)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ypk) & np.isfinite(yhr)
    kw = dict(max_iter=500, learning_rate=0.04, max_depth=4, min_samples_leaf=50, random_state=0)
    mk = HistGradientBoostingRegressor(**kw).fit(Xtr[ok], ypk[ok])
    mh = HistGradientBoostingRegressor(**kw).fit(Xtr[ok], yhr[ok])
    train_last = max(control_paths(k)[0].max() for k in ORDER)
    out = {}
    for seed in SEEDS:
        d = pd.read_parquet(core.RUNS / BASE / f"predictions_official_s{seed}.parquet")
        d["_o"] = pd.to_datetime(d["origin_last_input_utc"], utc=True)
        d = d.sort_values(["_o", "horizon_hours"]).reset_index(drop=True)
        origins = pd.DatetimeIndex(sorted(d["_o"].unique()))
        if origins.min() <= train_last:
            raise SystemExit("official window overlaps training")
        path = d["pred_kms"].to_numpy(float).reshape(len(origins), HORIZON)
        F = features(origins, path, companions(origins), ace)
        if not np.isfinite(F).all():
            raise SystemExit("non-finite features")
        q = bump(path, mk.predict(F), mh.predict(F))
        out[seed] = pd.DataFrame({
            "origin_last_input_utc": d["origin_last_input_utc"].to_numpy(),
            "horizon_hours": d["horizon_hours"].astype(int).to_numpy(),
            "pred_kms": q.ravel()})
    return out


def build_f1() -> dict[int, pd.DataFrame]:
    for rel, digest in F1_INPUTS.items():
        if core.sha256(LB / rel) != digest:
            raise SystemExit(f"input identity mismatch: {rel}")
    with np.load(LB / "releases/e1604/prepared_inputs.npz") as z:
        Xtr, ypk, yhr = z["train_x"], z["train_peak"], z["train_hour"]
    with np.load(LB / "releases/f1-20261026/hsspeak_f1_inputs.npz") as z:
        train_origins = pd.to_datetime(z["train_origins_ns"], utc=True)
        F1, base1, origins = z["f1_x"], z["f1_base"], pd.to_datetime(z["f1_origins_ns"], utc=True)
    ok = np.asarray(train_origins + pd.Timedelta(hours=HORIZON) < F1_CUTOFF)
    kw = dict(max_iter=500, learning_rate=0.04, max_depth=4, min_samples_leaf=50, random_state=0)
    mk = HistGradientBoostingRegressor(**kw).fit(Xtr[ok], ypk[ok])
    mh = HistGradientBoostingRegressor(**kw).fit(Xtr[ok], yhr[ok])
    out = {}
    for i, seed in enumerate(SEEDS):
        q = bump(base1[i], mk.predict(F1[i]), mh.predict(F1[i]))
        out[seed] = pd.DataFrame({
            "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON),
            "horizon_hours": np.tile(np.arange(1, HORIZON + 1), len(origins)),
            "pred_kms": q.ravel()})
    return out


def predictions() -> pd.DataFrame:
    frames = []
    for parts in (build_f1(), build()):
        m = parts[0].copy()
        m["pred_kms"] = np.mean([p["pred_kms"].to_numpy() for p in parts.values()], axis=0)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)
