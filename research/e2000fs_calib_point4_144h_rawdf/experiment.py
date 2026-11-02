"""e2000fs_calib_point4_144h_rawdf: the point4 direct-24 XGBoost recipe on the raw-only df.parquet (2026-11-02 rebuild).

Research layer. The only input is data-v2/input/df.parquet, built from data-v2/raw by data-v2/notebooks/raw_to_input_df.ipynb;
nothing here imports data-v2 or another study.

Same model as research/e2000fs_calib_point4_144h/experiment.py: three 24-output vector-leaf boosters per period
(blocks h 1-24, 25-48, 49-72), each on 21 constant + 24 x 9 varying = 237 columns, squared error with weight 4 on
y >= 550 km/s normalised by the mean weight per horizon, training rows fit_F* (origin + 144 h <= cutoff), 900 rounds.
Two changes follow from the new table and nothing else:
* the SUVI dark-area columns are named suvi_t0xx_* instead of ch_ch_t0xx_*;
* y keeps only SWEPAM status-0 values and is never filled, so a missing target gets weight 0 in the loss (masked
  squared error) instead of a filled value; an origin with no observed target in a block is not a training row.

    python experiment.py --jobs 4             # all nine blocks, then results/predictions.parquet
    python experiment.py --period F1 --block 0 --out DIR    # one block
    python experiment.py --smoke --out DIR    # 2 rounds on F1 block 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA = Path("/home/t-lab01/sunrun/data-v2/input/df.parquet")
RESULTS = HERE / "results"
H, BLOCK = 72, 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # name -> query origins
HIGH_SPEED, HIGH_WEIGHT = 550.0, 4.0
ROUNDS = 900
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
              objective="reg:squarederror", tree_method="hist", device="cpu", multi_strategy="multi_output_tree",
              eval_metric="rmse", seed=42, verbosity=0, nthread=1)
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last", "ace_temperature_k_m24",
            "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area", "suvi_t035_meridian30_equatorial_area",
            "suvi_t035_equatorial_area", "suvi_t055_meridian15_area", "suvi_t055_meridian30_equatorial_area",
            "suvi_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]


def masked_weighted_l2(observed: np.ndarray):
    """point4's weighted squared error, with weight 0 on missing targets; the mean weight per horizon is over observed rows."""
    def objective(pred, matrix):
        target = matrix.get_label().reshape(pred.shape)
        w = np.where(target >= HIGH_SPEED, HIGH_WEIGHT, 1.0) * observed
        w = (w / (w.sum(axis=0) / observed.sum(axis=0))).astype(np.float32)
        return ((pred - target) * w).astype(np.float32), w
    return objective


def table(rows: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Long rows -> (origins, constants[origin, feature], varying[origin, h, feature], y[origin, h])."""
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    assert len(rows) == n * H and (rows["h"].to_numpy().reshape(n, H) == np.arange(1, H + 1)).all(), "rows are not a full origin x 72 h grid"
    origins = pd.DatetimeIndex(rows["origin"].to_numpy()[::H], tz="UTC")
    const = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0, :]
    vary = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return origins, const, vary, rows["y"].to_numpy(np.float64).reshape(n, H)


def design(const: np.ndarray, vary: np.ndarray, start: int) -> np.ndarray:
    return np.hstack([const, vary[:, start:start + BLOCK, :].reshape(len(const), -1)]).astype(np.float32)


def fit_block(df: pd.DataFrame, period: str, block: int, rounds: int, out: Path) -> None:
    t0 = time.perf_counter()
    start = block * BLOCK
    _, c_tr, e_tr, y_tr = table(df[df[f"fit_{period}"]])
    origins, c_qu, e_qu, _ = table(df[(df["set"] == "query") & (df["period"] == period)])
    assert len(origins) == PERIODS[period], (period, len(origins))
    y = y_tr[:, start:start + BLOCK]
    keep = np.isfinite(y).any(axis=1)
    x, y = design(c_tr, e_tr, start)[keep], y[keep]
    observed = np.isfinite(y).astype(np.float64)
    matrix = xgb.DMatrix(x, label=np.where(observed > 0, y, 0.0).astype(np.float32), nthread=1)
    model = xgb.train(dict(PARAMS, base_score=float(np.nanmean(y))), matrix, num_boost_round=rounds,
                      obj=masked_weighted_l2(observed), verbose_eval=False)
    pred = model.inplace_predict(design(c_qu, e_qu, start))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, origins=origins.asi8)
    print(json.dumps({"period": period, "block": block, "train_origins": int(len(y)), "dropped_no_target": int((~keep).sum()),
                      "observed_fraction": round(float(observed.mean()), 4), "queries": len(origins), "rounds": rounds,
                      "seconds": round(time.perf_counter() - t0)}), flush=True)


def run_all(jobs: int, out: Path) -> None:
    parts = [(p, b) for p in PERIODS for b in range(3)]

    def one(part):
        p, b = part
        return subprocess.run([sys.executable, __file__, "--period", p, "--block", str(b), "--out", str(out)]).returncode

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        codes = list(pool.map(one, parts))
    if any(codes):
        raise SystemExit(f"blocks failed: {dict(zip(parts, codes))}")


def merge(out: Path) -> None:
    frames = []
    for period in PERIODS:
        parts = [np.load(out / f"{period}_b{b}.npz") for b in range(3)]
        pred = np.hstack([np.asarray(z["pred"], dtype=np.float64) for z in parts])
        assert pred.shape == (PERIODS[period], H) and np.isfinite(pred).all()
        np.savez_compressed(out / f"{period}_predictions.npz", pred=pred, origins=parts[0]["origins"])
        o = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()}))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    print(f"{len(frame)} rows -> {out / 'predictions.parquet'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODS))
    ap.add_argument("--block", type=int, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke or a.period:
        df = pd.read_parquet(DATA, columns=["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y", *CONSTANT, *VARYING])
        fit_block(df, a.period or "F1", a.block or 0, 2 if a.smoke else ROUNDS, a.out)
        return
    run_all(a.jobs, a.out)
    merge(a.out)


if __name__ == "__main__":
    main()
