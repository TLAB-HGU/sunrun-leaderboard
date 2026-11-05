"""e2000fs_calib_point4_144h: the point4 direct-24 XGBoost recipe, with df.parquet as its only input.

Research layer. The input table is built by the DB layer (data-v2/builders/input_df.py); nothing here imports data-v2 or another study.

Model: three 24-output vector-leaf boosters per period (blocks h 1-24, 25-48, 49-72), each on 21 constant + 24 x 9 varying = 237 columns.
Loss: squared error with weight 4 on y >= 550 km/s, normalised by the mean weight per horizon. Training rows: fit_F* (origin + 144 h <= cutoff).

    python experiment.py --jobs 3             # all nine blocks, then results/predictions.parquet and results/check.json
    python experiment.py --period F1 --block 0 --out DIR    # one block
    python experiment.py --smoke --out DIR    # 2 rounds on F1 block 0
    python experiment.py --check              # compare results/ with the published forecast in results/reference/
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
            "ch_ch_t035_top2_area", "ch_ch_t035_meridian15_equatorial_area", "ch_ch_t035_meridian30_equatorial_area",
            "ch_ch_t035_equatorial_area", "ch_ch_t055_meridian15_area", "ch_ch_t055_meridian30_equatorial_area",
            "ch_ch_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]


def weighted_l2(pred, matrix):
    target = matrix.get_label().reshape(pred.shape)
    w = np.where(target >= HIGH_SPEED, HIGH_WEIGHT, 1.0)
    w = (w / w.mean(axis=0)).astype(np.float32)
    return ((pred - target) * w).astype(np.float32), w


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
    assert np.isfinite(y_tr).all(), "training targets must be matured"
    origins, c_qu, e_qu, _ = table(df[(df["set"] == "query") & (df["period"] == period)])
    assert len(origins) == PERIODS[period], (period, len(origins))
    y = y_tr[:, start:start + BLOCK]
    matrix = xgb.DMatrix(design(c_tr, e_tr, start), label=y.astype(np.float32), nthread=1)
    model = xgb.train(dict(PARAMS, base_score=float(np.nanmean(y))), matrix, num_boost_round=rounds, obj=weighted_l2, verbose_eval=False)
    pred = model.inplace_predict(design(c_qu, e_qu, start))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, origins=origins.asi8)
    print(json.dumps({"period": period, "block": block, "train_rows": len(y), "queries": len(origins), "rounds": rounds,
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


def check(out: Path) -> bool:
    """Every value against the published forecast (results/reference, copied from the 2026-10-31 refit)."""
    report, same = {}, True
    for period in PERIODS:
        a, b = np.load(out / f"{period}_predictions.npz"), np.load(RESULTS / "reference" / f"{period}_predictions.npz")
        assert np.array_equal(a["origins"], b["origins"])
        d = np.abs(a["pred"] - b["pred"])
        report[period] = {"values": int(d.size), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
        same &= float(d.max()) == 0.0
    report["identical"] = same
    (out / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(PERIODS))
    ap.add_argument("--block", type=int, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.out) else 1)
    if a.smoke or a.period:
        df = pd.read_parquet(DATA, columns=["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y", *CONSTANT, *VARYING])
        fit_block(df, a.period or "F1", a.block or 0, 2 if a.smoke else ROUNDS, a.out)
        return
    run_all(a.jobs, a.out)
    merge(a.out)
    sys.exit(0 if check(a.out) else 1)


if __name__ == "__main__":
    main()
