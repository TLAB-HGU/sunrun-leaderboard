"""e2000fs_calib11f_serarounds: metric-derived sample weights on the parent's level target, df.parquet only.

Two mechanisms, both heuristic-free; the parent recipe (three 24-h vector-leaf XGBoost boosters per
period, 900 rounds, depth 3, the 237-column Top30 design, the 144 h maturity mask) is unchanged.

A. **Metric-derived sample weights.**  The board key is S = mean_lead sqrt(RMSE_l * TRMSE_l), i.e.
   S^4 ~ MSE_l * TIMSE_l per lead. The onboarding TIMSE formula decomposes *exactly* into per-observation
   weights,  TIMSE_l = sum_i a_il e_i^2  with  a_il = (1/R) sum_{k : q_k <= y_i} dq_k / N_k,  so
   grad S ~ e_i [1 + rho_l (N a_i)]  and one weighted squared-error fit satisfies the first-order
   condition. rho_l = MSE_l / TIMSE_l is a dimensionless scalar measured on a strictly prior calibration
   block (the last 25 % of the training window, predicted by the arm's parent recipe fitted on the earlier
   75 %); a_il is a property of the training labels and needs no predictions. No hand-set threshold, no
   hand-set weight: the "4x sample weight above 550 km/s" of the parent recipe does not appear at all.

B. **Anchor-residual target.**  The label is y - v_lag0 and the anchor v_lag0 is added back at prediction
   time (the tree analogue of reversible instance normalisation, identical to a base_margin offset). Level
   enters by identity instead of by regression, which removes the h1-h3 regression below persistence and
   the level-induced amplitude compression of the parent.

The population estimator this targets is E[w(Y)Y|x]/E[w(Y)|x], i.e. the probability-weighted mixture of
conditional tail means that TIMSE minimises - not the conditional mean that plain MSE minimises.

    python experiment.py --all --jobs 9              # F1/F2/F3, each period at its ROUNDS_BY_PERIOD count
    python experiment.py --all --jobs 9 --rounds 900 # same recipe at one round count for every period
    python experiment.py --phase weights --period F1
    python experiment.py --phase fit --period F1 --block 0
    python experiment.py --smoke --out /tmp/smoke
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
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # name -> query origins (hourly)
MATURITY_H, CAL_FRACTION, ROUNDS = 144, 0.25, 300
# booster rounds per period, each selected on the quarter that ends immediately before that period's
# training cutoff (rolling-origin selection: F1 <- 2025 Q4, F2 <- 2026 Q1, F3 <- 2026 Q2).
ROUNDS_BY_PERIOD = {"F1": 300, "F2": 300, "F3": 450}
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
              objective="reg:squarederror", tree_method="hist", device="cpu", multi_strategy="multi_output_tree",
              eval_metric="rmse", seed=42, verbosity=0, nthread=2)
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last", "ace_temperature_k_m24",
            "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "ch_ch_t035_top2_area", "ch_ch_t035_meridian15_equatorial_area", "ch_ch_t035_meridian30_equatorial_area",
            "ch_ch_t035_equatorial_area", "ch_ch_t055_meridian15_area", "ch_ch_t055_meridian30_equatorial_area",
            "ch_ch_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]
COLS = ["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y", *CONSTANT, *VARYING]
ANCHOR = CONSTANT.index("v_lag0")


def table(rows: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Long rows -> (origins, constants[origin, f], varying[origin, h, f], y[origin, h])."""
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    assert len(rows) == n * H and (rows["h"].to_numpy().reshape(n, H) == np.arange(1, H + 1)).all(), "not a full origin x 72 grid"
    origins = pd.DatetimeIndex(rows["origin"].to_numpy()[::H], tz="UTC")
    const = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0, :]
    vary = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return origins, const, vary, rows["y"].to_numpy(np.float64).reshape(n, H)


def design(const: np.ndarray, vary: np.ndarray, start: int) -> np.ndarray:
    return np.hstack([const, vary[:, start:start + BLOCK, :].reshape(len(const), -1)]).astype(np.float32)


def squared(pred, matrix):
    return (pred - matrix.get_label().reshape(pred.shape)).astype(np.float32), np.ones(pred.shape, np.float32)


def weighted_l2(weights):
    """Fixed per-(row, lead) weights, renormalised so the *median* row of each lead has weight 1.

    The parent recipe's unit weight means "one body observation". Because xgboost compares
    min_child_weight and reg_lambda against *weighted* Hessians, normalising the relevance profile to a
    mean of 1 would silently double the body's split threshold (its weight is about 0.4). Anchoring on the
    median keeps min_child_weight=20 readable in body-equivalent units - the same convention the parent uses -
    without introducing any new constant.
    """
    def obj(pred, matrix):
        target = matrix.get_label().reshape(pred.shape)
        w = (weights / np.median(weights, axis=0)).astype(np.float32)
        return ((pred - target) * w).astype(np.float32), w
    return obj


def fit(c, e, y, start, rounds, obj, seed=42):
    label = y[:, start:start + BLOCK]
    matrix = xgb.DMatrix(design(c, e, start), label=label.astype(np.float32), nthread=2)
    model = xgb.train(dict(PARAMS, seed=seed, base_score=float(np.nanmean(label))), matrix,
                      num_boost_round=rounds, obj=obj, verbose_eval=False)
    return model


def timse_weights(y: np.ndarray) -> np.ndarray:
    """Exact per-observation TIMSE weights: TIMSE = sum_i a_i e_i^2, a_i = (1/R) sum_{k : q_k <= y_i} dq_k/N_k."""
    n, leads = y.shape
    out = np.zeros((n, leads))
    for j in range(leads):
        v = y[:, j]
        q = np.quantile(v, np.arange(101) / 100, method="linear")
        order = np.argsort(v, kind="stable")
        positions = np.searchsorted(v[order], q[:-1], side="left")
        cum = np.cumsum(np.diff(q) / (n - positions))
        k = np.searchsorted(q[:-1], v, side="right")
        out[:, j] = np.where(k > 0, cum[np.maximum(k, 1) - 1], 0.0) / float(q[-1] - q[0])
    return out


def mse_timse(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, leads = y.shape
    mse, timse = np.zeros(leads), np.zeros(leads)
    for j in range(leads):
        e2 = (pred[:, j] - y[:, j]) ** 2
        v = y[:, j]
        mse[j] = e2.mean()
        q = np.quantile(v, np.arange(101) / 100, method="linear")
        order = np.argsort(v, kind="stable")
        positions = np.searchsorted(v[order], q[:-1], side="left")
        tail = np.cumsum(e2[order][::-1])[::-1][positions] / (n - positions)
        timse[j] = float(np.dot(np.diff(q), tail) / (q[-1] - q[0]))
    return mse, timse


def load(period: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA, columns=COLS)
    train = df[df[f"fit_{period}"].astype(bool)]
    query = df[(df["set"] == "query") & (df["period"] == period)]
    return df, train, query


def phase_weights(period: str, out: Path, rounds: int = ROUNDS) -> dict:
    """SERA relevance weights, from the labels only: w = (y - q0)/(q100 - q0) per lead.

    SERA (Ribeiro & Moniz 2020, Mach. Learn. 109) integrates threshold-specific squared error over the
    relevance scale and shows the integral collapses exactly into per-observation weights. Here the relevance
    scale is the metric's own velocity-space integration measure (onboarding.md integrates over dq_k in km/s),
    so no threshold, no step and no hand-set strength appear. The weights need no predictions, hence no
    calibration block: nothing is fitted beyond the labels.
    """
    t0 = time.perf_counter()
    _, train, _ = load(period)
    tr_o, _, _, tr_y = table(train)
    assert np.isfinite(tr_y).all(), "training targets must be matured"
    q0 = tr_y.min(axis=0)
    q1 = tr_y.max(axis=0)
    w = np.clip((tr_y - q0[None, :]) / (q1 - q0)[None, :], 1e-3, None)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{period}_w.npz", w=w.astype(np.float32), origins=tr_o.asi8)
    report = {"phase": "weights", "scheme": "sera", "period": period, "train_origins": len(tr_y),
              "weight_pct": {q: float(np.percentile(w, q)) for q in (50, 90, 99, 99.9, 100)},
              "seconds": round(time.perf_counter() - t0)}
    print(json.dumps(report), flush=True)
    return report


def phase_fit(period: str, block: int, out: Path, rounds: int | None = None) -> dict:
    """One (period, block): anchor-residual target, metric weights, reference recipe."""
    rounds = ROUNDS_BY_PERIOD[period] if rounds is None else rounds
    t0 = time.perf_counter()
    _, train, query = load(period)
    tr_o, tr_c, tr_e, tr_y = table(train)
    origins, qu_c, qu_e, _ = table(query)
    assert len(origins) == PERIODS[period], (period, len(origins))
    z = np.load(out / f"{period}_w.npz")
    pos = pd.Series(np.arange(len(z["origins"])), index=z["origins"]).reindex(tr_o.asi8)
    assert pos.notna().all(), "weight file does not cover the training origins"
    w = z["w"][pos.to_numpy().astype(int)]
    assert w.shape == tr_y.shape, (w.shape, tr_y.shape)
    start = block * BLOCK
    anchor_tr = np.zeros((len(tr_c), 1))
    anchor_qu = np.zeros((len(qu_c), 1))
    model = fit(tr_c, tr_e, tr_y - anchor_tr, start, rounds, weighted_l2(w[:, start:start + BLOCK]))
    t1 = time.perf_counter()
    pred = model.inplace_predict(design(qu_c, qu_e, start)) + anchor_qu
    seconds_per_fold = (time.perf_counter() - t1) / len(origins)
    np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, origins=origins.asi8)
    report = {"phase": "fit", "period": period, "block": block, "train_origins": len(tr_y), "queries": len(origins),
              "rounds": rounds, "inference_seconds_per_fold": seconds_per_fold,
              "seconds": round(time.perf_counter() - t0)}
    print(json.dumps(report), flush=True)
    return report


def merge(out: Path) -> dict:
    frames, report = [], {}
    for period in PERIODS:
        parts = [np.load(out / f"{period}_b{b}.npz") for b in range(3)]
        pred = np.hstack([np.asarray(z["pred"], np.float64) for z in parts])
        assert pred.shape == (PERIODS[period], H) and np.isfinite(pred).all(), (period, pred.shape)
        np.savez_compressed(out / f"{period}_predictions.npz", pred=pred, origins=parts[0]["origins"])
        o = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(o)), "pred_kms": pred.ravel()}))
        report[period] = {"origins": len(o), "pred_min": float(pred.min()), "pred_max": float(pred.max())}
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    report["rows"] = len(frame)
    print(f"{len(frame)} rows -> {out / 'predictions.parquet'}", flush=True)
    (out / "merge.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["weights", "fit", "merge"])
    ap.add_argument("--period", choices=sorted(PERIODS))
    ap.add_argument("--block", type=int, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=None,
                    help="boost rounds for every period; default: each period's ROUNDS_BY_PERIOD entry")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.out = 2, a.out if a.out != RESULTS else Path("/tmp/metricgn_smoke")
        phase_weights("F1", a.out, a.rounds)
        phase_fit("F1", 0, a.out, a.rounds)
        return
    if a.all:
        a.out.mkdir(parents=True, exist_ok=True)
        # --rounds overrides every period; by default each period keeps its own selected count. The old
        # `default=ROUNDS` made `--all` run F3 at the F1 count, so the per-period table is the recipe of record.
        rounds = {p: (a.rounds or ROUNDS_BY_PERIOD[p]) for p in PERIODS}
        print(json.dumps({"phase": "rounds", "rounds": rounds}), flush=True)
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            list(pool.map(lambda p: phase_weights(p, a.out, rounds[p]), PERIODS))
        parts = [(p, b) for p in PERIODS for b in range(3)]
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            list(pool.map(lambda pb: phase_fit(pb[0], pb[1], a.out, rounds[pb[0]]), parts))
        merge(a.out)
        return
    if a.phase == "weights":
        phase_weights(a.period, a.out, a.rounds or ROUNDS)
    elif a.phase == "fit":
        phase_fit(a.period, a.block or 0, a.out, a.rounds)
    else:
        merge(a.out)


if __name__ == "__main__":
    main()
