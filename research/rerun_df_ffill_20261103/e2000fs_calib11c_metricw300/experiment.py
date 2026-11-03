"""Retrain the eight frozen calib11 recipes on the campaign's causal-ffill table.

Only input-column aliases and missing-target masking adapt the original recipes.
This file is self-contained so it can be copied to a per-experiment study directory.
The input must already have the campaign's causal feature and training-y filling.
Residual missing targets are masked; query truth never enters training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd
import xgboost as xgb

H, BLOCK = 72, 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6",
            "ace_temperature_k_last", "ace_temperature_k_m24", "ace_bt_nt_last", "ace_bt_nt_m6",
            "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area",
            "suvi_t035_meridian30_equatorial_area", "suvi_t035_equatorial_area",
            "suvi_t055_meridian15_area", "suvi_t055_meridian30_equatorial_area", "suvi_t055_equatorial_area",
            "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed",
           "strip_v550_dark", "strip_v650_dark", "strip_v750_logmed"]
PARAMS = {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
          "min_child_weight": 20, "reg_lambda": 5, "objective": "reg:squarederror", "tree_method": "hist",
          "device": "cpu", "multi_strategy": "multi_output_tree", "seed": 42, "verbosity": 0}
# (weight scheme, normalization, residual anchor, F1/F2/F3 round counts)
RECIPES = {
    "e2000fs_calib11_metricgn": ("metric", "mean", True, (900, 900, 900)),
    "e2000fs_calib11b_metricwt": ("metric", "mean", False, (900, 900, 900)),
    "e2000fs_calib11c_metricw300": ("metric", "mean", False, (300, 300, 300)),
    "e2000fs_calib11d_sera": ("sera", "mean", False, (300, 300, 300)),
    "e2000fs_calib11e_seramed": ("sera", "median", False, (300, 300, 300)),
    "e2000fs_calib11f_serarounds": ("sera", "median", False, (300, 300, 450)),
    "e2000fs_calib11g_serarounds2": ("sera", "median", False, (300, 300, 1200)),
    "e2000fs_calib11h_noenlil": ("sera", "median", False, (300, 300, 1200)),
}


def constants(experiment):
    cols = list(CONSTANT)
    if experiment == "e2000fs_calib11h_noenlil":
        cols = [c for c in cols if c not in ("enlil_n_pending", "enlil_kp")]
    return cols


def feature_names(experiment, block):
    return constants(experiment) + [f"{f}@h{h}" for h in range(block * BLOCK + 1, (block + 1) * BLOCK + 1)
                                    for f in VARYING]


def table(rows, experiment):
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    if not n or len(rows) != n * H or not np.array_equal(
            rows.h.to_numpy().reshape(n, H), np.tile(np.arange(1, H + 1), (n, 1))):
        raise ValueError("Expected complete, unique origin x 72-hour grids")
    origin_values = pd.to_datetime(rows.origin, utc=True).array.asi8.reshape(n, H)
    if not (origin_values == origin_values[:, :1]).all() or len(np.unique(origin_values[:, 0])) != n:
        raise ValueError("Origin grid contains duplicate or inconsistent rows")
    cols = constants(experiment)
    c = rows[cols].to_numpy(np.float32).reshape(n, H, len(cols))[:, 0, :].copy()
    e = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    y = rows.y.to_numpy(np.float64).reshape(n, H)
    return pd.DatetimeIndex(origin_values[:, 0], tz="UTC"), c, e, y


def load(data, experiment, period):
    columns = ["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y",
               *constants(experiment), *VARYING]
    df = pd.read_parquet(data, columns=columns, use_threads=False)
    train = df.loc[df[f"fit_{period}"]]
    if not train["set"].eq("train").all():
        raise ValueError("Fit mask includes query rows")
    cutoff = pd.Timestamp(CUTOFFS[period], tz="UTC")
    if not (pd.to_datetime(train.origin, utc=True) + pd.Timedelta(hours=144) <= cutoff).all():
        raise ValueError("Training mask violates the period's 144-hour maturity cutoff")
    training = table(train, experiment)
    query = table(df.loc[df["set"].eq("query") & df.period.eq(period)], experiment)
    if len(query[0]) != PERIODS[period]:
        raise ValueError(f"{period} has {len(query[0])} query origins; expected {PERIODS[period]}")
    return training, query


def design(c, e, block):
    start = block * BLOCK
    return np.hstack([c, e[:, start:start + BLOCK].reshape(len(c), -1)]).astype(np.float32)


def normalized_weights(weights, observed, normalization):
    values = np.where(observed, weights, np.nan)
    denominator = np.nanmedian(values, axis=0) if normalization == "median" else np.nanmean(values, axis=0)
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("Every output requires positive finite observed training weights")
    return np.where(observed, weights / denominator, 0.0).astype(np.float32)


def masked_objective(weights):
    def objective(pred, matrix):
        target = matrix.get_label().reshape(pred.shape)
        return ((pred - target) * weights).astype(np.float32), weights
    return objective


def fit(x, y, weights, normalization, rounds, threads, model_path):
    observed = np.isfinite(y)
    keep = observed.any(axis=1)
    if not keep.any() or not observed.sum(axis=0).all():
        raise ValueError("Each output must have observed training labels")
    w = normalized_weights(weights[keep], observed[keep], normalization)
    label = np.where(observed[keep], y[keep], 0.0).astype(np.float32)
    matrix = xgb.DMatrix(x[keep], label=label, nthread=threads)
    history = {}
    mask = observed[keep]

    def observed_rmse(pred, dmat):
        errors = pred - dmat.get_label().reshape(pred.shape)
        return "observed_rmse", float(np.sqrt(np.mean(errors[mask] ** 2)))

    model = xgb.train(dict(PARAMS, nthread=threads, base_score=float(np.nanmean(y[keep])),
                           disable_default_eval_metric=True), matrix, num_boost_round=rounds,
                      obj=masked_objective(w), evals=[(matrix, "train")], custom_metric=observed_rmse,
                      evals_result=history, verbose_eval=False)
    model.save_model(model_path)
    Path(model_path).with_suffix(".loss.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, {"train_origins": int(keep.sum()), "dropped_no_target": int((~keep).sum()),
                   "observed_targets": int(mask.sum()), "observed_per_lead": mask.sum(axis=0).tolist()}


def timse_weights(y):
    """Original exact TIMSE weights, restricted to observed labels independently per lead."""
    out = np.zeros_like(y, dtype=np.float64)
    for j in range(y.shape[1]):
        valid = np.isfinite(y[:, j])
        v = y[valid, j]
        if not len(v) or np.ptp(v) == 0:
            raise ValueError(f"TIMSE needs an observed nonconstant target at lead {j + 1}")
        q = np.quantile(v, np.arange(101) / 100, method="linear")
        positions = np.searchsorted(np.sort(v), q[:-1], side="left")
        cum = np.cumsum(np.diff(q) / (len(v) - positions))
        k = np.searchsorted(q[:-1], v, side="right")
        out[valid, j] = np.where(k > 0, cum[np.maximum(k, 1) - 1], 0.0) / (q[-1] - q[0])
    return out


def mse_timse(pred, y):
    mse, timse = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    a = timse_weights(y)
    for j in range(y.shape[1]):
        valid = np.isfinite(y[:, j])
        if not np.isfinite(pred[valid, j]).all():
            raise ValueError("Nonfinite calibration prediction")
        errors = (pred[valid, j] - y[valid, j]) ** 2
        mse[j], timse[j] = errors.mean(), np.dot(a[valid, j], errors)
    return mse, timse


def sera_weights(y):
    low, high = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
    if not np.isfinite(low).all() or (high <= low).any():
        raise ValueError("SERA needs observed nonconstant targets for every lead")
    return np.where(np.isfinite(y), np.clip((y - low) / (high - low), 1e-3, None), 0.0)


def make_weights(training, experiment, period, rounds, threads, out):
    origins, c, e, y = training
    if RECIPES[experiment][0] == "sera":
        weights, extras = sera_weights(y), {}
    else:
        split = int(len(y) * 0.75)
        if split < 2 or len(y) - split < 2:
            raise ValueError("Insufficient origins for the original 75/25 calibration split")
        cal_pred = np.empty((len(y) - split, H))
        for block in range(3):
            sl = slice(block * BLOCK, (block + 1) * BLOCK)
            model, _ = fit(design(c[:split], e[:split], block), y[:split, sl],
                           np.ones_like(y[:split, sl]), "mean", rounds, threads,
                           out / f"{period}_cal_b{block}.ubj")
            cal_pred[:, sl] = model.inplace_predict(design(c[split:], e[split:], block))
        mse, timse = mse_timse(cal_pred, y[split:])
        if (timse <= 0).any():
            raise ValueError("Zero calibration TIMSE makes the original rho undefined")
        rho = mse / timse
        weights = np.where(np.isfinite(y), 1 + rho * np.isfinite(y).sum(axis=0) * timse_weights(y), 0)
        extras = {"rho": rho, "cal_mse": mse, "cal_timse": timse, "cal_origins": origins[split:].asi8}
    np.savez_compressed(out / f"{period}_w.npz", w=weights.astype(np.float32), origins=origins.asi8, **extras)
    return weights.astype(np.float32)


def source_identity(data):
    stat = data.stat()
    return {"path": str(data.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def train_period(data, experiment, period, blocks, rounds, threads, out, phase="all"):
    t0 = time.perf_counter()
    training, query = load(data, experiment, period)
    origins, c, e, y = training
    query_origins, qc, qe, _ = query
    identity = {"experiment": experiment, "period": period, "data": source_identity(data),
                "rounds": rounds, "normalization": RECIPES[experiment][1], "scheme": RECIPES[experiment][0]}
    weight_meta = out / f"{period}_weights.json"
    if phase == "fit":
        if json.loads(weight_meta.read_text()) != identity:
            raise ValueError("Weight artifact belongs to a different input or recipe")
        with np.load(out / f"{period}_w.npz") as z:
            if not np.array_equal(z["origins"], origins.asi8):
                raise ValueError("Weight origins differ from training origins")
            weights = z["w"]
    else:
        weights = make_weights(training, experiment, period, rounds, threads, out)
        weight_meta.write_text(json.dumps(identity, indent=2) + "\n")
    if phase == "weights":
        return
    anchor = RECIPES[experiment][2]
    anchor_train = c[:, 0, None].astype(np.float64) if anchor else np.zeros((len(c), 1))
    anchor_query = qc[:, 0, None].astype(np.float64) if anchor else np.zeros((len(qc), 1))
    if not np.isfinite(anchor_query).all():
        raise ValueError("Residual query anchor missing after feature forward filling")
    for block in blocks:
        start = time.perf_counter()
        sl = slice(block * BLOCK, (block + 1) * BLOCK)
        model, fit_report = fit(design(c, e, block), (y - anchor_train)[:, sl], weights[:, sl],
                                RECIPES[experiment][1], rounds, threads, out / f"{period}_b{block}.ubj")
        query_x = design(qc, qe, block)
        inference_start = time.perf_counter()
        pred = model.inplace_predict(query_x) + anchor_query
        inference_seconds = time.perf_counter() - inference_start
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Predictions must be finite and positive; no clipping is part of these recipes")
        np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, origins=query_origins.asi8)
        report = {**identity, **fit_report, "block": block, "query_origins": len(query_origins),
                  "training_origin_start": str(origins.min()), "training_origin_end": str(origins.max()),
                  "query_origin_start": str(query_origins.min()), "query_origin_end": str(query_origins.max()),
                  "features": feature_names(experiment, block), "feature_count": len(feature_names(experiment, block)),
                  "params": dict(PARAMS, nthread=threads), "residual_anchor": anchor,
                  "git_commit": git_commit(),
                  "inference_seconds_per_fold": inference_seconds / len(query_origins),
                  "seconds": time.perf_counter() - start, "period_elapsed_seconds": time.perf_counter() - t0,
                  "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}}
        (out / f"{period}_b{block}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("experiment", "period", "block", "rounds", "train_origins", "seconds")}), flush=True)


def merge(out):
    frames, reports = [], []
    for period, count in PERIODS.items():
        parts = [np.load(out / f"{period}_b{b}.npz") for b in range(3)]
        for z in parts[1:]:
            if not np.array_equal(z["origins"], parts[0]["origins"]):
                raise ValueError("Prediction block origins disagree")
        pred = np.hstack([z["pred"] for z in parts])
        if pred.shape != (count, H) or not np.isfinite(pred).all():
            raise ValueError("Invalid assembled prediction grid")
        origin = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origin.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(origin)), "pred_kms": pred.ravel()}))
        for block in range(3):
            report = json.loads((out / f"{period}_b{block}.json").read_text())
            expected_rounds = RECIPES[report["experiment"]][3][list(PERIODS).index(period)]
            if (report["period"] != period or report["block"] != block or report["rounds"] != expected_rounds
                    or report["features"] != feature_names(report["experiment"], block)):
                raise ValueError("Cannot assemble smoke or mismatched recipe artifacts")
            reports.append(report)
    identity = {(r["experiment"], json.dumps(r["data"], sort_keys=True)) for r in reports}
    if len(identity) != 1:
        raise ValueError("Cannot assemble predictions from different experiments or inputs")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    timing = sum(r["inference_seconds_per_fold"] * r["query_origins"] for r in reports) / sum(PERIODS.values())
    (out / "merge.json").write_text(json.dumps({"rows": len(frame), "experiment": reports[0]["experiment"],
                                                "inference_seconds_per_fold": timing}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experiment", choices=RECIPES, required=True)
    ap.add_argument("--period", choices=PERIODS)
    ap.add_argument("--block", type=int, choices=range(3))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--phase", choices=("all", "weights", "fit", "merge"), default="all")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.threads < 1 or (a.block is not None and a.period is None):
        ap.error("threads must be positive; block requires period")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == "merge":
        merge(a.out)
        return
    periods = [a.period or "F1"] if a.smoke or a.period else list(PERIODS)
    blocks = [a.block or 0] if a.smoke else ([a.block] if a.block is not None else list(range(3)))
    for period in periods:
        rounds = 2 if a.smoke else RECIPES[a.experiment][3][list(PERIODS).index(period)]
        train_period(a.data, a.experiment, period, blocks, rounds, a.threads, a.out, a.phase)
    if not a.smoke and not a.period and a.phase in ("all", "fit"):
        merge(a.out)


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_calib11c_metricw300'])
    main()
