"""Training-only frozen calibration recipes; no official-target calibration."""
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

H, BLOCK, GROUP_ROUNDS = 72, 24, 900
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
EXPERIMENTS = ("e2000fs_calib2_capacity_depth5", "e2000fs_calib2_mixture2comp", "e2000fs_calib_tweedie15", "e2000fs_calib3_era", "e2000fs_calib2_detroute")
PARAMS = {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
          "min_child_weight": 20, "reg_lambda": 5, "objective": "reg:squarederror", "tree_method": "hist",
          "device": "cpu", "multi_strategy": "multi_output_tree", "eval_metric": "rmse", "seed": 42, "verbosity": 0}
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last", "ace_temperature_k_m24",
            "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area", "suvi_t035_meridian30_equatorial_area",
            "suvi_t035_equatorial_area", "suvi_t055_meridian15_area", "suvi_t055_meridian30_equatorial_area",
            "suvi_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark", "strip_v650_dark",
           "strip_v750_logmed"]


def objective_for(y, kind="point4", row_weights=None):
    observed = np.isfinite(y)
    safe = np.where(observed, y, 0.)
    weights = np.where(safe >= 550., 4., 1.)
    if row_weights is None:
        weights /= np.sum(weights * observed, axis=0) / observed.sum(axis=0)
    else:
        weights *= np.asarray(row_weights)[:, None]

    def objective(pred, matrix):
        if kind == "tweedie":
            mu = np.clip(pred.astype(np.float64), 1., 3000.)
            scale = 500. ** 1.5
            grad = scale * (mu ** -.5 - safe * mu ** -1.5)
            hess = np.maximum(scale * (-.5 * mu ** -1.5 + 1.5 * safe * mu ** -2.5), 1e-6 * scale)
        else:
            grad, hess = (pred - safe) * weights, weights
        return np.where(observed, grad, 0.).astype(np.float32), np.where(observed, hess, 0.).astype(np.float32)
    return objective


def fit_head(x, y, rounds, threads, out, name, kind="point4", depth=3, base=None, row_weights=None):
    observed = np.isfinite(y)
    if not observed.any(axis=0).all():
        raise ValueError("Each output needs observed training targets")
    matrix = xgb.DMatrix(x, label=np.where(observed, y, 0.).astype(np.float32), nthread=threads)
    params = dict(PARAMS, nthread=threads, max_depth=depth, base_score=float(np.nanmean(y)) if base is None else base,
                  disable_default_eval_metric=True)
    def metric(pred, matrix):
        return "observed_mse", float(((pred-y)[observed] ** 2).mean())
    history = {}
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective_for(y, kind, row_weights),
                      evals=[(matrix, "train")], custom_metric=metric, evals_result=history, verbose_eval=False)
    model.save_model(out / f"{name}.ubj")
    (out / f"{name}_loss.json").write_text(json.dumps(history, indent=2)+"\n")
    return model


def fit_recipe(experiment, x, xq, y, origins, block, rounds, threads, out, prefix):
    keep = np.isfinite(y).any(axis=1)
    x, y, origins = x[keep], y[keep], origins[keep]
    timing = 0.
    def predict(model, values):
        nonlocal timing
        start = time.perf_counter()
        result = model.inplace_predict(values)
        timing += time.perf_counter() - start
        return result
    def head(label, target=y, values=x, **kwargs):
        return fit_head(values, target, rounds, threads, out, prefix+"_"+label, **kwargs)
    def mixture():
        level = predict(head("level"), xq)
        indicator = np.where(np.isfinite(y), (y >= 550.).astype(float), np.nan)
        probability = np.clip(predict(head("probability", indicator), xq), 0., 1.)
        event = (y >= 550.).any(axis=1)
        if not event.any() or not np.isfinite(y[event]).any(axis=0).all():
            raise ValueError("Mixture event subset lacks observed support")
        center = np.nanmean(y[event], axis=0)
        gap = predict(head("gap", y[event]-center, x[event], base=0.), xq)+center
        (out / f"{prefix}_event_center.json").write_text(json.dumps(center.tolist())+"\n")
        return np.clip(level+probability*(gap-level), 1., 3000.)
    if experiment.endswith("capacity_depth5"):
        pred = predict(head("capacity", depth=5), xq)
    elif experiment.endswith("tweedie15"):
        pred = predict(head("tweedie", kind="tweedie"), xq)
    elif experiment.endswith("mixture2comp"):
        pred = mixture()
    elif experiment.endswith("detroute"):
        body = np.clip(predict(head("tweedie", kind="tweedie"), xq), 1., 3000.)
        tail = mixture()
        width = y.shape[1]
        def detector_design(values):
            return np.column_stack([np.repeat(values, width, axis=0), np.tile(np.arange(block*BLOCK+1, block*BLOCK+width+1), len(values))]).astype(np.float32)
        observed = np.isfinite(y).ravel()
        dx = detector_design(x)[observed]
        matrix = xgb.DMatrix(dx, label=(y.ravel()[observed] >= 550.).astype(np.float32), nthread=threads)
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
                  "max_depth": 4, "eta": .05, "subsample": .8, "colsample_bytree": .7, "min_child_weight": 20,
                  "reg_lambda": 5., "seed": 42, "nthread": threads}
        history = {}
        model = xgb.train(params, matrix, num_boost_round=2 if rounds == 2 else 400,
                          evals=[(matrix,"train")], evals_result=history, verbose_eval=False)
        model.save_model(out / f"{prefix}_detector.ubj")
        (out / f"{prefix}_detector_loss.json").write_text(json.dumps(history)+"\n")
        probability = predict(model, detector_design(xq)).reshape(len(xq), width)
        pred = np.where(probability >= .234, tail, body)
    elif experiment.endswith("era"):
        months = origins.month.to_numpy()
        groups = np.unique(months)
        weights = np.ones(len(groups))
        for iteration in range(3):
            row_weights = weights[np.searchsorted(groups, months)]
            row_weights /= row_weights.mean()
            model = head(f"era{iteration}", row_weights=row_weights)
            inpred = model.inplace_predict(x)
            loss = np.array([np.nanmean((inpred[months == g]-y[months == g])**2) for g in groups])
            if loss.mean() > 0:
                weights *= np.exp(.5*(loss/loss.mean()-1.))
                weights /= weights.mean()
        pred = predict(model, xq)
        (out / f"{prefix}_era_weights.json").write_text(json.dumps(dict(zip(map(str,groups), weights.tolist())))+"\n")
    else:
        raise ValueError("Unsupported calibration recipe")
    return np.clip(pred, 1., 3000.), {}, {"predict_seconds": timing, "train_rows": len(y),
        "dropped_no_target": int((~keep).sum()), "observed_per_lead": np.isfinite(y).sum(axis=0).tolist()}


def table(rows):
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    if not n or len(rows) != n * H or not np.array_equal(rows.h.to_numpy().reshape(n, H), np.tile(np.arange(1, H + 1), (n, 1))):
        raise ValueError("Expected unique complete origin x 72h grid")
    origins = pd.to_datetime(rows.origin, utc=True).array.asi8.reshape(n, H)
    if not (origins == origins[:, :1]).all() or len(np.unique(origins[:, 0])) != n:
        raise ValueError("Duplicate or inconsistent origin grid")
    c = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0].copy()
    e = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return pd.DatetimeIndex(origins[:, 0], tz="UTC"), c, e, rows.y.to_numpy(np.float64).reshape(n, H)


def features(block):
    return CONSTANT + [f"{f}@h{h}" for h in range(block * BLOCK + 1, (block + 1) * BLOCK + 1) for f in VARYING]


def design(c, e, block):
    return np.hstack([c, e[:, block * BLOCK:(block + 1) * BLOCK].reshape(len(c), -1)]).astype(np.float32)


def load(data, period):
    df = pd.read_parquet(data, columns=["set", "origin", "h", "period", f"fit_{period}", "y", *CONSTANT, *VARYING], use_threads=False)
    train = df.loc[df[f"fit_{period}"]]
    cutoff = pd.Timestamp(CUTOFFS[period], tz="UTC")
    if not train["set"].eq("train").all() or not (pd.to_datetime(train.origin, utc=True) + pd.Timedelta(hours=144) <= cutoff).all():
        raise ValueError("Training mask violates period maturity")
    training = table(train)
    query = table(df.loc[df["set"].eq("query") & df.period.eq(period)])
    if len(query[0]) != PERIODS[period]:
        raise ValueError("Unexpected official query grid")
    return training, query


def train_period(data, experiment, period, blocks, rounds, threads, out):
    training, query = load(data, period)
    origins, c, e, y = training
    qo, qc, qe, _ = query
    stat = data.stat()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, check=False)
    for block in blocks:
        t0 = time.perf_counter()
        prefix = f"{period}_b{block}"
        x, xq = design(c, e, block), design(qc, qe, block)
        target = y[:, block * BLOCK:(block + 1) * BLOCK]
        pred, extra, info = fit_recipe(experiment, x, xq, target, origins, block, rounds, threads, out, prefix)
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Invalid distribution forecast")
        np.savez_compressed(out / f"{prefix}.npz", pred=pred, origins=qo.asi8, **extra)
        report = {**info, "experiment": experiment, "period": period, "block": block, "rounds": rounds,
                  "seed": 42, "recipe": experiment, "official_target_fits": False,
                  "features": features(block), "feature_count": len(features(block)), "queries": len(qo),
                  "params": dict(PARAMS, nthread=threads),
                  "train_origin_start": str(origins.min()), "train_origin_end": str(origins.max()),
                  "data": {"path": str(data.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                  "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                  "inference_seconds_per_fold": info["predict_seconds"] / len(qo), "seconds": time.perf_counter() - t0,
                  "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}}
        (out / f"{prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("experiment", "period", "block", "rounds", "train_rows", "seconds")}), flush=True)


def merge(out, experiment):
    frames, seconds, identities = [], 0., set()
    for period, count in PERIODS.items():
        parts = []
        for block in range(3):
            with np.load(out / f"{period}_b{block}.npz") as z:
                parts.append({"pred": z["pred"], "origins": z["origins"]})
            report = json.loads((out / f"{period}_b{block}.json").read_text())
            expected_rounds = GROUP_ROUNDS
            if (report["experiment"] != experiment or report["period"] != period or report["block"] != block
                    or report["rounds"] != expected_rounds or report["features"] != features(block)):
                raise ValueError("Mismatched or smoke artifacts cannot be assembled")
            identities.add(json.dumps(report["data"], sort_keys=True))
            seconds += report["predict_seconds"]
        if not all(np.array_equal(z["origins"], parts[0]["origins"]) for z in parts):
            raise ValueError("Prediction origins disagree")
        pred = np.hstack([z["pred"] for z in parts])
        if pred.shape != (count, H) or not np.isfinite(pred).all():
            raise ValueError("Invalid assembled forecast")
        origins = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), count), "pred_kms": pred.ravel()}))
    if len(identities) != 1:
        raise ValueError("Input snapshots differ across blocks")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "predictions.parquet", index=False)
    (out / "merge.json").write_text(json.dumps({"experiment": experiment, "rows": len(frame),
                                                "inference_seconds_per_fold": seconds / sum(PERIODS.values())}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--period", choices=PERIODS)
    parser.add_argument("--block", type=int, choices=range(3))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--phase", choices=("all", "merge"), default="all")
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if a.threads < 1 or (a.block is not None and a.period is None):
        parser.error("threads must be positive; block requires period")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == "merge":
        merge(a.out, a.experiment)
        return
    rounds = 2 if a.smoke else GROUP_ROUNDS
    periods = [a.period or "F1"] if a.smoke or a.period else list(PERIODS)
    blocks = [a.block or 0] if a.smoke else ([a.block] if a.block is not None else list(range(3)))
    for period in periods:
        train_period(a.data, a.experiment, period, blocks, rounds, a.threads, a.out)
    if not a.smoke and not a.period:
        merge(a.out, a.experiment)


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_calib3_era'])
    main()
