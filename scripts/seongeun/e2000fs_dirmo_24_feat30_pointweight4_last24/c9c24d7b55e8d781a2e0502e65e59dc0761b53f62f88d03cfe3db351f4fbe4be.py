"""Retrain the archived Top30 recipes on the campaign's feature-only ffill table.

The 14 named recipes preserve their own estimator, loss, rounds and readout.
Only input aliases, current F1/F2/F3 fit masks and missing-label masks change.
Each invocation may fit one block/lead or all periods; complete parts can be
assembled with --merge. No official query target participates in fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

HERE = Path(__file__).resolve().parent
H, BLOCK = 72, 24
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6",
            "ace_temperature_k_last", "ace_temperature_k_m24", "ace_bt_nt_last", "ace_bt_nt_m6",
            "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last", "ace_sis_gt10_flux_last",
            "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area", "suvi_t035_meridian30_equatorial_area",
            "suvi_t035_equatorial_area", "suvi_t055_meridian15_area", "suvi_t055_meridian30_equatorial_area",
            "suvi_t055_equatorial_area", "img_age_h", "enlil_n_pending", "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed",
           "strip_v550_dark", "strip_v650_dark", "strip_v750_logmed"]
XGB_PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                  min_child_weight=20, reg_lambda=5, objective="reg:squarederror", tree_method="hist",
                  device="cpu", multi_strategy="multi_output_tree", seed=42, verbosity=0)
LGB_PARAMS = dict(n_estimators=900, objective="regression", learning_rate=0.03, max_depth=3, num_leaves=8,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_weight=20,
                  min_child_samples=20, reg_lambda=5, random_state=42, device_type="cpu", verbosity=-1)
LGB_TUNED = dict(n_estimators=400, objective="regression", learning_rate=0.011665375590900416, max_depth=6,
                 num_leaves=39, subsample=0.8158360436957313, subsample_freq=1, colsample_bytree=0.796156879318864,
                 min_child_samples=76, min_child_weight=1.7495117760617724, reg_alpha=12.889193670860928,
                 reg_lambda=9.785792859825623, random_state=42, device_type="cpu", verbosity=-1)
# A missing entry is rejected, never replaced by a generic estimator.
RECIPES = {
    "e2000fs_calib_point4_144h": ("xgb", "point", 0, False),
    "e2000fs_calib_point4_144h_rawdf": ("xgb", "point", 0, False),
    "e2000fs_dirmo_24_feat30": ("xgb", "plain", 3, True),
    "e2000fs_dirmo_24_feat30_pointweight4": ("xgb", "point", 0, True),
    "e2000fs_dirmo_24_feat30_pointweight4_last24": ("xgb", "point", 2, True),
    "e2000fs_dirmo_24_feat30_pointweight4_last48": ("xgb", "point", 1, True),
    "e2000fs_feat30_wavelet": ("xgb", "wavelet", 3, True),
    "e2000fs_feat30_shape": ("xgb", "shape", 3, True),
    "e2000fs_feat30_extreme": ("xgb", "event", 0, True),
    "e2000fs_dirmo_24_feat30_lgbm": ("lgb", "plain", 3, True),
    "e2000fs_dirmo_24_feat30_lgbm_HPO": ("lgb_tuned", "plain", 3, True),
    "e2000fs_feat30_pointweight4": ("lgb_tuned", "point", 0, True),
    "e2000fs_feat30_lgbm_extreme4": ("lgb_tuned", "event", 0, True),
    "e2000fs_feat30_extreme_histprior": ("histogram", "histogram", 0, True),
}
EDGES = np.array([240., 290., 340., 390., 440., 490., 520., 550., 580., 610., 640., 680., 730., 800., 900., 1000., 1150.])


def haar24(values):
    x = np.moveaxis(np.asarray(values, dtype=np.float64), 1, -1)
    details = []
    for _ in range(3):
        even, odd = x[..., ::2], x[..., 1::2]
        details.append((even - odd) / np.sqrt(2.))
        x = (even + odd) / np.sqrt(2.)
    return np.moveaxis(np.concatenate([x, *reversed(details)], axis=-1), -1, 1)


def design(constant, varying, wavelet=False):
    pieces = [constant, varying.reshape(len(constant), -1)]
    if wavelet:
        pieces += [np.diff(varying, axis=1).reshape(len(constant), -1), haar24(varying).reshape(len(constant), -1)]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def feature_names(block, wavelet=False):
    names = [*CONSTANT, *[f"{n}@h{h}" for h in range(block * BLOCK + 1, (block + 1) * BLOCK + 1) for n in VARYING]]
    if wavelet:
        names += [f"diff_{n}@h{h}" for h in range(block * BLOCK + 2, (block + 1) * BLOCK + 1) for n in VARYING]
        names += [f"haar_{n}@coefficient{i}" for i in range(BLOCK) for n in VARYING]
    return names


def table(rows, block):
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // BLOCK
    if not n or len(rows) != n * BLOCK:
        raise ValueError("Expected a nonempty origin x 24 block")
    hs = rows.h.to_numpy().reshape(n, BLOCK)
    origin_rows = pd.DatetimeIndex(rows.origin)
    origins = origin_rows[::BLOCK]
    if not np.array_equal(hs, np.broadcast_to(np.arange(block * BLOCK + 1, (block + 1) * BLOCK + 1), hs.shape)):
        raise ValueError("Missing or duplicate horizons")
    if not np.array_equal(origin_rows.asi8.reshape(n, BLOCK), np.broadcast_to(origins.asi8[:, None], (n, BLOCK))):
        raise ValueError("Mixed origins within a block")
    c = rows[CONSTANT].to_numpy(np.float32).reshape(n, BLOCK, len(CONSTANT))
    if not np.array_equal(c, np.broadcast_to(c[:, :1], c.shape), equal_nan=True):
        raise ValueError("Constant inputs vary by horizon")
    e = rows[VARYING].to_numpy(np.float32).reshape(n, BLOCK, len(VARYING))
    return origins, c[:, 0].copy(), e, rows.y.to_numpy(np.float64).reshape(n, BLOCK)


def load_block(data, period, block, smoke=False):
    columns = ["set", "origin", "h", "period", f"fit_{period}", "y", *CONSTANT, *VARYING]
    horizon = [("h", ">=", block * BLOCK + 1), ("h", "<=", (block + 1) * BLOCK)]
    rows = pq.read_table(data, columns=columns,
                         filters=[[(f"fit_{period}", "==", True), *horizon],
                                  [("set", "==", "query"), ("period", "==", period), *horizon]],
                         use_threads=False).to_pandas()
    train = rows[rows[f"fit_{period}"]]
    query = rows[(rows["set"] == "query") & (rows["period"] == period)]
    cutoff = pd.Timestamp(CUTOFFS[period], tz="UTC")
    if not train["set"].eq("train").all() or not (train.origin + pd.Timedelta(hours=144) <= cutoff).all():
        raise ValueError("Training rows violate the new df maturity mask")
    if smoke:
        selected = train.origin.drop_duplicates().iloc[-256:]
        train = train[train.origin.isin(selected)]
        selected = query.origin.drop_duplicates().iloc[:8]
        query = query[query.origin.isin(selected)]
    tr, qu = table(train, block), table(query, block)
    if not smoke and len(qu[0]) != PERIODS[period]:
        raise ValueError(f"Unexpected query grid for {period}: {len(qu[0])}")
    return tr, qu


def event_weights(y):
    observed = np.isfinite(y)
    high = ((y >= 550) & observed).any(axis=1)
    ramp = ((y[:, 6:] - y[:, :-6] >= 100) & observed[:, 6:] & observed[:, :-6]).any(axis=1)
    w = np.where(high | ramp, 4., 1.)
    return w / w.mean()


def loss_weights(y, kind):
    observed = np.isfinite(y)
    if not observed.any(axis=0).all():
        raise ValueError("A target horizon has no observed training values")
    if kind == "point":
        raw = np.where(y >= 550, 4., 1.) * observed
        return raw / (raw.sum(axis=0) / observed.sum(axis=0))
    if kind == "event":
        return event_weights(y)[:, None] * observed
    return observed.astype(np.float64)


def masked_objective(y, kind):
    observed = np.isfinite(y)
    target = np.where(observed, y, 0.)
    weights = loss_weights(y, kind)

    def objective(pred, matrix):
        error = (pred - target) * observed
        gradient = error * weights
        hessian = weights.copy()
        if kind == "shape":
            edges = observed[:, :-1] & observed[:, 1:]
            delta = np.diff(error, axis=1) * edges
            gradient[:, :-1] -= .25 * delta
            gradient[:, 1:] += .25 * delta
            hessian[:, :-1] += .5 * edges
            hessian[:, 1:] += .5 * edges
        return gradient.astype(np.float32), hessian.astype(np.float32)
    return objective


def softmax(values):
    z = np.asarray(values, dtype=np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def bin_index(y):
    return np.clip(np.searchsorted(EDGES, y, side="right") - 1, 0, len(EDGES) - 2)


def soft_labels(y):
    from scipy.special import ndtr
    pinned = np.clip(y, EDGES[0], EDGES[-1])
    index = bin_index(pinned)
    band = np.clip(.5 * (EDGES[index + 1] - EDGES[index]), 10., 50.)[:, None]
    q = np.clip(ndtr((EDGES[None, 1:] - pinned[:, None]) / band)
                - ndtr((EDGES[None, :-1] - pinned[:, None]) / band), 0., None)
    return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)


def representatives(y):
    values = y[np.isfinite(y)]
    indices = bin_index(values)
    out = (EDGES[:-1] + EDGES[1:]) / 2.
    for i in range(len(out)):
        if (indices == i).any():
            out[i] = values[indices == i].mean()
    return np.maximum(out, 1.)


def fit_block(experiment, data, out, period, block, threads=1, smoke=False, lead=None):
    engine, kind, first_weighted, clip = RECIPES[experiment]
    if threads < 1:
        raise ValueError("threads must be positive")
    if lead is not None and (engine == "xgb" or not block * BLOCK < lead <= (block + 1) * BLOCK):
        raise ValueError("--lead selects a scalar model within its --block")
    started = time.perf_counter()
    tr, qu = load_block(data, period, block, smoke)
    tr_o, c, e, y = tr
    origins, qc, qe, _ = qu
    x, xq = design(c, e, kind == "wavelet"), design(qc, qe, kind == "wavelet")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{period}_b{block}" if lead is None else f"{period}_h{lead:02}"
    if (out / f"{suffix}.json").exists() or (out / f"{suffix}.npz").exists():
        raise FileExistsError(f"Completed part already exists: {suffix}")
    inference_seconds, model_paths, fit_reports = 0., [], []
    train_ns = tr_o.asi8
    if engine == "xgb":
        keep = np.isfinite(y).any(axis=1)
        x, y = x[keep], y[keep]
        if not len(y):
            raise ValueError("No observed training targets")
        weights_kind = kind if kind != "point" or block >= first_weighted else "plain"
        rounds = 2 if smoke else 900
        params = dict(XGB_PARAMS, nthread=threads, base_score=float(np.nanmean(y)), disable_default_eval_metric=1)
        matrix = xgb.DMatrix(x, label=np.nan_to_num(y, nan=0).astype(np.float32), nthread=threads)
        curve = {}
        def metric(pred, dm):
            return "observed_rmse", float(np.sqrt(np.nanmean((pred - y) ** 2)))
        model = xgb.train(params, matrix, num_boost_round=rounds, obj=masked_objective(y, weights_kind),
                          evals=[(matrix, "train")], custom_metric=metric, evals_result=curve, verbose_eval=False)
        tick = time.perf_counter(); pred = model.inplace_predict(xq); inference_seconds += time.perf_counter() - tick
        path = out / f"{suffix}.ubj"; model.save_model(path); model_paths.append(str(path))
        pd.DataFrame({"round": np.arange(1, rounds + 1), "train_observed_rmse": curve["train"]["observed_rmse"]}).to_csv(out / f"{suffix}_loss.csv", index=False)
        fit_reports.append({"train_origins": int(keep.sum()), "observed_targets": int(np.isfinite(y).sum()), "rounds": rounds,
                            "parameters": params, "loss": weights_kind})
        leads = list(range(block * BLOCK + 1, (block + 1) * BLOCK + 1))
    else:
        leads = [lead] if lead is not None else list(range(block * BLOCK + 1, (block + 1) * BLOCK + 1))
        pred = np.empty((len(origins), len(leads)))
        reps = representatives(y) if engine == "histogram" else None
        if reps is not None:
            (out / f"{suffix}_representatives.json").write_text(json.dumps(reps.tolist()) + "\n")
        for position, hour in enumerate(leads):
            j = hour - block * BLOCK - 1
            keep = np.isfinite(y[:, j])
            if not keep.any():
                raise ValueError(f"No observed target for h{hour}")
            labels, features = y[keep, j], x[keep]
            name = f"{period}_h{hour:02}"
            if engine == "histogram":
                k = len(EDGES) - 1; h_ref = (1. / k) * (1. - 1. / k)
                params = {k_: v for k_, v in XGB_PARAMS.items() if k_ != "multi_strategy"}
                params.update(objective="multi:softprob", num_class=k, min_child_weight=20 * h_ref,
                              reg_lambda=5 * h_ref, nthread=threads, base_score=0., disable_default_eval_metric=1)
                soft = soft_labels(labels).astype(np.float64)
                w = np.where(bin_index(labels) >= 7, 4., 1.); w = (w / w.mean())[:, None]
                def objective(margins, matrix):
                    p = softmax(margins)
                    return (p - soft) * w, np.maximum(p * (1. - p), 1e-6) * w
                matrix = xgb.DMatrix(features, label=np.zeros(len(labels)), nthread=threads)
                rounds = 2 if smoke else 300
                model = xgb.train(params, matrix, num_boost_round=rounds, obj=objective, verbose_eval=False)
                qm = xgb.DMatrix(xq, nthread=threads)
                tick = time.perf_counter(); values = softmax(model.predict(qm, output_margin=True)) @ reps
                inference_seconds += time.perf_counter() - tick
                path = out / f"{name}.ubj"; model.save_model(path)
            else:
                import lightgbm as lgb
                params = dict(LGB_PARAMS if engine == "lgb" else LGB_TUNED, n_jobs=threads)
                if smoke: params["n_estimators"] = 2
                rounds = params["n_estimators"]
                weight = None
                if kind == "point":
                    weight = np.where(labels >= 550, 4., 1.); weight /= weight.mean()
                elif kind == "event":
                    weight = event_weights(y)[keep]; weight /= weight.mean()
                curve = {}
                model = lgb.LGBMRegressor(**params)
                model.fit(features, labels, sample_weight=weight, eval_set=[(features, labels)], eval_names=["train"],
                          eval_metric="l2", callbacks=[lgb.record_evaluation(curve)])
                tick = time.perf_counter(); values = model.predict(xq); inference_seconds += time.perf_counter() - tick
                path = out / f"{name}.txt"; model.booster_.save_model(str(path))
                pd.DataFrame({"round": np.arange(1, len(curve["train"]["l2"]) + 1),
                              "train_observed_mse": curve["train"]["l2"]}).to_csv(out / f"{name}_loss.csv", index=False)
            pred[:, position] = values; model_paths.append(str(path))
            fit_reports.append({"lead": hour, "train_origins": int(keep.sum()), "rounds": rounds, "parameters": params, "loss": kind})
    if clip:
        pred = np.clip(pred, 1., 3000.)
    if not np.isfinite(pred).all() or not (pred > 0).all():
        raise ValueError("Nonpositive/nonfinite predictions; no replacement allowed")
    np.savez_compressed(out / f"{suffix}.npz", pred=pred, origins=origins.asi8, leads=np.array(leads))
    report = {"status": "complete", "experiment": experiment, "period": period, "block": block, "leads": leads,
              "smoke": smoke, "source": str(Path(data).resolve()), "feature_names": feature_names(block, kind == "wavelet"),
              "feature_count": xq.shape[1], "train_origins_before_target_mask": len(tr_o), "target_observed_fraction": float(np.isfinite(tr[3]).mean()),
              "training_origin_first": str(pd.Timestamp(train_ns.min(), tz="UTC")),
              "training_origin_last": str(pd.Timestamp(train_ns.max(), tz="UTC")), "training_cutoff": CUTOFFS[period],
              "maturity_hours": 144, "query_origins": len(origins), "query_origin_first": str(origins.min()), "query_origin_last": str(origins.max()),
              "fits": fit_reports, "model_files": model_paths, "postprocessing": "clip(1,3000)" if clip else "none",
              "inference_seconds_per_fold": inference_seconds / len(origins), "elapsed_seconds": time.perf_counter() - started,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    if engine.startswith("lgb"):
        report["versions"]["lightgbm"] = lgb.__version__
    (out / f"{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["experiment", "period", "block", "smoke", "elapsed_seconds", "inference_seconds_per_fold"]}), flush=True)
    return report


def merge(out, periods=None):
    out = Path(out); frames = []; reports = []
    identity = None
    period_origins = {}
    for period in periods or PERIODS:
        for block in range(3):
            stem = out / f"{period}_b{block}"
            if stem.with_suffix(".npz").exists():
                parts = [stem]
            else:
                parts = [out / f"{period}_h{h:02}" for h in range(block * BLOCK + 1, (block + 1) * BLOCK + 1)]
            for path in parts:
                report = json.loads(path.with_suffix(".json").read_text())
                if report["smoke"]:
                    raise ValueError("Smoke outputs cannot be assembled as full predictions")
                current_identity = (report["experiment"], report["source"], report["maturity_hours"])
                if identity is not None and current_identity != identity:
                    raise ValueError("Prediction parts mix experiment or input identities")
                identity = current_identity
                z = np.load(path.with_suffix(".npz")); o = pd.DatetimeIndex(z["origins"], tz="UTC"); leads = z["leads"]; values = z["pred"]
                if len(o) != PERIODS[period] or values.shape != (len(o), len(leads)):
                    raise ValueError("Unexpected prediction shape")
                expected_leads = list(range(block * BLOCK + 1, (block + 1) * BLOCK + 1)) if len(parts) == 1 else [int(path.name.rsplit("_h", 1)[1])]
                if report["period"] != period or report["block"] != block or list(leads) != expected_leads or report["leads"] != expected_leads:
                    raise ValueError("Part period, block or leads disagree")
                if not o.is_unique or not o.is_monotonic_increasing:
                    raise ValueError("Prediction origins are not unique and ordered")
                if period in period_origins and not np.array_equal(period_origins[period], o.asi8):
                    raise ValueError("Prediction blocks use different origin grids")
                period_origins[period] = o.asi8
                frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(o.strftime("%Y-%m-%dT%H:%M:%SZ"), len(leads)),
                                            "horizon_hours": np.tile(leads, len(o)), "pred_kms": values.ravel()})); reports.append(report)
    frame = pd.concat(frames, ignore_index=True).sort_values(["origin_last_input_utc", "horizon_hours"])
    if frame.duplicated(["origin_last_input_utc", "horizon_hours"]).any() or not np.isfinite(frame.pred_kms).all() or not (frame.pred_kms > 0).all():
        raise ValueError("Invalid assembled predictions")
    if len(frame) != sum(PERIODS[p] for p in periods or PERIODS) * H:
        raise ValueError("Incomplete forecast grid")
    frame.to_parquet(out / "predictions.parquet", index=False)
    (out / "merge.json").write_text(json.dumps({"rows": len(frame), "periods": list(periods or PERIODS),
                                               "parts": len(reports), "inference_seconds_per_fold":
                                               sum(r["inference_seconds_per_fold"] * r["query_origins"] for r in reports) /
                                               sum(PERIODS[p] for p in periods or PERIODS)}, indent=2) + "\n")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(RECIPES))
    parser.add_argument("--data", type=Path, default=HERE / "input/df_ffill.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period", choices=list(PERIODS))
    parser.add_argument("--block", type=int, choices=range(3))
    parser.add_argument("--lead", type=int, choices=range(1, 73))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args.out, [args.period] if args.period else None); return
    if args.lead is not None and args.block is not None and (args.lead - 1) // BLOCK != args.block:
        parser.error("--lead does not belong to --block")
    periods = [args.period] if args.period else (["F1"] if args.smoke else list(PERIODS))
    blocks = [args.block] if args.block is not None else ([(args.lead - 1) // BLOCK] if args.lead else ([0] if args.smoke else list(range(3))))
    for period in periods:
        for block in blocks:
            fit_block(args.experiment, args.data, args.out, period, block, args.threads, args.smoke, args.lead)
    if not args.smoke and args.block is None and args.lead is None:
        merge(args.out, periods)


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_dirmo_24_feat30_pointweight4_last24'])
    main()
