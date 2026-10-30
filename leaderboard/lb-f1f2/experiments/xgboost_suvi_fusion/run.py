"""Local feature, training and evaluation entry point; never uploads results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.xgboost_suvi_fusion"
    from experiments.xgboost_suvi_fusion import report
else:
    from . import report


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.is_dir() or path.suffix == ".parquet" else pd.read_csv(path)


def load_ace(paths: list[str]) -> pd.DataFrame:
    """Combine hourly inputs in priority order and fill using past values only."""
    parts = []
    for path in paths:
        table = read_table(path)
        index = pd.to_datetime(table["timestamp_utc"], utc=True)
        column = "filled_speed_kms" if "filled_speed_kms" in table else "ace_speed_kms"
        values = pd.to_numeric(table[column], errors="coerce")
        values = values.where(np.isfinite(values) & values.gt(0))
        missing = table["was_missing"].astype(bool) if "was_missing" in table else values.isna()
        parts.append(pd.DataFrame({"filled_speed_kms": values.to_numpy(), "was_missing": missing.to_numpy()}, index=index))
    if not parts:
        raise ValueError("at least one ACE input is required")
    series = pd.concat(parts)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) == 0 or series["filled_speed_kms"].first_valid_index() is None:
        raise ValueError("ACE inputs contain no valid speed")
    if not series.index.equals(series.index.floor("h")):
        raise ValueError("ACE timestamps must be hourly")
    series = series.loc[series["filled_speed_kms"].first_valid_index():]
    series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))
    series["was_missing"] = (series["was_missing"].astype("boolean").fillna(True) | series["filled_speed_kms"].isna()).astype(int)
    series["filled_speed_kms"] = series["filled_speed_kms"].ffill()
    return series.rename_axis("timestamp_utc").reset_index()


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    size = 0
    for item in files:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size += item.stat().st_size
    return {"path": str(path.resolve()), "bytes": size, "files": len(files), "sha256": digest.hexdigest()}


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def make_features(args, ace: pd.DataFrame):
    from . import features
    if not args.suvi_meta or not args.images_root or not args.feature_cache:
        raise ValueError("feature extraction requires --suvi-meta, --images-root and --feature-cache")
    hourly = features.build_hourly_cache(
        read_table(args.suvi_meta), Path(args.images_root), Path(args.feature_cache),
        workers=args.image_workers,
    )
    cutoff = pd.Timestamp(args.train_cutoff)
    medians = features.fit_image_medians(hourly, cutoff)
    timestamps = pd.DatetimeIndex(ace["timestamp_utc"])
    start = max(timestamps.min(), hourly.index.min()) + pd.Timedelta(hours=119)
    origins = timestamps[timestamps >= start]
    frame = features.build_origin_features(ace, hourly, origins, medians)
    return frame


def data_identity(args) -> dict:
    cache_manifest = Path(args.feature_cache).with_suffix(Path(args.feature_cache).suffix + ".json")
    identity = {
        "leaderboard_revision": args.leaderboard_revision,
        "ace_sha256": [file_identity(path)["sha256"] for path in args.ace],
        "suvi_metadata_sha256": file_identity(args.suvi_meta)["sha256"],
        "suvi_cache_source_sha256": json.loads(cache_manifest.read_text())["source_sha256"],
    }
    if args.origin_features:
        identity["origin_features_sha256"] = file_identity(args.origin_features)["sha256"]
    return identity


def load_origin_features(path) -> pd.DataFrame:
    frame = read_table(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name="origin_last_input_utc")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("origin feature index must be unique and sorted")
    return frame


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("features", "tune", "final", "evaluate"), default="features")
    result.add_argument("--ace", nargs="+", required=True)
    result.add_argument("--suvi-meta")
    result.add_argument("--images-root")
    result.add_argument("--folds")
    result.add_argument("--feature-cache")
    result.add_argument("--origin-features", help="precomputed output from features mode")
    result.add_argument("--output", required=True)
    result.add_argument("--train-cutoff", default="2025-07-31T23:00:00Z")
    result.add_argument("--predictions", help="fusion predictions for evaluate mode")
    result.add_argument("--ace-only-predictions")
    result.add_argument("--leader-predictions")
    result.add_argument("--config", help="frozen final model configuration JSON")
    result.add_argument("--trials", type=int, default=60)
    result.add_argument("--study", default="suvi-fusion")
    result.add_argument("--storage", default="sqlite:///suvi-fusion.sqlite3")
    result.add_argument("--device", default="cuda")
    result.add_argument("--image-workers", type=int, default=8)
    result.add_argument("--timing-folds", type=int, default=16)
    result.add_argument("--leaderboard-revision", default="bcf5d5417d99eaa331ddca40581d48953e09a58e")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ace = load_ace(args.ace)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        if not args.folds or not args.predictions:
            raise ValueError("evaluate requires --folds and --predictions")
        inputs = {"fusion": args.predictions, "ace_only": args.ace_only_predictions, "leader": args.leader_predictions}
        result = report.evaluate_predictions({name: read_table(path) for name, path in inputs.items() if path}, ace, read_table(args.folds))
        result["input_manifest"] = [file_identity(path) for path in [*args.ace, args.folds, *[p for p in inputs.values() if p]]]
        save_json(output, result)
    else:
        frame = load_origin_features(args.origin_features) if args.origin_features else make_features(args, ace)
        if args.mode == "features":
            frame.to_parquet(output)
            save_json(output.with_suffix(".manifest.json"), {"rows": len(frame), "columns": list(frame.columns),
                       "median_fit_cutoff": args.train_cutoff,
                       "inputs": [file_identity(path) for path in [*args.ace, args.suvi_meta]],
                       "output": file_identity(output)})
        else:
            run_model(args, ace, frame)
    return 0


def run_model(args, ace, frame):
    from .modeling import (
        DirectForecaster,
        ensure_device,
        prediction_metrics,
        select_feasible_trial,
        split_origins,
    )
    from .tune import run_study

    ensure_device(args.device)
    target = ace.set_index("timestamp_utc")["filled_speed_kms"]
    observed = ace.set_index("timestamp_utc")["was_missing"].eq(0)
    if args.mode == "tune":
        if pd.Timestamp(args.train_cutoff) != pd.Timestamp("2025-07-31T23:00:00Z"):
            raise ValueError("tuning uses the fixed July 2025 training cutoff")
        study = run_study(frame, target, observed, storage=args.storage, study_name=args.study,
                          n_trials=args.trials, device=args.device)
        validation_origins = split_origins(frame.index, "2025-08-01", "2026-02-28T23:00:00Z")
        naive = pd.DataFrame({
            "origin_last_input_utc": np.repeat(validation_origins, 72),
            "horizon_hours": np.tile(np.arange(1, 73), len(validation_origins)),
            "pred_kms": np.repeat(target.reindex(validation_origins).to_numpy(), 72),
        })
        reference = prediction_metrics(naive, target, observed)
        candidates = [{"number": item.number, "metrics": item.user_attrs.get("metrics")}
                      for item in study.trials if item.state.name == "COMPLETE"]
        selected = select_feasible_trial(candidates, reference)
        trial = study.trials[selected["number"]] if selected is not None else study.best_trial
        params = dict(trial.params)
        weights = {name: params.pop(name) for name in ("peak_weight", "rise_weight")}
        config = {
            "data": {"lookback_hours": 120, "mask": "disk_radius_0.98", "channel": "Fe195",
                     "median_fit_cutoff": args.train_cutoff, "train_end_utc": "2026-05-31T23:00:00Z",
                     **data_identity(args)},
            "hyperparameters": {"estimator": params, **weights, "rounds_by_horizon": trial.user_attrs["rounds_by_horizon"],
                                "libraries": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "optuna")},
                                "python": platform.python_version(), "seed": 42, "horizon_hours": 72,
                                "output_clip_kms": [1.0, 3000.0], "early_stopping_rounds": 50,
                                "tuning_device": trial.user_attrs.get("devices", args.device)},
        }
        result = {"config": config, "selected_trial": trial.number,
                  "validation_metrics": trial.user_attrs["metrics"],
                  "validation_reference": reference,
                  "validation_gate_passed": selected is not None,
                  "selection": ("minimum validation MSE among persistence-feasible trials"
                                if selected is not None else
                                "fallback minimum validation MSE; no trial passed the validation event gate"),
                  "holdout_policy": "single pass/fail evaluation; never used for model selection"}
        save_json(args.output, result)
        return
    if not args.config:
        raise ValueError("final mode requires --config from tune mode")
    frozen = json.loads(Path(args.config).read_text())
    config = frozen["config"]
    if args.train_cutoff != config["data"]["median_fit_cutoff"]:
        raise ValueError("feature median cutoff differs from frozen configuration")
    current_identity = data_identity(args)
    if any(config["data"].get(key) != value for key, value in current_identity.items()):
        raise ValueError("input data identity differs from frozen configuration")
    hp = config["hyperparameters"]
    rounds = {int(h): int(n) for h, n in hp["rounds_by_horizon"].items()}
    def fit_predict(columns, cutoff, origins, timed=False):
        model = DirectForecaster(hp["estimator"], device=args.device,
                                 peak_weight=hp["peak_weight"], rise_weight=hp["rise_weight"])
        model.fit(frame.loc[:, columns], target, cutoff, rounds_by_horizon=rounds)
        inputs = frame.loc[origins, columns]
        if not timed:
            return model.predict(inputs), None
        model.predict(inputs.iloc[:1])
        sample_size = min(args.timing_folds, len(inputs))
        if sample_size < 1:
            raise ValueError("timing-folds must be positive")
        sample_positions = np.linspace(0, len(inputs) - 1, sample_size, dtype=int)
        elapsed = 0.0
        for position in sample_positions:
            one = inputs.iloc[position:position + 1]
            started = time.perf_counter()
            model.predict(one)
            elapsed += time.perf_counter() - started
        return model.predict(inputs), elapsed / sample_size

    ace_columns = [c for c in frame.columns if c.startswith("ace_")]
    if not ace_columns:
        raise ValueError("feature frame has no ace_ columns for the ablation")
    holdout = split_origins(frame.index, "2026-03-01", "2026-05-31T23:00:00Z")
    if not len(holdout):
        raise ValueError("features do not cover the holdout")
    holdout_predictions = {name: fit_predict(columns, "2026-02-28T23:00:00Z", holdout)[0]
                           for name, columns in (("fusion", list(frame.columns)), ("ace_only", ace_columns))}
    holdout_report = report.evaluate_predictions(holdout_predictions, ace, pd.DataFrame({"origin_last_input_utc": holdout}))
    report_path = Path(args.output).with_suffix(".report.json")
    if not holdout_report["gates"]["ace_only"]["passed"]:
        save_json(report_path, {"holdout": holdout_report, "status": "holdout_gate_failed", "eligible_for_submission": False})
        return
    if not args.folds:
        raise ValueError("final mode requires --folds after holdout passes")
    folds = report.normalize_origins(read_table(args.folds))
    origins = pd.DatetimeIndex(folds["origin_last_input_utc"])
    cutoff = config["data"]["train_end_utc"]
    if origins.min() < pd.Timestamp(cutoff):
        raise ValueError("leaderboard origins must not precede final training cutoff")
    prediction, seconds = fit_predict(list(frame.columns), cutoff, origins, timed=True)
    ace_prediction, _ = fit_predict(ace_columns, cutoff, origins)
    predictions = {"fusion": prediction, "ace_only": ace_prediction}
    if args.leader_predictions:
        predictions["leader"] = read_table(args.leader_predictions)
    final_report = report.evaluate_predictions(predictions, ace, folds)
    prediction.to_parquet(args.output, index=False)
    ace_prediction.to_parquet(Path(args.output).with_suffix(".ace_only.parquet"), index=False)
    save_json(report_path, {"holdout": holdout_report, "leaderboard": final_report,
                          "eligible_for_submission": final_report["eligible_for_submission"]})
    payload = report.submission_payload(config, seconds)
    save_json(Path(args.output).with_suffix(".submission.json"), payload)


if __name__ == "__main__":
    raise SystemExit(main())
