"""Official leaderboard scores and observed-only peak diagnostics."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scorer.scoring import KEY, build_truth, score, validate

PEAK_HORIZONS = (6, 12, 24, 48, 72)


def normalize_origins(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table[KEY[0]] = pd.to_datetime(table[KEY[0]], utc=True)
    return table


def _binary(actual, predicted) -> dict:
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def peak_metrics(pred: pd.DataFrame, truth: pd.DataFrame, series: pd.DataFrame) -> dict:
    merged = normalize_origins(truth).merge(normalize_origins(pred)[KEY + ["pred_kms"]], on=KEY, validate="one_to_one")
    merged = merged[merged["was_missing"].eq(0)].copy()
    base = series.set_index(pd.to_datetime(series["timestamp_utc"], utc=True))["filled_speed_kms"]
    merged["origin_kms"] = merged[KEY[0]].map(base)
    if merged["origin_kms"].isna().any():
        raise ValueError("ACE does not cover prediction origins")
    by_horizon = {}
    for h in PEAK_HORIZONS:
        rows = merged[merged["horizon_hours"].eq(h)]
        actual = rows["target_kms"].ge(600)
        predicted = rows["pred_kms"].ge(600)
        bias = rows.loc[actual, "pred_kms"] - rows.loc[actual, "target_kms"]
        by_horizon[str(h)] = {
            "observed_count": len(rows), "peak_count": int(actual.sum()),
            "peak": _binary(actual, predicted),
            "peak_bias_kms": float(bias.mean()) if len(bias) else None,
            "rise": _binary((rows["target_kms"] - rows["origin_kms"]).ge(100),
                            (rows["pred_kms"] - rows["origin_kms"]).ge(100)),
        }
    selected = [by_horizon[str(h)] for h in (24, 48, 72)]
    supported = all(row["peak_count"] for row in selected)
    return {"by_horizon": by_horizon, "gate_horizons_supported": supported,
            "macro_f1_24_48_72": float(np.mean([r["peak"]["f1"] for r in selected])) if supported else None,
            "abs_mean_peak_bias_24_48_72": abs(float(np.mean([r["peak_bias_kms"] for r in selected]))) if supported else None}


def comparison_gate(candidate: dict, reference: dict, *, observed_required: bool = True) -> dict:
    c, r = candidate["official"], reference["official"]
    cp, rp = candidate["peaks"], reference["peaks"]
    supported = cp["gate_horizons_supported"] and rp["gate_horizons_supported"]
    checks = {"mse_improved": c["mse"] < r["mse"],
              "observed_mse_improved": (c["mse_observed"] is not None and r["mse_observed"] is not None and c["mse_observed"] < r["mse_observed"]) if observed_required else True,
              "peak_f1_improved": bool(supported and cp["macro_f1_24_48_72"] >= rp["macro_f1_24_48_72"] + 0.02),
              "peak_bias_improved": bool(supported and cp["abs_mean_peak_bias_24_48_72"] <= rp["abs_mean_peak_bias_24_48_72"] - 10)}
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_predictions(predictions: dict[str, pd.DataFrame], series: pd.DataFrame, folds: pd.DataFrame) -> dict:
    folds = normalize_origins(folds)
    if "regime" not in folds:
        folds["regime"] = "unclassified"
    truth = build_truth(series, folds)
    normalized = {name: normalize_origins(pred) for name, pred in predictions.items()}
    speed = series.set_index(pd.to_datetime(series["timestamp_utc"], utc=True))["filled_speed_kms"]
    origins = pd.DatetimeIndex(folds[KEY[0]])
    last = speed.reindex(origins).to_numpy()
    base = pd.DataFrame({KEY[0]: np.repeat(origins, 72), "horizon_hours": np.tile(np.arange(1, 73), len(origins))})
    normalized.setdefault("naive", base.assign(pred_kms=np.repeat(last, 72)))
    mean = speed.rolling(648, min_periods=648).mean().reindex(origins).to_numpy()
    if np.isfinite(mean).all():
        pred = mean[:, None] + (last - mean)[:, None] * np.exp(-np.arange(1, 73)[None, :] / 48)
        normalized.setdefault("mean_reversion", base.assign(pred_kms=pred.ravel()))
    for name, pred in normalized.items():
        errors = validate(pred, folds)
        if errors:
            raise ValueError(f"{name}: {'; '.join(errors)}")
    result = {name: {"official": score(pred, truth, folds), "peaks": peak_metrics(pred, truth, series)}
              for name, pred in normalized.items()}
    gates = {}
    if "fusion" in result:
        for reference in ("ace_only", "leader"):
            if reference in result:
                gates[reference] = comparison_gate(result["fusion"], result[reference], observed_required=reference == "leader")
    return {"models": result, "gates": gates,
            "eligible_for_submission": bool("leader" in gates and "ace_only" in gates and all(g["passed"] for g in gates.values()))}


def submission_payload(config: dict, inference_seconds_per_fold: float) -> dict:
    if set(config) != {"data", "hyperparameters"} or not all(isinstance(v, dict) and v for v in config.values()):
        raise ValueError("config requires non-empty data and hyperparameters mappings")
    if not np.isfinite(inference_seconds_per_fold) or inference_seconds_per_fold <= 0:
        raise ValueError("inference timing must be positive and finite")
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {"config_sha256": digest, "script_path": f"scripts/seongeun/xgboost-suvi-fusion/{digest}.py",
            "meta": {"team_member": "seongeun", "experiment": "xgboost-suvi-fusion",
                     "description": "Direct 72-hour XGBoost with 120-hour ACE and masked SUVI Fe195 features",
                     "no_future_leakage": True, "config": config,
                     "inference_seconds_per_fold": float(inference_seconds_per_fold)}}
