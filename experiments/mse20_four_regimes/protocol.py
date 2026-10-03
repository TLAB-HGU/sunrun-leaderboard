"""Shared causal splits, immutable manifests and bounded experiment utilities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import common, hpo
from experiments.xgboost_suvi_fusion.events import event_metrics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "store/mse20-four-regimes/run-20261003"
KEY = ["origin_last_input_utc", "horizon_hours"]
SPLITS = {
    "development": {"cutoff": "2025-07-31T23:00:00Z", "end": "2025-11-30T23:00:00Z", "step": 6},
    "selection": {"cutoff": "2025-11-30T23:00:00Z", "end": "2026-02-28T23:00:00Z", "step": 1},
    "audit": {"cutoff": "2026-02-28T23:00:00Z", "end": "2026-05-31T23:00:00Z", "step": 1},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def verify_ref(entry):
    if digest(entry["path"]) != entry["sha256"]:
        raise ValueError(f"stale artifact: {entry['path']}")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def check_deadline(run_dir=RUN, stage="deadline", reserve_seconds=0):
    runtime = json.loads((Path(run_dir) / "runtime.json").read_text())
    field = {"search": "search_stop_utc", "optimization": "optimization_stop_utc"}.get(stage, "deadline_utc")
    remaining = (pd.Timestamp(runtime[field]) - pd.Timestamp.now(tz="UTC")).total_seconds()
    if remaining <= reserve_seconds:
        raise TimeoutError(f"{stage} budget exhausted; {remaining:.1f}s left, reserve={reserve_seconds}")
    return remaining


def require_gate0(run_dir=RUN):
    record = json.loads((Path(run_dir) / "baseline-verification.json").read_text())
    if record.get("status") != "baseline_verified" or record.get("objective_achieved") is not False:
        raise ValueError("full baseline verification required before optimization")


def slice_frames(frames, end):
    """Return only source/feature rows available through the supplied split end."""
    end = pd.Timestamp(end)
    result = {}
    for name, frame in frames.items():
        if isinstance(frame, (pd.Series, pd.DataFrame)):
            if isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.loc[frame.index <= end].copy()
            elif isinstance(frame, pd.DataFrame) and "timestamp_utc" in frame:
                frame = frame.loc[pd.to_datetime(frame.timestamp_utc, utc=True) <= end].copy()
        result[name] = frame
    return result


def load_frames(end="2026-05-31T23:00:00Z"):
    """Old feature builders are causal; model/search sees only explicitly sliced rows."""
    frames = hpo.load_inputs([str(ROOT / "store/ch-v1/ace.parquet")],
                            str(ROOT / "store/ch-v1/ch-hourly.parquet"),
                            str(ROOT / "store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet"))
    return slice_frames(frames, end)


def split_origins(index, split, step=None):
    definition = SPLITS[split]
    cutoff, end = pd.Timestamp(definition["cutoff"]), pd.Timestamp(definition["end"])
    expected = pd.date_range(cutoff + pd.Timedelta(hours=73), end - pd.Timedelta(hours=72), freq="h")
    # Coverage gaps are errors, not permission to silently change validation rows.
    missing = expected.difference(pd.DatetimeIndex(index))
    if len(missing):
        raise ValueError(f"{split}: {len(missing)} missing origins, first={missing[0]}")
    return expected[::int(step or definition["step"])]


def canonical_predictions(predictions, origins=None):
    p = predictions[KEY + ["pred_kms"]].copy()
    p[KEY[0]] = pd.to_datetime(p[KEY[0]], utc=True)
    if p.duplicated(KEY).any() or not np.isfinite(p.pred_kms).all():
        raise ValueError("duplicate/nonfinite predictions")
    if not p.pred_kms.between(1, 3000).all():
        raise ValueError("forecast range violated")
    if origins is None:
        origins = pd.DatetimeIndex(p[KEY[0]].unique()).sort_values()
    expected = pd.MultiIndex.from_product([pd.DatetimeIndex(origins), range(1, 73)], names=KEY)
    actual = pd.MultiIndex.from_frame(p[KEY])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("incomplete origin/horizon support")
    p = p.sort_values(KEY).reset_index(drop=True)
    p[KEY[0]] = p[KEY[0]].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return p


def score_development(predictions, frames, origins):
    """No official labels. All requested windows and identical observed masks."""
    p = canonical_predictions(predictions, origins)
    times = pd.to_datetime(p[KEY[0]], utc=True) + pd.to_timedelta(p[KEY[1]], unit="h")
    y = frames["speed"].reindex(pd.DatetimeIndex(times)).to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("incomplete development truth")
    mask = frames["observed"].reindex(pd.DatetimeIndex(times))
    if mask.isna().any() or not mask.isin([True, False, 0, 1]).all():
        raise ValueError("incomplete or invalid development observation mask")
    observed = mask.to_numpy(bool)
    squared = np.square(y - p.pred_kms.to_numpy(float))
    events = event_metrics(p, frames["speed"], frames["observed"], origins=origins,
                          bootstrap=0, missing_policy="forward_fill")
    if events["evaluated_windows"] != len(origins):
        raise ValueError("event evaluator skipped requested windows")
    return {"mse": float(squared.mean()), "mse_observed": float(squared[observed].mean()),
            **{k: events.get(k) for k in ("precision", "recall", "tp", "fp", "fn", "height_mae_kms", "time_mae_hours", "evaluated_windows", "unique_observed_peak_timestamps")}}


def feasible(metrics, reference):
    return all(metrics.get(k) is not None and reference.get(k) is not None
               and metrics[k] >= reference[k] for k in ("precision", "recall"))


def blend(predictions, weights):
    canonical = [canonical_predictions(p) for p in predictions]
    if len(canonical) != len(weights) or not np.isclose(sum(weights), 1):
        raise ValueError("invalid blend weights")
    first = canonical[0].copy()
    if any(not first[KEY].equals(p[KEY]) for p in canonical[1:]):
        raise ValueError("blend keys differ")
    first["pred_kms"] = sum(float(w) * p.pred_kms.to_numpy(float) for w, p in zip(weights, canonical))
    return first
