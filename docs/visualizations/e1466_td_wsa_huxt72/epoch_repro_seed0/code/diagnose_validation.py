"""Diagnose why seed-0 validation loss plateaus or rises."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/home/t-lab01/sunrun/research/e1466-epoch-repro-20261026")
STATE = ROOT / "run/state/tdwsahuxt-state/td-state-f0ebb023d70b732f128b18980a41d3cd1ba0723d9d2b4315d3b22f6ee7befcc1"
FOLDS = pd.to_datetime(
    ["2025-11-15", "2025-12-15", "2026-02-01", "2026-03-15", "2026-04-15", "2026-05-29"],
    utc=True,
).as_unit("ns")


def normalization(handle, train_ids, seed=0):
    rng = np.random.default_rng(seed * 1009 + 0)
    sample_n = min(96, len(train_ids))
    selected = np.sort(rng.choice(train_ids, size=sample_n, replace=False))
    total = squared = count = 0.0
    for index in selected:
        values = np.asarray(handle["fields"][int(index)], dtype=np.float64)
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        count += values.size
    mean = total / count
    variance = max(squared / count - mean * mean, 1e-8)
    return mean, float(np.sqrt(variance))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": float(np.quantile(values, .01)),
        "q99": float(np.quantile(values, .99)),
    }


def smooth_l1(pred, target, beta=1.0):
    diff = np.abs(pred - target)
    return np.where(diff < beta, .5 * diff ** 2 / beta, diff - .5 * beta)


def main() -> None:
    origins = np.load(STATE / "origins_ns.npy")
    origins_ts = pd.to_datetime(origins, unit="ns", utc=True)
    rows = []
    with h5py.File(STATE / "lead_fields.h5", "r") as handle:
        for fold, cutoff in enumerate(FOLDS):
            if fold == 0:
                eligible = np.flatnonzero(origins < cutoff.value)
                split = max(4, int(len(eligible) * .8))
                train_ids, val_ids = eligible[:split], eligible[split:]
            else:
                previous = FOLDS[fold - 1].value
                train_ids = np.flatnonzero(origins < previous)
                val_ids = np.flatnonzero((origins >= previous) & (origins < cutoff.value))
            mean, std = normalization(handle, train_ids, seed=0)

            # sample fields to keep memory bounded
            rng = np.random.default_rng(100 + fold)
            train_sample = np.sort(rng.choice(train_ids, size=min(32, len(train_ids)), replace=False))
            train_values = np.stack([np.asarray(handle["fields"][int(i)], dtype=np.float32) for i in train_sample])
            val_values = np.stack([np.asarray(handle["fields"][int(i)], dtype=np.float32) for i in val_ids])

            train_z = (train_values - mean) / std
            val_z = (val_values - mean) / std
            # constant baseline: train per-lead mean, then constant 0 in z-space
            baseline_val = smooth_l1(np.zeros_like(val_z), val_z).mean()
            baseline_train = smooth_l1(np.zeros_like(train_z), train_z).mean()

            # per-origin mean absolute deviation from the train mean
            train_mean = train_z.mean(axis=(0, 1, 2, 3), keepdims=True)
            val_dev = np.abs(val_z - train_mean).mean(axis=(1, 2, 3))
            train_dev = np.abs(train_z - train_mean).mean(axis=(1, 2, 3))
            order = np.argsort(val_dev)[::-1]

            rows.append({
                "fold": fold,
                "cutoff": str(cutoff),
                "train_origins": int(len(train_ids)),
                "valid_origins": int(len(val_ids)),
                "normalization_mean": mean,
                "normalization_std": std,
                "train_field": stats(train_values),
                "valid_field": stats(val_values),
                "train_z": stats(train_z),
                "valid_z": stats(val_z),
                "baseline_train_smooth_l1": float(baseline_train),
                "baseline_valid_smooth_l1": float(baseline_val),
                "valid_dev_mean": float(val_dev.mean()),
                "train_dev_mean": float(train_dev.mean()),
                "valid_dev_p90": float(np.quantile(val_dev, .9)),
                "valid_dev_max": float(val_dev.max()),
                "worst_valid_origins": [
                    {"origin": str(origins_ts[val_ids[i]]), "dev": float(val_dev[i])}
                    for i in order[:5]
                ],
                "best_valid_origins": [
                    {"origin": str(origins_ts[val_ids[i]]), "dev": float(val_dev[i])}
                    for i in order[-5:]
                ],
            })
            print(f"fold {fold}: train={len(train_ids)} val={len(val_ids)} "
                  f"norm={mean:.2f}/{std:.2f} val_z mean={val_z.mean():.3f} "
                  f"baseline val={baseline_val:.4f} train={baseline_train:.4f} "
                  f"dev val={val_dev.mean():.3f} train={train_dev.mean():.3f}", flush=True)
    (ROOT / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("wrote", ROOT / "diagnostics.json")


if __name__ == "__main__":
    main()
