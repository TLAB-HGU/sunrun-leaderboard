"""Reproduce the E1604 HSSPEAK25 original-contract comparator from prepared features.

F2 (2026-06..09) is the original release, unchanged. F1 (2026-03..05, leaderboard rule since 2026-10-26) refits the same
two heads on the release training rows whose 72 h label window ends before 2026-02-26 00:00 UTC and corrects the
E0472-F1 base (experiments/f1_resubmit_20261026/e0472_f1.py: E0472 recipe with every fit before 2026-02-26).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

CONFIG = {'data': {'experiment': 'E1604', 'base': 'E0472', 'contract': 'original retrospective comparator; ACE observation through origin, not ACE obs+2h', 'sources': ['SW-001 ACE', 'SW-012 frozen GONG/PFSS/WSA/HUXt trajectory', 'E0472 frozen forecasts'], 'training_last_target_utc': '2026-05-31 23:00:00+00:00', 'training_cutoff_note': 'Original recipe uses all six fold labels through 2026-05-31 23:00 UTC, equal to first forecast origin; not the newer targets-before-May29 contract', 'reproduction': 'prepared causal features and base forecasts; upstream generation excluded', 'input_sha256': {'prepared_inputs.npz': 'f02d5e04095c28d71c34065b18699765ec4beff3018acc0b9efac736259f5ead'}, 'original_prediction_sha256': ['92ee473ab1d009814e763179cf6ffb34b68384ed4740e09b2a7ebbc04d7b1e48', 'ed0ff67b04d6cf8eb0fa5d1d66c7abc76b12e06cafc68c30d2e9a901baff9938', 'd656cfb1f6ddffd1de8a576f0a2b8261a608a1393bc1907ee95e82f6b8faf121']}, 'hyperparameters': {'model': 'two HistGradientBoostingRegressor heads', 'max_iter': 500, 'learning_rate': 0.04, 'max_depth': 4, 'min_samples_leaf': 50, 'random_state': 0, 'width_hours': 12.0, 'shrink': 0.15, 'min_lead': 25, 'seeds': [0, 1, 2], 'ensemble': 'equal prediction mean', 'timing_scope': 'batched peak/hour head inference, Gaussian correction, and seed averaging; prepared features exclude upstream forecasts'}}
CONFIG_SHA256 = '4a0e7bb648ab92c9bb75c76443d5576f5bd4576481edc8161c44f8c0321e7f5b'
F1_INPUT_SHA256 = {'hsspeak_f1_inputs.npz': '795df2238e423b8246f2c972a1360c2c2cda1721a59da6f662a733c6a1501210'}
F1_CUTOFF = pd.Timestamp('2026-02-26T00:00:00Z')


def predict(models, x, base):
    peak = models[0].predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
    hour = models[1].predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
    hh = np.arange(1, 73, dtype=float)
    b = np.exp(-((hh[None, None, :] - np.clip(hour, 1, 72)[:, :, None]) ** 2) / (2 * 12.0 ** 2))
    b /= np.nanmax(b, axis=2, keepdims=True)
    inc = 0.15 * (peak[:, :, None] - np.nanmin(base, axis=2, keepdims=True)) * b
    inc[:, :, :24] = 0.0
    per_seed = base + np.nan_to_num(inc)
    return per_seed, per_seed.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--per-seed-dir")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    release = root / "releases/e1604"
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == CONFIG_SHA256 == Path(__file__).stem
    for name, digest in CONFIG["data"]["input_sha256"].items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == digest, name
    with np.load(release / "prepared_inputs.npz") as z:
        train_x, yp, yh = z["train_x"], z["train_peak"], z["train_hour"]
        x, base, origins = z["official_x"], z["base"], pd.to_datetime(z["origins_ns"], utc=True)
    assert x.shape == (3, 2833, 29) and base.shape == (3, 2833, 72)
    f1_release = root / "releases/f1-20261026"
    for name, digest in F1_INPUT_SHA256.items():
        assert hashlib.sha256((f1_release / name).read_bytes()).hexdigest() == digest, name
    with np.load(f1_release / "hsspeak_f1_inputs.npz") as z:
        train_origins = pd.to_datetime(z["train_origins_ns"], utc=True)
        x1, base1, origins1 = z["f1_x"], z["f1_base"], pd.to_datetime(z["f1_origins_ns"], utc=True)
    assert len(train_origins) == len(train_x) and x1.shape == (3, 2137, 29) and base1.shape == (3, 2137, 72)
    f1_rows = np.asarray(train_origins + pd.Timedelta(hours=72) < F1_CUTOFF)
    kw = {k: CONFIG["hyperparameters"][k] for k in ("max_iter", "learning_rate", "max_depth", "min_samples_leaf", "random_state")}
    with threadpool_limits(limits=1):
        models = [HistGradientBoostingRegressor(**kw).fit(train_x, y) for y in (yp, yh)]
        models_f1 = [HistGradientBoostingRegressor(**kw).fit(train_x[f1_rows], y[f1_rows]) for y in (yp, yh)]
        predict(models, x[:, :20], base[:, :20])
        start = time.perf_counter()
        per_seed, pred = predict(models, x, base)
        per_seed1, pred1 = predict(models_f1, x1, base1)
        elapsed = time.perf_counter() - start
    for values in (per_seed, per_seed1):
        assert np.isfinite(values).all() and (values > 0).all() and (values <= 3000).all()
    origins_all = origins1.append(origins)
    keys = {"origin_last_input_utc": np.repeat(origins_all.strftime("%Y-%m-%dT%H:%M:%SZ"), 72),
            "horizon_hours": np.tile(np.arange(1, 73), len(origins_all))}
    pd.DataFrame({**keys, "pred_kms": np.concatenate([pred1.ravel(), pred.ravel()])}).to_parquet(a.output, index=False)
    if a.per_seed_dir:
        folder = Path(a.per_seed_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for seed in range(3):
            values = np.concatenate([per_seed1[seed].ravel(), per_seed[seed].ravel()])
            pd.DataFrame({**keys, "pred_kms": values}).to_parquet(folder / f"predictions_f1f2_s{seed}.parquet", index=False)
    report = {"inference_seconds_per_fold": elapsed / len(origins_all), "origins_measured": len(origins_all),
              "seconds": elapsed, "rows": pred.size + pred1.size, "f1_train_rows": int(f1_rows.sum()),
              "scope": CONFIG["hyperparameters"]["timing_scope"]}
    Path(a.timing).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
