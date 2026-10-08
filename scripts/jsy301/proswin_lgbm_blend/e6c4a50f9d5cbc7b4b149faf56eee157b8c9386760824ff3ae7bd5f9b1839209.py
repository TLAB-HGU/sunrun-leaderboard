"""PROSWIN (SUVI images) + LightGBM (L1 plasma/field, coronal holes) per-horizon blend, 72 h ahead.

Leaderboard entry of team member jsy301. For every evaluation origin T the forecast
uses only data with timestamp <= T:

- PROSWIN: Swin-Tiny encoder on the GOES-18 SUVI Fe171/Fe195 image at T plus
  physical inputs (24 recent hourly ACE speeds, 27-day recurrence window,
  sunspot numbers, latitude, cycle year), 72-wide heads predicting the change
  from the speed at T with a learned per-horizon weight on that speed. Trained
  with the project config ``config_suvi_mimo_72h_causal.yaml`` (labels up to
  2025-06-30, early stopping on 2025-07..12); speed inputs are rebuilt from
  past observations only (gaps carried forward).
- LightGBM: one model per horizon (1..72) on the change from the speed at T;
  inputs: speed history and 27-day recurrence, ACE magnetic field / density /
  temperature / energetic particle summaries (incl. values one rotation before
  the target hour), coronal-hole measures of the SUVI images with 12..120 h
  history, 32 PCA components of a frozen ImageNet Swin embedding. Labels up to
  2025-06-30; early stopping on 2025-03-27..06-27.
- Blend: per horizon w * PROSWIN + (1 - w) * LightGBM, w on a 0.05 grid fitted
  on origins 2025-08-01..2025-12-31 (forward-filled ACE truth) and smoothed over
  5 neighbouring horizons. Origins without a SUVI image use LightGBM alone.

Usage (needs the project checkout with its data, SUVI images and the trained model):

    <project>/proswin-repo/.venv/bin/python scripts/jsy301/proswin_lgbm_blend/<sha>.py \
        --project-root /path/to/proswin --output predictions.parquet [--folds folds.parquet]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_HOURS = 72
DEFAULT_FOLDS_REPO = "tlabtlab/sunrun-lb-store"
CONFIG = {
    "data": {
        "evaluation_folds": "tlabtlab/sunrun-lb-store/folds.parquet",
        "frequency": "1h",
        "input_rule": "only values with timestamp <= origin; gaps forward-filled from the last observation",
        "inputs": [
            "ACE SWEPAM speed/density/temperature (collector solar.csv + NOAA SWPC real-time archive)",
            "ACE MAG Bt/By/Bz GSM (solar.csv + NOAA SWPC real-time archive), ACE EPAM/SIS (solar.csv)",
            "GOES-18 SUVI L1b Fe171 and Fe195, hourly, 224x224 north-up",
            "SILSO monthly sunspot number, OMNI Earth heliographic latitude",
        ],
        "labels": "ACE bulk speed, gaps <=30 h linearly interpolated (labels only)",
        "train_labels_end": "2025-06-30T23:00:00Z",
        "proswin_early_stopping": ["2025-07-01T00:00:00Z", "2025-12-31T23:00:00Z"],
        "lgbm_early_stopping": ["2025-03-27T00:00:00Z", "2025-06-27T23:00:00Z"],
        "blend_fit_origins": ["2025-08-01T00:00:00Z", "2025-12-31T23:00:00Z"],
    },
    "hyperparameters": {
        "proswin": {
            "project_config": "config_suvi_mimo_72h_causal.yaml",
            "encoder": "swin_tiny_patch4_window7_224 (ImageNet)",
            "channels": ["suvi_171", "suvi_195"],
            "horizons": "1-72",
            "residual_from": "speed at origin, learned weight per horizon",
            "mu_head": {"hidden": 131, "dropout": 0.53},
            "learning_rate": {"encoder": 1e-05, "head": 0.0001},
            "batch_size": 16,
            "early_stopping_epochs": 30,
            "seed": 42,
        },
        "lightgbm": {
            "objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "max_rounds": 3000, "early_stopping_rounds": 100, "seed": 0,
            "target": "speed(T+h) - speed(T)", "feature_set": "all+",
        },
        "blend": {"grid_step": 0.05, "smoothing_horizons": 5, "missing_image": "lightgbm only"},
    },
}
CONFIG_SHA256 = "e6c4a50f9d5cbc7b4b149faf56eee157b8c9386760824ff3ae7bd5f9b1839209"

TRAIN = ("2022-08-22 21:00", "2025-06-27 23:00")
EARLY_STOP_FROM = "2025-03-27 00:00"
BLEND_FIT = ("2025-08-01 00:00", "2025-12-31 23:00")


def canonical_sha() -> str:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_config_identity() -> None:
    actual = canonical_sha()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual} ({Path(__file__).stem})")


# ---------------------------------------------------------------------------
# PROSWIN
# ---------------------------------------------------------------------------

def proswin_forecasts(project: Path, anchors: pd.DatetimeIndex, device_name: str) -> tuple[pd.DataFrame, float]:
    """Mean forecasts (rows = anchors with an image, columns = horizons) and seconds per origin."""
    import logging

    import torch

    from proswin.data_management.data_loader_creator import get_prediction_loader
    from proswin.model.solar_swin_transformer import load_model
    from proswin.utils.configurator import ConfigLoader
    from proswin.utils.factory import build_dataset

    repo = project / "proswin-repo"
    config = ConfigLoader(repo, config_name=CONFIG["hyperparameters"]["proswin"]["project_config"],
                          enable_logging=False, initialize_ddp=False)
    config.logger = logging.getLogger("proswin")
    model_name = config.model_config["model_name"]
    model = load_model(config.paths["models"] / f"{model_name}_sigma_best.pth")
    device = torch.device(device_name)
    model.to(device).eval()
    # causal_speed_inputs in the project config rebuilds the speed inputs from past observations
    dataset = build_dataset(config, require_targets=False)
    dates = anchors.intersection(dataset.target_dates)
    loader = get_prediction_loader(dataset=dataset, dates=dates, input_scaler=model.physical_feature_scaler,
                                   output_scaler=model.distribution_transformer, batch_size=64, num_workers=8)
    batches = [{k: v for k, v in b.items()} for b in loader]  # data loading (excluded from timing)
    dt = model.distribution_transformer
    outputs, issue = [], []
    elapsed = 0.0
    with torch.no_grad():
        for batch in batches:
            inputs = {k: batch[k].to(device) for k in ("images", "physical_features")}
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            mu, sigma = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            outputs.append((mu.float().cpu().numpy(), sigma.float().cpu().numpy()))
            issue.append(batch["target_date"].numpy())
    mu = np.concatenate([o[0] for o in outputs])
    sigma = np.concatenate([o[1] for o in outputs])
    # Mean of the shifted log-normal in km/s (mu is already bias-corrected in eval mode)
    scale, mean_ = float(dt.target_scaler.scale_[0]), float(dt.target_scaler.mean_[0])
    log_scale = mu * scale + mean_
    shape = sigma * scale
    mean_kms = dt.loc_shift + np.exp(log_scale + shape ** 2 / 2)
    index = pd.to_datetime(np.concatenate(issue), unit="s")
    frame = pd.DataFrame(mean_kms, index=index, columns=range(1, HORIZON_HOURS + 1)).sort_index()
    return frame[~frame.index.duplicated()], elapsed / max(len(frame), 1)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def lgbm_forecasts(project: Path, predict_at: pd.DatetimeIndex) -> tuple[pd.DataFrame, float]:
    import lightgbm as lgb

    sys.path.insert(0, str(project / "scripts"))
    import quick_feature_tests as q

    causal, _ = q.load_speed()
    l1 = q.load_l1()
    train_anchors = pd.date_range(*TRAIN, freq="h")
    anchors = train_anchors.union(predict_at)
    base = q.base_features(anchors, causal)
    l1_feats = q.l1_features(anchors, l1, causal)
    ch = q.ch_features(anchors, lags=(12, 24, 36, 48, 72, 96, 120))
    emb = q.embedding_features(anchors, train_anchors)
    labels = pd.read_csv(project / "data/physical_features/solar_wind_speed.csv", index_col=0,
                         parse_dates=True)["solar_wind_speed"]
    now_all = causal.reindex(anchors).to_numpy()
    pos = pd.Index(anchors)
    i_train = pos.get_indexer(train_anchors)
    stop = train_anchors >= pd.Timestamp(EARLY_STOP_FROM)
    i_fit, i_stop, i_pred = i_train[~stop], i_train[stop], pos.get_indexer(predict_at)
    hp = CONFIG["hyperparameters"]["lightgbm"]
    params = {k: hp[k] for k in ("objective", "learning_rate", "num_leaves", "min_data_in_leaf",
                                 "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2")}
    params.update(verbose=-1, num_threads=32, seed=hp["seed"])
    preds = np.full((len(predict_at), HORIZON_HOURS), np.nan)
    elapsed = 0.0
    for h in range(1, HORIZON_HOURS + 1):
        x = pd.concat([base, q.horizon_features(anchors, causal, h), l1_feats, ch, emb,
                       q.horizon_l1_features(anchors, l1, h)], axis=1).to_numpy(dtype=np.float32)
        y = labels.reindex(anchors + pd.Timedelta(hours=h)).to_numpy() - now_all
        ok_fit = np.isfinite(y[i_fit]) & np.isfinite(now_all[i_fit])
        ok_stop = np.isfinite(y[i_stop]) & np.isfinite(now_all[i_stop])
        train_set = lgb.Dataset(x[i_fit][ok_fit], y[i_fit][ok_fit])
        model = lgb.train(params, train_set, num_boost_round=hp["max_rounds"],
                          valid_sets=[lgb.Dataset(x[i_stop][ok_stop], y[i_stop][ok_stop], reference=train_set)],
                          callbacks=[lgb.early_stopping(hp["early_stopping_rounds"], verbose=False)])
        start = time.perf_counter()
        preds[:, h - 1] = model.predict(x[i_pred], num_iteration=model.best_iteration) + now_all[i_pred]
        elapsed += time.perf_counter() - start
        if h % 12 == 0:
            print(f"lightgbm horizon {h} done", flush=True)
    return pd.DataFrame(preds, index=predict_at, columns=range(1, HORIZON_HOURS + 1)), elapsed / len(predict_at)


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def fit_weights(pro: pd.DataFrame, tree: pd.DataFrame, truth: pd.Series) -> pd.Series:
    step = CONFIG["hyperparameters"]["blend"]["grid_step"]
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    weights = {}
    for h in range(1, HORIZON_HOURS + 1):
        y = truth.reindex(pro.index + pd.Timedelta(hours=h)).to_numpy()
        a, b = pro[h].to_numpy(), tree.loc[pro.index, h].to_numpy()
        ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
        errs = [np.mean((w * a[ok] + (1 - w) * b[ok] - y[ok]) ** 2) for w in grid]
        weights[h] = float(grid[int(np.argmin(errs))])
    raw = pd.Series(weights)
    k = CONFIG["hyperparameters"]["blend"]["smoothing_horizons"]
    return raw.rolling(k, center=True, min_periods=1).mean().round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", default="predictions.parquet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights-out", help="optional CSV with the fitted blend weights")
    args = parser.parse_args()
    verify_config_identity()
    project = Path(args.project_root).resolve()
    sys.path.insert(0, str(project / "proswin-repo" / "src"))
    sys.path.insert(0, str(project / "scripts"))

    if args.folds:
        folds_path = args.folds
    else:
        from huggingface_hub import hf_hub_download
        folds_path = hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    folds = pd.read_parquet(folds_path) if str(folds_path).endswith(".parquet") else pd.read_csv(folds_path)
    raw_origins = folds["origin_last_input_utc"]
    origins = pd.DatetimeIndex(pd.to_datetime(raw_origins, utc=True)).tz_localize(None)
    fit_origins = pd.date_range(*BLEND_FIT, freq="h")
    predict_at = fit_origins.union(origins)

    pro, pro_seconds = proswin_forecasts(project, predict_at, args.device)
    tree, tree_seconds = lgbm_forecasts(project, predict_at)

    import quick_feature_tests as q
    _, truth = q.load_speed()
    weights = fit_weights(pro.loc[pro.index.intersection(fit_origins)], tree, truth)
    if args.weights_out:
        weights.rename("proswin_weight").to_csv(args.weights_out, index_label="horizon_hours")

    start = time.perf_counter()
    t = tree.loc[origins].to_numpy()
    p = pro.reindex(origins).to_numpy()
    w = weights.reindex(range(1, HORIZON_HOURS + 1)).to_numpy()[None, :]
    blended = np.where(np.isfinite(p), w * p + (1 - w) * t, t)
    blend_seconds = (time.perf_counter() - start) / len(origins)

    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(raw_origins.to_numpy(), HORIZON_HOURS),
        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
        "pred_kms": blended.ravel(),
    })
    output.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "n_folds": int(len(origins)),
        "origins_without_image": int((~np.isfinite(p).all(axis=1)).sum()),
        "inference_seconds_per_fold": pro_seconds + tree_seconds + blend_seconds,
        "inference_seconds_parts": {"proswin": pro_seconds, "lightgbm": tree_seconds, "blend": blend_seconds},
        "proswin_weight_h1_h24_h48_h72": [float(weights[h]) for h in (1, 24, 48, 72)],
    }))


if __name__ == "__main__":
    main()
