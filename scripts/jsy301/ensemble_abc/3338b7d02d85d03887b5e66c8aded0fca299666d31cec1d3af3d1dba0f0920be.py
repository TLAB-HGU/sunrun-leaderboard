"""A/B/C ensemble: numeric trees since 2000 (A) + PROSWIN (B1) + image-feature trees (B2), combined per horizon.

Leaderboard entry of team member jsy301. For every evaluation origin T the forecast
uses only data with timestamp <= T (gaps carried forward; OMNI-filled hours are
treated as gaps). Labels, early stopping, member selection and combination weights
use nothing after 2026-05-31 23:00 UTC (the leaderboard targets start 2026-06-01).

Members (each predicts speed(T+h) for h = 1..72):
- A  : LightGBM per horizon (17 trained horizons, linear interpolation of the predicted
       change) on ACE speed/density/temperature/field/particle history from the
       collector's solar.csv since 2000, 27-day recurrence aligned with the target hour,
       sunspot numbers, Earth heliographic latitude, cycle year. Fit anchors every 3 h
       up to 73 h before the early-stopping window 2026-03-01 .. 05-28.
- B1 : PROSWIN (Swin-Tiny on the GOES-18 SUVI Fe171/Fe195 image at T + 24 recent hourly
       speeds, 27-day recurrence window, sunspots, latitude, cycle year; 72-wide mu/sigma
       heads predicting the change from the speed at T with a learned per-horizon weight).
       Trained with ``configs/config_ens_b1_FINAL.yaml`` (labels <= 2025-11-30 for the
       fit, early stopping on 2025-12-01 .. 2026-05-31, brightness limits from images
       <= 2026-05-31); forecasts with past-only inputs via scripts/predict_causal.py.
- B2 : LightGBM per horizon on image measures (coronal-hole area / darkness / position
       with 12..120 h history, longitude strips aligned with the target hour, 32 PCA
       components of a frozen ImageNet Swin embedding fitted on the training anchors)
       plus speed history; anchors from 2022-08-22 (first SUVI hour).
- Combination: per-horizon weights on the simplex (0.05 grid, smoothed over 5
  neighbouring horizons) or the gate model C (softmax weights from 5 horizon basis
  functions and per-lead-band context: PROSWIN sigma, coronal-hole area near the
  meridian, speed at T and its 24-h trend, member disagreement, verified 1-6 h errors
  of the forecasts issued 12 h earlier). Both fitted on out-of-fold forecasts of the
  fold models (origins 2025-08-01 .. 2026-05-28 only); which one is used is a decision
  recorded in CONFIG (gate C only if it beat the fixed weights forward-chained).
  Origins whose PROSWIN forecast is missing (no image) use the remaining members with
  renormalised weights.

Usage (needs the project checkout with its data, SUVI images, the trained PROSWIN FINAL
model and the out-of-fold store results_ens/oof/):

    <project>/proswin-repo/.venv/bin/python scripts/jsy301/ensemble_abc/<sha>.py \
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
        "input_rule": "only values with timestamp <= origin; gaps carried forward from the last ACE measurement; OMNI-filled hours are gaps",
        "inputs": [
            "ACE SWEPAM speed/density/temperature, MAG Bt/By/Bz GSM, EPAM/SIS particles (collector solar.csv since 2000 + NOAA SWPC real-time archive for recent hours)",
            "GOES-18 SUVI L1b Fe171 and Fe195, hourly, 224x224 north-up (2022-08-22 on)",
            "SILSO monthly sunspot number, OMNI Earth heliographic latitude, solar-cycle year",
        ],
        "labels": "ACE bulk speed forward-filled over gaps (scorer rule); member A uses gap-interpolated labels (<= 30 h) whose next measurement is before the cutoff",
        "label_cutoff": "2026-05-31T23:00:00Z",
        "final_fold": {
            "tree_fit_anchors_end": "2026-02-25T23:00:00Z",
            "tree_early_stopping": ["2026-03-01T00:00:00Z", "2026-05-28T23:00:00Z"],
            "proswin_train_end": "2025-11-30T23:00:00Z",
            "proswin_validation": ["2025-12-01T00:00:00Z", "2026-05-31T23:00:00Z"],
        },
        "out_of_fold": {
            "folds": {"F1": ["2024-07-01", "2024-12-31"], "F2": ["2025-01-01", "2025-06-30"],
                      "F3": ["2025-07-01", "2025-12-31"], "F4": ["2026-01-01", "2026-05-28"]},
            "rule": "each fold's members are trained with labels <= the fold's cutoff (6 months before the fold's first origin for PROSWIN; 73 h before the 3-month early-stopping window for the trees)",
            "tuning_origins": ["2025-08-01T00:00:00Z", "2026-05-28T23:00:00Z"],
        },
    },
    "hyperparameters": {
        "members": ["A", "B1", "B2"],
        "combination": "fixed_per_horizon_weights",
        "member_A": {
            "model": "lightgbm per horizon", "trained_horizons": [1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],
            "start_year": 2000, "fit_stride_hours": 3, "label": "interp", "sunspot": "raw12", "particles": True,
            "recency_weighting": None, "target": "speed(T+h) - speed(T)",
        },
        "member_B1": {
            "project_config": "config_ens_b1_FINAL.yaml", "variant": "B1",
            "encoder": "swin_tiny_patch4_window7_224 (ImageNet)", "channels": ["suvi_171", "suvi_195"],
            "residual_from": "speed at origin, learned weight per horizon", "mu_head": {"hidden": 131, "dropout": 0.53},
            "learning_rate": {"encoder": 1e-05, "head": 0.0001}, "batch_size": 16, "early_stopping_epochs": 30, "seed": 42,
        },
        "member_B2": {
            "model": "lightgbm per horizon", "trained_horizons": "same as A", "anchors_from": "2022-08-22T21:00:00Z",
            "embedding_pca_components": 32, "coronal_hole_lags_hours": [12, 24, 36, 48, 72, 96, 120],
        },
        "lightgbm": {
            "objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "max_rounds": 3000, "early_stopping_rounds": 100, "seed": 0,
        },
        "fixed_weights": {"grid_step": 0.05, "smoothing_horizons": 5},
        "gate_model_c": {"horizon_basis": 5, "lead_bands": "per band context weights", "l2_to_fixed_weights": 1.0,
                         "context": ["sigma_b1", "ch_cm30", "speed_now", "trend24", "disagree", "recent_err_A", "recent_err_B1"]},
    },
}
CONFIG_SHA256 = "3338b7d02d85d03887b5e66c8aded0fca299666d31cec1d3af3d1dba0f0920be"


def canonical_sha() -> str:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_config_identity() -> None:
    actual = canonical_sha()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: expected {CONFIG_SHA256}, got {actual} ({Path(__file__).stem})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--output", default="predictions.parquet")
    parser.add_argument("--refit-trees", action="store_true", help="refit A and B2 on the FINAL fold instead of reusing results_ens/oof/*/FINAL.parquet")
    args = parser.parse_args()
    verify_config_identity()
    project = Path(args.project_root).resolve()
    for p in (project / "scripts", project / "scripts" / "ensemble", project / "proswin-repo" / "src"):
        sys.path.insert(0, str(p))

    import combiner
    import member_a
    import member_b2
    import oof
    from data import load_bundle
    from folds import BY_NAME, FINAL, LEADERBOARD_ORIGINS, tuning_origins
    from run_stage3 import CONTEXT_NAMES, context_features, gate_eval, load_member

    hp = CONFIG["hyperparameters"]
    members, b1_name, use_c = list(hp["members"]), hp["member_B1"]["variant"], hp["combination"] == "gate_model_c"
    if args.folds:
        folds_path = args.folds
    else:
        from huggingface_hub import hf_hub_download
        folds_path = hf_hub_download(DEFAULT_FOLDS_REPO, "folds.parquet", repo_type="dataset")
    folds = pd.read_parquet(folds_path) if str(folds_path).endswith(".parquet") else pd.read_csv(folds_path)
    raw_origins = folds["origin_last_input_utc"]
    origins = pd.DatetimeIndex(pd.to_datetime(raw_origins, utc=True)).tz_localize(None)
    assert origins.min() == LEADERBOARD_ORIGINS[0] and origins.max() == LEADERBOARD_ORIGINS[1]

    bundle = load_bundle()
    assert origins.min() + pd.Timedelta(hours=1) > FINAL.label_cutoff  # every target is after the label cutoff
    H = list(range(1, HORIZON_HOURS + 1))
    variant = {k: hp["member_A"][k] for k in ("start_year", "label", "sunspot", "particles")}
    variant["recency_half_life_years"] = hp["member_A"]["recency_weighting"]
    variant["pooled"] = False

    # 1. members on the leaderboard origins (+ the F4 OOF forecasts as lead-in context for gate C)
    preds, timing, sigma_b1 = {}, {}, None
    for member in sorted(set(members) | {"A", b1_name}):
        path = oof.OOF_ROOT / "oof" / member / "FINAL.parquet"
        if member in ("A", "B2") and (args.refit_trees or not path.exists()):
            m = (member_a.fit_member(bundle, FINAL, variant) if member == "A" else member_b2.fit_member(bundle, FINAL))
            t0 = time.perf_counter()
            fin = m.predict(bundle, origins)
            timing[member] = (time.perf_counter() - t0) / len(origins)
            sfin, man = None, m.manifest
        else:
            fin, sfin, man = oof.read_oof(oof.OOF_ROOT, member, "FINAL")
            # per-origin inference time measured when the FINAL forecasts were produced (stage 4); the PROSWIN
            # forward was timed separately on the FINAL model (480 origins, batch 64, RTX A5000): 2.155 ms
            stage4 = oof.OOF_ROOT / "stage4" / "summary.json"
            measured = json.loads(stage4.read_text())["timing_s_per_origin"] if stage4.exists() else {}
            timing[member] = float(measured.get(member, 0.002155))
        assert pd.Timestamp(man["max_label_time"]) <= FINAL.label_cutoff, member
        f4, s4, _ = oof.read_oof(oof.OOF_ROOT, member, "F4")
        preds[member] = pd.concat([f4, fin]).sort_index()
        if member == b1_name and sfin is not None:
            sigma_b1 = pd.concat([s4, sfin]).sort_index()
        print(f"{member}: {len(fin)} leaderboard origins, {int(fin.reindex(origins).isna().any(axis=1).sum())} without forecast", flush=True)

    # 2. combination fitted on the out-of-fold forecasts inside the tuning window only
    oof_preds, oof_sigma = {}, None
    for member in set(members) | {"A", b1_name}:
        oof_preds[member], s = load_member(member)
        if member == b1_name:
            oof_sigma = s
    tuning = tuning_origins(BY_NAME["F3"]).union(tuning_origins(BY_NAME["F4"]))
    sub = {m: preds[m] for m in members}
    if use_c:
        model, _, _, _ = gate_eval(oof_preds, members, bundle, oof_sigma, b1_name, tuning, tuning_origins(BY_NAME["F4"]),
                                   float(hp["gate_model_c"]["l2_to_fixed_weights"]))
        t0 = time.perf_counter()
        ctx = context_features(bundle, preds, sigma_b1, origins, b1_name)
        final = combiner.gate_blend(model, sub, origins, ctx)
    else:
        w = combiner.fit_fixed_weights({m: oof_preds[m] for m in members}, bundle.truth, tuning,
                                       step=hp["fixed_weights"]["grid_step"], smooth=hp["fixed_weights"]["smoothing_horizons"])
        t0 = time.perf_counter()
        final = combiner.blend(sub, w, origins)
    timing["combine"] = (time.perf_counter() - t0) / len(origins)

    values = final.reindex(origins)[H].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "NaN in the final forecast"
    output = pd.DataFrame({
        "origin_last_input_utc": np.repeat(raw_origins.to_numpy(), HORIZON_HOURS),
        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)),
        "pred_kms": values.ravel(),
    })
    output.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "n_folds": int(len(origins)),
        "members": members, "combination": hp["combination"],
        "origins_without_proswin": int(preds[b1_name].reindex(origins).isna().any(axis=1).sum()),
        "inference_seconds_per_fold": float(sum(timing.values())),
        "inference_seconds_parts": timing,
    }))


if __name__ == "__main__":
    main()
