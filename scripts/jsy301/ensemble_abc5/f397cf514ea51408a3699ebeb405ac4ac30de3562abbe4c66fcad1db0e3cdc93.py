"""ensemble_abc5: abc4 + member X (shallow pooled trees), seed-bagged members, blend weights on a longer window. Team member jsy301.

Reproduces the leaderboard forecasts from the project checkout: the FINAL member forecasts of release
releases/ensemble_abc5 (made by scripts/exp/exp.py promote EXP-0043; A_e, B2_e and X each averaged over LightGBM
seeds 0-4, their boosters under releases/ensemble_abc5/models), the background-wind member rebuilt from the WSA-Enlil
runs, and the release's fixed per-horizon weights and gain factors.
Every input of an origin O uses data with timestamp <= O; every fitted quantity uses labels <= 2026-05-31 23:00.

    <project>/proswin-repo/.venv/bin/python scripts/jsy301/ensemble_abc5/<sha>.py --project-root <project> --output predictions.parquet
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
CONFIG = json.loads(r'''{
 "data": {
  "evaluation_folds": "tlabtlab/sunrun-lb-store/folds.parquet",
  "frequency": "1h",
  "input_rule": "only values with timestamp <= origin; ACE gaps carried forward; OMNI-filled hours are gaps",
  "inputs": [
   "ACE SWEPAM/MAG/EPAM/SIS (solar.csv since 2000 + NOAA SWPC real-time archive)",
   "GOES-18 SUVI Fe171/Fe195 coronal-hole measures and longitude strips (2022-08 on)",
   "SILSO monthly sunspot number, OMNI Earth heliographic latitude, solar-cycle phase",
   "NASA DONKI WSA-Enlil CME runs with modelCompletionTime <= origin (2010 on)",
   "NOAA/SWPC WSA-Enlil background runs (NCEI archive), usable from run date 00:00 + 24.5 h"
  ],
  "labels": "ACE speed forward-filled (scorer rule), labels <= 2026-05-31T23:00:00Z",
  "tuning_window": [
   "2025-08-01T00:00:00Z",
   "2026-05-28T23:00:00Z"
  ],
  "weight_window": [
   "2024-07-01T00:00:00Z",
   "2026-05-28T23:00:00Z"
  ],
  "out_of_fold": "4 folds (2024-07..2026-05), members trained with labels before each fold; blend weights fitted on all out-of-fold origins 2024-07..2026-05, gain on the tuning window",
  "experiment": "EXP-0043 (project experiment registry), release ensemble_abc5"
 },
 "hyperparameters": {
  "members": [
   "A_e",
   "B2_e",
   "BKG",
   "X"
  ],
  "seeds": [0, 1, 2, 3, 4],
  "combination": "fixed_per_horizon_weights (simplex grid 0.05, smoothed over 5 horizons, weight_window)",
  "gain_layer": {
   "enabled": true,
   "g_h_at_1_24_48_72": [0.949, 1.181, 1.196, 1.209]
  },
  "member_A": {
   "start_year": 2000,
   "recency_half_life_years": null,
   "label": "interp",
   "sunspot": "raw12",
   "particles": true,
   "pooled": false,
   "enlil": "six"
  },
  "member_B2": {
   "enlil": "six"
  },
  "member_X": {
   "model": "one LightGBM regression for all 72 horizons (horizon as input), target = speed at the target hour",
   "inputs": "B2 image inputs + ACE L1 summaries + sunspots/latitude (no DONKI, no background run)",
   "max_depth": 3,
   "num_leaves": 8,
   "rounds": 900,
   "learning_rate": 0.03,
   "min_data_in_leaf": 20,
   "feature_fraction": 0.7,
   "bagging_fraction": 0.8,
   "lambda_l2": 5.0,
   "train_origins": "every 3 h from 2022-10-01"
  },
  "background_wind": {
   "availability_hours": 24.5,
   "bias_correction_hours": 24
  }
 }
}''')
CONFIG_SHA256 = "f397cf514ea51408a3699ebeb405ac4ac30de3562abbe4c66fcad1db0e3cdc93"


def canonical_sha() -> str:
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--folds")
    ap.add_argument("--output", default="predictions.parquet")
    a = ap.parse_args()
    if canonical_sha() != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError("config identity mismatch")
    project = Path(a.project_root).resolve()
    for p in (project / "scripts", project / "scripts" / "ensemble"):
        sys.path.insert(0, str(p))
    import combiner, correction as C, enlil_bkg  # noqa: E401
    from data import load_bundle
    from folds import FINAL, LEADERBOARD_ORIGINS

    if a.folds:
        folds = pd.read_csv(a.folds) if str(a.folds).endswith(".csv") else pd.read_parquet(a.folds)
    else:
        from huggingface_hub import hf_hub_download
        folds = pd.read_parquet(hf_hub_download("tlabtlab/sunrun-lb-store", "folds.parquet", repo_type="dataset"))
    raw_origins = folds["origin_last_input_utc"]
    origins = pd.DatetimeIndex(pd.to_datetime(raw_origins, utc=True)).tz_localize(None)
    assert origins.min() == LEADERBOARD_ORIGINS[0] and origins.max() == LEADERBOARD_ORIGINS[1]
    H = list(range(1, HORIZON_HOURS + 1))
    hp = CONFIG["hyperparameters"]
    members = hp["members"]
    rel = project / "releases" / "ensemble_abc5"
    man = json.loads((rel / "manifest.json").read_text())
    assert man["experiment"] == "EXP-0043"
    bundle = load_bundle()
    assert origins.min() + pd.Timedelta(hours=1) > FINAL.label_cutoff
    preds = {}
    for m in members:
        if m == "BKG":
            preds[m] = pd.DataFrame(enlil_bkg.corrected_paths(enlil_bkg.load_runs(), origins, bundle.speed_causal), index=origins, columns=H)
        else:
            p = pd.read_parquet(rel / "members" / f"{m}.parquet")
            p.columns = [int(c) for c in p.columns]
            preds[m] = p.reindex(origins)
    w = pd.read_csv(rel / "weights.csv", index_col=0)
    w.index = w.index.astype(int)
    t0 = time.perf_counter()
    pred = combiner.blend(preds, w[members], origins)[H].to_numpy(dtype=float)
    now = bundle.speed_causal.reindex(origins).to_numpy(dtype=float)
    if hp["gain_layer"]["enabled"]:
        g = pd.read_csv(rel / "gain.csv", index_col=0)["g"].to_numpy(dtype=float)
        pred = C.apply_gain(pred, now, g)
    seconds = (time.perf_counter() - t0) / len(origins)
    out = pd.DataFrame({"origin_last_input_utc": np.repeat(raw_origins.to_numpy(), HORIZON_HOURS),
                        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)), "pred_kms": pred.ravel()})
    out.to_parquet(a.output, index=False)
    print(json.dumps({"output": str(Path(a.output).resolve()), "n_folds": int(len(origins)), "members": members,
                      "combination_seconds_per_fold": seconds}))


if __name__ == "__main__":
    main()
