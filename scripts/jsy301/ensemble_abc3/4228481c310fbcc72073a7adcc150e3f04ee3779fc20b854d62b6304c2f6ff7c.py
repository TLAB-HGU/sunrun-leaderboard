"""ensemble_abc3: A/B/C ensemble with correction layers (gain, amount model). Team member jsy301.

Reproduces the leaderboard forecasts from the project checkout: member forecasts of the FINAL fold
(results_ens/oof/<member>/FINAL.parquet, background-wind runs), the fixed per-horizon weights, the gain
factors and the amount model stored by scripts/ensemble/run_final3.py under results_ens/final3/.
Every input of an origin O uses data with timestamp <= O; every fitted quantity uses labels <= 2026-05-31 23:00.

    <project>/proswin-repo/.venv/bin/python scripts/jsy301/ensemble_abc3/<sha>.py --project-root <project> --output predictions.parquet
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
   "NASA DONKI WSA-Enlil CME runs with modelCompletionTime <= origin (2010 on)"
  ],
  "labels": "ACE speed forward-filled (scorer rule), labels <= 2026-05-31T23:00:00Z",
  "tuning_window": [
   "2025-08-01T00:00:00Z",
   "2026-05-28T23:00:00Z"
  ],
  "out_of_fold": "4 folds (2024-07..2026-05), members trained with labels before each fold; amount model fit on F1+F2+F3, its weight on F4"
 },
 "hyperparameters": {
  "members": [
   "A_e",
   "B2_e"
  ],
  "combination": "fixed_per_horizon_weights (simplex grid 0.05, smoothed over 5 horizons, tuning window)",
  "gain_layer": {
   "enabled": true,
   "g_h_at_1_24_48_72": [
    0.983,
    1.194,
    1.25,
    1.187
   ]
  },
  "amount_model": null,
  "member_A": {
   "start_year": 2000,
   "recency_half_life_years": null,
   "label": "interp",
   "sunspot": "raw12",
   "particles": true,
   "pooled": false,
   "enlil": "six"
  },
  "proswin": {
   "included": false,
   "config": "config_ens_b1_FINAL.yaml"
  },
  "background_wind": null
 }
}''')
CONFIG_SHA256 = "4228481c310fbcc72073a7adcc150e3f04ee3779fc20b854d62b6304c2f6ff7c"


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
    for p in (project / "scripts", project / "scripts" / "ensemble", project / "proswin-repo" / "src"):
        sys.path.insert(0, str(p))
    import combiner, correction as C, enlil_bkg, lightgbm as lgb, oof  # noqa: E401
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
    art = project / "proswin-repo" / "results_ens" / "final3"
    bundle = load_bundle()
    assert origins.min() + pd.Timedelta(hours=1) > FINAL.label_cutoff
    bkg = enlil_bkg.load_runs() if hp.get("background_wind") else None
    preds = {}
    for m in members:
        if m == "BKG":
            preds[m] = pd.DataFrame(enlil_bkg.corrected_paths(bkg, origins, bundle.speed_causal), index=origins, columns=H)
        else:
            p, _, man = oof.read_oof(oof.OOF_ROOT, m, "FINAL")
            assert pd.Timestamp(man["max_label_time"]) <= FINAL.label_cutoff
            preds[m] = p.reindex(origins)
    w = pd.read_csv(art / "weights.csv", index_col=0)
    w.index = w.index.astype(int)
    t0 = time.perf_counter()
    pred = combiner.blend(preds, w[members], origins)[H].to_numpy(dtype=float)
    now = bundle.speed_causal.reindex(origins).to_numpy(dtype=float)
    g = pd.read_csv(art / "gain.csv", index_col=0)["g"].to_numpy(dtype=float)
    if hp["gain_layer"]["enabled"]:
        pred = C.apply_gain(pred, now, g)
    am = hp.get("amount_model")
    if am:
        inputs = tuple(C.INPUT_SETS[am["inputs"]])
        X = C.features(origins, pred, bundle, inputs, bkg)
        cols = (art / "amount" / "columns.txt").read_text().split("\n")
        x = X[cols].to_numpy(np.float32)
        if am["method"] == "A":
            corr = lgb.Booster(model_file=str(art / "amount" / "residual.txt")).predict(x)
        else:
            p_ev = lgb.Booster(model_file=str(art / "amount" / "event_cls.txt")).predict(x)
            s = lgb.Booster(model_file=str(art / "amount" / "event_size.txt")).predict(x)
            c = lgb.Booster(model_file=str(art / "amount" / "quiet_size.txt")).predict(x)
            corr = p_ev * s + (1 - p_ev) * c
        pred = C.apply_w(pred, corr.reshape(-1, HORIZON_HOURS), float(am["w"]))
    seconds = (time.perf_counter() - t0) / len(origins)
    out = pd.DataFrame({"origin_last_input_utc": np.repeat(raw_origins.to_numpy(), HORIZON_HOURS),
                        "horizon_hours": np.tile(np.arange(1, HORIZON_HOURS + 1), len(origins)), "pred_kms": pred.ravel()})
    out.to_parquet(a.output, index=False)
    print(json.dumps({"output": str(Path(a.output).resolve()), "n_folds": int(len(origins)), "members": members,
                      "combination_seconds_per_fold": seconds}))


if __name__ == "__main__":
    main()
