"""Frozen CH-only candidate entry point; config identity is supplied in CONFIG."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.jsy301.ch_xgboost_runner import run  # noqa: E402

CONFIG = {
    "data": {"source": "pinned sun-db SUVI Fe195 numeric features",
             "feature_set": "ch_only", "threshold": 0.35,
             "feature_version": "ch-v1-t035-045-055-opening-closing3-minarea0005",
             "training_cutoff": "2025-07-31T23:00:00Z", "missing_target_policy": "forward_fill",
             "evaluation": "pinned folds June-September 2026"},
    "hyperparameters": {
        "model": "direct XGBoost one model per horizon", "peak_weight": 2.7306188803953546,
        "rise_weight": 2.2897531246059444, "weight_cap": 5,
        "xgboost": {"colsample_bytree": 0.5170598539249687, "gamma": 1.5769581497512377,
                    "learning_rate": 0.04753240502605287, "max_bin": 64, "max_depth": 6,
                    "min_child_weight": 17.82660255824357, "n_estimators": 2000,
                    "reg_alpha": 0.036192080233385465, "reg_lambda": 0.6585238724526555,
                    "subsample": 0.9034803828102056},
        "rounds_by_horizon": [44,43,68,42,58,43,37,38,29,24,35,41,41,29,42,29,28,29,30,28,28,34,32,39,58,75,43,46,49,24,60,40,34,96,48,70,74,46,60,68,79,60,100,53,35,51,25,29,19,50,71,26,19,61,37,56,25,20,54,37,53,79,13,37,56,17,39,63,57,115,33,26]}}

CONFIG_SHA256 = "09d1b7615f64984225e4d4ddac79f4914759936567310a4f3c22624ff621999c"


if __name__ == "__main__":
    actual = hashlib.sha256(json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: {actual}")
    run(CONFIG)
