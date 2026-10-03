"""Frozen ACE+CH candidate entry point; config identity is supplied in CONFIG."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.jsy301.ch_xgboost_runner import run  # noqa: E402

CONFIG = {
    "data": {"source": "pinned sun-db ACE + SUVI Fe195 numeric features",
             "feature_set": "ace_ch", "threshold": 0.35,
             "feature_version": "ch-v1-t035-045-055-opening-closing3-minarea0005",
             "training_cutoff": "2025-07-31T23:00:00Z", "missing_target_policy": "forward_fill",
             "evaluation": "pinned folds June-September 2026"},
    "hyperparameters": {
        "model": "direct XGBoost one model per horizon", "peak_weight": 2.825080620111566,
        "rise_weight": 2.0856123011459125, "weight_cap": 5,
        "xgboost": {"colsample_bytree": 0.7643984683359885, "gamma": 1.8988452481227993,
                    "learning_rate": 0.03372142497594221, "max_bin": 64, "max_depth": 4,
                    "min_child_weight": 6.405996928403651, "n_estimators": 2000,
                    "reg_alpha": 0.004208086201731013, "reg_lambda": 0.2911346465714742,
                    "subsample": 0.7018386128294355},
        "rounds_by_horizon": [101,120,140,87,112,110,178,146,131,133,170,145,138,186,138,123,119,132,95,137,107,89,90,85,98,136,139,93,93,82,79,90,82,91,127,114,115,111,116,85,83,127,171,185,64,173,55,47,295,22,114,27,34,99,64,102,12,54,98,12,47,20,27,64,61,69,68,57,63,61,74,29]}}

CONFIG_SHA256 = "5247ee2ce72d69d5d8953697f8df47ccea96a5cf3176df7df1247613b6076838"


if __name__ == "__main__":
    actual = hashlib.sha256(json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError(f"config identity mismatch: {actual}")
    run(CONFIG)
