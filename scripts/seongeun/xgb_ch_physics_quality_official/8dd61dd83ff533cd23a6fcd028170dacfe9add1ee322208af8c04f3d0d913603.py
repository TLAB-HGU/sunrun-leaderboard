"""Frozen winner-physical replay; newer raw snapshot requires full equivalence proof."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.seongeun.ch_breakthrough_runner import run

CONFIG = json.loads(r'''{"data":{"ace_history_hours":72,"evaluation":"2026-06-01..2026-09-30 official 2833x72 folds; exploratory previously inspected","evaluation_snapshot":"bcf5d5417d99eaa331ddca40581d48953e09a58e","feature_arm":"physical_ace_ch_quality","feature_branch":"physical","horizons_hours":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72],"input_sha256":{"ace":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","ch_hourly":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","raw_ace_years":"695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736"},"mean_reversion_window_hours":648,"missing_policy":"causal forward fill; original observation flags retained","original_experiment_raw_ace_years_sha256":"c3f309bbd11a6137e4064c1ba2f2e1b12ba37f8f2d44395978946ab20f721abe","point_in_time_vintage_verified":false,"recurrence_days":[],"reproduction_raw_snapshot_created_utc":"2026-10-03T05:18:39.245782+00:00","source":"sun-db ACE+single Fe195 coronal-hole numeric geometry (no image neural network)","source_sha256":{"common":"23ec9a09ead017f4682749fdcb70d370d992f67a799eccd8c2c451b072c9b4c2","hpo":"b2aea0a61bcf43e9a4bfe9ea8e272884c013eaf623ed6883320513e3d4ffeef5","hpo_gpu":"0f29043055082e8e6feae49e6d000214f37cfcee33906007ba19042242595309","physical_quality":"a08a070ae2a9b244f078cc79597f04782e74dfdbf5fde301998c6b36f33aeec6","recurrence":"ce166d9acfee0f9ce7c2ffb635aa3bb5f25182119e0bfad291bb6169557bdfbb"},"training_cutoff":"2026-05-31T23:00:00+00:00"},"hyperparameters":{"max_train_origins":6000,"model":"pooled horizon XGBoost residual atop 648h mean-reversion tau48h","n_jobs":4,"output_range_kms":[1,3000],"seed":42,"xgboost":{"colsample_bytree":0.7,"learning_rate":0.05,"max_depth":4,"min_child_weight":6.0,"n_estimators":80,"reg_lambda":5.0,"subsample":0.9}}}''')
CONFIG_SHA256 = "8dd61dd83ff533cd23a6fcd028170dacfe9add1ee322208af8c04f3d0d913603"


if __name__ == "__main__":
    if Path(__file__).stem != CONFIG_SHA256:
        raise RuntimeError("submission script filename/config identity mismatch")
    run(CONFIG, expected_config_sha256=CONFIG_SHA256)
