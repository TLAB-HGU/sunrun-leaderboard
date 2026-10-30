"""Contract for D2 dev-only HPO (leaderboard/experiments/d2_hpo/).

- HPO uses dev origins only: no selection/audit/official date or truth string
  may appear in hpo_search.py; VAL_LO sits strictly inside the dev window;
  hpo_results.json (when present) attests 60 completed trials on the late-dev
  holdout with a fixed seed.
- runner.py keeps the existing_all rows/features/purge and only changes model
  params (BEST_PARAMS schema); its grid/write helpers honour the expos output
  contract (every origin x 72 horizons, exact column names).
- The live future-poison proof runs inside the expos run (poison.json); here
  we pin its interface (3 period probes on the 3h grid, exact-key + positive
  control semantics in poison_check).
"""
import json
import sys
from pathlib import Path

import pandas as pd

D2 = Path(__file__).resolve().parents[1] / "experiments" / "d2_hpo"
sys.path.insert(0, str(D2))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import runner as d2  # noqa: E402
import info_ablation as ia  # noqa: E402

DEV_END = pd.Timestamp("2025-11-27T23:00:00Z")
FORBIDDEN_DATE_STRINGS = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                          "2026-05-29", "2026-05-31", "2026-09-26",
                          "truth.parquet", "folds.parquet", "naive.parquet",
                          "baseline-replay", "official-predictions")


def test_hpo_search_is_dev_only_static():
    src = (D2 / "hpo_search.py").read_text()
    for s in FORBIDDEN_DATE_STRINGS:
        assert s not in src, f"hpo_search.py must not reference {s}"
    assert "VAL_LO_UTC" in src and "N_TRIALS = 60" in src and "SEED = 0" in src
    import hpo_search as hs
    dev_lo, dev_end = pd.Timestamp(hs.DEV_LO_UTC), pd.Timestamp(hs.DEV_END_UTC)
    val_lo = pd.Timestamp(hs.VAL_LO_UTC)
    assert dev_lo < val_lo < dev_end  # late-dev holdout strictly inside dev
    assert hs.N_TRIALS == 60 and hs.SEED == 0
    assert set(hs.FIXED) == {"reg_lambda", "tree_method", "n_jobs", "random_state"}


def test_hpo_results_attest_dev_only_when_present():
    p = D2 / "hpo_results.json"
    if not p.exists():
        return  # HPO still running; static guards above already hold
    r = json.loads(p.read_text())
    assert r["n_completed"] == 60 and r["seed"] == 0
    assert r["split"]["dev_only"] is True
    vlo, vend = (pd.Timestamp(t) for t in r["split"]["hpo_val"])
    assert pd.Timestamp(r["split"]["dev"][0]) <= vlo < vend <= DEV_END
    assert r["best_params"] and r["best_val_mse"] < r["baseline_val_mse"]
    assert len(r["trials"]) == 60


def test_runner_params_schema_only():
    tuned = {"n_estimators", "max_depth", "learning_rate",
             "subsample", "colsample_bytree", "min_child_weight"}
    assert set(d2.BEST_PARAMS) == tuned | {"reg_lambda", "tree_method", "n_jobs", "random_state"}
    b = d2.BEST_PARAMS
    assert b["reg_lambda"] == 5 and b["tree_method"] == "hist" and b["random_state"] == 0
    assert 200 <= b["n_estimators"] <= 1200 and 3 <= b["max_depth"] <= 8
    assert 0.02 <= b["learning_rate"] <= 0.15 and 0.6 <= b["subsample"] <= 1.0
    assert 0.5 <= b["colsample_bytree"] <= 1.0 and 5 <= b["min_child_weight"] <= 100


def test_runner_keeps_existing_all_rows_and_features():
    import inspect
    src = (D2 / "runner.py").read_text()
    for s in FORBIDDEN_DATE_STRINGS:
        assert s not in src, f"runner.py must not reference {s}"
    assert 'ia.arms(df.columns)["existing_all"]' in src
    assert 'df["target_t"] < lo - pd.Timedelta("72h")' in src
    assert set(inspect.signature(d2.full_grid).parameters) == {"origins"}


def test_full_grid_full_horizon_contract():
    origins = pd.DatetimeIndex(["2025-09-15T00:00:00Z", "2025-09-15T03:00:00Z"])
    g = d2.full_grid(origins)
    assert len(g) == 2 * 72
    assert not g.duplicated(["origin", "h"]).any()
    assert sorted(g["h"].unique()) == list(range(1, 73))
    assert (g.groupby("origin")["h"].size() == 72).all()


def test_write_output_columns(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(d2, "OUT", tmp_path)
    origins = pd.DatetimeIndex(["2025-09-15T00:00:00Z"])
    g = d2.full_grid(origins)
    d2.write(g, np.zeros(len(g)), "dev")
    p = pd.read_parquet(tmp_path / "predictions_dev.parquet")
    assert list(p.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(p) == 72 and sorted(p["horizon_hours"]) == list(range(1, 73))
    assert p["origin_last_input_utc"].eq("2025-09-15T00:00:00Z").all()


def test_poison_interface_covers_three_periods():
    assert set(d2.PROBES) == {"dev", "selection", "audit"}
    for iso in d2.PROBES.values():
        o = pd.Timestamp(iso)
        assert o.tz is not None and (o.hour % 3 == 0) and o.minute == 0
    import inspect
    assert callable(d2.poison_check)
    assert "positive_control" in inspect.getsource(d2.poison_check)
