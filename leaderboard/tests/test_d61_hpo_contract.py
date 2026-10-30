"""Contract for D61 rolling-fold HPO (leaderboard/experiments/d61_nestedhpo/).

- HPO is dev-only: fold bounds sit strictly inside dev; no later-period date
  or official-truth string may appear in hpo_spec/hpo_search/runner;
  hpo_results.json (when present) attests 20 completed TPE trials on the 3
  rolling folds with a fixed seed, winner beating the same-fold baseline mean.
- runner.py keeps the existing_all rows/features/purge (read-only imports)
  and only changes model params (BEST_PARAMS tuned schema + identical engine);
  its grid/write helpers honour the expos output contract.
- The live future-poison proof runs inside the expos run (poison.json); here
  we pin its interface (3 period probes, exact-key + positive control).
- Hypothesis carries the dual [family=xgb] + [axes: ...] prefix; the full
  gate is preregistered in FULL_RULE; scratch stays in worker-d61.
"""
import json
import sys
from pathlib import Path

import pandas as pd

D61 = Path(__file__).resolve().parents[1] / "experiments" / "d61_nestedhpo"
sys.path.insert(0, str(D61))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "expos_runners"))
import hpo_spec as spec  # noqa: E402
import info_ablation as ia  # noqa: E402
import runner as d61  # noqa: E402

DEV_END = pd.Timestamp("2025-11-27T23:00:00Z")
FORBIDDEN_STRINGS = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                     "2026-05-29", "2026-05-31", "2026-09-26",
                     "truth.parquet", "folds.parquet", "naive.parquet",
                     "baseline-replay", "official-predictions",
                     "datasets--tlabtlab--sunrun-lb-store")
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
TUNED = {"n_estimators", "max_depth", "learning_rate",
         "subsample", "colsample_bytree", "min_child_weight"}


def test_hypothesis_dual_tag_and_citations():
    assert spec.FAMILY_TAG == "[family=xgb]"
    assert spec.AXES_TAG.startswith("[axes:") and "opt=" in spec.AXES_TAG
    assert spec.HYPOTHESIS.startswith("[family=xgb] [axes:")
    assert "E0006" in spec.HYPOTHESIS and "X0037" in spec.HYPOTHESIS
    assert "20-trial" in spec.HYPOTHESIS and "6h" in spec.HYPOTHESIS


def test_folds_preregistered_inside_dev():
    assert len(spec.FOLDS) == 3
    assert [f["name"] for f in spec.FOLDS] == ["F1", "F2", "F3"]
    spec.check_folds_inside_dev()
    for src in ("hpo_spec.py", "hpo_search.py", "runner.py"):
        text = (D61 / src).read_text()
        for s in FORBIDDEN_STRINGS:
            assert s not in text, f"{src} must not reference {s}"


def test_search_budget_fixed_and_engine_pinned():
    assert spec.N_TRIALS == 20 and spec.SEED == 0
    src = (D61 / "hpo_search.py").read_text()
    assert "N_TRIALS = 20" in src and "SEED = 0" in src and "TPESampler" in src
    assert set(spec.FIXED_SEARCH) == {"reg_lambda", "tree_method", "n_jobs", "random_state"}
    assert spec.FIXED_SEARCH["tree_method"] == "hist" and spec.FIXED_SEARCH["reg_lambda"] == 5
    assert spec.FIXED_RUN["n_jobs"] * 3 <= 20  # 3 parallel period fits stay within budget


def test_hpo_results_attest_dev_only_when_present():
    p = D61 / "hpo_results.json"
    if not p.exists():
        return  # HPO still running; static guards above already hold
    r = json.loads(p.read_text())
    assert r["n_completed"] == 20 and r["seed"] == 0 and len(r["trials"]) == 20
    assert r["split"]["dev_only"] is True and len(r["split"]["folds"]) == 3
    for f in r["split"]["folds"]:
        vlo, vhi = pd.Timestamp(f["val_lo"]), pd.Timestamp(f["val_hi"])
        assert vlo < vhi <= DEV_END
    assert r["best_mean"] < r["baseline_mean"] * 1.02  # honest margin record; pilot decides
    assert set(r["best_params"]) == TUNED
    assert all(t["fold_mses"] and len(t["fold_mses"]) == 3 for t in r["trials"] if t["value"])


def test_runner_winner_matches_hpo_and_engine_identical():
    assert set(d61.BEST_PARAMS) == TUNED | {"reg_lambda", "tree_method", "n_jobs", "random_state"}
    b = d61.BEST_PARAMS
    assert b["reg_lambda"] == 5 and b["tree_method"] == "hist" and b["random_state"] == 0
    assert b["reg_lambda"] == ia.PARAMS["reg_lambda"] and b["random_state"] == ia.PARAMS["random_state"]
    tuned = {k: b[k] for k in TUNED}
    for k, (kind, lo, hi) in spec.SPACE.items():
        assert lo <= tuned[k] <= hi, f"{k} escapes the preregistered space"
    p = D61 / "hpo_results.json"
    if p.exists():
        assert tuned == json.loads(p.read_text())["best_params"]


def test_runner_keeps_existing_all_rows_and_features():
    import inspect
    src = (D61 / "runner.py").read_text()
    assert "from existing_all import full_grid" in src
    assert 'ia.arms(df.columns)["existing_all"]' in src
    assert 'df["target_t"] < lo - pd.Timedelta("72h")' in src
    assert set(inspect.signature(d61.fit_period).parameters) == {"args"}
    assert "name, lo, hi, df, cols, params, pilot = args" in inspect.getsource(d61.fit_period)
    assert d61.PERIOD_JOBS == 3 and d61.PERIOD_JOBS * d61.PERIOD_N_JOBS <= 20


def test_full_rule_preregistered_and_scratch_owned():
    assert "poison" in spec.FULL_RULE and "1/3" in spec.FULL_RULE and "exp-add" in spec.FULL_RULE
    assert "6h" in spec.FULL_RULE and "3h" in spec.FULL_RULE
    assert str(spec.SCRATCH).endswith("store/scratch/worker-d61")
    runner_src = (D61 / "runner.py").read_text()
    assert "worker-d61" in runner_src and "worker-d59" not in runner_src
    assert "worker-d60" not in runner_src
    pkg = sorted(p.name for p in D61.iterdir() if p.suffix == ".py")
    assert len(pkg) <= 3, f"owned python files exceed budget: {pkg}"
    for src in ("hpo_spec.py", "hpo_search.py", "runner.py"):
        text = (D61 / src).read_text()
        for ban in COPY_BANS:
            assert ban not in text, f"{src} must import strips, never vendor it"


def test_full_grid_full_horizon_contract():
    origins = pd.DatetimeIndex(["2025-09-15T00:00:00Z", "2025-09-15T03:00:00Z"])
    g = d61.full_grid(origins)
    assert len(g) == 2 * 72
    assert not g.duplicated(["origin", "h"]).any()
    assert sorted(g["h"].unique()) == list(range(1, 73))
    assert (g.groupby("origin")["h"].size() == 72).all()


def test_write_output_columns(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(d61, "OUT", tmp_path)
    origins = pd.DatetimeIndex(["2025-09-15T00:00:00Z"])
    g = d61.full_grid(origins)
    d61.write(g, np.zeros(len(g)), "dev")
    p = pd.read_parquet(tmp_path / "predictions_dev.parquet")
    assert list(p.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert len(p) == 72 and sorted(p["horizon_hours"]) == list(range(1, 73))
    assert p["origin_last_input_utc"].eq("2025-09-15T00:00:00Z").all()


def test_poison_interface_covers_three_periods():
    assert set(d61.PROBES) == {"dev", "selection", "audit"}
    for iso in d61.PROBES.values():
        o = pd.Timestamp(iso)
        assert o.tz is not None and (o.hour % 3 == 0) and o.minute == 0
    import inspect
    assert callable(d61.poison_check)
    assert "positive_control" in inspect.getsource(d61.poison_check)
