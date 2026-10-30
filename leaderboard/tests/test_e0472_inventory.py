"""Synthetic evidence contracts for the historical audit (no scientific data access)."""
import importlib.util
import hashlib
import json
import sqlite3
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "e0472_inventory", Path(__file__).resolve().parents[1] /
    "experiments/e0472_ablation_hpo/inventory.py")
inv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inv)


def registration(eid, reference=None, hypothesis="CPU recipe"):
    return {"event": "register", "id": eid, "reference": reference,
            "direction": "D1", "hypothesis": hypothesis, "seq": 1}


def score(eid, mse=1000, f1=0.4, cell=2000, seq=3):
    return {"event": "lbeval", "id": eid, "seq": seq,
            "summary": {"mse_seedmean": mse, "f1_seedmean": f1,
                        "n_seeds": 3, "mse_by_seed": [mse - 1, mse, mse + 1],
                        "cells_seedmean": {c: cell for c in inv.CELLS}}}


def result_rows(tmp_path, candidates):
    events = [registration("E0472", "E0400"), score("E0472")]
    for eid, extra in candidates.items():
        events.extend([registration(eid, "E0472"), {"event": "run", "id": eid, "rc": 0, "seq": 2}, *extra])
    experiments, directions = inv.reduce_events(events)
    metadata = {eid: [("metadata.json", {"device": "cpu"})] for eid in candidates}
    return {r["id"]: r for r in inv.build_rows(experiments, directions, metadata, tmp_path)}


def test_reduction_keeps_directions_and_score_history_separate():
    experiments, directions = inv.reduce_events([
        {"event": "direction", "id": "D1", "title": "original", "parent": "D0"},
        {"event": "direction", "id": "D1", "state": "closed"},
        registration("E0472", "E0400"), score("E0472"), score("E0472", mse=990),
    ])
    assert directions["D1"]["parent"] == "D0"
    assert directions["D1"]["title"] == "original"
    assert experiments["E0472"]["registration"]["reference"] == "E0400"
    assert len(experiments["E0472"]["lbeval"]) == 2


def test_reference_is_comparator_not_recipe_parent(tmp_path):
    events = [registration("E0400"), registration("E0472", "E0400"),
              registration("E0500", "E0472"), registration("E0501", "E0500"),
              registration("E0502", "E0400", "mentions E0472 but unrelated recipe")]
    exp, dirs = inv.reduce_events(events)
    rows = {r["id"]: r for r in inv.build_rows(exp, dirs, root=tmp_path)}
    assert rows["E0400"]["relation"] == "baseline_reference_chain"
    assert rows["E0500"]["relation"] == "direct_reference"
    assert rows["E0501"]["relation"] == "indirect_reference"
    assert rows["E0502"]["relation"] == "mentions_baseline"
    assert all(r["official_rank"] is None for r in rows.values())


def test_oracle_score_is_never_ranked(tmp_path):
    rows = result_rows(tmp_path, {"E0500": [registration("E0500", "E0472", "ORACLE upper bound"), score("E0500", 100)]})
    assert rows["E0500"]["oracle_or_oracle_mention"]
    assert rows["E0500"]["official_rank"] is None


def test_strict_F1_and_each_cell_and_seed_gates(tmp_path):
    rows = result_rows(tmp_path, {
        "E0500": [score("E0500", 950, .3)],
        "E0501": [score("E0501", 940, .4, 2001)],
        "E0502": [score("E0502", 900, .4, 1999)],
    })
    assert "F1_lower" in rows["E0500"]["adoption_failures"]
    assert rows["E0500"]["effect_strength"] == "weak_under_90"
    assert len(rows["E0501"]["adoption_failures"]) == 4
    assert rows["E0502"]["metric_adoption"] == "metric_gates_pass_review_required"
    assert rows["E0502"]["official_rank"] == 1


def test_missing_metrics_and_run_vs_score(tmp_path):
    incomplete = score("E0500", 900)
    del incomplete["summary"]["cells_seedmean"][inv.CELLS[0]]
    rows = result_rows(tmp_path, {"E0500": [incomplete], "E0501": []})
    assert "missing_high_cells" in rows["E0500"]["comparison_blockers"]
    assert rows["E0501"]["run_status"] == "run_succeeded"
    assert rows["E0501"]["protocol"] == "unscored"
    assert rows["E0501"]["delta_MSE_vs_registered_E0472"] is None


def test_stale_score_excluded_after_rerun(tmp_path):
    rows = result_rows(tmp_path, {"E0500": [score("E0500", 900), {"event": "run", "id": "E0500", "rc": 0, "seq": 5}]})
    assert "score_predates_latest_run" in rows["E0500"]["comparison_blockers"]


def test_trial_extraction_preserves_split_and_failure_and_does_not_invent_grid_scores():
    data = {"split": {"dev_only": True}, "seed": 2,
            "trials": [{"number": 0, "value": 42, "params": {"depth": 3}},
                       {"number": 1, "value": None, "state": "FAIL"}],
            "grid": [1, 2, 3], "best_trial": 0,
            "seeds": {"2": {"k_scores": {"15": 41.5}}}}
    rows = inv.extract_trials(data, "a.json", ["E0001"])
    assert len(rows) == 3
    assert rows[0]["json_pointer"] == "/trials/0"
    assert rows[0]["context"]["split"]["dev_only"]
    assert rows[1]["state"] == "FAIL" and rows[1]["value"] is None
    assert rows[2]["json_pointer"] == "/seeds/2/k_scores/15"
    assert inv.extract_trials({"best_trial": 3, "best_params": {"depth": 4}}, "summary.json") == []


def test_device_requires_metadata_or_matching_code_hash(tmp_path):
    reg = registration("E0500", "E0472", "CPU baseline, GPU candidate")
    assert inv.infer_device(reg, [], tmp_path)[0] == "unknown"
    assert inv.infer_device(reg, [("meta", {"fit": {"device": "cpu"}})], tmp_path)[0] == "cpu"
    assert inv.infer_device(reg, [("meta", {"base": {"device": "cpu"}, "candidate": {"device": "cuda"}})], tmp_path)[0] == "mixed"


def test_exported_CONFIG_requires_matching_digest_and_registered_sources(tmp_path):
    folder = tmp_path / "leaderboard/scripts/seongeun/xgb_d76_lead_moe_resid_dv_enlil"
    folder.mkdir(parents=True)
    config = {"data": {"source_sha256": {"lean.py": "registered"}}, "hyperparameters": {"device": "cpu"}}
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (folder / (digest + ".py")).write_text("CONFIG = json.loads(" + repr(payload) + ")\nraise RuntimeError('must not import')\n")
    assert len(list(inv.exported_baseline_metadata(tmp_path, {"code": {"lean.py": "registered"}}))) == 1
    assert not list(inv.exported_baseline_metadata(tmp_path, {"code": {"lean.py": "different"}}))


def test_sqlite_trials_preserve_failures_and_decode_categories(tmp_path):
    path = tmp_path / "trials.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE studies(study_id, study_name);
            CREATE TABLE trials(trial_id, number, study_id, state);
            CREATE TABLE trial_params(trial_id, param_name, param_value, distribution_json);
            CREATE TABLE trial_values(trial_id, objective, value, value_type);
            INSERT INTO studies VALUES(1, 'old-hpo');
            INSERT INTO trials VALUES(1, 0, 1, 'COMPLETE');
            INSERT INTO trials VALUES(2, 1, 1, 'FAIL');
            INSERT INTO trial_values VALUES(1, 0, 12.5, 'FINITE');
        """)
        conn.execute("INSERT INTO trial_params VALUES(1, 'weight', 1, ?)",
                     (json.dumps({"name": "CategoricalDistribution", "attributes": {"choices": ["uniform", "distance"]}}),))
    rows = inv.sqlite_trials(path, tmp_path)
    assert len(rows) == 2
    assert rows[0]["params"] == {"weight": "distance"}
    assert rows[0]["value"] == 12.5
    assert rows[1]["state"] == "FAIL" and rows[1]["value"] is None
