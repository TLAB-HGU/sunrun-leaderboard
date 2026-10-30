import json

import pytest

from board import _latest, load_results, ranked, render
from submit import canonical_config, script_location, store_submission, validate_meta


CONFIG = {
    "data": {"source": "ace", "features": ["speed"]},
    "hyperparameters": {"tau": 48, "window": 648},
}


def meta(**overrides):
    value = {
        "team_member": "alice",
        "experiment": "mean_reversion",
        "description": "test",
        "no_future_leakage": True,
        "f1_training_data_end_utc": "2026-02-26T00:00:00Z",
        "config": CONFIG,
        "inference_seconds_per_fold": 0.001,
    }
    value.update(overrides)
    return value


def test_config_hash_is_stable_across_key_order():
    reordered = {
        "hyperparameters": {"window": 648, "tau": 48},
        "data": {"features": ["speed"], "source": "ace"},
    }
    canonical, digest = canonical_config(CONFIG)
    other_canonical, other_digest = canonical_config(reordered)
    assert json.loads(canonical) == CONFIG
    assert other_canonical == canonical
    assert other_digest == digest


def test_config_hash_changes_with_data_or_hyperparameters():
    _, digest = canonical_config(CONFIG)
    _, data_digest = canonical_config({**CONFIG, "data": {"source": "suvi"}})
    _, hp_digest = canonical_config({**CONFIG, "hyperparameters": {"tau": 24}})
    assert digest not in (data_digest, hp_digest)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "fast"])
def test_timing_must_be_positive_and_finite(value):
    with pytest.raises(SystemExit, match="positive finite"):
        validate_meta(meta(inference_seconds_per_fold=value))


@pytest.mark.parametrize("value", ["2026-02-26T00:00:00Z", "2026-02-25", "2026-02-26T09:00:00+09:00",
                                   __import__("datetime").datetime(2026, 2, 1)])
def test_f1_training_end_on_or_before_cutoff_is_accepted(value):
    assert validate_meta(meta(f1_training_data_end_utc=value))[3] <= "2026-02-26T00:00:00Z"


@pytest.mark.parametrize("value", ["2026-02-26T01:00:00Z", "2026-05-31", "not a date"])
def test_f1_training_end_after_cutoff_or_invalid_is_rejected(value):
    with pytest.raises(SystemExit, match="f1_training_data_end_utc"):
        validate_meta(meta(f1_training_data_end_utc=value))


def test_meta_requires_f1_training_end():
    value = meta()
    del value["f1_training_data_end_utc"]
    with pytest.raises(SystemExit, match="f1_training_data_end_utc"):
        validate_meta(value)


def test_meta_requires_structured_config_and_safe_names():
    with pytest.raises(SystemExit, match="config.data"):
        validate_meta(meta(config={"data": {}, "hyperparameters": {"tau": 48}}))
    with pytest.raises(SystemExit, match="team_member"):
        validate_meta(meta(team_member="../alice"))
    with pytest.raises(SystemExit, match="description"):
        validate_meta(meta(description=[]))
    with pytest.raises(SystemExit, match="unsupported sections"):
        validate_meta(meta(config={**CONFIG, "notes": {"owner": "alice"}}))


def test_script_path_is_derived_from_full_identity():
    _, digest = canonical_config(CONFIG)
    path, raw, page = script_location("alice", "mean_reversion", digest)
    assert path == f"scripts/alice/mean_reversion/{digest}.py"
    assert path in raw
    assert path in page


def test_results_are_unique_by_member_experiment_and_config():
    base = {"team_member": "alice", "experiment": "model", "submitted_utc": "20260101", "mse": 2.0}
    rows = _latest([
        {**base, "submitted_utc": "20260101", "config_sha256": "a" * 64},
        {**base, "submitted_utc": "20260102", "config_sha256": "b" * 64},
        {**base, "submitted_utc": "20260103", "config_sha256": "a" * 64},
        {**base, "submitted_utc": "20260101"},
        {**base, "submitted_utc": "20260102"},
    ])
    assert len(rows) == 3
    assert {r["submitted_utc"] for r in rows} == {"20260102", "20260103"}


def test_board_renders_new_and_legacy_metadata():
    metric = {
        "team_member": "alice", "experiment": "model", "submitted_utc": "20260101T000000Z",
        "mse": 2.0, "mse_observed": 2.0, "skill_vs_naive": 0.1,
        "regime_balanced_mse": None, "vs_naive": {"beats_reference": True},
        "by_regime": {name: {"mse": None, "reliable": False} for name in __import__("board").REGIMES},
    }
    page = render([
        metric,
        {**metric, "experiment": "new", "config_sha256": "a" * 64,
         "inference_seconds_per_fold": 0.001234, "code_url": "https://example.com/code.py"},
    ])
    assert "legacy" in page
    assert "0.001234" in page
    assert "https://example.com/code.py" in page


def test_f2_only_results_rank_below_f1_f2_results():
    regimes = {name: {"mse": None, "reliable": False} for name in __import__("board").REGIMES}
    base = {"team_member": "alice", "submitted_utc": "20261026T000000Z", "skill_vs_naive": 0.1,
            "vs_naive": {"beats_reference": True}, "by_regime": regimes}
    full = {"mse_f1": 5000.0, "mse_f2": 3000.0, "mse_ge550": 9000.0, "mse_f1_ge550": 9500.0, "mse_f2_ge550": 8500.0}
    rows = [
        {**base, "experiment": "old_good", "config_sha256": "a" * 64, "mse": 1000.0},
        {**base, "experiment": "new_worse", "config_sha256": "b" * 64, "mse": 4100.0, **full},
        {**base, "experiment": "new_better", "config_sha256": "c" * 64, "mse": 4000.0, **full},
        {**base, "experiment": "old_bad", "config_sha256": "d" * 64, "mse": 2000.0},
    ]
    assert [r["experiment"] for r in ranked(rows)] == ["new_better", "new_worse", "old_good", "old_bad"]
    page = render(rows)
    assert page.index("new_better") < page.index("new_worse") < page.index("old_good") < page.index("old_bad")
    assert "F2 only" in page and "9,500.0" in page
    assert '<td data-v="1000.0">1,000.0</td>' in page  # an F2-only row's old MSE is shown as its F2 MSE


def test_local_preview_loads_flat_and_nested_results(tmp_path):
    legacy = tmp_path / "legacy.json"
    nested = tmp_path / "alice/model/hash.json"
    nested.parent.mkdir(parents=True)
    legacy.write_text('{"experiment": "legacy"}')
    nested.write_text('{"experiment": "new"}')
    assert {r["experiment"] for r in load_results(tmp_path)} == {"legacy", "new"}


class FakeApi:
    def __init__(self, conflict=False):
        self.conflict = conflict
        self.commits = []
        self.head = 0

    def repo_info(self, repo, repo_type):
        self.head += 1
        return type("Info", (), {"sha": f"head-{self.head}"})()

    def create_commit(self, **kwargs):
        if self.conflict:
            self.conflict = False
            import httpx
            from huggingface_hub.utils import HfHubHTTPError
            response = httpx.Response(409, request=httpx.Request("POST", "https://huggingface.co"))
            raise HfHubHTTPError("conflict", response=response)
        self.commits.append(kwargs)


def result_record():
    return {"team_member": "alice", "experiment": "model", "config_sha256": "a" * 64}


def test_store_submission_commits_prediction_and_result_atomically(monkeypatch):
    monkeypatch.setattr("submit.all_results", lambda api, repo, revision=None: [])
    api = FakeApi()
    store_submission(api, "repo", result_record(), b"prediction", replace=False)
    assert len(api.commits) == 1
    commit = api.commits[0]
    assert commit["parent_commit"] == "head-1"
    assert {op.path_in_repo.split("/", 1)[0] for op in commit["operations"]} == {"submissions", "results"}


def test_concurrent_same_key_is_rechecked_and_rejected(monkeypatch):
    calls = 0

    def results(api, repo, revision=None):
        nonlocal calls
        calls += 1
        return [] if calls == 1 else [result_record()]

    monkeypatch.setattr("submit.all_results", results)
    with pytest.raises(SystemExit, match="already exists"):
        store_submission(FakeApi(conflict=True), "repo", result_record(), b"prediction", replace=False)
