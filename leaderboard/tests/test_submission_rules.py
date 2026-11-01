import json
import re
from pathlib import Path

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
        "f1_training_data_end_utc": "2025-12-29T00:00:00Z",
        "f2_training_data_end_utc": "2026-03-29T00:00:00Z",
        "f3_training_data_end_utc": "2026-06-29T00:00:00Z",
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


@pytest.mark.parametrize("value", ["2025-12-29T00:00:00Z", "2025-12-28", "2025-12-29T09:00:00+09:00",
                                   __import__("datetime").datetime(2025, 12, 1)])
def test_f1_training_end_on_or_before_cutoff_is_accepted(value):
    assert validate_meta(meta(f1_training_data_end_utc=value))[3] <= "2025-12-29T00:00:00Z"


@pytest.mark.parametrize("value", ["2025-12-29T01:00:00Z", "2026-02-26", "not a date"])
def test_f1_training_end_after_cutoff_or_invalid_is_rejected(value):
    with pytest.raises(SystemExit, match="f1_training_data_end_utc"):
        validate_meta(meta(f1_training_data_end_utc=value))


@pytest.mark.parametrize("period, ok, late", [("f2", "2026-03-29T00:00:00Z", "2026-03-29T01:00:00Z"),
                                              ("f3", "2026-06-29T00:00:00Z", "2026-06-30")])
def test_f2_and_f3_training_ends_follow_the_period_start_minus_72h(period, ok, late):
    field = f"{period}_training_data_end_utc"
    assert validate_meta(meta(**{field: ok}))[4][period] == ok
    with pytest.raises(SystemExit, match=field):
        validate_meta(meta(**{field: late}))
    # the F1 cutoff does not bind later periods, and a later period's cutoff does not bind F1
    assert validate_meta(meta(f1_training_data_end_utc="2025-12-01"))[4][period] == ok


@pytest.mark.parametrize("field", ["f1_training_data_end_utc", "f2_training_data_end_utc", "f3_training_data_end_utc"])
def test_meta_requires_every_training_end(field):
    value = meta()
    del value[field]
    with pytest.raises(SystemExit, match=field):
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


def test_rows_without_a_complete_f1_rank_below_complete_rows_and_legacy_rows_last():
    regimes = {name: {"mse": None, "reliable": False} for name in __import__("board").REGIMES}
    base = {"team_member": "alice", "submitted_utc": "20261031T000000Z", "skill_vs_naive": 0.1,
            "vs_naive": {"beats_reference": True}, "by_regime": regimes}
    full = {"mse_f1": 5000.0, "mse_f2": 3000.0, "mse_f3": 3500.0, "mse_ge550": 9000.0, "mse_f1_ge550": 9500.0,
            "mse_f2_ge550": 8500.0, "mse_f3_ge550": 8700.0}
    partial = {"mse_f1": 100.0, "coverage_f1": 0.35, "coverage_f2": 1.0, "coverage_f3": 1.0, "mse_ge550": 9000.0,
               "mse_f1_ge550": 100.0, "mse_f2_ge550": 8100.0, "mse_f3_ge550": 8300.0, "n_pairs_f2": 100, "n_pairs_f3": 300}
    rows = [
        {**base, "experiment": "legacy_good", "config_sha256": "a" * 64, "mse": 1000.0, "mse_f1": 700.0, "mse_f2": 900.0},
        {**base, "experiment": "new_worse", "config_sha256": "b" * 64, "mse": 4100.0, **full},
        {**base, "experiment": "new_better", "config_sha256": "c" * 64, "mse": 4000.0, **full},
        {**base, "experiment": "march_start_good", "config_sha256": "d" * 64, "mse": 500.0, "mse_f2": 3000.0,
         "mse_f3": 3400.0, **partial},
        {**base, "experiment": "march_start_bad", "config_sha256": "e" * 64, "mse": 600.0, "mse_f2": 4000.0,
         "mse_f3": 4400.0, **partial},
    ]
    assert [r["experiment"] for r in ranked(rows)] == [
        "new_better", "new_worse", "march_start_good", "march_start_bad", "legacy_good"]
    page = render(rows)
    assert (page.index("new_better") < page.index("new_worse") < page.index("march_start_good")
            < page.index("march_start_bad") < page.index("legacy_good"))
    assert "no F1" in page and "legacy" in page and "9,500.0" in page
    # an incomplete F1 (March only) and the totals that include it are not shown; the pooled F2+F3 value orders the rows
    assert "<td data-v=\"100.0\"" not in page and "<td data-v=\"500.0\"" not in page
    assert '<td data-v="3000.0"><b class="lb-1">3,000.0</b></td>' in page
    assert '<td data-v="700.0"' not in page and "900.0" not in page  # legacy values mean other windows: not shown


def test_board_keeps_the_period_and_event_columns():
    metric = {"team_member": "alice", "submitted_utc": "20260101T000000Z", "skill_vs_naive": 0.1, "mse_observed": 2.0}
    full = {"mse_f1": 5000.0, "mse_f2": 3000.0, "mse_f3": 3300.0, "mse_ge550": 9000.0, "mse_f1_ge550": 9500.0,
            "mse_f2_ge550": 8500.0, "mse_f3_ge550": 8800.0, "csi": 0.25, "timing_mae_h": 6.5, "timing_bias_h": -1.0}
    page = render([
        {**metric, "experiment": "old_windows", "config_sha256": "a" * 64, "mse": 1000.0},
        {**metric, "experiment": "scored", "config_sha256": "b" * 64, "mse": 4000.0, **full,
         "extras": {"end_to_end_seconds": 7200}},
    ])
    labels = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", page)
    assert labels == ["#", "experiment",
                      "MSE", "F1 MSE", "F2 MSE", "F3 MSE", "MSE ≥550", "F1 ≥550", "F2 ≥550", "F3 ≥550",
                      "CSI", "timing MAE (h)", "MSE obs.", "skill vs naive",
                      "member", "sec/fold", "config", "end-to-end (h)", "code"]
    assert page.index("scored") < page.index("old_windows") and "legacy" in page
    assert all(text in page for text in ("9,500.0", "8,800.0", "3,300.0", "0.250", "6.5", "2.00"))


def test_every_publish_marks_the_best_and_second_best_of_each_skill_column():
    """The page a publisher writes carries its own marks, so a leaderboard refresh no longer drops them."""
    base = {"team_member": "alice", "submitted_utc": "20260101T000000Z", "mse_ge550": 9000.0,
            "mse_f1_ge550": 9000.0, "mse_f2_ge550": 9000.0, "mse_f3_ge550": 9000.0}
    page = render([
        {**base, "experiment": name, "config_sha256": sha * 64, "mse": mse, "mse_f1": f1, "mse_f2": f2, "mse_f3": f2 + 100, "csi": csi,
         "timing_mae_h": mae, "mse_observed": obs, "skill_vs_naive": skill, "inference_seconds_per_fold": cost,
         "extras": {"end_to_end_seconds": e2e}}
        for name, sha, mse, f1, f2, csi, mae, obs, skill, cost, e2e in (
            ("best", "a", 4000.0, 5000.0, 3000.0, 0.3, 6.0, 3900.0, 0.5, 0.003, 7200),
            ("second", "b", 4100.0, 5100.0, 3100.0, 0.2, 5.0, 4000.0, 0.4, 0.001, 3600),
            ("third", "c", 4200.0, 5200.0, 3200.0, 0.1, 8.0, 4100.0, 0.3, 0.002, 1800))
    ])
    assert len(re.findall(r'<(b|u) class="lb-[12]">', page)) == 28  # 8 skill columns × 2 plus 4 ≥550 ties, so 12 + 8 + 8
    assert '<td data-v="4000.0"><b class="lb-1">4,000.0</b></td>' in page   # lower is better: MSE, F1/F2 MSE
    assert '<td data-v="4100.0"><u class="lb-2">4,100.0</u></td>' in page
    assert '<td data-v="4200.0">4,200.0</td>' in page
    assert page.count('<td data-v="9000.0"><b class="lb-1">9,000.0</b></td>') == 12  # 4 columns × 3-way ties share the best mark
    assert '<td data-v="0.3"><b class="lb-1">0.300</b></td>' in page         # higher is better: CSI, skill
    assert '<td data-v="0.2"><u class="lb-2">0.200</u></td>' in page
    assert '<td data-v="5.0" title=""><b class="lb-1">5.0</b></td>' in page  # lower is better: timing MAE
    assert '<td data-v="6.0" title=""><u class="lb-2">6.0</u></td>' in page
    assert '<td data-v="0.001">0.001000</td>' in page and '<td data-v="1.0" title="">1.00</td>' in page  # cost
    # A status write re-applies the same rule to the page a publisher wrote, so it never fights it. Only the
    # canonical checkout ships lbstatus.py next to the board.
    assert pytest.importorskip("lbstatus").mark_leaders(page) == page



    root = Path(__file__).resolve().parents[1]
    root = root if (root / "lb-f1f2").is_dir() else root.parent
    copies = [root / name / "board.py" for name in (".", "model-release-publish", "lb-f1f2", "lb-maint")]
    assert len({path.read_text() for path in copies}) == 1
    clients = [root / name / "submit.py" for name in (".", "model-release-publish", "lb-f1f2")]
    assert len({path.read_text() for path in clients}) == 1
    scorers = [root / name / "scorer/scoring.py" for name in (".", "model-release-publish", "lb-f1f2", "lb-maint")]
    assert len({path.read_text() for path in scorers}) == 1


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


def test_board_shows_end_to_end_hours_only_when_provided():
    regimes = {name: {"mse": None, "reliable": False} for name in __import__("board").REGIMES}
    base = {"team_member": "alice", "submitted_utc": "20261027T000000Z", "skill_vs_naive": 0.1, "mse": 4000.0,
            "vs_naive": {"beats_reference": True}, "by_regime": regimes, "mse_f1": 5000.0, "mse_f2": 3000.0,
            "mse_f3": 3200.0}
    page = render([
        {**base, "experiment": "timed", "config_sha256": "a" * 64,
         "extras": {"end_to_end_seconds": 5400.0, "end_to_end_note": "select + F1 + F2 fits"}},
        {**base, "experiment": "untimed", "config_sha256": "b" * 64},
    ])
    assert "end-to-end (h)" in page
    assert '<td data-v="1.5" title="select + F1 + F2 fits">1.50</td>' in page
    assert '<td data-v="" title=""><span class="dash">–</span></td>' in page


def test_extras_attach_to_matching_results_only(monkeypatch, tmp_path):
    import submit
    extra = tmp_path / "e.json"
    extra.write_text('{"end_to_end_seconds": 60.0}')
    key = f"extras/alice/model/{'a' * 64}.json"
    monkeypatch.setattr(submit, "_download", lambda repo, name, revision=None: str(extra) if name == key else None)
    api = type("Api", (), {"list_repo_files": lambda self, repo, repo_type, revision=None: [key, "results/x.json"]})()
    rows = submit.with_extras(api, "repo", [result_record(), {**result_record(), "experiment": "other"}])
    assert rows[0]["extras"] == {"end_to_end_seconds": 60.0}
    assert "extras" not in rows[1]
