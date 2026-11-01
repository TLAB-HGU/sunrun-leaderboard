import json
import re
from pathlib import Path

import pytest

from board import _latest, code_link, load_results, ranked, render
from submit import (architecture_location, canonical_config, publish, script_location, store_submission,
                    validate_meta)


CONFIG = {
    "data": {"source": "ace", "features": ["speed"]},
    "hyperparameters": {"tau": 48, "window": 648},
}
COMMIT = "1" * 40
ARCHITECTURE = f"https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{COMMIT}/research/mean_reversion/experiment.py"


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
        "architecture_url": ARCHITECTURE,
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
    assert ">unscored</td>" in page
    assert "0.001234" not in page
    assert "https://example.com/code.py" in page


def _competition(score):
    return {"version": __import__("board").COMPETITION_VERSION, **{
        scope: {"s": score, "timse": score * score, "rmse": score}
        for scope in ("total", "f1", "f2", "f3")}}


def test_current_scores_rank_by_total_s_and_old_results_remain_unscored():
    base = {"team_member": "alice", "submitted_utc": "20261031T000000Z"}
    rows = [{**base, "experiment": "old", "mse": 1.0},
            {**base, "experiment": "second", "mse": 10.0, "competition": _competition(5.0)},
            {**base, "experiment": "best", "mse": 100.0, "competition": _competition(2.0)}]
    assert [r["experiment"] for r in ranked(rows)] == ["best", "second", "old"]
    page = render(rows)
    assert page.index('>best</td>') < page.index('>second</td>') < page.index('>old</td>')
    assert '>unscored</td>' in page
    assert '<th class="">MSE</th>' not in page


def test_board_keeps_total_and_period_competition_columns():
    metric = {"team_member": "alice", "submitted_utc": "20260101T000000Z"}
    page = render([{**metric, "experiment": "scored", "competition": _competition(10.0),
                    "extras": {"end_to_end_seconds": 7200}}])
    labels = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", page)
    assert labels == ["#", "experiment", *[f"{p} {m}" for p in ("Total", "F1", "F2", "F3")
                                           for m in ("S", "TIMSE", "RMSE")],
                      "lead curves", "member", "end-to-end (h)", "code"]
    assert "10.00" in page and "100.00" in page and "2.00" in page


def test_removed_config_column_keeps_the_row_identity_on_the_code_cell():
    """No config column (user instruction 2026-11-01): the SHA rides on the code cell so its architecture link stays
    rewritable, and no cell shows it."""
    sha = "c" * 64
    page = render([{"team_member": "alice", "experiment": "kept", "submitted_utc": "20260101T000000Z",
                    "config_sha256": sha, "mse": 4000.0, "mse_f1": 5000.0, "mse_f2": 3000.0, "mse_f3": 3300.0}])
    assert "config" not in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", page)
    assert f'<td data-config="{sha}" class="l">' in page  # identity for architecture_code.py, not a visible cell
    assert "<code>" not in page


def test_every_publish_marks_the_best_and_second_best_of_each_skill_column():
    base = {"team_member": "alice", "submitted_utc": "20260101T000000Z"}
    page = render([{**base, "experiment": name, "competition": _competition(score),
                    "inference_seconds_per_fold": 0.001,
                    "extras": {"end_to_end_seconds": 3600}}
                   for name, score in (("best", 2.0), ("second", 3.0), ("third", 4.0))])
    assert len(re.findall(r'<(b|u)(?: data-metric-version="[^"]+")? class="lb-[12]">', page)) == 24
    assert page.count('<b data-metric-version="lead-timse-6h-v1" class="lb-1">2.00</b>') == 8
    assert page.count('<u data-metric-version="lead-timse-6h-v1" class="lb-2">3.00</u>') == 8
    assert '<td data-v="0.001">0.001000</td>' not in page
    assert '<td data-v="1.0" title="">1.00</td>' in page
    assert pytest.importorskip("lbstatus").mark_leaders(page) == page


def test_published_client_copies_are_synchronized():
    root = Path(__file__).resolve().parents[1]
    root = root if (root / "lb-f1f2").is_dir() else root.parent
    copies = [root / name / "board.py" for name in (".", "model-release-publish", "lb-f1f2", "lb-maint")]
    assert len({path.read_text() for path in copies}) == 1
    clients = [root / name / "submit.py" for name in (".", "model-release-publish", "lb-f1f2")]
    assert len({path.read_text() for path in clients}) == 1
    # submit.py writes the code-column mapping on every submission, so every client carries that module too
    mapping_writers = [root / name / "architecture_code.py" for name in (".", "model-release-publish", "lb-f1f2")]
    assert len({path.read_text() for path in mapping_writers}) == 1
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


def test_architecture_link_is_required_and_must_be_a_pinned_study_source():
    value = meta()
    del value["architecture_url"]
    with pytest.raises(SystemExit, match="architecture_url"):
        validate_meta(value)
    replay = script_location("alice", "mean_reversion", "a" * 64)[2].replace("/main/", f"/{COMMIT}/")
    with pytest.raises(SystemExit, match="replay script"):
        architecture_location(meta(architecture_url=replay))
    for url in (f"https://github.com/TLAB-HGU/sunrun-leaderboard/blob/main/research/mean_reversion/experiment.py",
                f"https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{COMMIT}/research/mean_reversion/",
                f"https://example.com/{COMMIT}/research/mean_reversion/experiment.py",
                ""):
        with pytest.raises(SystemExit, match="architecture_url"):
            architecture_location(meta(architecture_url=url))
    assert architecture_location(meta()) == ARCHITECTURE
    assert architecture_location(meta(architecture_url=ARCHITECTURE + "#L1-L169")) == ARCHITECTURE + "#L1-L169"


def test_code_cell_opens_the_architecture_and_never_the_replay_script():
    script = script_location("alice", "mean_reversion", "a" * 64)[2]
    row = {"team_member": "alice", "experiment": "mean_reversion", "config_sha256": "a" * 64,
           "code_url": script, "architecture_url": ARCHITECTURE}
    assert code_link(row) == (ARCHITECTURE, "architecture")
    assert code_link({k: v for k, v in row.items() if k != "architecture_url"}) == (script, "script")
    page = render([{**row, "submitted_utc": "20260101T000000Z", "mse": 2.0}])
    assert f'href="{ARCHITECTURE}"' in page and ">architecture</a>" in page and script not in page


def test_the_mapping_wins_over_the_recorded_link_and_matches_full_config_only():
    script = script_location("alice", "mean_reversion", "a" * 64)[2]
    mapped = ARCHITECTURE.replace("mean_reversion/experiment.py", "mean_reversion/lane.py")
    mapping = {"models": [{"member": "alice", "experiment": "mean_reversion", "config_sha256": "a" * 64,
                           "url": mapped}]}
    row = {"team_member": "alice", "experiment": "mean_reversion", "config_sha256": "a" * 64,
           "code_url": script, "architecture_url": ARCHITECTURE}
    assert code_link(row, mapping) == (mapped, "architecture")
    other = {**row, "config_sha256": "b" * 64}
    assert code_link(other, mapping) == (ARCHITECTURE, "architecture")
    page = render([{**row, "submitted_utc": "20260101T000000Z", "mse": 2.0}], architecture=mapping)
    assert f'href="{mapped}"' in page and ARCHITECTURE not in page


def test_publish_records_the_architecture_link_in_the_same_commit_as_the_page(monkeypatch):
    import architecture_code
    kept = {"member": "other", "experiment": "kept", "config_sha256": "b" * 64, "url": "https://example.com/kept.py"}
    monkeypatch.setattr("submit.space_file",
                        lambda space, name, revision: {"models": [kept]} if name == "architecture-code.json" else {})
    api = FakeApi()
    api.create_repo = lambda *args, **kwargs: None
    script = script_location("alice", "mean_reversion", "a" * 64)[2]
    row = {"team_member": "alice", "experiment": "mean_reversion", "config_sha256": "a" * 64, "submitted_utc":
           "20260101T000000Z", "mse": 4000.0, "mse_f1": 5000.0, "mse_f2": 3000.0, "mse_f3": 3300.0,
           "mse_ge550": 9000.0, "mse_f1_ge550": 9500.0, "mse_f2_ge550": 8500.0, "mse_f3_ge550": 8800.0,
           "code_url": script, "architecture_url": ARCHITECTURE}
    entry = architecture_code.entry("alice", "mean_reversion", "a" * 64, ARCHITECTURE)
    publish(api, [row], "space", architecture={"models": [entry]})
    files = {op.path_in_repo: op.path_or_fileobj for op in api.commits[0]["operations"]}
    mapping = json.loads(files["architecture-code.json"])
    assert [e["experiment"] for e in mapping["models"]] == ["mean_reversion", "kept"]  # the Space entry is kept
    page = files["index.html"].decode()
    assert f'href="{ARCHITECTURE}"' in page and ">architecture</a>" in page and script not in page


def test_publish_keeps_the_space_visibility_policy_in_step_with_the_page(monkeypatch):
    import board
    monkeypatch.setattr("submit.space_file", lambda space, name, revision: {})
    api = FakeApi()
    api.create_repo = lambda *args, **kwargs: None
    publish(api, [], "space")
    files = {op.path_in_repo: op.path_or_fileobj for op in api.commits[0]["operations"]}
    assert json.loads(files["visibility.json"]) == board.visibility_policy()
    # a Space that already carries the same policy gets no second copy
    monkeypatch.setattr("submit.space_file",
                        lambda space, name, revision: board.visibility_policy() if name == "visibility.json" else {})
    api = FakeApi()
    api.create_repo = lambda *args, **kwargs: None
    publish(api, [], "space")
    assert "visibility.json" not in {op.path_in_repo for op in api.commits[0]["operations"]}
