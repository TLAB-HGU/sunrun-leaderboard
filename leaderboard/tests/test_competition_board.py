"""Competition rendering and status refresh integration."""
import re

import pytest

import board
import lbstatus


def record(name="model", score=10.0):
    return {
        "team_member": "alice", "experiment": name, "submitted_utc": "20261101T000000Z",
        "config_sha256": "a" * 64,
        "competition": {"version": board.COMPETITION_VERSION, **{
            scope: {"s": score + i, "timse": 100.0 + score + i, "rmse": 8.0 + score + i,
                    "replacement_fraction": 0.25, "bias": -1.5}
            for i, scope in enumerate(("total", "f1", "f2", "f3"))}},
        "metric_plot_url": "https://raw.githubusercontent.com/TLAB-HGU/sunrun-leaderboard/" + "1" * 40
                           + "/leaderboard/assets/metrics/model.png",
        "architecture_url": "https://github.com/TLAB-HGU/sunrun-leaderboard/blob/" + "1" * 40
                            + "/research/model/experiment.py",
    }


def test_metrics_order_units_images_and_identity_survive_status_refresh(monkeypatch):
    monkeypatch.setattr(board, "visibility_policy", lambda: {})
    page = board.render([record()], architecture={})
    headers = [label for label, _ in board.ranking(page)[2]]
    assert headers == ["#", "experiment", *[f"{p} {m}" for p in ("Total", "F1", "F2", "F3")
                                           for m in ("S", "TIMSE", "RMSE")],
                       "lead curves", "member", "sec/fold", "end-to-end (h)", "code"]
    assert "(km/s)²" in page and "every 6 hours" in page
    assert 'loading="lazy"' in page and 'target="_blank"' in page
    assert 'Persistence replacements: 25.00% of valid pairs; Mean forecast bias: -1.50 km/s' in page
    assert f'data-config="{"a" * 64}"' in page
    assert lbstatus.trim_ranking_columns(page) == page
    pending = lbstatus.upsert([], "alice", "next", "대기", at="2026-11-01T00:00:00Z")
    refreshed = lbstatus.mark_leaders(lbstatus.inject(page, pending))
    assert [label for label, _ in board.ranking(refreshed)[2]] == headers
    assert refreshed.count('class="lead-curves"') == 1
    status = re.search(r'<tr style="color:var\(--mut\)">(.*?)</tr>', refreshed).group(1)
    assert len(board.CELL.findall(status)) == len(headers)
    assert lbstatus.mark_leaders(lbstatus.inject(refreshed, pending)) == refreshed


def test_ranking_uses_only_complete_current_scores_and_keeps_unscored_below():
    old = {k: v for k, v in record("old").items() if k != "competition"}
    old["mse"] = 0
    best, second = record("best", 2), record("second", 5)
    best["mse"], second["mse"] = 10000, 1
    assert [r["experiment"] for r in board.ranked([old, second, best])] == ["best", "second", "old"]
    assert board.competition_metrics(old) is None


@pytest.mark.parametrize("invalid", [None, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_or_partial_scores_cannot_rank(invalid):
    row = record()
    row["competition"]["f3"]["timse"] = invalid
    assert board.competition_metrics(row) is None
    page = board.render([row], architecture={})
    assert ">unscored</td>" in page and 'class="lb-1"' not in page


def test_wrong_version_and_unpinned_image_are_not_published():
    row = record()
    row["competition"]["version"] = "old"
    assert board.competition_metrics(row) is None
    row = record()
    row["metric_plot_url"] = row["metric_plot_url"].replace("1" * 40, "main")
    assert '<img ' not in board.render([row], architecture={})


def test_visibility_and_architecture_rewrite_preserve_current_metrics(monkeypatch):
    monkeypatch.setattr(board, "visibility_policy", lambda: {
        "hidden": [{"member": "alice", "experiment": "hidden"}],
        "bold": [{"member": "alice", "experiment": "kept"}],
    })
    page = board.render([record("hidden", 1), record("kept", 2)], architecture={})
    assert '>hidden</td>' not in page
    assert '<td>1</td>' in page and '<strong class="lb-experiment">kept</strong>' in page
    target = "https://github.com/TLAB-HGU/sunrun-leaderboard/blob/" + "2" * 40 + "/research/kept/model.py"
    mapping = {"models": [{"member": "alice", "experiment": "kept", "config_sha256": "a" * 64, "url": target}]}
    updated = board.apply_architecture_links(page, mapping)
    assert target in updated and board.script_cells(updated) == []
    assert '<img ' in updated and lbstatus.mark_leaders(updated) == updated
