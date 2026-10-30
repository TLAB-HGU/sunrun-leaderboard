import json
import re

import pytest

import lbstatus
from lbstatus import (ROWS_BEGIN, ROWS_END, dump, entries_of, inject, main, mark_leaders, normalize_status,
                      on_leaderboard, remove, split, upsert)

RANKING = ('<main><h1>x</h1>{legacy}<div class="wrap"><table><thead><tr><th class="">#</th><th class="l">experiment</th>'
           '<th class="l">member</th><th class="">sec/fold</th><th class="">MSE</th><th class="">CSI</th>'
           '<th class="">skill vs naive</th><th class="">low_high_low</th></tr></thead><tbody>'
           '<tr><td>1</td><td class="l">a</td><td class="l">s</td><td data-v="9.0">9.0</td><td data-v="100.0">100.0</td>'
           '<td data-v="0.05">0.050</td><td data-v="0.6">+60.0%</td><td>1,500.0</td></tr>'
           '<tr><td>2</td><td class="l">b</td><td class="l">s</td><td data-v="1.0">1.0</td><td data-v="200.0">200.0</td>'
           '<td data-v="0.08">0.080</td><td data-v="0.6">+60.0%</td><td class="dash" title="t">–</td></tr>'
           '<tr><td>3</td><td class="l">c</td><td class="l">s</td><td data-v="">–</td><td data-v="300.0">300.0</td>'
           '<td data-v="0.07">0.070</td><td data-v="0.1">+10.0%</td><td>1,200.0</td></tr>'
           '</tbody></table></div></main>')
PAGE = RANKING.format(legacy="")
LEGACY = ('<!-- sunrun-status:begin --><section class="xs"><div class="wrap"><table><thead><tr><td>status</td></tr>'
          '</thead><tbody></tbody></table></div></section><!-- sunrun-status:end -->')


def test_status_aliases_and_rejection():
    assert normalize_status("running") == "진행"
    assert normalize_status(" 대기 ") == "대기"
    with pytest.raises(ValueError):
        normalize_status("started")


def test_upsert_tracks_lifecycle_timestamps():
    es = upsert([], "seongeun", "E2001", "대기", at="2026-10-28T00:00:00Z", description="d", agent="codex")
    es = upsert(es, "seongeun", "E2001", "진행", at="2026-10-28T01:00:00Z", agent="")
    es = upsert(es, "seongeun", "E2001", "완료", at="2026-10-28T03:00:00Z", note="MSE=1.0")
    (e,) = es
    assert (e["created_utc"], e["started_utc"], e["finished_utc"]) == (
        "2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z", "2026-10-28T03:00:00Z")
    assert e["agent"] == "codex" and e["description"] == "d" and e["note"] == "MSE=1.0"
    assert [h[0] for h in e["history"]] == ["대기", "진행", "완료"]


def test_rerun_of_finished_experiment_clears_finish_fields():
    es = upsert([], "seongeun", "E2001", "실패", at="2026-10-28T00:00:00Z", note="rc=1")
    (e,) = upsert(es, "seongeun", "E2001", "진행", at="2026-10-28T02:00:00Z")
    assert "finished_utc" not in e and "note" not in e and e["started_utc"] == "2026-10-28T02:00:00Z"
    assert e["created_utc"] == "2026-10-28T02:00:00Z"


def test_upsert_rejects_unsafe_names_and_does_not_mutate_input():
    with pytest.raises(ValueError):
        upsert([], "seongeun", "bad name", "대기")
    es = upsert([], "seongeun", "E1", "대기", at="2026-10-28T00:00:00Z")
    upsert(es, "seongeun", "E1", "진행", at="2026-10-28T01:00:00Z")
    assert es[0]["status"] == "대기"
    assert remove(es, "seongeun", "E1") == []


def test_leaderboard_result_clears_entry_only_if_submitted_after_registration():
    es = upsert([], "seongeun", "E2001", "진행", at="2026-10-28T10:00:00Z")
    assert on_leaderboard(es, {("seongeun", "E2001"): "20261028T095959000Z"}) == es  # an older submission
    assert on_leaderboard(es, {("other", "E2001"): "20261028T120000000Z"}) == es
    (e,) = on_leaderboard(es, {("seongeun", "E2001"): "20261028T120000123Z"})
    assert e["status"] == "완료" and e["note"] == "on the leaderboard"
    assert "E2001" not in inject(PAGE, [e])


def test_split_moves_finished_entries_to_history():
    es = upsert([], "seongeun", "A", "진행", at="2026-10-28T00:00:00Z")
    es = upsert(es, "seongeun", "B", "실패", at="2026-10-28T01:00:00Z")
    active, history = split(es, [{"experiment": "old", "status": "완료"}])
    assert [e["experiment"] for e in active] == ["A"]
    assert [e["experiment"] for e in history] == ["old", "B"]


def test_status_file_format_has_no_legacy_entries_key():
    # `run` processes started with the first version read status.json["entries"]; they must fail, not render.
    assert "entries" not in json.loads(dump("active", []))
    assert entries_of(dump("active", [{"x": 1}]), "active") == [{"x": 1}]
    assert entries_of('{"entries": [{"x": 1}]}', "active") == [{"x": 1}]
    assert entries_of(None, "finished") == []


def _rows(page):
    return re.findall(r'<tr style="color:var\(--mut\)">(.*?)</tr>', page)


def test_active_entries_become_top_rows_of_the_ranking_table():
    es = upsert([], "seongeun", "E2", "대기", at="2026-10-28T00:00:00Z", description='<b>"x"</b>')
    es = upsert(es, "seongeun", "E1", "진행", at="2026-10-28T01:00:00Z")
    page = inject(RANKING.format(legacy=LEGACY), es)
    assert "sunrun-status:begin" not in page and "<section" not in page  # the separate table is gone
    assert page.index("<tbody>" + ROWS_BEGIN) < page.index("<td>1</td>")
    first, second = _rows(page)
    assert '<td class="l" title="진행 since 2026-10-28 01:00 UTC">E1 (진행)</td><td class="l">seongeun</td>' in first
    assert "E2 (대기)" in second and "&lt;b&gt;&quot;x&quot;&lt;/b&gt;" in second and "<b>" not in second
    assert first.startswith("<td></td>") and first.count('<span class="dash">–</span>') == 5


def test_rows_are_replaced_in_place_and_removed_when_finished():
    es = upsert([], "seongeun", "E1", "진행", at="2026-10-28T00:00:00Z")
    page = inject(PAGE, es)
    assert inject(page, es) == page  # sync is a no-op on an up-to-date page
    done = inject(page, upsert(es, "seongeun", "E1", "완료", at="2026-10-28T01:00:00Z"))
    assert done.count(ROWS_BEGIN) == done.count(ROWS_END) == 1 and "E1 (" not in done and _rows(done) == []


def _marked(page):
    return re.findall(r'<(b|u) class="lb-[12]">(.*?)</\1>', page)


def test_mark_leaders_bolds_best_and_underlines_second_per_skill_column():
    page = mark_leaders(PAGE)
    assert len(_marked(page)) == 9
    assert '<td data-v="100.0"><b class="lb-1">100.0</b></td>' in page        # MSE: lower is better
    assert '<td data-v="200.0"><u class="lb-2">200.0</u></td>' in page
    assert '<td data-v="0.08"><b class="lb-1">0.080</b></td>' in page        # CSI: higher is better
    assert '<td data-v="0.07"><u class="lb-2">0.070</u></td>' in page
    assert page.count('<b class="lb-1">+60.0%</b>') == 2                     # ties share the mark
    assert '<u class="lb-2">+10.0%</u>' in page
    assert '<td><b class="lb-1">1,200.0</b></td>' in page and '<td><u class="lb-2">1,500.0</u></td>' in page
    assert '<td data-v="9.0">9.0</td>' in page and '<td data-v="1.0">1.0</td>' in page  # sec/fold is not marked
    assert mark_leaders(page) == page


def test_marks_ignore_status_rows_and_survive_row_refresh():
    es = upsert([], "seongeun", "E1", "진행", at="2026-10-28T00:00:00Z")
    page = mark_leaders(inject(PAGE, es))
    assert _marked(page) == _marked(mark_leaders(PAGE))
    again = mark_leaders(inject(page, upsert(es, "seongeun", "E1", "완료")))
    assert _marked(again) == _marked(page)


def test_run_requires_a_command():
    with pytest.raises(SystemExit):
        main(["run", "E1", "--desc", "x"])


def test_run_keeps_its_options_and_posts_running_then_result(monkeypatch):
    posted = []
    monkeypatch.setattr(lbstatus, "commit", lambda mutate, *a: posted.append(mutate([])[0]))
    assert main(["run", "E1", "--desc", "probe", "--sync-every", "0", "--", "sh", "-c", "exit 3"]) == 3
    assert [(e["status"], e.get("description")) for e in posted] == [("진행", "probe"), ("실패", None)]
    assert posted[0]["command"] == "sh -c 'exit 3'" and posted[1]["note"] == "rc=3"
    posted.clear()
    assert main(["run", "E1", "--sync-every", "0", "--", "true"]) == 0
    assert [e["status"] for e in posted] == ["진행", "진행"]  # stays listed until the result is on the leaderboard
    assert posted[1]["note"].startswith("run finished (rc=0)")


def test_trim_ranking_columns_preserves_metrics_and_status_rows():
    entries = upsert([], "seongeun", "E2001", "진행", at="2026-10-28T00:00:00Z")
    original = inject(PAGE, entries)
    trimmed = lbstatus.trim_ranking_columns(original)
    assert lbstatus._ranking(trimmed)[2] == lbstatus._ranking(original)[2][:-1]
    before_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", original, re.S)
    after_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", trimmed, re.S)
    assert len(before_rows) == len(after_rows)
    for before, after in zip(before_rows, after_rows):
        cells = lambda row: re.findall(r"<t[hd]\b[^>]*>.*?</t[hd]>", row, re.S)
        assert cells(after) == cells(before)[:7]
    assert ROWS_BEGIN in trimmed and ROWS_END in trimmed
    assert lbstatus.trim_ranking_columns(trimmed) == trimmed
    assert lbstatus.trim_ranking_columns("<p>No ranking</p>") == "<p>No ranking</p>"


def test_hide_rows_matches_exact_identity_and_preserves_active_status():
    entries = upsert([], "seongeun", "E2001", "대기", at="2026-10-28T00:00:00Z")
    original = inject(PAGE, entries)
    policy = {"hidden": [{"member": "s", "experiment": "a"}, {"member": "other", "experiment": "b"}]}
    hidden = lbstatus.hide_ranking_rows(original, policy)
    rows = re.findall(r"<tr\b[^>]*>.*?</tr>", hidden, re.S)
    assert len(rows) == 4  # header, active experiment, b, c
    assert '>a</td>' not in hidden
    assert '>b</td>' in hidden and '>c</td>' in hidden
    assert '<td>1</td><td class="l">b</td>' in hidden
    assert '<td>2</td><td class="l">c</td>' in hidden
    assert 'E2001 (대기)' in hidden and ROWS_BEGIN in hidden
    assert lbstatus.hide_ranking_rows(hidden, policy) == hidden
    assert lbstatus.hide_ranking_rows(original, {}) == original
    assert lbstatus.hide_ranking_rows('<p>No ranking</p>', policy) == '<p>No ranking</p>'


def test_bold_experiment_names_only_changes_selected_name_cell():
    policy = {"bold": [{"member": "s", "experiment": "b"}, {"member": "other", "experiment": "a"}]}
    expected = PAGE.replace('<td class="l">b</td>', '<td class="l"><strong class="lb-experiment">b</strong></td>')
    actual = lbstatus.bold_experiment_names(PAGE, policy)
    assert actual == expected
    assert lbstatus.bold_experiment_names(actual, policy) == actual
    assert lbstatus.bold_experiment_names(actual, {}) == PAGE
    assert lbstatus.bold_experiment_names('<p>No table</p>', policy) == '<p>No table</p>'
