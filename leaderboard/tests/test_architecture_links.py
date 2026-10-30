import importlib.util
import json
from pathlib import Path
import re

import pytest

import lbstatus

SHA_A = "a" * 64
SHA_B = "b" * 64
URL_A = "https://github.com/TLAB-HGU/sunrun-leaderboard/blob/" + "1" * 40 + "/source/model_a.py"
URL_B = "https://github.com/TLAB-HGU/sunrun-leaderboard/blob/" + "2" * 40 + "/source/model_b.py"
MAPPING = {"models": [
    {"member": "s", "experiment": "model", "config_sha256": SHA_A, "url": URL_A},
    {"member": "s", "experiment": "model", "config_sha256": SHA_B, "url": URL_B},
]}


def row(config=SHA_A, member="s", title=True, experiment="model"):
    config_title = f' title="{config}"' if title else ""
    code = f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/main/scripts/{member}/{experiment}/{config}.py'
    return (f'<tr><td>1</td><td class="l"><strong class="lb-experiment">{experiment}</strong></td>'
            f'<td>{member}</td><td{config_title}><code>{config[:12]}</code></td>'
            f'<td class="l"><a href="{code}">script</a></td>'
            '<td data-v="100.0"><b class="lb-1">100.0</b></td></tr>')


def page(*rows):
    return ('<main><table><thead><tr>'
            + ''.join(f'<th>{name}</th>' for name in ("#", "experiment", "member", "config", "code", "MSE"))
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></main>')


def code_cells(html):
    return [list(lbstatus.CELL.finditer(r))[4].group(2)
            for r in re.findall(r'<tr\b[^>]*>.*?</tr>', html, re.S) if '<td' in r]


def test_links_match_member_experiment_full_config_and_only_change_code_cells():
    original = page(row(), row(SHA_B), row(member="other"), row("c" * 64), row(experiment="other"))
    actual = lbstatus.apply_architecture_links(original, MAPPING)
    before_rows = re.findall(r'<tr\b[^>]*>.*?</tr>', original, re.S)
    after_rows = re.findall(r'<tr\b[^>]*>.*?</tr>', actual, re.S)
    for before, after in zip(before_rows, after_rows):
        old, new = list(lbstatus.CELL.finditer(before)), list(lbstatus.CELL.finditer(after))
        assert [c.group(0) for i, c in enumerate(old) if i != 4] == [c.group(0) for i, c in enumerate(new) if i != 4]
    codes = code_cells(actual)
    assert URL_A in codes[0] and '>architecture</a>' in codes[0]
    assert URL_B in codes[1] and '>architecture</a>' in codes[1]
    assert codes[2:] == code_cells(original)[2:]
    assert lbstatus.apply_architecture_links(actual, MAPPING) == actual


def test_script_sha_fallback_is_exact_and_does_not_guess_from_short_hash():
    original = page(row(title=False), row(config=SHA_A[:12], title=False))
    actual = lbstatus.apply_architecture_links(original, MAPPING)
    assert URL_A in code_cells(actual)[0]
    assert code_cells(actual)[1] == code_cells(original)[1]
    conflict = page(row().replace(f'title="{SHA_A}"', f'title="{SHA_B}"'))
    assert URL_B in code_cells(lbstatus.apply_architecture_links(conflict, MAPPING))[0]


def test_only_ranking_table_changes_and_unmapped_pages_remain_intact():
    first, second = page(row()), page(row(SHA_B))
    actual = lbstatus.apply_architecture_links(first + second, MAPPING)
    assert actual.endswith(second)
    assert lbstatus.apply_architecture_links(first, {}) == first
    assert lbstatus.apply_architecture_links('<p>No table</p>', MAPPING) == '<p>No table</p>'
    no_code = first.replace('<th>code</th>', '<th>other</th>')
    assert lbstatus.apply_architecture_links(no_code, MAPPING) == no_code


def test_status_refresh_reads_architecture_mapping_at_the_same_revision(monkeypatch):
    import huggingface_hub
    original = page(row(), row(experiment="hidden"))
    visibility = {"hidden": [{"member": "s", "experiment": "hidden"}],
                  "bold": [{"member": "s", "experiment": "model"}]}
    calls, commits = [], []
    monkeypatch.setattr(lbstatus, "load", lambda space: (lbstatus.dump("active", []), lbstatus.dump("finished", []), original, "revision"))
    monkeypatch.setattr(lbstatus, "leaderboard_submissions", lambda *args: {})

    def file(repo, name, revision):
        calls.append((repo, name, revision))
        return json.dumps(visibility if name == "visibility.json" else MAPPING)

    class Api:
        def create_commit(self, **kwargs):
            commits.append(kwargs)

    monkeypatch.setattr(lbstatus, "_space_file", file)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert lbstatus.commit(lambda entries: entries, space="test/space")
    assert calls == [("test/space", "visibility.json", "revision"),
                     ("test/space", "architecture-code.json", "revision")]
    assert commits[0]["parent_commit"] == "revision"
    operations = commits[0]["operations"]
    assert len(operations) == 1 and operations[0].path_in_repo == "index.html"
    updated = operations[0].path_or_fileobj.decode()
    assert URL_A in updated and '>architecture</a>' in updated
    assert '<strong class="lb-experiment">model</strong>' in updated
    assert '>hidden</strong>' not in updated
    assert lbstatus.ROWS_BEGIN in updated


@pytest.mark.parametrize("relative", ["board.py", "model-release-publish/board.py"])
def test_standard_publishers_load_mapping_and_preserve_visibility(tmp_path, monkeypatch, relative):
    source = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location("architecture_board", source)
    board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(board)
    local_source = tmp_path / relative
    local_source.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(board, "__file__", str(local_source))
    (tmp_path / "architecture-code.json").write_text(json.dumps(MAPPING))
    (tmp_path / "visibility.json").write_text(json.dumps({
        "hidden": [{"member": "s", "experiment": "hidden"}],
        "bold": [{"member": "s", "experiment": "model"}],
    }))
    records = [{"team_member": "s", "experiment": experiment, "config_sha256": config,
                "submitted_utc": "2026-10-29T00:00:00Z", "mse": 100 + i, "code_url": f"https://example.org/{i}.py"}
               for i, (experiment, config) in enumerate([("model", SHA_A), ("model", SHA_B),
                                                          ("model", "c" * 64), ("hidden", SHA_A)])]
    result = board.render(records)
    assert result.count('>architecture</a>') == 2
    assert URL_A in result and URL_B in result
    assert 'https://example.org/2.py' in result and '>script</a>' in result
    assert '<strong class="lb-experiment">model</strong>' in result
    assert '>hidden</td>' not in result
