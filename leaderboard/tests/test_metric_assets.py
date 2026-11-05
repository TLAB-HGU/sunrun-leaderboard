import base64
import copy
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metric_assets as assets


def record(name="sample"):
    return {"team_member": "alice", "experiment": name, "config_sha256": "a" * 64,
            "competition": {"version": "lead-timse-6h-v1", **{
                scope: {"s": 20, "timse": 400, "rmse": 20, "by_lead": [
                    {"lead_hours": h, "s": 10 + h * factor, "timse": 100 + h * factor * 10,
                     "rmse": 12 + h * factor} for h in range(1, 73)]}
                for scope, factor in zip(("total", "f1", "f2", "f3"), (1, 1.1, 0.9, 1.2))}}}


def test_plot_is_reproducible_and_named_by_identity(tmp_path):
    original = record("../an experiment")
    first = assets.generate_plot(original, tmp_path / "one")
    second = assets.generate_plot(original, tmp_path / "two")
    assert first.name == second.name
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.parent == tmp_path / "one"
    assert first.name == f"alice--an_experiment--{'a' * 64}.png"


@pytest.mark.parametrize("change", ["missing", "nonfinite", "negative"])
def test_invalid_curve_rejected(tmp_path, change):
    item = record()
    rows = item["competition"]["f2"]["by_lead"]
    if change == "missing":
        rows.pop()
    else:
        rows[0]["s"] = float("nan") if change == "nonfinite" else -1
    with pytest.raises(ValueError):
        assets.generate_plot(item, tmp_path)


def test_dry_run_never_authenticates_or_mutates_input(tmp_path, monkeypatch):
    original = record()
    original["metric_plot_url"] = "https://old/image.png"
    before = copy.deepcopy(original)
    monkeypatch.setattr(assets, "GitHubAPI", lambda *_: pytest.fail("network during dry run"))
    result = assets.publish_plots([original], tmp_path, dry_run=True)
    assert original == before
    assert "metric_plot_url" not in result[0]
    assert Path(result[0]["metric_plot_path"]).is_file()


class FakeAPI:
    def __init__(self, conflict=False):
        self.calls = []
        self.files = {}
        self.blobs = {}
        self.head = "1" * 40
        self.conflict = conflict
        self.commits = 0
        self.pending = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if path == "git/ref/heads/main":
            return {"object": {"sha": self.head}}
        if method == "GET" and path.startswith("git/commits/"):
            return {"tree": {"sha": "root"}}
        if path.startswith("git/trees/"):
            name = path.rsplit("/", 1)[1]
            next_tree = {"root": "leaderboard", "leaderboard": "assets", "assets": "metrics"}.get(name)
            entries = ([{"path": next_tree, "type": "tree", "sha": next_tree}] if next_tree
                       else [{"path": n, "type": "blob", "sha": sha, "size": len(self.blobs[sha])}
                             for n, sha in self.files.items()])
            return {"tree": entries}
        if path == "git/blobs":
            data = base64.b64decode(payload["content"])
            sha = f"{len(self.blobs) + 1:040d}"
            self.blobs[sha] = data
            return {"sha": sha}
        if path == "git/trees":
            self.pending = payload["tree"]
            return {"sha": "new_tree"}
        if path == "git/commits":
            self.commits += 1
            assert payload["parents"] == [self.head]
            return {"sha": str(self.commits + 1) * 40}
        if path == "git/refs/heads/main":
            assert payload["force"] is False
            if self.conflict:
                self.conflict = False
                self.head = "9" * 40
                raise urllib.error.HTTPError("https://api.github.com", 422, "race", {}, None)
            self.head = payload["sha"]
            self.files.update({Path(e["path"]).name: e["sha"] for e in self.pending})
            return {}
        raise AssertionError((method, path))


def test_batch_publish_pins_urls_and_skips_existing_blobs(tmp_path, monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(assets, "GitHubAPI", lambda repo: api)
    originals = [record("one"), record("two"), {"experiment": "unscored"}]
    result = assets.publish_plots(originals, tmp_path)
    assert api.commits == 1 and len(api.blobs) == 2
    assert all(f"/{api.head}/leaderboard/assets/metrics/" in r["metric_plot_url"] for r in result[:2])
    assert result[2] == originals[2]
    assert "metric_plot_url" not in originals[0]
    call_count = len(api.calls)
    repeated = assets.publish_plots(result, tmp_path)
    assert repeated == result
    assert not any(method != "GET" for method, _, _ in api.calls[call_count:])


def test_branch_race_retries_against_new_head_without_reupload(tmp_path, monkeypatch):
    api = FakeAPI(conflict=True)
    monkeypatch.setattr(assets, "GitHubAPI", lambda repo: api)
    result = assets.publish_plots([record()], tmp_path)
    assert api.commits == 2
    assert len([c for c in api.calls if c[1] == "git/blobs"]) == 1
    assert f"/{api.head}/" in result[0]["metric_plot_url"]
    parents = [p["parents"] for method, path, p in api.calls if path == "git/commits"]
    assert parents == [["1" * 40], ["9" * 40]]
