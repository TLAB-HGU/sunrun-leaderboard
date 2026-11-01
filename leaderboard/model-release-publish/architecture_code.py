"""Keep the leaderboard's code column on architecture sources, never on a submitted replay script.

    python architecture_code.py check                  # ranking rows whose code cell is not an architecture link (exit 1)
    python architecture_code.py add <member> <exp> <config_sha256> <url>   # record an entry in the local mapping
    python architecture_code.py publish [--dry-run]     # upload the mapping and rewrite the code cells of the Space
    python architecture_code.py verify                  # every mapped URL answers 200

`architecture-code.json` maps (member, experiment, full config_sha256) to the study source that defines the model:
`research/<study>/<file>` (optionally `#Lx-Ly`) at a pinned commit of TLAB-HGU/sunrun-leaderboard. board.render labels
that cell "architecture" and only falls back to the submitted replay script, labelled "script", when the row has no
entry. submit.py writes its own entry on every submission, so a new result needs no hand curation; this tool repairs
rows published before that rule and `check` reports any that remain.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from board import apply_architecture_links, script_cells  # noqa: E402  (board ships with every client copy)

SPACE, DATASET = "tlabtlab/sunrun-leaderboard", "tlabtlab/sunrun-lb-store"
FILE = "architecture-code.json"
GITHUB_REPO = "TLAB-HGU/sunrun-leaderboard"
# A pinned commit, not a branch: `main` moves and the linked lines would drift away from the submitted config.
PINNED_URL = re.compile(rf"^https://github\.com/{GITHUB_REPO}/blob/(?P<ref>[0-9a-f]{{7,40}})/"
                        r"(?P<path>[^#?\s/][^#?\s]*/[^#?\s/]*\.[A-Za-z0-9]+)(?P<lines>#L\d+(?:-L\d+)?)?$")
# The replay script submit.py requires for reproducibility, keyed by the same config SHA: never an architecture link.
REPLAY_SCRIPT = re.compile(r"^scripts/(?P<member>[^/]+)/(?P<experiment>[^/]+)/(?P<sha>[0-9a-f]{64})\.py$")


def mapping_path() -> Path:
    for candidate in (HERE / FILE, HERE.parent / FILE):
        if candidate.is_file():
            return candidate
    return HERE / FILE


def load(path: Path | None = None) -> dict:
    path = path or mapping_path()
    return json.loads(path.read_text()) if path.is_file() else {"models": []}


def entry(member: str, experiment: str, config_sha256: str, url: str) -> dict:
    return {"member": member, "experiment": experiment, "config_sha256": config_sha256, "url": url}


def upsert(mapping: dict, new: dict) -> dict:
    """One entry per (member, experiment, config_sha256): the new link replaces an earlier one for the same key."""
    key = lambda e: (e["member"], e["experiment"], e["config_sha256"])
    models = {key(e): e for e in mapping.get("models", [])}
    models[key(new)] = new
    return {**mapping, "models": [models[k] for k in sorted(models)]}


def is_architecture_url(url: str) -> bool:
    """A pinned repository file, not a branch, not a directory, not the replay script it was submitted with."""
    match = PINNED_URL.match(url or "")
    if not match:
        return False
    script = REPLAY_SCRIPT.match(match.group("path"))
    return script is None


def why_not_architecture(url: str) -> str:
    match = PINNED_URL.match(url or "")
    if not match:
        return (f"must be a pinned repository file: https://github.com/{GITHUB_REPO}/blob/<commit>/research/<study>/"
                f"<file>[#Lx-Ly], got {url!r}")
    return ("points at the submitted replay script; link the study source that defines the model instead: "
            f"{match.group('path')}")


def raw_url(url: str) -> str:
    match = PINNED_URL.match(url)
    return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{match.group('ref')}/{match.group('path')}"


def reachable(url: str, timeout: int = 15) -> bool:
    try:
        with urllib.request.urlopen(raw_url(url), timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def script_rows(page: str) -> list[tuple[str, str]]:
    """(experiment, link or "") of every ranking row whose code cell is not an architecture link."""
    return script_cells(page)


def space_file(space: str, name: str, revision: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        with open(hf_hub_download(space, name, repo_type="space", revision=revision), encoding="utf-8") as f:
            return f.read()
    except EntryNotFoundError:
        return None


def space_files(space: str) -> tuple[str | None, str | None, str]:
    """(index.html, architecture-code.json, revision) at the Space's head."""
    from huggingface_hub import HfApi
    revision = HfApi().repo_info(space, repo_type="space").sha
    return space_file(space, "index.html", revision), space_file(space, FILE, revision), revision


def publish(space: str = SPACE, mapping: dict | None = None, dry_run: bool = False,
            attempts: int = 5) -> tuple[bool, list[tuple[str, str]]]:
    """Write the mapping and rewrite the code cells of `index.html` in one commit, then report rows still on a script.

    Another publisher can land first (submit.py refreshes the page on every result), so a rejected commit is retried
    against the new head instead of overwriting it.
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    mapping = mapping or load()
    api = HfApi()
    for attempt in range(attempts):
        page, on_space, parent = space_files(space)
        if page is None:
            raise SystemExit(f"the Space has no index.html: {space}")
        merged = {**json.loads(on_space or '{"models": []}')}
        for item in mapping.get("models", []):  # the local file is authoritative for the entries it holds
            merged = upsert(merged, item)
        merged.setdefault("models", [])
        merged["models"] = [{k: e[k] for k in ("member", "experiment", "config_sha256", "url")}
                            for e in merged["models"]]
        new_page = apply_architecture_links(page, merged)
        rows = script_rows(new_page)
        if dry_run:
            return False, rows
        body = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        ops = [CommitOperationAdd(name, content.encode())
               for name, content, current in ((FILE, body, on_space), ("index.html", new_page, page))
               if content != current]
        if not ops:
            return False, rows
        try:
            api.create_commit(repo_id=space, repo_type="space", parent_commit=parent,
                              commit_message="Code column: architecture sources", operations=ops)
            return True, rows
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="report ranking rows whose code cell is not an architecture link")
    check.add_argument("--space", default=SPACE)
    add = sub.add_parser("add", help="record a mapping entry in the local architecture-code.json")
    add.add_argument("member"), add.add_argument("experiment"), add.add_argument("config_sha256"), add.add_argument("url")
    pub = sub.add_parser("publish", help="upload the mapping and rewrite the Space's code cells")
    pub.add_argument("--space", default=SPACE)
    pub.add_argument("--dry-run", action="store_true")
    ver = sub.add_parser("verify", help="check every mapped URL answers 200")
    a = ap.parse_args()

    if a.cmd == "check":
        page, _, _ = space_files(a.space)
        rows = script_rows(page or "")
        for experiment, link in rows:
            print(f"script: {experiment:40} {link or '(no link)'}")
        print(f"{len(rows)} row(s) without an architecture link")
        return 1 if rows else 0
    if a.cmd == "add":
        if not is_architecture_url(a.url):
            raise SystemExit(why_not_architecture(a.url))
        mapping = upsert(load(), entry(a.member, a.experiment, a.config_sha256, a.url))
        mapping_path().write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
        print(f"recorded {a.member}/{a.experiment} {a.config_sha256[:12]} -> {a.url}")
        return 0
    if a.cmd == "publish":
        changed, rows = publish(a.space, dry_run=a.dry_run)
        for experiment, link in rows:
            print(f"script: {experiment:40} {link or '(no link)'}")
        state = "would publish" if a.dry_run else ("published" if changed else "already up to date")
        print(f"{state}; {len(rows)} row(s) still without an architecture link")
        return 1 if rows else 0
    bad = [(e["experiment"], e["url"]) for e in load().get("models", []) if not reachable(e["url"])]
    for experiment, url in bad:
        print(f"unreachable: {experiment:40} {url}")
    print(f"{len(load().get('models', []))} entries, {len(bad)} unreachable")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
