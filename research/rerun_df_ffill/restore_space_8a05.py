"""Restore the Space visibility policy of revision 8a05fa25f3 (2026-11-03 15:18 UTC) and re-render the page.

A jsy301 submission at 2026-11-04 00:45 UTC came from a client holding the 2026-11-01 visibility.json and
uploaded it with its page. This puts the 8a05 policy back byte for byte and redraws the ranking from every
stored result at the dataset head (so the jsy301 result and all fresh-df results submitted since 8a05 show),
then re-adds the active status rows from the Space's status.json, in the same order lbstatus.commit uses.
Stored results, status.json and architecture-code.json are not modified.
"""
from __future__ import annotations

import fcntl
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "leaderboard"))
import board  # noqa: E402
import lbstatus  # noqa: E402
import submit  # noqa: E402

SPACE = "tlabtlab/sunrun-leaderboard"
DATASET = "tlabtlab/sunrun-lb-store"
POLICY_REVISION = "8a05fa25f3"
STATUS_MARK = re.compile(r" \((?:진행|대기)\)$")


def read(repo, name, revision, kind):
    return Path(hf_hub_download(repo, name, repo_type=kind, revision=revision)).read_text()


def full_revision(api, short):
    return next(c.commit_id for c in api.list_repo_commits(SPACE, repo_type="space") if c.commit_id.startswith(short))


def table_rows(page):
    """(member, experiment, submitted_utc or None, is_status_row) for every rendered ranking row."""
    found = board.ranking(page)
    if not found:
        raise RuntimeError("Ranking table missing")
    start, stop, cols = found
    labels = [name for name, _ in cols]
    out = []
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", page[start:stop], re.DOTALL):
        cells = list(board.CELL.finditer(row))
        if len(cells) != len(cols):
            continue
        text = lambda label: html.unescape(re.sub(r"<[^>]+>", "", cells[labels.index(label)].group(2))).strip()
        experiment = text("experiment")
        dated = re.search(r'data-submitted-utc="([^"]*)"', row)
        out.append((text("member"), STATUS_MARK.sub("", experiment), dated[1] if dated and dated[1] else None,
                    bool(STATUS_MARK.search(experiment))))
    return out


def main():
    api = HfApi()
    policy_rev = full_revision(api, POLICY_REVISION)
    policy_raw = read(SPACE, "visibility.json", policy_rev, "space")
    policy = json.loads(policy_raw)
    reference_rows = table_rows(read(SPACE, "index.html", policy_rev, "space"))
    data_revision, records = None, []
    for attempt in range(4):
        current_data = api.repo_info(DATASET, repo_type="dataset").sha
        if current_data != data_revision:
            files = [f for f in api.list_repo_files(DATASET, repo_type="dataset", revision=current_data)
                     if f.startswith("results/") and f.endswith(".json")]
            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(lambda f, v=current_data: json.loads(read(DATASET, f, v, "dataset")), files))
            records = submit.with_extras(api, DATASET, records, revision=current_data)
            data_revision = current_data
        parent = api.repo_info(SPACE, repo_type="space").sha
        architecture = json.loads(read(SPACE, "architecture-code.json", parent, "space"))
        active = lbstatus.entries_of(read(SPACE, "status.json", parent, "space"), "active")
        board.visibility_policy = lambda p=policy: p
        page = board.render(records, architecture=architecture)
        page = lbstatus.bold_experiment_names(lbstatus.mark_leaders(lbstatus.hide_ranking_rows(
            lbstatus.inject(lbstatus.trim_ranking_columns(page), active), policy)), policy)
        page = lbstatus.apply_architecture_links(page, architecture)
        if api.repo_info(DATASET, repo_type="dataset").sha != data_revision:
            continue
        try:
            result = api.create_commit(
                repo_id=SPACE, repo_type="space", parent_commit=parent,
                commit_message="Restore 8a05 visibility policy; re-render fresh-df and jsy301 results",
                operations=[CommitOperationAdd("visibility.json", policy_raw.encode()),
                            CommitOperationAdd("index.html", page.encode())])
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == 3:
                raise
            time.sleep(1 + attempt)
            continue
        latest = api.repo_info(SPACE, repo_type="space").sha
        if read(SPACE, "visibility.json", latest, "space") != policy_raw:
            raise RuntimeError("Space visibility.json differs from the 8a05 policy after commit")
        rows = table_rows(read(SPACE, "index.html", latest, "space"))
        results = [r for r in rows if not r[3]]
        status = [r for r in rows if r[3]]
        old_visible = [r for r in results if board.hidden_before(r[0], r[2], policy)]
        ref_results = {(m, e, s) for m, e, s, st in reference_rows if not st}
        now_results = {(m, e, s) for m, e, s, _ in results}
        receipt = {"commit_url": result.commit_url, "space_revision": latest, "policy_revision": policy_rev,
                   "dataset_revision": data_revision, "stored_results": len(records),
                   "visible_result_rows": len(results), "status_rows": len(status), "active_entries": len(active),
                   "reference_result_rows": len(ref_results),
                   "added_since_reference": sorted(f"{m}/{e} {s}" for m, e, s in now_results - ref_results),
                   "reference_rows_missing": sorted(f"{m}/{e} {s}" for m, e, s in ref_results - now_results),
                   "visible_old_input_rows": len(old_visible)}
        (HERE / "restore-8a05-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        if old_visible or len(status) != len(active):
            raise RuntimeError("Restored page failed its visibility/status checks; see receipt")
        return
    raise RuntimeError("Leaderboard kept changing; retry from fresh heads")


if __name__ == "__main__":
    with (HERE / ".publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
