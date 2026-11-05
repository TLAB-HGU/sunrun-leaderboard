"""Withdraw pre-campaign seongeun scores without hiding fresh run status/results."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "leaderboard"))
import board
import lbstatus
import submit

SPACE = "tlabtlab/sunrun-leaderboard"
DATASET = "tlabtlab/sunrun-lb-store"
RULE = {"member": "seongeun", "submitted_utc": "20261103T052654000Z"}


def read(repo, name, revision, kind):
    return Path(hf_hub_download(repo, name, repo_type=kind, revision=revision)).read_text()


def main():
    api = HfApi()
    data_revision, records = None, []
    for attempt in range(4):
        current_data = api.repo_info(DATASET, repo_type="dataset").sha
        if current_data != data_revision:
            files = [f for f in api.list_repo_files(DATASET, repo_type="dataset", revision=current_data)
                     if f.startswith("results/") and f.endswith(".json")]
            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(lambda f, version=current_data: json.loads(read(DATASET, f, version, "dataset")), files))
            records = submit.with_extras(api, DATASET, records, revision=current_data)
            data_revision = current_data
        parent = api.repo_info(SPACE, repo_type="space").sha
        policy = json.loads(read(SPACE, "visibility.json", parent, "space"))
        if RULE not in policy.setdefault("hidden_before", []):
            policy["hidden_before"].append(RULE)
        architecture = json.loads(read(SPACE, "architecture-code.json", parent, "space"))
        status = json.loads(read(SPACE, "status.json", parent, "space"))
        board.visibility_policy = lambda p=policy: p
        page = board.render(records, architecture=architecture)
        page = lbstatus.bold_experiment_names(lbstatus.mark_leaders(lbstatus.inject(page, status.get("active", []))), policy)
        page = lbstatus.apply_architecture_links(page, architecture)
        if api.repo_info(DATASET, repo_type="dataset").sha != data_revision:
            continue
        try:
            result = api.create_commit(repo_id=SPACE, repo_type="space", parent_commit=parent,
                                       commit_message="Withdraw old-input seongeun scores; retain fresh-df runs",
                                       operations=[CommitOperationAdd("visibility.json", (json.dumps(policy, indent=2) + "\n").encode()),
                                                   CommitOperationAdd("index.html", page.encode())])
            local = HERE.parents[1] / "leaderboard/visibility.json"
            local_policy = json.loads(local.read_text())
            if RULE not in local_policy.setdefault("hidden_before", []):
                local_policy["hidden_before"].append(RULE)
            local.write_text(json.dumps(local_policy, indent=2, ensure_ascii=False) + "\n")
            latest = api.repo_info(SPACE, repo_type="space").sha
            actual = read(SPACE, "index.html", latest, "space")
            visible_old = []
            found = board.ranking(actual)
            if not found:
                raise RuntimeError("Published ranking table missing")
            start, stop, cols = found
            import html
            import re
            labels = [name for name, _ in cols]
            for row in re.findall(r"<tr\b[^>]*>.*?</tr>", actual[start:stop], re.DOTALL):
                cells = list(board.CELL.finditer(row))
                if len(cells) != len(cols):
                    continue
                member = html.unescape(re.sub(r"<[^>]+>", "", cells[labels.index("member")].group(2)))
                dated = re.search(r'data-submitted-utc="([^"]+)"', row)
                if dated and board.hidden_before(member, dated[1], policy):
                    visible_old.append(row)
            if visible_old:
                raise RuntimeError("Old-input score rows remain visible")
            removed = sum(board.hidden_before(r["team_member"], r.get("submitted_utc"), policy) for r in records)
            receipt = {"commit_url": result.commit_url, "legacy_submissions_hidden": removed,
                       "fresh_status_entries_preserved": len(status.get("active", [])),
                       "stored_results_unchanged": len(records), "visible_old_score_rows": 0}
            (HERE / "old-input-withdrawal.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)
            return
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == 3:
                raise
    raise RuntimeError("Leaderboard kept changing; retry withdrawal from fresh heads")


if __name__ == "__main__":
    main()
