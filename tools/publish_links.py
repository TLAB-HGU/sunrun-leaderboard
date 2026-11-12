"""Record visualisation links on the leaderboard Space and redraw the page in one commit.

    /home/t-lab01/sunrun/.venv/bin/python tools/publish_links.py --commit <40-hex GitHub commit> \
        seongeun/e2000fs_calib3_era/20261104T013907870Z=e2000fs_calib3_era ...

Each argument is member/experiment/submitted_utc=<experiment folder under research>; the pictures are in
research/<folder>/visualization. The link is pinned to the given
commit of TLAB-HGU/sunrun-leaderboard. Entries are upserted into the Space's visualizations.json; the page is
rendered from every stored result under the Space's own visibility policy and architecture mapping, and the active
대기/진행 rows are re-added in the same commit, so the page never shows a gap. Holds a local lock so two link
publishers on this machine do not interleave.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path("/home/t-lab01/sunrun")
sys.path.insert(0, str(ROOT / "leaderboard"))
import board  # noqa: E402
import lbstatus  # noqa: E402
import submit  # noqa: E402

LOCK = ROOT / "leaderboard/.publish.lock"
REPO = "https://github.com/TLAB-HGU/sunrun-leaderboard/tree"


def entries(commit: str, specs: list[str]) -> list[dict]:
    out = []
    for spec in specs:
        ident, folder = spec.split("=", 1)
        member, experiment, submitted = ident.split("/")
        assert (ROOT / "research" / folder / "visualization/manifest.json").exists(), folder
        out.append({"member": member, "experiment": experiment, "submitted_utc": submitted,
                    "url": f"{REPO}/{commit}/research/{folder}/visualization"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", required=True)
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args()
    new = entries(args.commit, args.specs)
    api = HfApi()
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(5):
            revision = api.repo_info(submit.DATASET, repo_type="dataset").sha
            files = [f for f in api.list_repo_files(submit.DATASET, repo_type="dataset", revision=revision)
                     if f.startswith("results/") and f.endswith(".json")]
            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(lambda f: json.loads(Path(submit._download(submit.DATASET, f,
                                                                                   revision=revision)).read_text()),
                                        files))
            records = submit.with_extras(api, submit.DATASET, records, revision=revision)
            parent = api.repo_info(submit.SPACE, repo_type="space").sha
            policy = submit.space_file(submit.SPACE, "visibility.json", parent)
            architecture = submit.space_file(submit.SPACE, "architecture-code.json", parent)
            on_space = submit.space_file(submit.SPACE, board.VISUALIZATION_FILE, parent)
            mapping = on_space
            for item in new:
                mapping = board.upsert_visualization(mapping, item)
            status = lbstatus.entries_of(lbstatus._space_file(submit.SPACE, lbstatus.STATUS_FILE, parent), "active")
            page = submit.render_with_policy(records, policy, architecture, mapping)
            page = lbstatus.bold_experiment_names(lbstatus.mark_leaders(lbstatus.hide_ranking_rows(
                lbstatus.inject(lbstatus.trim_ranking_columns(page), status), policy)), policy)
            page = lbstatus.apply_architecture_links(page, architecture)
            if api.repo_info(submit.DATASET, repo_type="dataset").sha != revision:
                continue
            body = json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
            try:
                result = api.create_commit(
                    repo_id=submit.SPACE, repo_type="space", parent_commit=parent,
                    commit_message=f"Link {len(new)} submissions to research/<experiment>/visualization",
                    operations=[CommitOperationAdd("index.html", page.encode()),
                                CommitOperationAdd(board.VISUALIZATION_FILE, body.encode())])
            except HfHubHTTPError as exc:
                if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == 4:
                    raise
                time.sleep(1 + attempt)
                continue
            (ROOT / "leaderboard" / board.VISUALIZATION_FILE).write_text(body)  # local fallback mirrors the Space
            print(json.dumps({"commit_url": result.commit_url, "dataset_revision": revision,
                              "linked": len(mapping["models"]), "active_status_rows": len(status)}))
            return
    raise RuntimeError("Leaderboard kept changing; retry from fresh heads")


if __name__ == "__main__":
    main()
