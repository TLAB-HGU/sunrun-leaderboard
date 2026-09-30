"""Score a predictions file locally against the private HF store, record it, refresh the static leaderboard.

    python submit.py predictions.parquet meta.yaml

Needs `hf auth login` with access to the dataset (default tlabtlab/sunrun-lb-store).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "scorer"), str(HERE)]
from board import render  # noqa: E402
from scoring import score, validate  # noqa: E402

DATASET = "tlabtlab/sunrun-lb-store"
SPACE = "tlabtlab/sunrun-leaderboard"
REQUIRED_META = ("team_member", "experiment", "description", "no_future_leakage")


def _download(repo: str, name: str, tries: int = 4) -> str:
    """Fresh download with retries: right after an upload the Hub can briefly fail the HEAD call."""
    for i in range(tries):
        try:
            return hf_hub_download(repo, name, repo_type="dataset", force_download=True)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _store(name: str, repo: str) -> pd.DataFrame:
    return pd.read_parquet(_download(repo, name))


def all_results(api: HfApi, repo: str) -> list[dict]:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("results/")]
    return [json.loads(Path(_download(repo, f)).read_text()) for f in files]


def publish(api: HfApi, results: list[dict], space: str) -> str:
    try:
        api.create_repo(space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    except Exception:  # some plans cannot host private Spaces; the page only exposes metrics
        api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=b"---\ntitle: Solar wind 72h leaderboard\nsdk: static\napp_file: index.html\n---\n",
                    path_in_repo="README.md", repo_id=space, repo_type="space")
    api.upload_file(path_or_fileobj=render(results).encode(), path_in_repo="index.html", repo_id=space, repo_type="space")
    return f"https://huggingface.co/spaces/{space}"


def submit(pred_path: str, meta: dict, dataset: str = DATASET, space: str = SPACE, dry_run: bool = False) -> dict:
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise SystemExit(f"meta needs: {missing} (no_future_leakage must be true)")
    api = HfApi()
    pred = pd.read_parquet(pred_path) if str(pred_path).endswith(".parquet") else pd.read_csv(pred_path)
    folds, truth, naive = (_store(n, dataset) for n in ("folds.parquet", "truth.parquet", "naive.parquet"))
    errs = validate(pred, folds)
    if errs:
        raise SystemExit("submission rejected:\n- " + "\n- ".join(errs))
    res = score(pred, truth, folds, naive)
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now % 1 * 1000):03d}Z"
    tag = f"{ts}_{meta['team_member']}_{meta['experiment']}"
    rec = {"submitted_utc": ts, **{k: meta[k] for k in ("team_member", "experiment", "description")}, **res}
    if not dry_run:
        # raw file first so a scored row always has its source
        api.upload_file(path_or_fileobj=str(pred_path), path_in_repo=f"submissions/{tag}{Path(pred_path).suffix}", repo_id=dataset, repo_type="dataset")
        api.upload_file(path_or_fileobj=json.dumps(rec, ensure_ascii=False).encode(), path_in_repo=f"results/{tag}.json", repo_id=dataset, repo_type="dataset")
        rec["_leaderboard"] = publish(api, all_results(api, dataset), space)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("meta")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--dry-run", action="store_true", help="validate and score only; upload nothing")
    a = ap.parse_args()
    r = submit(a.predictions, yaml.safe_load(Path(a.meta).read_text()), a.dataset, a.space, a.dry_run)
    vs = r["vs_naive"]
    print(f"MSE={r['mse']:.1f}  MSE(obs)={r['mse_observed']:.1f}  skill vs naive={r['skill_vs_naive']:+.3f}  "
          f"beats naive (95% CI)={vs['beats_reference']}  diff CI={[round(x, 1) for x in vs['ci95']]}")
    if "_leaderboard" in r:
        print("leaderboard:", r["_leaderboard"])
