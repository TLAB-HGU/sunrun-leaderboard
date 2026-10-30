"""Score a predictions file locally against the private HF store, record it, refresh the static leaderboard.

    export HF_TOKEN=hf_...            # token handed out by the maintainer; no HF account or org membership needed
    python submit.py predictions.parquet meta.yaml [--dry-run]
    python submit.py --show           # print the current leaderboard in the terminal

The token only needs read/write on the dataset (default tlabtlab/sunrun-lb-store) and the static Space.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "scorer"), str(HERE)]
from board import _latest, render  # noqa: E402
from scoring import score, validate  # noqa: E402

DATASET = "tlabtlab/sunrun-lb-store"
SPACE = "tlabtlab/sunrun-leaderboard"
GITHUB_REPO = "TLAB-HGU/sunrun-leaderboard"
GITHUB_BRANCH = "main"
REQUIRED_META = ("team_member", "experiment", "description", "no_future_leakage", "config", "inference_seconds_per_fold")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def require_token() -> None:
    from huggingface_hub import get_token
    if not get_token():
        raise SystemExit("No Hugging Face token found. Set it first:  export HF_TOKEN=<token from the maintainer>")


def text_table(results: list[dict]) -> str:
    rows = sorted(_latest(results), key=lambda r: r["mse"])
    out = [f"{'#':>2} {'experiment':<24} {'member':<12} {'config':<12} {'sec/fold':>10} {'MSE':>10} {'skill':>8} {'beats naive':>11} code"]
    for i, r in enumerate(rows, 1):
        beat = (r.get("vs_naive") or {}).get("beats_reference")
        config = r.get("config_sha256", "legacy")[:12]
        seconds = r.get("inference_seconds_per_fold")
        seconds_text = f"{seconds:.6f}" if seconds is not None else "-"
        out.append(f"{i:>2} {r['experiment'][:24]:<24} {r['team_member'][:12]:<12} {config:<12} {seconds_text:>10} "
                   f"{r['mse']:>10,.1f} {r.get('skill_vs_naive') or 0:>+8.1%} {'yes' if beat else 'no':>11} "
                   f"{r.get('code_url') or '-'}")
    return "\n".join(out)


def canonical_config(config: dict) -> tuple[str, str]:
    """Return canonical JSON and its stable SHA-256 identity."""
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    for section in ("data", "hyperparameters"):
        if not isinstance(config.get(section), dict) or not config[section]:
            raise ValueError(f"config.{section} must be a non-empty mapping")
    unexpected = set(config).difference(("data", "hyperparameters"))
    if unexpected:
        raise ValueError(f"config has unsupported sections: {sorted(unexpected)}")
    try:
        canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config must contain JSON-compatible finite values: {exc}") from exc
    return canonical, hashlib.sha256(canonical.encode()).hexdigest()


def script_location(member: str, experiment: str, config_sha256: str) -> tuple[str, str, str]:
    path = f"scripts/{member}/{experiment}/{config_sha256}.py"
    raw = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
    page = f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{path}"
    return path, raw, page


def validate_meta(meta: dict) -> tuple[dict, str, float]:
    if not isinstance(meta, dict):
        raise SystemExit("meta must be a YAML mapping")
    missing = [k for k in REQUIRED_META if k not in meta or meta[k] is None or meta[k] == ""]
    if missing:
        raise SystemExit(f"meta needs: {missing}")
    if meta["no_future_leakage"] is not True:
        raise SystemExit("meta.no_future_leakage must be true")
    if not isinstance(meta["description"], str) or not meta["description"].strip():
        raise SystemExit("meta.description must be a non-empty string")
    for field in ("team_member", "experiment"):
        if not isinstance(meta[field], str) or not SAFE_NAME.fullmatch(meta[field]):
            raise SystemExit(f"meta.{field} must match {SAFE_NAME.pattern}")
    try:
        canonical, config_sha = canonical_config(meta["config"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    seconds = meta["inference_seconds_per_fold"]
    if isinstance(seconds, bool):
        raise SystemExit("meta.inference_seconds_per_fold must be a positive finite number")
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise SystemExit("meta.inference_seconds_per_fold must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise SystemExit("meta.inference_seconds_per_fold must be a positive finite number")
    return json.loads(canonical), config_sha, seconds


def require_script(member: str, experiment: str, config_sha: str, remote: bool) -> str:
    path, raw_url, page_url = script_location(member, experiment, config_sha)
    if not remote:
        if not (HERE / path).is_file():
            raise SystemExit(f"experiment script is missing: {path}")
        return page_url
    try:
        with urllib.request.urlopen(raw_url, timeout=15) as response:
            if response.status != 200:
                raise SystemExit(f"experiment script is not available on GitHub main: {page_url}")
            response.read(1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"cannot verify experiment script on GitHub main: {page_url} ({exc})") from exc
    return page_url


def _download(repo: str, name: str, tries: int = 4, revision: str | None = None) -> str:
    """Fresh download with retries: right after an upload the Hub can briefly fail the HEAD call."""
    for i in range(tries):
        try:
            return hf_hub_download(repo, name, repo_type="dataset", revision=revision, force_download=True)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _store(name: str, repo: str) -> pd.DataFrame:
    return pd.read_parquet(_download(repo, name))


def all_results(api: HfApi, repo: str, revision: str | None = None) -> list[dict]:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision) if f.startswith("results/")]
    return [json.loads(Path(_download(repo, f, revision=revision)).read_text()) for f in files]


def publish(api: HfApi, results: list[dict], space: str) -> str:
    try:
        api.create_repo(space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    except Exception:  # some plans cannot host private Spaces; the page only exposes metrics
        api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.create_commit(
        repo_id=space,
        repo_type="space",
        commit_message="Refresh leaderboard",
        operations=[
            CommitOperationAdd("README.md", b"---\ntitle: Solar wind 72h leaderboard\nsdk: static\napp_file: index.html\n---\n"),
            CommitOperationAdd("index.html", render(results).encode()),
        ],
    )
    return f"https://huggingface.co/spaces/{space}"


def store_submission(api: HfApi, repo: str, rec: dict, pred_bytes: bytes, replace: bool,
                     max_attempts: int = 3) -> None:
    """Atomically store one keyed result, retrying if another submission advances the repo."""
    member, experiment, config_sha = (rec[k] for k in ("team_member", "experiment", "config_sha256"))
    key_path = f"{member}/{experiment}/{config_sha}"
    for attempt in range(max_attempts):
        parent = api.repo_info(repo, repo_type="dataset").sha
        results = all_results(api, repo, revision=parent)
        duplicate = any(r.get("team_member") == member and r.get("experiment") == experiment
                        and r.get("config_sha256") == config_sha for r in results)
        if duplicate and not replace:
            raise SystemExit("submission rejected: (team_member, experiment, config_sha256) already exists; use --replace")
        operations = [
            CommitOperationAdd(f"submissions/{key_path}.parquet", pred_bytes),
            CommitOperationAdd(f"results/{key_path}.json", json.dumps(rec, ensure_ascii=False).encode()),
        ]
        try:
            api.create_commit(repo_id=repo, repo_type="dataset", commit_message=f"Submit {member}/{experiment}",
                              operations=operations, parent_commit=parent)
            return
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in (409, 412) or attempt == max_attempts - 1:
                raise
    raise RuntimeError("unreachable")


def submit(pred_path: str, meta: dict, dataset: str = DATASET, space: str = SPACE, dry_run: bool = False,
           replace: bool = False) -> dict:
    config, config_sha, seconds = validate_meta(meta)
    code_url = require_script(meta["team_member"], meta["experiment"], config_sha, remote=not dry_run)
    require_token()
    api = HfApi()
    pred = pd.read_parquet(pred_path) if str(pred_path).endswith(".parquet") else pd.read_csv(pred_path)
    folds, truth, naive = (_store(n, dataset) for n in ("folds.parquet", "truth.parquet", "naive.parquet"))
    errs = validate(pred, folds)
    if errs:
        raise SystemExit("submission rejected:\n- " + "\n- ".join(errs))
    res = score(pred, truth, folds, naive)
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now % 1 * 1000):03d}Z"
    rec = {"submitted_utc": ts, **{k: meta[k] for k in ("team_member", "experiment", "description")},
           "config": config, "config_sha256": config_sha, "inference_seconds_per_fold": seconds,
           "code_url": code_url, **res}
    if not dry_run:
        # One Hub commit keeps the keyed prediction and its score consistent.
        buf = io.BytesIO()
        pred.to_parquet(buf, index=False)
        store_submission(api, dataset, rec, buf.getvalue(), replace)
        rec["_leaderboard"] = publish(api, all_results(api, dataset), space)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="?")
    ap.add_argument("meta", nargs="?")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--dry-run", action="store_true", help="validate and score only; upload nothing")
    ap.add_argument("--replace", action="store_true", help="replace an existing (member, experiment, config) result")
    ap.add_argument("--show", action="store_true", help="print the current leaderboard and exit")
    a = ap.parse_args()
    if a.show:
        require_token()
        print(text_table(all_results(HfApi(), a.dataset)))
        raise SystemExit(0)
    if not (a.predictions and a.meta):
        ap.error("predictions and meta are required (or use --show)")
    r = submit(a.predictions, yaml.safe_load(Path(a.meta).read_text()), a.dataset, a.space, a.dry_run, a.replace)
    vs = r["vs_naive"]
    print(f"MSE={r['mse']:.1f}  MSE(obs)={r['mse_observed']:.1f}  skill vs naive={r['skill_vs_naive']:+.3f}  "
          f"beats naive (95% CI)={vs['beats_reference']}  diff CI={[round(x, 1) for x in vs['ci95']]}")
    if "_leaderboard" in r:
        print("\n" + text_table(all_results(HfApi(), a.dataset)))
        print("\npage (needs an HF login of an org member):", r["_leaderboard"])
