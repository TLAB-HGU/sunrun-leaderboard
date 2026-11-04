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
import board  # noqa: E402  (render reads board.visibility_policy at call time)
from board import competition_metrics, ranked, render, visibility_policy
from competition import prepare_context, score_competition  # noqa: E402
from metric_assets import publish_plots  # noqa: E402
import architecture_code  # noqa: E402  (the code-column mapping every submission writes)

DATASET = "tlabtlab/sunrun-lb-store"
SPACE = "tlabtlab/sunrun-leaderboard"
GITHUB_REPO = "TLAB-HGU/sunrun-leaderboard"
GITHUB_BRANCH = "main"
REQUIRED_META = ("team_member", "experiment", "description", "no_future_leakage", "f1_training_data_end_utc",
                 "f2_training_data_end_utc", "f3_training_data_end_utc", "config", "inference_seconds_per_fold",
                 "architecture_url")
# period start - 72h: predictions for a period use only models trained on data before its cutoff
TRAIN_END = {"f1": pd.Timestamp("2025-12-29T00:00:00Z"), "f2": pd.Timestamp("2026-03-29T00:00:00Z"),
             "f3": pd.Timestamp("2026-06-29T00:00:00Z")}
F1_TRAIN_END = TRAIN_END["f1"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def require_token() -> None:
    from huggingface_hub import get_token
    if not get_token():
        raise SystemExit("No Hugging Face token found. Set it first:  export HF_TOKEN=<token from the maintainer>")


def text_table(results: list[dict]) -> str:
    labels = [f"{scope.upper()} {metric.upper()}" for metric in ("s", "timse", "rmse")
              for scope in ("total", "f1", "f2", "f3")]
    out = ["rank experiment member " + " | ".join(labels)]
    for i, record in enumerate(ranked(results), 1):
        metrics = competition_metrics(record) or {}
        values = [metrics.get(scope, {}).get(metric) for metric in ("s", "timse", "rmse")
                  for scope in ("total", "f1", "f2", "f3")]
        cells = " | ".join("-" if value is None else f"{value:.3f}" for value in values)
        rank = str(i) if metrics else "unscored"
        out.append(f"{rank} {record['experiment']} {record['team_member']} {cells}")
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


def replay_script_path(member: str, experiment: str, config: dict) -> str:
    """Repository path where a study writes its replay script; the key in the file name stays inside submit.py."""
    return script_location(member, experiment, canonical_config(config)[1])[0]


def architecture_location(meta: dict) -> str:
    """The code-column link of a submission: the study source that defines the model, pinned to a commit.

    The submitted replay script stays required (scripts/<member>/<experiment>/<config_sha256>.py) but is never shown
    on the board: the code cell must open the architecture, so `meta.architecture_url` names it explicitly and
    submit.py records it in the Space's architecture-code.json entry for this result."""
    url = meta.get("architecture_url")
    if not isinstance(url, str) or not url.strip():
        raise SystemExit("meta.architecture_url must be a non-empty string")
    url = url.strip()
    if not architecture_code.is_architecture_url(url):
        raise SystemExit(f"meta.architecture_url {architecture_code.why_not_architecture(url)}")
    return url


def validate_meta(meta: dict) -> tuple[dict, str, float, str]:
    if not isinstance(meta, dict):
        raise SystemExit("meta must be a YAML mapping")
    missing = [k for k in REQUIRED_META if k not in meta or meta[k] is None or meta[k] == ""]
    if missing:
        raise SystemExit(f"meta needs: {missing}")
    if meta["no_future_leakage"] is not True:
        raise SystemExit("meta.no_future_leakage must be true")
    if not isinstance(meta["description"], str) or not meta["description"].strip():
        raise SystemExit("meta.description must be a non-empty string")
    architecture_location(meta)
    for field in ("team_member", "experiment"):
        if not isinstance(meta[field], str) or not SAFE_NAME.fullmatch(meta[field]):
            raise SystemExit(f"meta.{field} must match {SAFE_NAME.pattern}")
    ends = {}
    for period, limit in TRAIN_END.items():
        field = f"{period}_training_data_end_utc"
        try:
            end = pd.Timestamp(str(meta[field]))
        except ValueError as exc:
            raise SystemExit(f"meta.{field} must be a UTC timestamp, e.g. {limit:%Y-%m-%dT%H:%M:%SZ}") from exc
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end > limit:
            raise SystemExit(f"meta.{field} must be <= {limit:%Y-%m-%dT%H:%M:%SZ}: "
                             f"{period.upper()} predictions may only use models trained on data before it")
        ends[period] = f"{end:%Y-%m-%dT%H:%M:%SZ}"
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
    return json.loads(canonical), config_sha, seconds, ends["f1"], ends


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


def with_extras(api: HfApi, repo: str, results: list[dict], revision: str | None = None) -> list[dict]:
    """Attach optional display-only extras (``extras/<member>/<experiment>/<config_sha256>.json``) as ``r["extras"]``.

    Extras are facts added after scoring, such as end-to-end compute time; the scored result files stay untouched.
    """
    files = {f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision) if f.startswith("extras/")}
    out = []
    for r in results:
        path = f"extras/{r['team_member']}/{r['experiment']}/{r.get('config_sha256', 'legacy')}.json"
        out.append({**r, "extras": json.loads(Path(_download(repo, path, revision=revision)).read_text())}
                   if path in files else r)
    return out


def space_file(space: str, name: str, revision: str) -> dict:
    """A JSON policy file at the Space's head, or {} on a Space that has none yet."""
    try:
        with open(hf_hub_download(space, name, repo_type="space", revision=revision), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def render_with_policy(results: list[dict], policy: dict, architecture: dict) -> str:
    """Render under `policy` instead of the local visibility.json, restoring the module hook afterwards."""
    local = board.visibility_policy
    board.visibility_policy = lambda: policy
    try:
        return render(results, architecture=architecture)
    finally:
        board.visibility_policy = local


def publish(api: HfApi, results: list[dict], space: str, architecture: dict | None = None,
            attempts: int = 3, dataset: str | None = None, push_policy: bool = False) -> str:
    """Render the page and record this submission's architecture link in one commit, retrying if another publish
    lands first. The code cells must open the architecture, never the replay script, so the mapping is uploaded with
    the page it renders instead of being curated by hand afterwards.

    The Space's visibility.json is the policy of record: the page is rendered with it, so a client holding a stale
    local copy cannot re-hide or re-show rows (a 2026-11-04 submission from an old checkout did exactly that).
    The local visibility.json is uploaded only when the Space has none yet, or when `push_policy` is set on purpose
    (`submit.py --push-policy`)."""
    try:
        api.create_repo(space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    except Exception:  # some plans cannot host private Spaces; the page only exposes metrics
        api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    for attempt in range(attempts):
        parent = api.repo_info(space, repo_type="space").sha
        if dataset is not None:
            revision = api.repo_info(dataset, repo_type="dataset").sha
            results = with_extras(api, dataset, all_results(api, dataset, revision=revision), revision=revision)
        on_space = space_file(space, architecture_code.FILE, parent)
        mapping = on_space
        for item in (architecture or {}).get("models", []):
            mapping = architecture_code.upsert(mapping, item)
        body = json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
        on_space_policy = space_file(space, "visibility.json", parent)
        policy = visibility_policy() if push_policy or not on_space_policy else on_space_policy
        operations = [
            CommitOperationAdd("README.md", b"---\ntitle: Solar wind 72h leaderboard\nsdk: static\napp_file: index.html\n---\n"),
            CommitOperationAdd("index.html", render_with_policy(results, policy, mapping).encode()),
        ]
        if body != json.dumps(on_space, indent=2, ensure_ascii=False) + "\n":
            operations.append(CommitOperationAdd(architecture_code.FILE, body.encode()))
        if policy and policy != on_space_policy:  # only a bootstrap or an explicit --push-policy reaches this
            operations.append(CommitOperationAdd("visibility.json",
                                                 (json.dumps(policy, indent=2, ensure_ascii=False) + "\n").encode()))
        try:
            api.create_commit(repo_id=space, repo_type="space", commit_message="Refresh leaderboard",
                              operations=operations, parent_commit=parent)
            return f"https://huggingface.co/spaces/{space}"
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


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
    config, config_sha, seconds, f1_end, train_ends = validate_meta(meta)
    code_url = require_script(meta["team_member"], meta["experiment"], config_sha, remote=not dry_run)
    architecture_url = architecture_location(meta)
    if not dry_run and not architecture_code.reachable(architecture_url):
        raise SystemExit(f"meta.architecture_url does not resolve on GitHub: {architecture_url}")
    require_token()
    api = HfApi()
    pred = pd.read_parquet(pred_path) if str(pred_path).endswith(".parquet") else pd.read_csv(pred_path)
    data_revision = api.repo_info(dataset, repo_type="dataset").sha
    observation_path = Path(_download(dataset, "competition-observations.parquet", revision=data_revision))
    observations = pd.read_parquet(observation_path)
    try:
        competition = score_competition(pred, prepare_context(observations))
    except (ValueError, FloatingPointError) as exc:
        raise SystemExit(f"submission rejected: {exc}") from exc
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now % 1 * 1000):03d}Z"
    rec = {"submitted_utc": ts, **{k: meta[k] for k in ("team_member", "experiment", "description")},
           "f1_training_data_end_utc": f1_end, "f2_training_data_end_utc": train_ends["f2"],
           "f3_training_data_end_utc": train_ends["f3"], "config": config, "config_sha256": config_sha, "inference_seconds_per_fold": seconds,
           "code_url": code_url, "architecture_url": architecture_url, "competition": competition,
           "competition_dataset_revision": data_revision}
    if not dry_run:
        rec = publish_plots([rec], HERE / "assets" / "metrics")[0]
        # One Hub commit keeps the keyed prediction and its score consistent.
        buf = io.BytesIO()
        pred.to_parquet(buf, index=False)
        store_submission(api, dataset, rec, buf.getvalue(), replace)
        entry = architecture_code.entry(rec["team_member"], rec["experiment"], config_sha, architecture_url)
        rec["_leaderboard"] = publish(api, [], space, architecture={"models": [entry]}, dataset=dataset)
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
    ap.add_argument("--push-policy", action="store_true",
                    help="upload the local visibility.json as the Space policy, re-render the page and exit")
    a = ap.parse_args()
    if a.push_policy:
        require_token()
        print(publish(HfApi(), [], a.space, dataset=a.dataset, push_policy=True))
        raise SystemExit(0)
    if a.show:
        require_token()
        print(text_table(with_extras(HfApi(), a.dataset, all_results(HfApi(), a.dataset))))
        raise SystemExit(0)
    if not (a.predictions and a.meta):
        ap.error("predictions and meta are required (or use --show)")
    r = submit(a.predictions, yaml.safe_load(Path(a.meta).read_text()), a.dataset, a.space, a.dry_run, a.replace)
    for scope in ("total", "f1", "f2", "f3"):
        metrics = r["competition"][scope]
        print(f"{scope.upper()}: S={metrics['s']:.3f} TIMSE={metrics['timse']:.3f} "
              f"RMSE={metrics['rmse']:.3f} persistence={metrics['replacement_fraction']:.2%}")
    if "_leaderboard" in r:
        print("\n" + text_table(with_extras(HfApi(), a.dataset, all_results(HfApi(), a.dataset))))
        print("\npage (needs an HF login of an org member):", r["_leaderboard"])
