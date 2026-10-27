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
from board import end_to_end_hours, event_value, f2_mse, has_f1, ranked, render  # noqa: E402
from scoring import score, validate  # noqa: E402

DATASET = "tlabtlab/sunrun-lb-store"
SPACE = "tlabtlab/sunrun-leaderboard"
GITHUB_REPO = "TLAB-HGU/sunrun-leaderboard"
GITHUB_BRANCH = "main"
REQUIRED_META = ("team_member", "experiment", "description", "no_future_leakage", "f1_training_data_end_utc", "config",
                 "inference_seconds_per_fold")
F1_TRAIN_END = pd.Timestamp("2026-02-26T00:00:00Z")  # F1 start - 72h; F1 predictions use only data before this
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def require_token() -> None:
    from huggingface_hub import get_token
    if not get_token():
        raise SystemExit("No Hugging Face token found. Set it first:  export HF_TOKEN=<token from the maintainer>")


def text_table(results: list[dict]) -> str:
    def num(x):
        return f"{x:,.1f}" if x is not None else "-"

    out = [f"{'#':>4} {'experiment':<24} {'member':<12} {'config':<12} {'sec/fold':>10} {'e2e(h)':>7} {'MSE':>10} {'F1':>10} {'F2':>10} "
           f"{'MSE>=550':>10} {'F1>=550':>10} {'F2>=550':>10} {'CSI':>6} {'TmMAE':>6} {'skill':>8} {'beats naive':>11} code"]
    for i, r in enumerate(ranked(results), 1):
        full = has_f1(r)
        metrics = [r.get(k) if full else None for k in ("mse", "mse_f1", "mse_f2", "mse_ge550", "mse_f1_ge550", "mse_f2_ge550")]
        metrics[2] = f2_mse(r)
        periods = " ".join(f"{num(x):>10}" for x in metrics)
        csi, tmae = event_value(r, "csi"), event_value(r, "timing_mae_h")
        events = f"{f'{csi:.3f}' if csi is not None else '-':>6} {num(tmae):>6}"
        beat = (r.get("vs_naive") or {}).get("beats_reference")
        config = r.get("config_sha256", "legacy")[:12]
        seconds = r.get("inference_seconds_per_fold")
        seconds_text = f"{seconds:.6f}" if seconds is not None else "-"
        e2e = end_to_end_hours(r)
        e2e_text = f"{e2e:.2f}" if e2e is not None else "-"
        out.append(f"{i if full else 'F2':>4} {r['experiment'][:24]:<24} {r['team_member'][:12]:<12} {config:<12} {seconds_text:>10} "
                   f"{e2e_text:>7} {periods} {events} {r.get('skill_vs_naive') or 0:>+8.1%} {'yes' if beat else 'no':>11} "
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
    for field in ("team_member", "experiment"):
        if not isinstance(meta[field], str) or not SAFE_NAME.fullmatch(meta[field]):
            raise SystemExit(f"meta.{field} must match {SAFE_NAME.pattern}")
    try:
        f1_end = pd.Timestamp(str(meta["f1_training_data_end_utc"]))
    except ValueError as exc:
        raise SystemExit("meta.f1_training_data_end_utc must be a UTC timestamp, e.g. 2026-02-26T00:00:00Z") from exc
    f1_end = f1_end.tz_localize("UTC") if f1_end.tzinfo is None else f1_end.tz_convert("UTC")
    if f1_end > F1_TRAIN_END:
        raise SystemExit(f"meta.f1_training_data_end_utc must be <= {F1_TRAIN_END:%Y-%m-%dT%H:%M:%SZ}: "
                         "F1 predictions may only use models trained on data before it")
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
    return json.loads(canonical), config_sha, seconds, f"{f1_end:%Y-%m-%dT%H:%M:%SZ}"


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
    config, config_sha, seconds, f1_end = validate_meta(meta)
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
           "f1_training_data_end_utc": f1_end, "config": config, "config_sha256": config_sha, "inference_seconds_per_fold": seconds,
           "code_url": code_url, **res}
    if not dry_run:
        # One Hub commit keeps the keyed prediction and its score consistent.
        buf = io.BytesIO()
        pred.to_parquet(buf, index=False)
        store_submission(api, dataset, rec, buf.getvalue(), replace)
        rec["_leaderboard"] = publish(api, with_extras(api, dataset, all_results(api, dataset)), space)
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
        print(text_table(with_extras(HfApi(), a.dataset, all_results(HfApi(), a.dataset))))
        raise SystemExit(0)
    if not (a.predictions and a.meta):
        ap.error("predictions and meta are required (or use --show)")
    r = submit(a.predictions, yaml.safe_load(Path(a.meta).read_text()), a.dataset, a.space, a.dry_run, a.replace)
    vs = r["vs_naive"]
    print(f"MSE={r['mse']:.1f}  F1={r['mse_f1']:.1f}  F2={r['mse_f2']:.1f}  "
          f"MSE>=550={r['mse_ge550']:.1f}  F1>=550={r['mse_f1_ge550']:.1f}  F2>=550={r['mse_f2_ge550']:.1f}")
    tm = "-" if r["timing_mae_h"] is None else f"{r['timing_mae_h']:.1f}h (bias {r['timing_bias_h']:+.1f}h)"
    print(f"HSS events: CSI={'-' if r['csi'] is None else format(r['csi'], '.3f')}  timing MAE={tm}  (TP/FP/FN={r['events']['']['tp']}/{r['events']['']['fp']}/{r['events']['']['fn']})")
    print(f"MSE(obs)={r['mse_observed']:.1f}  skill vs naive={r['skill_vs_naive']:+.3f}  "
          f"beats naive (95% CI)={vs['beats_reference']}  diff CI={[round(x, 1) for x in vs['ci95']]}")
    if "_leaderboard" in r:
        print("\n" + text_table(with_extras(HfApi(), a.dataset, all_results(HfApi(), a.dataset))))
        print("\npage (needs an HF login of an org member):", r["_leaderboard"])
