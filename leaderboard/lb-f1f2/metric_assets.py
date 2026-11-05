"""Render lead curves and publish immutable PNG links through GitHub's Git API.

Only image blobs and a commit based on the current remote main are published;
local branches, index, and repository history are never pushed or modified.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

import numpy as np

GITHUB_REPO = "TLAB-HGU/sunrun-leaderboard"
ASSET_DIRECTORY = "leaderboard/assets/metrics"
PLOT_VERSION = "lead-curves-v1"
SCOPES = (("total", "Total", "#171717"), ("f1", "F1", "#0072B2"),
          ("f2", "F2", "#D55E00"), ("f3", "F3", "#009E73"))
METRICS = (("s", "S (km/s)"), ("timse", "TIMSE ((km/s)²)"), ("rmse", "RMSE (km/s)"))


def _filename(record: dict) -> str:
    def safe(value, limit=60):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "legacy")).strip("._-")[:limit] or "legacy"
    return (f"{safe(record.get('team_member'))}--{safe(record.get('experiment'))}"
            f"--{safe(record.get('config_sha256'), 64)}.png")


def generate_plot(record: dict, output_dir: str | Path) -> Path:
    """Write one deterministic 3-panel PNG named by the submission identity."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    competition = record["competition"]
    series = {}
    for scope, _, _ in SCOPES:
        rows = competition[scope]["by_lead"]
        leads = [row["lead_hours"] for row in rows]
        values = np.asarray([[row[key] for key, _ in METRICS] for row in rows], dtype=float)
        if leads != list(range(1, 73)) or values.shape != (72, 3):
            raise ValueError(f"{scope}: expected exactly the 72 ordered lead hours")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{scope}: curves must have finite nonnegative values")
        series[scope] = values
    target = Path(output_dir) / _filename(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A fixed style and PNG metadata avoid timestamps and user matplotlib styles.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "text.color": "#171717", "axes.labelcolor": "#171717",
                         "xtick.color": "#171717", "ytick.color": "#171717"}):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
        try:
            leads = np.arange(1, 73)
            for column, (key, label) in enumerate(METRICS):
                ax = axes[column]
                for scope, name, color in SCOPES:
                    ax.plot(leads, series[scope][:, column], label=name, color=color,
                            linewidth=2.2 if scope == "total" else 1.3)
                ax.set_ylabel(label)
                ax.grid(alpha=0.22)
                ax.set_xlim(1, 72)
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[0].legend(ncol=4, loc="best", frameon=False)
            axes[-1].set_xticks([1, 12, 24, 36, 48, 60, 72])
            axes[-1].set_xlabel("Lead time (hours)")
            title = str(record.get("experiment", "Forecast"))
            fig.suptitle(title, fontsize=11, wrap=True)
            fig.savefig(target, dpi=140, metadata={"Software": PLOT_VERSION})
        finally:
            plt.close(fig)
    return target


def _github_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n", text=True,
                            capture_output=True, timeout=20, check=False,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode == 0:
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if fields.get("password"):
            return fields["password"]
    raise RuntimeError("GitHub credentials unavailable: set GH_TOKEN/GITHUB_TOKEN or configure git credentials")


class GitHubAPI:
    def __init__(self, repo: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Expected a GitHub owner/repository name")
        self.repo = repo
        self.token = _github_token()

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(f"https://api.github.com/repos/{self.repo}/{path}",
                                         data=data, method=method,
                                         headers={"Authorization": f"Bearer {self.token}",
                                                  "Accept": "application/vnd.github+json",
                                                  "X-GitHub-Api-Version": "2022-11-28",
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)


def _remote_assets(api: GitHubAPI, root_tree: str) -> dict[str, str]:
    """Return {name: size} for the image directory, avoiding truncated recursive trees."""
    tree_sha = root_tree
    for component in ASSET_DIRECTORY.split("/"):
        tree = api.request("GET", f"git/trees/{tree_sha}")
        if tree.get("truncated"):
            raise RuntimeError("GitHub returned a truncated tree; image publication stopped")
        entry = next((item for item in tree["tree"]
                      if item["path"] == component and item["type"] == "tree"), None)
        if entry is None:
            return {}
        tree_sha = entry["sha"]
    tree = api.request("GET", f"git/trees/{tree_sha}")
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated image tree")
    return {item["path"]: item.get("size") for item in tree["tree"] if item["type"] == "blob"}


def publish_plots(records: list[dict], output_dir: str | Path, repo: str = GITHUB_REPO,
                  dry_run: bool = False) -> list[dict]:
    """Return copied records with pinned metric_plot_url values after one batch commit.

    Dry runs generate local images and return metric_plot_path, never contact
    GitHub, and do not claim a published URL. Records without competition data
    pass through unchanged. A non-fast-forward branch race is retried three times.
    """
    output = [dict(record) for record in records]
    paths = {}
    for index, record in enumerate(output):
        if record.get("competition"):
            paths[index] = generate_plot(record, output_dir)
    if dry_run:
        for index, path in paths.items():
            output[index].pop("metric_plot_url", None)
            output[index]["metric_plot_path"] = str(path)
        return output
    if not paths:
        return output
    api = GitHubAPI(repo)
    files = {path.name: path.read_bytes() for path in paths.values()}
    blob_ids = {}
    for attempt in range(3):
        head = api.request("GET", "git/ref/heads/main")["object"]["sha"]
        root = api.request("GET", f"git/commits/{head}")["tree"]["sha"]
        existing = _remote_assets(api, root)
        changed = {name: data for name, data in files.items() if existing.get(name) != len(data)}
        if not changed:
            commit_sha = head
            break
        for name, data in changed.items():
            if name not in blob_ids:
                blob_ids[name] = api.request("POST", "git/blobs", {"content": base64.b64encode(data).decode(),
                                                                   "encoding": "base64"})["sha"]
        tree = api.request("POST", "git/trees", {"base_tree": root, "tree": [
            {"path": f"{ASSET_DIRECTORY}/{name}", "mode": "100644", "type": "blob", "sha": blob_ids[name]}
            for name in sorted(changed)]})
        commit = api.request("POST", "git/commits", {"message": "Leaderboard: publish lead metric curves",
                                                      "tree": tree["sha"], "parents": [head]})
        commit_sha = commit["sha"]
        try:
            api.request("PATCH", "git/refs/heads/main", {"sha": commit_sha, "force": False})
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (409, 422) or attempt == 2:
                raise
    for index, path in paths.items():
        output[index].pop("metric_plot_path", None)
        output[index]["metric_plot_url"] = (
            f"https://raw.githubusercontent.com/{repo}/{commit_sha}/{ASSET_DIRECTORY}/{path.name}")
    return output
