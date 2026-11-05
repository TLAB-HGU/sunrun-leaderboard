"""Back up and rescore stored forecasts, then publish versioned metrics and PNGs.

First run without --apply to inspect scores and preview.html. Existing evaluation
sets and prediction files remain unchanged. Dataset commits use a head lease.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "scorer")]
from board import render
from competition import VERSION, prepare_context, score_competition
from competition_data import download_observations
from metric_assets import publish_plots
from submit import publish

DATASET = "tlabtlab/sunrun-lb-store"
SPACE = "tlabtlab/sunrun-leaderboard"


def prediction_path(result_path: str, files: set[str]) -> str | None:
    stem = "submissions/" + result_path.removeprefix("results/").removesuffix(".json")
    return next((stem + ext for ext in (".parquet", ".csv") if stem + ext in files), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    observations, provenance = download_observations(out.parent / "monthly-txt")
    context = prepare_context(observations)
    observations_path = out / "competition-observations.parquet"
    observations.to_parquet(observations_path, index=False)
    provenance.update(version=VERSION, n_origins=len(context["origins"]),
                      origins_by_period={name: int((context["period"] == name).sum())
                                         for name in ("F1", "F2", "F3")})
    api = HfApi()
    head = api.repo_info(args.dataset, repo_type="dataset").sha
    files = set(api.list_repo_files(args.dataset, repo_type="dataset", revision=head))
    paths = sorted(path for path in files if path.startswith("results/") and path.endswith(".json"))
    backup = out / "backup" / head
    backup.mkdir(parents=True, exist_ok=True)
    print(f"Pinned dataset {head}; {len(paths)} records; {len(context['origins'])} six-hour origins", flush=True)

    def fetch(path):
        cached = backup / path
        if not cached.exists():
            source = Path(hf_hub_download(args.dataset, path, repo_type="dataset", revision=head))
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(source.read_bytes())
        return cached

    jobs = sorted(set(paths + [p for path in paths if (p := prediction_path(path, files))]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, jobs))
    records, missing = [], []
    for index, path in enumerate(paths, 1):
        old = json.loads(fetch(path).read_text())
        pred_path = prediction_path(path, files)
        record = dict(old)
        if pred_path is None:
            record.pop("competition", None)
            record.pop("metric_plot_url", None)
            record["competition_unscored_reason"] = "stored predictions unavailable"
            missing.append(path)
        else:
            cached = fetch(pred_path)
            pred = pd.read_parquet(cached) if pred_path.endswith(".parquet") else pd.read_csv(cached)
            # Historical results stay intact; new scores live in a versioned namespace.
            record["competition"] = score_competition(pred, context)
            record.pop("competition_unscored_reason", None)
            print(f"[{index}/{len(paths)}] {old['experiment']}: S={record['competition']['total']['s']:.3f}", flush=True)
        records.append(record)
    records = publish_plots(records, out / "images", dry_run=not args.apply)
    preview_records, local_images = [], {}
    for record in records:
        preview = dict(record)
        if record.get("metric_plot_path"):
            path = Path(record["metric_plot_path"]).resolve()
            placeholder = ("https://raw.githubusercontent.com/TLAB-HGU/sunrun-leaderboard/"
                           + "0" * 40 + "/leaderboard/assets/metrics/" + path.name)
            preview["metric_plot_url"] = placeholder
            local_images[placeholder] = path.as_uri()
        preview_records.append(preview)
    page = render(preview_records)
    for placeholder, local in local_images.items():
        page = page.replace(placeholder, local)
    (out / "preview.html").write_text(page)
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, allow_nan=False))
    receipt = {"dataset_parent": head, "version": VERSION, "records": len(records), "unscored": missing,
               "provenance": provenance, "applied": False}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    if not args.apply:
        print(f"Dry run complete: {out / 'preview.html'}; unscored={len(missing)}", flush=True)
        return
    space_head = api.repo_info(args.space, repo_type="space").sha
    old_page = Path(hf_hub_download(args.space, "index.html", repo_type="space", revision=space_head))
    (out / f"space-before-{space_head}.html").write_bytes(old_page.read_bytes())
    operations = [CommitOperationAdd("competition-observations.parquet", str(observations_path)),
                  CommitOperationAdd("competition-provenance.json", json.dumps(provenance, indent=2).encode())]
    operations += [CommitOperationAdd(path, json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
                   for path, record in zip(paths, records)]
    commit = api.create_commit(repo_id=args.dataset, repo_type="dataset", parent_commit=head,
                               operations=operations, commit_message="Leaderboard: six-hour lead-first S/TIMSE/RMSE")
    receipt.update(applied=True, dataset_commit=commit.oid)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    publish(api, [], args.space, dataset=args.dataset)
    print(f"Published {len(records)} records; unscored={len(missing)}; {SPACE}", flush=True)


if __name__ == "__main__":
    main()
