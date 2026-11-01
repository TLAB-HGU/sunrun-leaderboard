"""Rescore every stored submission against the F1/F2/F3 evaluation sets (2026-10-31 re-split).

    python rescore.py --sets DIR --out DIR [--limit N]            # dry run: writes rescored result JSONs under --out
    python rescore.py --sets DIR --out DIR --apply                # also commits sets + results to the dataset

--sets is the directory written by rebuild_eval_sets.py (folds/truth/naive parquet). Each stored prediction is
scored against the new sets with the shared scorer, skipping validate(): predictions from before the re-split
start in March and skip the origins of 2026-05-29..05-31, so F1 (and F2 by about 3%) is only partly covered.
The scorer reports that as coverage_f1/f2/f3, which board.py uses to list such rows below the complete ones.
The old scores are kept in each record as `legacy_f1f2`. --apply writes folds/truth/naive and all result
files in ONE dataset commit (atomic), after saving the current versions under --out/backup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "scorer")]
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402
from scoring import KEY, score  # noqa: E402

DATASET = "tlabtlab/sunrun-lb-store"


def rescore_record(old: dict, pred: pd.DataFrame, truth, folds, naive, stamp: str) -> dict:
    """New result record: identity fields of `old`, scores recomputed, old scores kept as legacy_f1f2."""
    new = score(pred[KEY + ["pred_kms"]], truth, folds, naive)
    kept = {k: v for k, v in old.items() if k not in new and k not in ("legacy_f1f2", "rescored_utc")}
    return {**kept, **new, "legacy_f1f2": old.get("legacy_f1f2") or {k: old[k] for k in old if k in new},
            "rescored_utc": stamp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="substring of the result path to rescore")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    sets, out = Path(a.sets), Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "backup").mkdir(exist_ok=True)
    folds, truth, naive = (pd.read_parquet(sets / f"{n}.parquet") for n in ("folds", "truth", "naive"))

    api = HfApi()
    head = api.repo_info(a.dataset, repo_type="dataset").sha
    files = set(api.list_repo_files(a.dataset, repo_type="dataset", revision=head))
    results = sorted(f for f in files if f.startswith("results/") and f.endswith(".json"))
    if a.only:
        results = [f for f in results if a.only in f]
    if a.limit:
        results = results[:a.limit]
    print(f"dataset head {head[:12]}: {len(results)} result files to rescore", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done, skipped = {}, []
    for i, path in enumerate(results, 1):
        sub = "submissions/" + path[len("results/"):-len(".json")] + ".parquet"
        old = json.loads(Path(hf_hub_download(a.dataset, path, repo_type="dataset", revision=head)).read_text())
        (out / "backup" / path.replace("/", "__")).write_text(json.dumps(old, ensure_ascii=False))
        if sub not in files:
            skipped.append((path, "no stored predictions"))
            continue
        pred = pd.read_parquet(hf_hub_download(a.dataset, sub, repo_type="dataset", revision=head))
        t0 = time.time()
        try:
            rec = rescore_record(old, pred, truth, folds, naive, stamp)
        except Exception as exc:  # keep going; the record stays as it was (legacy on the board)
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        done[path] = rec
        (out / "results" / path.replace("/", "__")).write_text(json.dumps(rec, ensure_ascii=False))
        cov = "/".join(f"{rec[f'coverage_{p}']:.3f}" for p in ("f1", "f2", "f3"))
        print(f"[{i}/{len(results)}] {path}  mse_f2={rec['mse_f2']:.1f} mse_f3={rec['mse_f3']:.1f} "
              f"coverage f1/f2/f3={cov}  ({time.time() - t0:.0f}s)", flush=True)
    (out / "skipped.json").write_text(json.dumps(skipped, indent=1))
    print(f"rescored {len(done)}, skipped {len(skipped)}: {skipped}", flush=True)
    if not a.apply:
        print("dry run: nothing written to the dataset")
        return

    # one atomic commit: evaluation sets + every rescored result; old versions are in --out/backup
    if api.repo_info(a.dataset, repo_type="dataset").sha != head:
        raise SystemExit("dataset changed while rescoring; rerun so no newer submission is overwritten")
    ops = [CommitOperationAdd(f"{n}.parquet", str(sets / f"{n}.parquet")) for n in ("folds", "truth", "naive")]
    ops += [CommitOperationAdd(p, json.dumps(r, ensure_ascii=False).encode()) for p, r in done.items()]
    api.create_commit(repo_id=a.dataset, repo_type="dataset", parent_commit=head, operations=ops,
                      commit_message="Re-split evaluation into F1 (01-03), F2 (04-06), F3 (07-09) and rescore stored predictions")
    print(f"committed {len(ops)} files to {a.dataset}")


if __name__ == "__main__":
    main()
