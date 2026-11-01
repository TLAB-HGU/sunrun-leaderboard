"""Complete the five baseline submissions on the F1/F2/F3 folds (2026-10-31 re-split).

    python fill_baselines.py --sets DIR --out DIR [--apply]

The stored baselines miss 1,416 F1 origins (2025-12-31T23Z..2026-02-28T22Z) and 71 origins of 2026-05-29..05-31.
Each baseline is recomputed with the configuration recorded in its result file (naive, seasonal naive 648,
mean reversion tau 48 / window 648, SES and AutoARIMA on the trailing 648 h) for the missing origins only, and the
recomputation is checked against the stored predictions on sampled overlapping origins before the two are merged.
--apply replaces each submission + result in one dataset commit; the previous versions are saved under --out/backup.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scorer")]
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402
from rescore import DATASET, rescore_record  # noqa: E402
from scoring import HORIZON, KEY, validate  # noqa: E402

WINDOW, TAU = 648, 48.0
NAMES = ("naive", "seasonal_naive_648", "mean_reversion", "ses", "auto_arima")
ARIMA = dict(season_length=1, seasonal=False, allowdrift=True, allowmean=True, approximation=False, ic="aicc",
             max_d=2, max_p=5, max_q=5, nmodels=94, stepwise=True, test="kpss")
H = np.arange(1, HORIZON + 1)


def _fit(args):
    kind, window = args
    from statsforecast.models import AutoARIMA, SimpleExponentialSmoothingOptimized
    model = SimpleExponentialSmoothingOptimized() if kind == "ses" else AutoARIMA(**ARIMA)
    return model.forecast(y=window, h=HORIZON)["mean"]


def compute(name: str, vals: np.ndarray, pos: np.ndarray, workers: int) -> np.ndarray:
    """(n_origins, 72) forecasts for the origins at integer positions `pos` of the hourly series `vals`."""
    if name == "naive":
        return np.repeat(vals[pos][:, None], HORIZON, axis=1)
    if name == "seasonal_naive_648":
        return vals[pos[:, None] + H[None, :] - 648]
    if name == "mean_reversion":
        mean = pd.Series(vals).rolling(WINDOW).mean().to_numpy()[pos]
        return mean[:, None] + (vals[pos] - mean)[:, None] * np.exp(-H / TAU)[None, :]
    jobs = [(name, vals[p + 1 - WINDOW:p + 1].astype(float)) for p in pos]  # workers see only data up to the origin
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return np.vstack(list(pool.map(_fit, jobs, chunksize=8)))


def frame(origins: pd.DatetimeIndex, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON),
                         "horizon_hours": np.tile(H, len(origins)), "pred_kms": pred.ravel()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=120, help="overlapping origins used to verify each recomputation")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    sets, out = Path(a.sets), Path(a.out)
    (out / "backup").mkdir(parents=True, exist_ok=True)
    folds, truth, naive = (pd.read_parquet(sets / f"{n}.parquet") for n in ("folds", "truth", "naive"))
    raw = pd.read_csv(HERE / "data" / "hourly_series.csv")
    series = raw.set_index(pd.to_datetime(raw["timestamp_utc"], utc=True))["filled_speed_kms"]
    vals, index = series.to_numpy(), series.index

    api = HfApi()
    head = api.repo_info(DATASET, repo_type="dataset").sha
    results = {p.split("/")[2]: p for p in api.list_repo_files(DATASET, repo_type="dataset", revision=head)
               if p.startswith("results/baseline/") and p.endswith(".json")}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ops, summary = [], {}
    for name in NAMES:
        rpath = results[name]
        spath = "submissions/" + rpath[len("results/"):-len(".json")] + ".parquet"
        old = json.loads(Path(hf_hub_download(DATASET, rpath, repo_type="dataset", revision=head)).read_text())
        stored = pd.read_parquet(hf_hub_download(DATASET, spath, repo_type="dataset", revision=head))
        (out / "backup" / f"{name}.json").write_text(json.dumps(old, ensure_ascii=False))
        stored.to_parquet(out / "backup" / f"{name}.parquet", index=False)
        have = set(stored["origin_last_input_utc"])
        wanted = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
        is_missing = ~wanted.strftime("%Y-%m-%dT%H:%M:%SZ").isin(have)
        todo, overlap = wanted[is_missing], wanted[~is_missing]
        t0 = time.time()
        new = compute(name, vals, index.get_indexer(todo), a.workers)
        # recomputation must reproduce what is already stored (sampled evenly over the overlapping origins)
        sample = overlap[np.linspace(0, len(overlap) - 1, a.check, dtype=int)]
        again = frame(sample, compute(name, vals, index.get_indexer(sample), a.workers))
        ref = stored.merge(again, on=KEY, suffixes=("", "_new"))
        per_origin = (ref["pred_kms"] - ref["pred_kms_new"]).abs().groupby(ref["origin_last_input_utc"]).max()
        diff, exact = float(per_origin.max()), float((per_origin < 1e-3).mean())
        # deterministic baselines must match exactly; AutoARIMA's stepwise search can pick another model on a few
        # origins with a different statsforecast build, so it needs >= 95% of the sampled origins identical
        assert len(per_origin) == a.check and exact >= (0.95 if name == "auto_arima" else 1.0), \
            f"{name}: recomputation differs from stored on {1 - exact:.1%} of origins (max {diff})"
        pred = pd.concat([stored[stored["origin_last_input_utc"].isin(set(wanted.strftime('%Y-%m-%dT%H:%M:%SZ')))],
                          frame(todo, new)], ignore_index=True).sort_values(KEY).reset_index(drop=True)
        errs = validate(pred, folds)
        assert not errs, f"{name}: {errs}"
        rec = rescore_record(old, pred, truth, folds, naive, stamp)
        rec["filled_utc"] = stamp
        rec["filled_origins"] = int(len(todo))
        rec["fill_check"] = {"sampled_origins": a.check, "identical_share": exact, "max_abs_diff_kms": diff}
        summary[name] = {"added_origins": int(len(todo)), "identical_share": exact, "max_diff_vs_stored": diff, "seconds": round(time.time() - t0),
                         **{k: rec[k] for k in ("mse", "mse_f1", "mse_f2", "mse_f3", "coverage_f1", "coverage_f2", "coverage_f3")}}
        print(name, json.dumps(summary[name]), flush=True)
        buf = pred.to_parquet(None, index=False)
        ops += [CommitOperationAdd(spath, buf), CommitOperationAdd(rpath, json.dumps(rec, ensure_ascii=False).encode())]
        (out / f"{name}.parquet").write_bytes(buf)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not a.apply:
        print("dry run: nothing written to the dataset")
        return
    if api.repo_info(DATASET, repo_type="dataset").sha != head:
        raise SystemExit("dataset changed while computing; rerun so no newer submission is overwritten")
    api.create_commit(repo_id=DATASET, repo_type="dataset", parent_commit=head, operations=ops,
                      commit_message="Complete the five baselines on F1 (Jan-Mar) and the 2026-05-29..31 origins")
    print(f"committed {len(ops)} files")


if __name__ == "__main__":
    main()
