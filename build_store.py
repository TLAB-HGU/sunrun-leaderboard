"""Build the leaderboard store: folds, truth, naive reference and baseline submissions.

    python build_store.py --out store [--end 2026-09-26]

`--end` is the exclusive target end (default: full evaluation end from labeling/config.py).
Baselines come from analysis/naive_baselines (Naive, Seasonal naive-648, SES, Auto ARIMA) plus
one new mean-reversion baseline whose decay time is tuned on targets BEFORE the evaluation start.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "scorer"), str(HERE / "labeling")]
from config import EVAL_END, EVAL_START  # noqa: E402
from regimes import build_folds  # noqa: E402
from scoring import HORIZON, KEY, build_truth, score, validate  # noqa: E402

BASE = HERE.parents[0] / "analysis" / "naive_baselines"
SERIES = HERE / "data" / "hourly_series.csv"  # written by refresh_data.py
EXTRA = HERE / "data" / "extra_es_arima.parquet"  # written by extend_baselines.py (origins after the study)
MEAN_WINDOW = 648
TAUS = [12, 24, 48, 72, 120, 240]


def mean_reversion(series: pd.Series, origins: pd.DatetimeIndex, tau: float) -> pd.DataFrame:
    last = series.loc[origins].to_numpy()
    mean = series.rolling(MEAN_WINDOW).mean().loc[origins].to_numpy()
    h = np.arange(1, HORIZON + 1)
    pred = mean[:, None] + (last - mean)[:, None] * np.exp(-h / tau)[None, :]
    return pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON),
                         "horizon_hours": np.tile(h, len(origins)), "pred_kms": pred.ravel()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "store"))
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "baselines").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SERIES)
    series = raw.set_index(pd.to_datetime(raw["timestamp_utc"], utc=True))["filled_speed_kms"]
    folds = build_folds(raw, a.start, a.end)
    truth = build_truth(raw, folds)

    fold_origins = pd.DatetimeIndex(pd.to_datetime(folds["origin_last_input_utc"], utc=True))
    h = np.arange(1, HORIZON + 1)
    keys = pd.DataFrame({"origin_last_input_utc": np.repeat(fold_origins.strftime("%Y-%m-%dT%H:%M:%SZ"), HORIZON), "horizon_hours": np.tile(h, len(fold_origins))})
    vals = series.to_numpy()
    pos = series.index.get_indexer(fold_origins)
    assert (pos >= 0).all()
    preds = {
        "naive": keys.assign(pred_kms=np.repeat(vals[pos], HORIZON)),
        "seasonal_naive_648": keys.assign(pred_kms=vals[pos[:, None] + h[None, :] - 648].ravel()),
    }
    # SES / Auto ARIMA: fitted study predictions + the extension for later origins
    ext = pd.concat([pd.read_csv(BASE / "extended" / "predictions.csv.gz")[["origin_last_input_utc", "horizon_hours", "naive_kms", "seasonal_naive_648_kms", "es_kms", "auto_arima_kms"]],
                     pd.read_parquet(EXTRA)], ignore_index=True)
    old = ext.dropna(subset=["naive_kms"])
    chk = keys.merge(old, on=KEY).merge(preds["naive"], on=KEY).merge(preds["seasonal_naive_648"], on=KEY, suffixes=("", "_s"))
    assert np.allclose(chk["naive_kms"], chk["pred_kms"]) and np.allclose(chk["seasonal_naive_648_kms"], chk["pred_kms_s"]), "recomputed naive/seasonal differ from study"
    print(f"naive/seasonal recomputation matches the study on {chk['origin_last_input_utc'].nunique()} shared origins")
    for k, c in {"ses": "es_kms", "auto_arima": "auto_arima_kms"}.items():
        preds[k] = keys.merge(ext[KEY + [c]], on=KEY, how="left").rename(columns={c: "pred_kms"})

    # tune the mean-reversion decay on targets strictly before the evaluation start (no eval peeking)
    dev_origins = pd.date_range(pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp(a.start, tz="UTC") - pd.Timedelta(hours=HORIZON + 1), freq="h")
    dev_origins = dev_origins[dev_origins >= series.index[MEAN_WINDOW - 1]]
    dev_truth = np.stack([series.loc[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=HORIZON)].to_numpy() for o in dev_origins])
    dev_mse = {t: float(((mean_reversion(series, dev_origins, t)["pred_kms"].to_numpy().reshape(-1, HORIZON) - dev_truth) ** 2).mean()) for t in TAUS}
    tau = min(dev_mse, key=dev_mse.get)
    preds["mean_reversion"] = mean_reversion(series, fold_origins, tau)

    naive = preds["naive"]
    folds.to_parquet(out / "folds.parquet", index=False)
    truth.to_parquet(out / "truth.parquet", index=False)
    naive[naive["origin_last_input_utc"].isin(folds["origin_last_input_utc"])].to_parquet(out / "naive.parquet", index=False)
    summary = {"mean_reversion_tau": tau, "dev_mse_by_tau": dev_mse, "n_folds": len(folds), "baselines": {}}
    naive_eval = pd.read_parquet(out / "naive.parquet")
    for k, p in preds.items():
        p = p[p["origin_last_input_utc"].isin(folds["origin_last_input_utc"])].reset_index(drop=True)
        errs = validate(p, folds)
        if errs:
            raise SystemExit(f"baseline {k} failed validation: {errs}")
        p.to_parquet(out / "baselines" / f"{k}.parquet", index=False)
        r = score(p, truth, folds, naive_eval)
        summary["baselines"][k] = {"mse": r["mse"], "mse_observed": r["mse_observed"], "skill_vs_naive": r["skill_vs_naive"], "vs_naive": r["vs_naive"]}
        print(f"{k:20s} mse={r['mse']:10.1f} observed={r['mse_observed']:10.1f} skill={r['skill_vs_naive']:+.4f} beats_naive={r['vs_naive']['beats_reference']}")
    (out / "baselines" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"tau={tau} (dev mse {dev_mse}); {len(folds)} folds -> {out}")


if __name__ == "__main__":
    main()
