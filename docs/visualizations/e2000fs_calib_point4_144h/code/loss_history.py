"""Per-round training and held-out loss of the nine e2000fs_calib_point4_144h boosters.

The submission is the point4 144 h control arm of the tweedie15 lane: `reg:squarederror` with a 4x weight on targets
>= 550 km/s (weights normalised by the batch mean per horizon), one `multi_output_tree` booster per 24 h block, 900
rounds, seed 42, training mask `origin + 144 h <= cutoff`. The archived run kept the predictions but not
`evals_result`, so this script re-runs the identical fit with an evaluation set attached and records the per-round
curve. Attaching an eval set without early stopping cannot change the model, so the refit must reproduce the archived
`F*_part{b}.npz` predictions exactly; the script stops otherwise.

    python loss_history.py --period F1 --block 0        # one booster -> loss_history/f1_b0.csv
    python loss_history.py --aggregate                  # nine CSVs -> loss_history.png + loss_receipt.json

The validation curve is the official period (F1 2026-01..03, F2 04..06, F3 07..09), which lies entirely outside the
training mask but is itself a published, already-scored window: the curve is a diagnostic and never chose the round.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import vizlib as V  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
STUDY = V.ROOT / "research/e2000fs_f123-20261031"
RUN = STUDY / "runs/point4_144h_v1"
HISTORY = OUT / "loss_history"
PERIODS = ("F1", "F2", "F3")
BLOCKS = (0, 1, 2)
PERIOD_WINDOW = {"F1": "2026-01-01 .. 2026-03-31", "F2": "2026-04-01 .. 2026-06-30", "F3": "2026-07-01 .. 2026-09-29"}
COLORS = ("#2f5f98", "#c2553f", "#4f8a4f")

sys.path.insert(0, str(STUDY))
import arms as A  # noqa: E402


def point4_rmse(pred, matrix):
    """Per-horizon-normalised point4 weighted RMSE in km/s (the objective's own weighting)."""
    target = matrix.get_label().reshape(pred.shape).astype(np.float64)
    weight = A.point_weights(target)
    return "point4_rmse", float(np.sqrt((((pred - target) ** 2) * weight).mean()))


def run_one(period: str, block: int, threads: int, rounds: int, verify: bool = True) -> dict:
    stage = A.Stage(period)
    y, x, xq = stage.y[:, A.sl(block)], stage.x[block], stage.xq[block]
    truth = stage.truth[:, A.sl(block)]
    params = A.params_of(threads, base_score=float(np.nanmean(y)))
    train = xgb.DMatrix(x, label=y.astype(np.float32))
    valid = xgb.DMatrix(xq, label=truth.astype(np.float32))
    history: dict = {}
    booster = xgb.train(dict(params, nthread=threads), train, num_boost_round=rounds, obj=A.obj_point4,
                        evals=[(train, "train"), (valid, "valid")], custom_metric=point4_rmse,
                        evals_result=history, verbose_eval=False)

    pred = np.clip(booster.inplace_predict(xq), 1.0, 3000.0)
    archived = np.load(RUN / f"{period}_part{block}.npz")["pred"][:, A.sl(block)]
    difference = float(np.abs(pred - archived).max())
    if verify:
        assert difference == 0.0, f"refit does not reproduce {period}_part{block}.npz (max |diff| {difference})"

    frame = pd.DataFrame({"round": np.arange(1, rounds + 1),
                          "train_rmse": history["train"]["rmse"], "valid_rmse": history["valid"]["rmse"],
                          "train_point4_rmse": history["train"]["point4_rmse"],
                          "valid_point4_rmse": history["valid"]["point4_rmse"]})
    HISTORY.mkdir(exist_ok=True)
    frame.to_csv(HISTORY / f"{period.lower()}_b{block}.csv", index=False)
    best = int(frame["valid_point4_rmse"].idxmin()) + 1
    receipt = {"period": period, "block": block, "threads": threads, "rounds": rounds,
               "train_rows": int(len(stage.idx)), "last_train_origin_utc": stage.origins.max().isoformat(),
               "cutoff_utc": stage.cutoff.isoformat(), "queries": int(len(stage.q_origins)),
               "validation_window": PERIOD_WINDOW[period],
               "final": {"train_rmse": float(frame["train_rmse"].iloc[-1]),
                         "valid_rmse": float(frame["valid_rmse"].iloc[-1]),
                         "train_point4_rmse": float(frame["train_point4_rmse"].iloc[-1]),
                         "valid_point4_rmse": float(frame["valid_point4_rmse"].iloc[-1])},
               "valid_point4_rmse_min": float(frame["valid_point4_rmse"].min()), "valid_min_round": best,
               "reproduces_archive": True, "max_abs_difference_kms": difference,
               "archived_part": {"file": str(RUN / f"{period}_part{block}.npz")}}
    (HISTORY / f"{period.lower()}_b{block}.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("period", "block", "train_rows", "queries")}
                     | {"valid_point4_rmse": round(receipt["final"]["valid_point4_rmse"], 4),
                        "min_round": best}), flush=True)
    return receipt


def aggregate(rounds: int) -> None:
    frames, receipts = [], []
    for period in PERIODS:
        for block in BLOCKS:
            path = HISTORY / f"{period.lower()}_b{block}.csv"
            receipt = json.loads((HISTORY / f"{period.lower()}_b{block}.json").read_text())
            assert len(pd.read_csv(path)) == receipt["rounds"]
            receipts.append(receipt)
            frames.append(pd.read_csv(path).assign(period=period, block=block))
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(OUT / "loss_history.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(12.6, 9.2), sharex=True, constrained_layout=True)
    for row, period in enumerate(PERIODS):
        for column, (split, metric) in enumerate((("train", "train_point4_rmse"), ("valid", "valid_point4_rmse"))):
            axis = axes[row, column]
            for block, color in zip(BLOCKS, COLORS):
                part = long[(long["period"] == period) & (long["block"] == block)]
                axis.plot(part["round"], part[metric], color=color, lw=1.2, label=f"b{block} h{block * 24 + 1}–{block * 24 + 24}")
            axis.grid(alpha=.25)
            axis.set_ylabel("point4 RMSE (km/s)", fontsize=9)
            if row == 0:
                axis.set_title(f"{split} — {'training rows' if split == 'train' else 'held-out period'}",
                               fontsize=10.5, loc="left")
            if column == 1:
                axis.legend(fontsize=8, loc="upper right", framealpha=.9)
            if row == 2:
                axis.set_xlabel("boosting round", fontsize=9.5)
    for row, period in enumerate(PERIODS):
        axes[row, 0].text(-0.16, 0.5, f"{period}\n{PERIOD_WINDOW[period]}", transform=axes[row, 0].transAxes,
                          fontsize=10, rotation=90, va="center", ha="center")
    fig.text(.012, -.015, "point4 RMSE = sqrt(mean(w (pred - truth)^2)) with w = 4 for targets >= 550 km/s and 1 "
                          "otherwise, normalised per horizon. Refits reproduce the archived boosters exactly "
                          "(max |diff| 0.0); no early stopping, 900 rounds fixed by the recipe.",
             fontsize=7.6, color="#555555")
    fig.savefig(OUT / "loss_history.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([{**{k: r[k] for k in ("period", "block", "train_rows", "queries", "rounds")},
                             **r["final"], "valid_point4_rmse_min": r["valid_point4_rmse_min"],
                             "valid_min_round": r["valid_min_round"],
                             "max_abs_difference_kms": r["max_abs_difference_kms"]} for r in receipts])
    summary.to_csv(OUT / "loss_summary.csv", index=False)
    payload = {"experiment": "e2000fs_calib_point4_144h", "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "recipe": {"objective": "reg:squarederror with 4x weight on targets >= 550 km/s, per-horizon normalised",
                          "rounds": rounds, "threads": 1, "seed": 42, "tree": "multi_output_tree, 24 outputs per block",
                          "training_mask": "origin + 144 h <= cutoff; cutoffs 2025-12-29 / 2026-03-29 / 2026-06-29"},
               "validation": "each period's official fold window; outside the training mask but already published",
               "reproduces_archive": all(r["reproduces_archive"] for r in receipts),
               "max_abs_difference_kms": max(r["max_abs_difference_kms"] for r in receipts),
               "blocks": receipts, "outputs": ["loss_history.csv", "loss_summary.csv", "loss_history.png"]}
    (OUT / "loss_receipt.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", choices=PERIODS)
    parser.add_argument("--block", type=int, choices=BLOCKS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=900)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="short run without the archive reproduction check")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.rounds)
        return
    assert args.period is not None and args.block is not None, "give --period and --block, or --aggregate"
    run_one(args.period, args.block, args.threads, args.rounds, verify=not args.smoke)


if __name__ == "__main__":
    main()
