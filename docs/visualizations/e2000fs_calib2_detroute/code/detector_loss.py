"""Record the per-round training and validation loss of the six submitted detector boosters.

The archived run kept the boosters (`detector_<stage>_b<band>.ubj`) and the exceedance probabilities but not
`evals_result`, so this script re-runs the frozen recipe once per booster with an eval set attached. Training is
unchanged by an eval set (no early stopping, no metric weights), so the refit must reproduce the archived
probability matrix exactly; the script stops otherwise. Nothing is selected with these curves — the 400 rounds are
the recipe and tau=0.234 was fixed a priori from the cost ratio.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib2_detroute/code/detector_loss.py
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detector_common import (BANDS, LANE, OUT, ROUNDS, RUN, STAGES, TAU, archived_booster,
                             archived_probabilities, load_features, official_arrays, receipt, refit_detector,
                             routed_score, sha)

BAND_LABEL = {0: "b0 · h1–24", 24: "b1 · h25–48", 48: "b2 · h49–72"}
BAND_COLOR = {0: "#1f77b4", 24: "#e8620c", 48: "#2ca02c"}
STAGE_LABEL = {"f1": "F1 model (target < 2026-02-26) · validated on F1 2026-03…05",
               "final": "F2 model (target < 2026-05-29) · validated on F2 2026-06…09"}
STAGE_COLOR = {"f1": "#1f77b4", "final": "#e8620c"}
MARKS = (25, 50, 100, 150, 200, 250, 300, 350, 400)


def main():
    started = time.time()
    data = load_features()
    official = official_arrays(data)
    probability_archived, probability_origins = archived_probabilities()
    assert probability_origins.equals(official["origins"])
    n_f1 = official["n_f1"]
    routed_archived = np.where(probability_archived >= TAU, official["tail"], official["body"])
    session = np.load(RUN / "forecast.npz")["predictions"].astype(np.float64)
    assert np.abs(routed_archived - session).max() == 0.0, "routing replay differs from the submitted forecast"
    scores_archived = routed_score(session, official["target"], n_f1)
    assert abs(scores_archived["total"] - receipt()["mse"]) < 1e-9

    history, summary, frames, refit = {}, [], {}, {}
    for stage in STAGES:
        staged = {k: np.zeros((len(data["y_f1"]) if stage == "f1" else len(data["y_f2"]), 72))
                  for k in (0,) + MARKS}
        for start in BANDS:
            name = f"{stage}_b{BANDS.index(start)}"
            tuning = refit_detector(data, stage, start, evals=True)
            booster = archived_booster(stage, start)
            # The refit must reproduce the archived booster on the applied rows, and through it the submitted
            # probability matrix; anything else means the recorded loss belongs to a different model.
            distance = float(np.max(np.abs(tuning["model"].inplace_predict(tuning["x_apply"]) -
                                            booster.inplace_predict(tuning["x_apply"]))))
            assert distance == 0.0, f"{name}: refit differs from the archived booster by {distance}"
            curve = tuning["history"]
            assert curve["round"].tolist() == list(range(1, ROUNDS + 1))
            curve.insert(0, "stage", stage)
            curve.insert(1, "band", start)
            curve.to_csv(OUT / "loss_history" / f"{name}.csv", index=False)
            history[name] = curve
            refit[name] = dict(max_abs_diff=distance, train_rows=tuning["train_rows"],
                               apply_rows=tuning["apply_rows"], base_rate=float(tuning["labels_apply"].mean()),
                               train_logloss=float(curve["train_logloss"].iloc[-1]),
                               validation_logloss=float(curve["validation_logloss"].iloc[-1]),
                               validation_logloss_min=float(curve["validation_logloss"].min()),
                               validation_logloss_min_round=int(curve.loc[curve["validation_logloss"].idxmin(),
                                                                         "round"]))
            summary.append(dict(stage=stage, band=start, rounds=ROUNDS, **{k: refit[name][k] for k in
                             ("train_logloss", "validation_logloss", "validation_logloss_min",
                              "validation_logloss_min_round", "base_rate", "max_abs_diff")}))
            # Staged probabilities: route the archived arms with the first k rounds of this band booster.
            for k in MARKS:
                staged[k][:, start:start + 24] = booster.inplace_predict(
                    tuning["x_apply"], iteration_range=(0, k)).reshape(-1, 24)
        frames[stage] = staged

    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "loss_summary.csv", index=False)

    curve_rows = []
    for stage, staged in frames.items():
        rows = slice(0, n_f1) if stage == "f1" else slice(n_f1, None)
        for k, probability in staged.items():
            routed = official["body"][rows] if k == 0 else np.where(probability >= TAU, official["tail"][rows],
                                                                    official["body"][rows])
            error = (routed - official["target"][rows]) ** 2
            high = official["target"][rows] >= 550.0
            curve_rows.append(dict(stage=stage, rounds=k, mse=float(error.mean()),
                                   mse_ge550=float(error[high].mean())))
    routing_curve = pd.DataFrame(curve_rows)
    routing_curve.to_csv(OUT / "routing_curve.csv", index=False)

    plot_loss(history)
    plot_routing(routing_curve)

    receipt_text = dict(
        experiment="e2000fs_calib2_detroute", booster="binary:logistic detector, 400 rounds, depth 4, eta 0.05",
        tau=TAU, rounds=ROUNDS,
        verification=dict(max_abs_diff=max(v["max_abs_diff"] for v in refit.values()),
                          routed_replay_matches_submission=True,
                          archived_routed_mse=scores_archived["total"],
                          leaderboard_mse=receipt()["mse"], mse_tolerance=1e-9),
        boosters=refit,
        outputs={name: sha(OUT / name) for name in
                 ("loss_history.png", "routing_curve.png", "loss_summary.csv", "routing_curve.csv",
                  *[f"loss_history/{stage}_b{index}.csv" for stage in STAGES for index in range(3)])},
        sources={name: {"file": str(path), "sha256": sha(path)} for name, path in (
            ("detector_f1_b0", RUN / "detector_f1_b0.ubj"), ("detector_f1_b1", RUN / "detector_f1_b1.ubj"),
            ("detector_f1_b2", RUN / "detector_f1_b2.ubj"), ("detector_final_b0", RUN / "detector_final_b0.ubj"),
            ("detector_final_b1", RUN / "detector_final_b1.ubj"), ("detector_final_b2", RUN / "detector_final_b2.ubj"),
            ("exceedance_prob", RUN / "exceedance_prob.npz"), ("forecast", RUN / "forecast.npz"),
            ("results", RUN / "results.json"), ("tweedie_lane", LANE))},
        seconds=round(time.time() - started, 1))
    (OUT / "loss_receipt.json").write_text(json.dumps(receipt_text, indent=2) + "\n")
    print(summary_frame.to_string(index=False))
    print(routing_curve.to_string(index=False))
    print(f"max |refit - archived| = {receipt_text['verification']['max_abs_diff']}")
    print(f"{receipt_text['seconds']} s")


def plot_loss(history):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), dpi=150, sharex=True)
    for row, stage in enumerate(STAGES):
        for column, metric in enumerate(("train_logloss", "validation_logloss")):
            ax = axes[row, column]
            for start in BANDS:
                curve = history[f"{stage}_b{BANDS.index(start)}"]
                ax.plot(curve["round"], curve[metric], color=BAND_COLOR[start], lw=1.4, label=BAND_LABEL[start])
            ax.set_title(f"{STAGE_LABEL[stage]} — {'train' if column == 0 else 'validation'} logloss",
                         fontsize=9, loc="left")
            ax.set_ylabel("binary logloss", fontsize=8.5)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=8)
            if row == 0 and column == 1:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
            if row == 1:
                ax.set_xlabel("detector boosting round (400 fixed by the recipe)", fontsize=8.5)
    fig.suptitle("e2000fs_calib2_detroute — detector loss recorded while fitting the submitted boosters",
                 fontsize=11, x=.012, ha="left")
    fig.text(.012, .012, "Validation is each stage's official scoring window, held out in time by the "
                         "origin+144 h ≤ cutoff mask; it is diagnostic only, not a round-selection curve.",
             fontsize=7.6, color="#555555")
    fig.tight_layout(rect=(0, .03, 1, .96))
    fig.savefig(OUT / "loss_history.png")
    plt.close(fig)


def plot_routing(routing_curve):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=150, sharex=True)
    for ax, metric, title in ((axes[0], "mse", "total MSE"), (axes[1], "mse_ge550", "MSE on truth ≥ 550 km/s")):
        for stage in STAGES:
            part = routing_curve[routing_curve.stage == stage]
            window = "F1 (2026-03…05)" if stage == "f1" else "F2 (2026-06…09)"
            ax.plot(part["rounds"], part[metric], color=STAGE_COLOR[stage], lw=1.6, marker="o", ms=3.2,
                    label=f"{window}")
            ax.axhline(float(part[metric].iloc[0]), color=STAGE_COLOR[stage], lw=.9, ls="--", alpha=.55)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("detector rounds used for routing (0 = body arm everywhere)", fontsize=8.5)
        ax.set_ylabel("MSE (km/s)²", fontsize=8.5)
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("e2000fs_calib2_detroute — official score of the routed archived arms as the detector trains",
                 fontsize=10.5, x=.012, ha="left")
    fig.text(.012, .012, "Dashed lines are the same window with the body arm everywhere (k=0). The official "
                         "window is the scored window, so this is diagnostic; tau and 400 rounds were fixed "
                         "before it was scored.", fontsize=7.4, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(OUT / "routing_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
