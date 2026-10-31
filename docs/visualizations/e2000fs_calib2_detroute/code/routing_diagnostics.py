"""Routing diagnostics of the submitted e2000fs_calib2_detroute forecast.

Recomputes, from the archived probability matrix and the two archived arm forecasts, the detection skill by
horizon band and the threshold sensitivity that `results.json` reports, and stops if either disagrees with the
archive. tau=0.234 was fixed before the window was scored (cost ratio 1233/(1233+4045)); the sensitivity sweep is
a diagnostic and did not choose the submitted threshold.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib2_detroute/code/routing_diagnostics.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detector_common import HIGH, OUT, RUN, TAU, archived_probabilities, load_features, official_arrays, \
    receipt, routed_score, sha

BANDS = ((0, "h1-24", "h1–24"), (24, "h25-48", "h25–48"), (48, "h49-72", "h49–72"))
TAUS = (0.05, 0.10, 0.15, 0.20, 0.234, 0.30, 0.40, 0.50, 0.60)


def main():
    data = load_features()
    official = official_arrays(data)
    probability, origins = archived_probabilities()
    assert origins.equals(official["origins"])
    target, n_f1 = official["target"], official["n_f1"]
    high = target >= HIGH
    routed = np.where(probability >= TAU, official["tail"], official["body"])
    assert np.abs(routed - np.load(RUN / "forecast.npz")["predictions"]).max() == 0.0
    archive = json.loads((RUN / "results.json").read_text())

    scores = {"body_tweedie": routed_score(official["body"], target, n_f1),
              "tail_mixture": routed_score(official["tail"], target, n_f1),
              "routed": routed_score(routed, target, n_f1)}
    for name, block in scores.items():
        for key in ("total", "f1", "f2", "ge550"):
            assert abs(block[key] - archive["metrics"][name][key]) < 1e-9, (name, key)
    assert abs(scores["routed"]["total"] - receipt()["mse"]) < 1e-9
    oracle = np.where(high, official["tail"], official["body"])
    scores["oracle_arm"] = routed_score(oracle, target, n_f1)

    detection = probability >= TAU
    band_rows = []
    for period, rows in (("F1", slice(0, n_f1)), ("F2", slice(n_f1, None))):
        for start, key_label, label in BANDS:
            part = detection[rows, start:start + 24]
            truth = high[rows, start:start + 24]
            tp = int((part & truth).sum())
            fp = int((part & ~truth).sum())
            fn = int((~part & truth).sum())
            hit, false = tp / max(tp + fn, 1), fp / max(tp + fp, 1)
            band_rows.append(dict(period=period, band=label, pairs=int(part.size), tp=tp, fp=fp, fn=fn,
                                  pod=hit, far=false, precision=tp / max(tp + fp, 1),
                                  share_of_high_routed=hit))
            key = f"{period}_{key_label}"
            archived = archive["detection_by_band_period"][key]
            for metric in ("pod", "far", "precision"):
                assert abs(band_rows[-1][metric] - archived[metric]) < 1e-12, (key, metric)
    band_frame = pd.DataFrame(band_rows)
    band_frame.to_csv(OUT / "routing_by_band.csv", index=False)

    sensitivity = []
    for tau in TAUS:
        chosen = np.where(probability >= tau, official["tail"], official["body"])
        score = routed_score(chosen, target, n_f1)
        sensitivity.append(dict(tau=tau, total=score["total"], f1=score["f1"], f2=score["f2"],
                                ge550=score["ge550"], f1_ge550=score["f1_ge550"], f2_ge550=score["f2_ge550"],
                                routed_share=float((probability >= tau).mean()),
                                precision=float(((probability >= tau) & high).sum() / max((probability >= tau).sum(), 1)),
                                pod=float(((probability >= tau) & high).sum() / high.sum())))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "routing_sensitivity.csv", index=False)
    for row in sensitivity.itertuples():
        archived = archive["sensitivity"].get(f"{row.tau:.3f}")
        if archived:
            for key, column in (("total", "total"), ("ge550", "ge550"), ("precision", "precision"), ("pod", "pod")):
                assert abs(getattr(row, column) - archived[key]) < 1e-9, (row.tau, key)

    plot(band_frame, sensitivity, scores)

    summary = dict(
        experiment="e2000fs_calib2_detroute", tau=TAU, tau_rule=archive["rule"],
        pairs=int(target.size), pairs_ge550=int(high.sum()),
        arms={name: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in block.items()}
              for name, block in scores.items()},
        sensitivity_sha256=sha(OUT / "routing_sensitivity.csv"),
        band_sha256=sha(OUT / "routing_by_band.csv"),
        note="recomputed from exceedance_prob.npz, the two archived arm forecasts and the official truth; every "
             "value that results.json also reports is asserted equal before writing")
    (OUT / "routing_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(band_frame.to_string(index=False))
    print(json.dumps(scores["routed"], indent=1))
    print(sensitivity[["tau", "total", "ge550", "f1_ge550", "f2_ge550", "precision", "pod"]].to_string(index=False))


def plot(band_frame, sensitivity, scores):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    axis = axes[0]
    width = 0.36
    positions = np.arange(len(BANDS))
    for offset, (period, color, alpha) in zip((-width / 2, width / 2),
                                              (("F1", "#4c72b0", 1.0), ("F2", "#e8620c", 1.0))):
        part = band_frame[band_frame.period == period]
        axis.bar(positions + offset, part["pod"], width * 0.9, color=color, alpha=alpha,
                 label=f"{period} · POD")
        axis.bar(positions + offset, -(part["far"]), width * 0.9, color=color, alpha=.45,
                 label=f"{period} · FAR (negative)")
        for position, precision in zip(positions + offset, part["precision"]):
            axis.plot([position], [precision], marker="D", color="black", ms=3.6)
    axis.axhline(0, color="#666666", lw=.8)
    axis.set_xticks(positions, [display for _, _, display in BANDS])
    axis.set_ylim(-0.65, 0.95)
    axis.set_ylabel("rate (POD up, FAR down, ◆ precision)")
    axis.set_xlabel("forecast horizon band")
    axis.set_title(f"Detection skill at tau = {TAU} on the official windows", fontsize=10.5, loc="left")
    axis.grid(alpha=.25, axis="y")
    axis.legend(fontsize=7.8, frameon=False, ncol=2)
    axis.tick_params(labelsize=8.5)

    axis = axes[1]
    axis.plot(sensitivity["tau"], sensitivity["total"], color="#222222", lw=1.6, marker="o", ms=3.2,
              label="total MSE")
    axis.plot(sensitivity["tau"], sensitivity["ge550"], color="#c44e52", lw=1.6, marker="s", ms=3.0,
              label="MSE, truth ≥ 550 km/s")
    axis.axvline(TAU, color="#4c72b0", lw=1.2, ls="--",
                 label=f"preregistered tau = {TAU} (cost ratio)")
    axis.axhline(scores["body_tweedie"]["total"], color="#222222", lw=.8, ls=":", alpha=.7)
    axis.axhline(scores["body_tweedie"]["ge550"], color="#c44e52", lw=.8, ls=":", alpha=.7)
    axis.set_xlabel("routing threshold tau")
    axis.set_ylabel("MSE (km/s)²")
    axis.set_title("Threshold sensitivity (dotted: body arm everywhere)", fontsize=10.5, loc="left")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8, frameon=False)
    axis.tick_params(labelsize=8.5)

    fig.suptitle("e2000fs_calib2_detroute — routing diagnostics on the officially scored windows",
                 fontsize=11, x=.012, ha="left")
    fig.text(.012, -.02, "Recomputed from the archived probabilities and arm forecasts; tau was never tuned on "
                         "these windows, so the sweep is a diagnostic only.", fontsize=7.6, color="#555555")
    fig.savefig(OUT / "routing_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
