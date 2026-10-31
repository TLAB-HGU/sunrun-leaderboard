"""Feature importance of the submitted e2000fs_calib2_detroute routing.

Two views of the same six detector boosters:

* **split counts** — how often each input split a node inside the boosters that produced the submitted routing.
* **permutation ΔMSE** — each feature is shuffled across origins, the three band detectors are re-predicted, the
  archive of the two arms is re-routed, and the official MSE is recomputed. The arms themselves are frozen
  archives, so this measures the detector's contribution to the routing only, not the arms' own errors.

The unshuffled re-prediction must reproduce the archived exceedance probabilities bit for bit, and the routed
forecast must reproduce the submitted forecast, before any number is written. The scored windows are public, so
the permutation table is a diagnostic; nothing here selected features, tau or rounds.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib2_detroute/code/detector_importance.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detector_common import (BANDS, HIGH, OUT, RUN, STAGES, TAU, archived_booster, archived_probabilities,
                             column_names, load_features, official_arrays, receipt, routed_score, sha)

REPEATS = 5
SEED = 20261031
STAGE_TITLE = {"f1": "F1 model (fit targets < 2026-02-26)", "final": "F2 model (fit targets < 2026-05-29)"}
HORIZON = "horizon_hours"


def long_matrix(matrix, start):
    """Origin-level design -> the detector's row-level design (24 rows per origin plus the horizon index)."""
    n = len(matrix)
    return np.hstack([np.repeat(matrix, 24, axis=0),
                      np.tile(np.arange(start + 1, start + 25), n).astype(np.float32)[:, None]])


def main():
    data = load_features()
    official = official_arrays(data)
    columns, constant, varying = column_names(data)
    probability_archived, probability_origins = archived_probabilities()
    assert probability_origins.equals(official["origins"])
    n_f1 = official["n_f1"]
    routed = np.where(probability_archived >= TAU, official["tail"], official["body"])
    assert np.abs(routed - np.load(RUN / "forecast.npz")["predictions"]).max() == 0.0
    base = routed_score(routed, official["target"], n_f1)
    assert abs(base["total"] - receipt()["mse"]) < 1e-9, base

    def columns_of(feature):
        """Column indices of one feature in an origin-level design matrix."""
        if feature in constant:
            return [constant.index(feature)], "constant"
        slot = varying.index(feature)
        return [len(constant) + lead * len(varying) + slot for lead in range(24)], "hourly"

    grids = {"f1": data["f1_bundle"], "final": data["off_bundle"]}
    truth = {"f1": data["y_f1"], "final": data["y_f2"]}
    arms = {"f1": (official["body"][:n_f1], official["tail"][:n_f1]),
            "final": (official["body"][n_f1:], official["tail"][n_f1:])}
    split_rows, permutation_rows, parity, model_hashes = [], [], {}, {}

    for stage in STAGES:
        designs = [data["tw"].design(grids[stage], data["names"], data["selected"], start) for start in BANDS]
        boosters = []
        for position, start in enumerate(BANDS):
            path = RUN / f"detector_{stage}_b{position}.ubj"
            model_hashes[str(path.relative_to(data["tw"].ROOT))] = sha(path)
            boosters.append(archived_booster(stage, start))

        def probabilities(mats):
            out = np.empty((len(truth[stage]), 72))
            for position, (booster, matrix, start) in enumerate(zip(boosters, mats, BANDS)):
                out[:, start:start + 24] = booster.inplace_predict(long_matrix(matrix, start)).reshape(-1, 24)
            return out

        def score(probability):
            high = truth[stage] >= HIGH
            chosen = np.where(probability >= TAU, arms[stage][1], arms[stage][0])
            error = chosen - truth[stage]
            return float(np.mean(error ** 2)), float(np.mean(error[high] ** 2))

        base_probability = probabilities(designs)
        offset = 0 if stage == "f1" else n_f1
        distance = float(np.max(np.abs(base_probability - probability_archived[offset:offset + len(truth[stage])])))
        assert distance == 0.0, f"{stage}: rebuilt detector does not reproduce the archived probabilities"
        base_total, base_high = score(base_probability)
        assert abs(base_total - base["f1" if stage == "f1" else "f2"]) < 1e-9
        parity[stage] = dict(origins=int(len(truth[stage])), mse=base_total,
                             mse_ge550=base_high, expected_mse=base["f1" if stage == "f1" else "f2"],
                             probability_parity="exact (max |diff| = 0.0 vs exceedance_prob.npz)")

        # Split counts per source feature over the three band boosters of this stage.
        counts = {name: 0 for name in columns}
        for booster in boosters:
            for key, value in booster.get_score(importance_type="weight").items():
                counts[columns[int(key[1:])]] += int(value)
        for name in constant + varying + [HORIZON]:
            split_rows.append(dict(stage=stage, feature=name, splits=counts[name]))

        rng = np.random.default_rng(SEED)
        for feature in constant + varying:
            cols, kind = columns_of(feature)
            totals, highs = [], []
            for _ in range(REPEATS):
                order = rng.permutation(len(truth[stage]))
                shuffled = []
                for matrix in designs:
                    copy = matrix.copy()
                    copy[:, cols] = copy[order][:, cols]
                    shuffled.append(copy)
                total, high = score(probabilities(shuffled))
                totals.append(total - base_total)
                highs.append(high - base_high)
            totals, highs = np.asarray(totals), np.asarray(highs)
            permutation_rows.append(dict(
                stage=stage, feature=feature, kind=kind, columns=len(cols), repeats=REPEATS,
                delta_mse_mean=float(totals.mean()), delta_mse_std=float(totals.std(ddof=1)),
                delta_mse_ge550_mean=float(highs.mean()), delta_mse_ge550_std=float(highs.std(ddof=1))))
        print(f"{stage}: {len(truth[stage])} origins, base MSE {base_total:.6f}, >=550 {base_high:.6f}", flush=True)

    table = pd.DataFrame(permutation_rows)
    splits = pd.DataFrame(split_rows)
    merged = table.merge(splits, on=["stage", "feature"], how="outer")
    for stage in STAGES:
        mask = merged.stage == stage
        merged.loc[mask, "rank_permutation"] = merged.loc[mask, "delta_mse_mean"].rank(ascending=False)
        merged.loc[mask, "rank_splits"] = merged.loc[mask, "splits"].rank(ascending=False, method="min")
    merged = merged.sort_values(["stage", "rank_permutation"])
    merged.to_csv(OUT / "feature_importance.csv", index=False)

    plot(merged)
    (OUT / "feature_importance_receipt.json").write_text(json.dumps(dict(
        experiment="e2000fs_calib2_detroute",
        method="permutation across origins of one feature's columns in the 237-column block design, then the three "
               "band detectors are re-predicted, the two archived arm forecasts are re-routed at tau=0.234 and the "
               "official MSE is recomputed; ΔMSE vs the submitted routing",
        note="the arm forecasts are frozen archives, so the permutation measures the routing component only; "
             "the scored windows are public, so this is diagnostic and was not used for selection",
        repeats=REPEATS, seed=SEED, tau=TAU,
        parity=parity,
        split_counts="xgboost get_dump over the 400 trees of each of the six archived boosters; design column -> "
                     "source feature, hourly features summed over their 24 lead copies",
        model_sha256=model_hashes,
        outputs={"feature_importance.csv": sha(OUT / "feature_importance.csv"),
                 "feature_importance.png": sha(OUT / "feature_importance.png")}), indent=2) + "\n")

    for stage in STAGES:
        part = merged[merged.stage == stage]
        print(f"\n{stage} top 10 by ΔMSE")
        print(part.nlargest(10, "delta_mse_mean")[
            ["feature", "kind", "delta_mse_mean", "delta_mse_std", "splits", "rank_splits"]].to_string(index=False))
    print("wrote feature_importance.{csv,png} and the receipt")


def plot(table):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=(1.35, 1.0)))
    colors = {"constant": "#c44e52", "hourly": "#4c72b0"}
    for row, stage in enumerate(STAGES):
        part = table[table.stage == stage]
        top = part.nlargest(20, "delta_mse_mean").sort_values("delta_mse_mean")
        axis = axes[row, 0]
        axis.barh(top["feature"], top["delta_mse_mean"], xerr=top["delta_mse_std"],
                  color=[colors[k] for k in top["kind"]], error_kw=dict(lw=.9, capsize=2))
        axis.axvline(0, color="#888888", lw=.8)
        axis.set_title(f"{STAGE_TITLE[stage]} — permutation ΔMSE", fontsize=10.5, loc="left")
        axis.set_xlabel("ΔMSE when the feature is shuffled across origins (km/s)²")
        axis.grid(alpha=.25, axis="x")
        axis.tick_params(labelsize=8)

        axis = axes[row, 1]
        top = part.nlargest(20, "splits").sort_values("splits")
        axis.barh(top["feature"], top["splits"], color=["#999999" if f == HORIZON else colors[k]
                                                        for f, k in zip(top["feature"], top["kind"])])
        axis.set_title(f"{STAGE_TITLE[stage]} — split counts", fontsize=10.5, loc="left")
        axis.set_xlabel("nodes split on the feature (three band boosters, 400 trees each)")
        axis.grid(alpha=.25, axis="x")
        axis.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors["constant"]),
               plt.Rectangle((0, 0), 1, 1, color=colors["hourly"]),
               plt.Rectangle((0, 0), 1, 1, color="#999999")]
    fig.legend(handles, ["constant feature", "hourly feature (24 lead copies)",
                         "horizon index (model-internal, not shuffled)"],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("e2000fs_calib2_detroute — detector importance\n"
                 f"routing re-run with one feature shuffled across origins; {REPEATS} shuffles, mean ± s.d.; "
                 "officially scored windows, diagnostic only", fontsize=11)
    fig.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
