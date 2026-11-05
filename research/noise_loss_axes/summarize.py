"""Aggregate pre-registered multi-seed loss comparisons; never fit forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ("sera_l2", "plain_l2", "gaussian", "lognormal", "mixture")
SEEDS = (42, 43, 44)


def aggregate(root=HERE / "results"):
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            path = root / arm / f"seed{seed}" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Incomplete experiment: {path}")
            data = json.loads(path.read_text())
            if data["arm"] != arm or data["seed"] != seed:
                raise ValueError(f"Experiment identity mismatch: {path}")
            row = {"arm": arm, "seed": seed}
            for key in ("W", "mae", "rmse", "bias", "n_pairs"):
                row[key] = data[key]
            distribution = data.get("distribution") or {}
            for key in ("nll", "coverage50", "coverage90", "coverage95"):
                row[key] = distribution.get(key, np.nan)
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = []
    for arm in ARMS:
        group = frame[frame.arm == arm]
        record = {"arm": arm, "seeds": list(SEEDS)}
        for key in ("W", "mae", "rmse", "bias", "nll", "coverage50", "coverage90", "coverage95"):
            values = group[key].dropna()
            if len(values):
                record[key + "_mean"] = float(values.mean())
                record[key + "_std"] = float(values.std(ddof=1))
                record[key + "_worst"] = float(values.max())
        summary.append(record)
    by_arm = {row["arm"]: row for row in summary}
    baseline = by_arm["sera_l2"]
    for row in summary:
        row["W_change_pct_vs_sera"] = 100 * (row["W_mean"] / baseline["W_mean"] - 1)
        row["point_candidate"] = bool(
            row["arm"] not in ("sera_l2", "plain_l2")
            and row["W_mean"] <= .99 * baseline["W_mean"]
            and row["mae_mean"] <= 1.02 * baseline["mae_mean"]
            and row["rmse_mean"] <= 1.02 * baseline["rmse_mean"])
        if "coverage90_mean" in row:
            row["absolute_coverage_pass"] = bool(
                abs(row["coverage90_mean"] - .90) <= .05
                and abs(row["coverage95_mean"] - .95) <= .05)
    axes = []
    for arm, parent in (("gaussian", "plain_l2"), ("lognormal", "gaussian"), ("mixture", "lognormal")):
        child, reference = by_arm[arm], by_arm[parent]
        record = {"arm": arm, "parent": parent,
                  "W_change_pct_vs_parent": 100 * (child["W_mean"] / reference["W_mean"] - 1),
                  "absolute_coverage_pass": child["absolute_coverage_pass"]}
        if parent != "plain_l2":
            record["nll_change_vs_parent"] = child["nll_mean"] - reference["nll_mean"]
            record["coverage_not_materially_worse"] = bool(all(
                abs(child[f"coverage{level}_mean"] - level / 100)
                <= abs(reference[f"coverage{level}_mean"] - level / 100) + .02
                for level in (90, 95)))
            record["distribution_hypothesis_supported"] = bool(
                record["nll_change_vs_parent"] < 0 and record["absolute_coverage_pass"]
                and record["coverage_not_materially_worse"])
        else:
            record["distribution_hypothesis_supported"] = bool(
                record["absolute_coverage_pass"] and child["rmse_mean"] <= reference["rmse_mean"])
            record["axis1_rule"] = "absolute coverage passes and mean RMSE does not worsen vs plain L2"
        axes.append(record)
    frame.to_csv(HERE / "seed_results.csv", index=False)
    output = {"scope": "internal 2025Q3; not official leaderboard scores", "summary": summary, "axes": axes,
              "baseline_update": False, "official_submission": False,
              "bootstrap_pending_candidates": [r["arm"] for r in summary if r["point_candidate"]]}
    (HERE / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(summary)[["arm", "W_mean", "W_std", "W_worst", "mae_mean", "rmse_mean", "W_change_pct_vs_sera", "point_candidate"]].to_string(index=False))
    print(json.dumps(axes, indent=2))
    return output


if __name__ == "__main__":
    aggregate()
