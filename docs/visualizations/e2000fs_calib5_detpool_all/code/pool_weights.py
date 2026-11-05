"""Pool weights of the submitted e2000fs_calib5_detpool_all forecast.

The submission is a linear stack on the `e2000fs_calib4_detpool_gpd` prediction: every 168 origins a non-negative
least-squares fit on the trailing 2,160 matured origins gives one weight per archived refit arm
(`pred = base + sum_j beta_j (arm_j - base)`, `beta >= 0`, `sum(beta) <= 1`, 1e-3 ridge, then a shrink ladder
until the trailing >=550 MSE does not exceed the base's). This script re-runs that lane from the archived arm
forecasts, refuses to continue unless it reproduces the archived submission `forecast.npz` bit for bit, and then
writes the weight tables and figures.

It also reads the legacy 81-arm lane (`research/e2000fs_calib5-20261031/runs/poolall_v1/weights.npz`, the original
F1+F2-window run) to show the arm-weight distribution of "attach every archived arm", which is what the
2026-10-31 result was about.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib5_detpool_all/code/pool_weights.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vizlib as V  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
RUNS = V.ROOT / "research/e2000fs_f123-20261031/runs"
BASE_FILE = RUNS / "detpool_gpd_v1/forecast.npz"
ARCHIVE = RUNS / "poolall_v1/forecast.npz"
LEGACY = V.ROOT / "research/e2000fs_calib5-20261031/runs/poolall_v1/weights.npz"
H = 72
# Arm set of the 2026-10-31 refit lane, in the order of research/e2000fs_f123-20261031/post.py (lane_poolall).
ALLOWED = ["pointweight4", "point4_144h", "tweedie15", "mixture", "era", "gpd", "spectral", "dirmo24", "detroute",
           "offset_roll", "pool_bodytail", "detpool_era", "detpool_mixture", "lgbm_extreme4", "lgbm_point4",
           "lgbm_recent"]
PERIOD_EDGES = {"F1": ("2026-01-01", "2026-03-31"), "F2": ("2026-04-01", "2026-06-30"), "F3": ("2026-07-01", "2026-09-30")}


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool_all = _import("calib5_pool_all", V.ROOT / "research/e2000fs_calib5-20261031/pool_all.py")
pool_lane = _import("calib4_pool_lane", V.ROOT / "research/e2000fs_calib4-20261031/pool_lane.py")
common = _import("f123_common", V.ROOT / "research/e2000fs_f123-20261031/common.py")


def load(name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Forecast of one refit arm on all 6,315 origins (same loader as post.py)."""
    folder = RUNS / f"{name}_v1"
    single = folder / "forecast.npz"
    if single.exists():
        z = np.load(single)
        return pd.DatetimeIndex(z["origins_ns"], tz="UTC"), z["predictions"].astype(np.float64)
    parts = [np.load(folder / f"{period}_predictions.npz") for period in ("F1", "F2", "F3")]
    origins = pd.DatetimeIndex(np.concatenate([z["origins"] for z in parts]), tz="UTC")
    return origins, np.vstack([z["pred"] for z in parts]).astype(np.float64)


def short(arm: str) -> str:
    return (arm.replace("e2000fs_", "").replace("/results/forecast.npz", "").replace("/forecast.npz", "")
            .replace("_v1", ""))


def main() -> None:
    base = np.load(BASE_FILE)
    origins = pd.DatetimeIndex(base["origins_ns"], tz="UTC")
    base_pred = base["predictions"].astype(np.float64)

    names, mats = [], []
    seen = [base_pred]
    for name in ALLOWED:
        try:
            arm_origins, arm_pred = load(name)
        except FileNotFoundError:
            continue
        if arm_pred.shape != base_pred.shape or not np.array_equal(arm_origins.asi8, origins.asi8):
            continue
        if not np.isfinite(arm_pred).all():
            continue
        if any(np.array_equal(arm_pred, other) for other in seen):
            continue
        seen.append(arm_pred)
        names.append(name)
        mats.append(arm_pred - base_pred)
    assert len(names) == 16, f"expected the 16-arm refit lane, found {len(names)}: {names}"
    delta = np.stack(mats)                               # (arms, origins, 72)

    target = common.targets(common.speed(), origins)      # the series the lane was fitted on
    high = target >= 550.0
    mature = pool_lane.matured_counts(origins)
    residual = target - base_pred

    betas = np.zeros((len(origins), len(names)))
    refits = []
    for start in range(0, len(origins), pool_all.REFIT):
        end_of_history = int(mature[start])
        begin_of_history = max(0, end_of_history - pool_all.WINDOW)
        stop = min(start + pool_all.REFIT, len(origins))
        entry = dict(refit_start_fold=start + 1, refit_start_origin_utc=origins[start].isoformat(),
                     refit_last_origin_utc=origins[stop - 1].isoformat(), origins=int(stop - start),
                     trailing_origins=int(end_of_history - begin_of_history),
                     trailing_high_pairs=int(high[begin_of_history:end_of_history].sum()))
        if end_of_history - begin_of_history < 48 or high[begin_of_history:end_of_history].sum() < 50:
            refits.append({**entry, "sum_beta": 0.0, "n_nonzero": 0, "skipped": True})
            continue
        beta = pool_all.fit_block(delta[:, begin_of_history:end_of_history].reshape(len(names), -1),
                                  residual[begin_of_history:end_of_history].ravel(),
                                  high[begin_of_history:end_of_history].ravel())
        betas[start:stop] = beta
        refits.append({**entry, "sum_beta": float(beta.sum()), "n_nonzero": int((beta > 0).sum()), "skipped": False})

    stacked = base_pred + np.einsum("np,pnh->nh", betas, delta)
    archived = np.load(ARCHIVE)["predictions"]
    assert np.array_equal(stacked.astype(np.float32), archived), \
        "the recomputed stack is not the archived submission forecast.npz"

    data = V.load(OUT, [ARCHIVE], [("origins_ns", "predictions")])   # scores must equal the leaderboard record too

    by_arm = pd.DataFrame({
        "arm": names,
        "mean_beta": betas.mean(axis=0),
        "share_origins_nonzero": (betas > 0).mean(axis=0),
        "mean_beta_when_nonzero": np.divide(betas.sum(axis=0), np.maximum((betas > 0).sum(axis=0), 1)),
        "max_beta": betas.max(axis=0),
        "first_nonzero_origin_utc": [origins[np.argmax(betas[:, j] > 0)].isoformat() if (betas[:, j] > 0).any() else ""
                                     for j in range(len(names))],
    }).sort_values("mean_beta", ascending=False).reset_index(drop=True)
    by_arm["share_of_total_weight"] = by_arm["mean_beta"] / by_arm["mean_beta"].sum()
    by_arm.to_csv(OUT / "weights_by_arm.csv", index=False)

    per_refit = pd.DataFrame(refits)
    per_refit["period"] = [V.PERIOD_OF_MONTH.get(origin[:7], "") for origin in per_refit["refit_start_origin_utc"]]
    wide = pd.DataFrame(betas, columns=names)
    wide.insert(0, "origin_last_input_utc", origins.strftime("%Y-%m-%dT%H:%M:%SZ"))
    wide["sum_beta"] = betas.sum(axis=1)
    wide.to_csv(OUT / "weights_by_origin.csv", index=False, float_format="%.6f")
    per_refit.to_csv(OUT / "weights_by_refit.csv", index=False)

    np.savez_compressed(OUT / "pool_weights.npz", betas=betas.astype(np.float32), arms=np.array(names),
                        origins_ns=origins.asi8)

    # ---- legacy 81-arm lane (original F1+F2 window): the distribution that motivated the refit lane
    legacy = np.load(LEGACY, allow_pickle=True)
    legacy_betas, legacy_arms = legacy["betas"], [short(a) for a in legacy["arms"].astype(str)]
    legacy_mean = legacy_betas.mean(axis=0)
    legacy_table = pd.DataFrame({"arm": legacy_arms, "mean_beta": legacy_mean,
                                 "share_origins_nonzero": (legacy_betas > 0).mean(axis=0)}) \
        .sort_values("mean_beta", ascending=False).reset_index(drop=True)
    legacy_table.to_csv(OUT / "legacy_81arm_weights.csv", index=False)
    legacy_archive = json.loads((V.ROOT / "research/e2000fs_calib5-20261031/runs/poolall_v1/results.json").read_text())

    # ---- figures
    plot_by_arm(by_arm)
    plot_over_time(origins, betas, names)
    plot_legacy(legacy_betas, legacy_arms, legacy_archive)

    receipt = {
        "experiment": data["receipt"]["experiment"],
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rule": pool_all.RULE,
        "constants": {"arms": len(names), "window": pool_all.WINDOW, "refit": pool_all.REFIT, "ridge": pool_all.RIDGE,
                      "shrink_ladder": list(pool_all.SHRINK), "high_threshold_kms": 550.0},
        "arm_order": names,
        "inputs": {
            "base": {"file": str(BASE_FILE)},
            "submission": {"file": str(ARCHIVE)},
            "legacy_weights": {"file": str(LEGACY)},
            "arms": {name: {"file": str(RUNS / f"{name}_v1")} for name in names},
        },
        "verification": {
            "recomputed_stack_equals_submission": True,
            "max_abs_difference_float32": float(np.abs(stacked.astype(np.float32) - archived).max()),
            "scores": data["scores"],
            "receipt_scores": {k: data["receipt"][k] for k in data["scores"]},
            "tolerance": 1e-9,
        },
        "weights": {
            "n_refit_blocks": int((~per_refit["skipped"]).sum()) if "skipped" in per_refit else 0,
            "mean_sum_beta": float(betas.sum(axis=1).mean()),
            "share_origin_arm_cells_nonzero": float((betas > 0).mean()),
            "share_origins_all_zero": float((betas.sum(axis=1) == 0).mean()),
            "top": by_arm.head(8).to_dict("records"),
        },
        "legacy_81arm_lane": {
            "arms": len(legacy_arms),
            "mean_sum_beta": float(legacy_betas.sum(axis=1).mean()),
            "share_origin_arm_cells_nonzero": float((legacy_betas > 0).mean()),
            "top": legacy_table.head(8).to_dict("records"),
            "official_mse": legacy_archive["official"]["mse"],
        },
        "outputs": ["weights_by_arm.csv", "weights_by_origin.csv", "weights_by_refit.csv", "pool_weights.npz",
                    "weights_by_arm.png", "weights_over_time.png", "legacy_81arm_weights.csv", "legacy_81arm_weights.png"],
    }
    (OUT / "weights_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"n_arms": len(names), "mean_sum_beta": receipt["weights"]["mean_sum_beta"],
                      "share_nonzero": receipt["weights"]["share_origin_arm_cells_nonzero"],
                      "top": [(r["arm"], round(r["mean_beta"], 4)) for r in by_arm.head(6).to_dict("records")]}, indent=1))


def plot_by_arm(by_arm: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    table = by_arm.sort_values("mean_beta")
    positions = np.arange(len(table))
    axis.barh(positions, table["mean_beta"], color="#3d6fb4", height=.68)
    axis.barh(positions, table["mean_beta"] * (table["share_origins_nonzero"] / table["share_origins_nonzero"].max()),
              color="#8fb3e0", height=.3)
    axis.set_yticks(positions)
    axis.set_yticklabels([short(a) for a in table["arm"]], fontsize=9)
    axis.set_xlabel("mean beta over the 6,315 origins", fontsize=9.5)
    axis.grid(axis="x", alpha=.25)
    for y, (beta, share) in enumerate(zip(table["mean_beta"], table["share_origins_nonzero"])):
        axis.text(beta + .002, y, f"{beta:.4f}  (active {share * 100:.0f}%)", va="center", fontsize=8)
    axis.set_title("Weight of each archived arm in the submitted stack", fontsize=11, loc="left")
    fig.text(.012, -.03, "Bar: mean beta over all origins; light bar: the same mean scaled by how often the arm was "
                         "active (beta > 0). Recomputed from the archived arm forecasts and verified bit for bit "
                         "against the submitted forecast.npz.", fontsize=7.6, color="#555555")
    fig.savefig(OUT / "weights_by_arm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_over_time(origins: pd.DatetimeIndex, betas: np.ndarray, names: list[str]) -> None:
    order = np.argsort(-betas.mean(axis=0))
    top = order[:6]
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 6.4), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})
    for axis in axes:
        for label, (first, last) in PERIOD_EDGES.items():
            axis.axvspan(pd.Timestamp(first), pd.Timestamp(last), color="#f0f0f0", zorder=0)
            axis.text(pd.Timestamp(first) + pd.Timedelta(days=4), 0.96, label, transform=axis.get_xaxis_transform(),
                      fontsize=9, color="#777777", va="top")
        axis.grid(alpha=.25)
    axes[0].plot(origins, betas.sum(axis=1), color="#333333", lw=1.2)
    axes[0].set_ylabel("sum of beta", fontsize=9.5)
    axes[0].set_title("Total stacked weight (capped at 1.0 by the shrink ladder)", fontsize=10.5, loc="left")
    for rank, column in enumerate(top):
        axes[1].plot(origins, betas[:, column], lw=1.3, label=short(names[column]))
    axes[1].set_ylabel("beta", fontsize=9.5)
    axes[1].set_xlabel("origin (t0) of the 6,315 leaderboard folds", fontsize=9.5)
    axes[1].set_title("The six arms with the largest mean weight", fontsize=10.5, loc="left")
    axes[1].legend(fontsize=8, ncol=3, loc="upper left", framealpha=.9)
    fig.text(.012, -.02, "Weights are piecewise constant over each 168-origin refit window; a refit uses only the "
                         "trailing 2,160 origins whose 72 h targets are matured at the window start.",
             fontsize=7.6, color="#555555")
    fig.savefig(OUT / "weights_over_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_legacy(betas: np.ndarray, arms: list[str], archive: dict) -> None:
    order = np.argsort(-betas.mean(axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    means = betas.mean(axis=0)
    axes[0].hist(means, bins=30, color="#b4533d", alpha=.85)
    axes[0].set_xlabel("mean beta per arm", fontsize=9.5)
    axes[0].set_ylabel("number of arms", fontsize=9.5)
    axes[0].set_title(f"Legacy lane: {len(arms)} arms, mean sum(beta) "
                      f"{betas.sum(axis=1).mean():.3f}", fontsize=10.5, loc="left")
    axes[0].grid(alpha=.25)
    top = order[:18]
    axes[1].barh(np.arange(len(top))[::-1], means[top], color="#b4533d", height=.7)
    axes[1].set_yticks(np.arange(len(top))[::-1])
    axes[1].set_yticklabels([arms[i] for i in top], fontsize=8)
    axes[1].set_xlabel("mean beta", fontsize=9.5)
    axes[1].set_title("Legacy lane: the 18 largest mean weights", fontsize=10.5, loc="left")
    axes[1].grid(axis="x", alpha=.25)
    fig.text(.012, -.03, f"Original 81-arm lane fitted on the older F1+F2 windows (official MSE "
                         f"{archive['official']['mse']:.1f}, >=550 {archive['official']['mse_ge550']:.1f}); "
                         f"{(betas > 0).mean() * 100:.1f}% of (origin, arm) cells are non-zero. "
                         "Read from the archived weights.npz of research/e2000fs_calib5-20261031.", fontsize=7.6,
             color="#555555")
    fig.savefig(OUT / "legacy_81arm_weights.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
