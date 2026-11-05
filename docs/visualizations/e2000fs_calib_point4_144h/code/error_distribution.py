"""The error distribution of the point4 144 h arm: how far, and in which direction, each forecast misses.

`error_by_bands.py` summarises the same residuals as MSE per target-speed band; this script keeps the full
distribution instead of its second moment. For the same 6,315 folds it histograms the signed residual
`forecast - truth` (km/s) of `point4_144h`, its paired `tweedie15` arm (same code path, mask, blocks, rounds
and seed, Tweedie loss) and the `detpool_gpd` stack — over all pairs, over the body below 550 km/s and over
the tail the 4x point weight acts on — and writes the bar chart, the bin counts and a receipt. Nothing is
retrained; the archived forecasts are only read.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib_point4_144h/code/error_distribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vizlib as V  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
RUNS = V.ROOT / "research/e2000fs_f123-20261031/runs"
ARMS = {"point4_144h": ("point4_144h", "#2f5f98"), "tweedie15": ("tweedie15", "#c2553f"),
        "detpool_gpd": ("detpool_gpd", "#4f8a4f")}
BIN_WIDTH = 50.0
EDGES = np.arange(-400.0, 400.1, BIN_WIDTH)  # 16 bins; ±400 km/s keeps >= 99.93 % of every arm
SUBSETS = [("all", "all pairs"), ("lt550", "targets < 550 km/s (body)"), ("ge550", "targets ≥ 550 km/s (tail)")]


def load(name: str) -> np.ndarray:
    folder = RUNS / f"{name}_v1"
    single = folder / "forecast.npz"
    if single.exists():
        z = np.load(single)
        return z["predictions"].astype(np.float64)
    parts = [np.load(folder / f"{period}_predictions.npz") for period in ("F1", "F2", "F3")]
    return np.vstack([z["pred"] for z in parts]).astype(np.float64)


def stats(residual: np.ndarray) -> dict:
    magnitude = np.abs(residual)
    return {
        "pairs": int(residual.size),
        "bias_kms": float(residual.mean()),
        "sd_kms": float(residual.std()),
        "mse": float((residual ** 2).mean()),
        "mae_kms": float(magnitude.mean()),
        "median_kms": float(np.median(residual)),
        "p05_kms": float(np.percentile(residual, 5)),
        "p95_kms": float(np.percentile(residual, 95)),
        "share_within_50_kms": float((magnitude <= 50).mean()),
        "share_within_100_kms": float((magnitude <= 100).mean()),
        "share_over_200_kms": float((magnitude > 200).mean()),
    }


def main() -> None:
    receipt = json.loads((OUT / "leaderboard-result.json").read_text())
    folds = pd.read_parquet(V.pinned("folds"))
    truth = pd.read_parquet(V.pinned("truth"))
    origins = pd.DatetimeIndex(folds["origin_last_input_utc"])
    target = truth.set_index(V.KEY)["target_kms"].unstack().loc[origins.strftime("%Y-%m-%dT%H:%M:%SZ")].to_numpy()
    assert target.shape == (len(origins), V.H) and np.isfinite(target).all()

    residuals = {}
    for label, (name, _) in ARMS.items():
        pred = load(name)
        assert pred.shape == target.shape, (name, pred.shape)
        residuals[label] = pred - target

    # Same scores as error_by_band.csv: point4_144h must equal the leaderboard record, detpool_gpd its float32 archive.
    for label, archive, tolerance in (("point4_144h", receipt["mse"], 1e-6), ("detpool_gpd", 4360.208035478823, 1e-3)):
        recomputed = stats(residuals[label].ravel())["mse"]
        assert abs(recomputed - archive) < tolerance, (label, recomputed, archive)
    band_receipt = json.loads((OUT / "error_by_band_receipt.json").read_text())
    bands = pd.read_csv(OUT / "error_by_band.csv")
    high = bands[bands["band_kms"].isin(["550–650", "650–800", "≥800"])]
    for label in ARMS:
        overall = stats(residuals[label].ravel())["mse"]
        assert abs(overall - band_receipt["overall_mse"][label]) < 1e-6, (label, overall)
        # the band table's ≥550 row, rebuilt from its own rows, for every arm
        weighted = float((high["pairs"] * high[f"mse_{label}"]).sum() / high["pairs"].sum())
        tail = stats(residuals[label][target >= 550.0])["mse"]
        assert abs(weighted - tail) < 1e-6, (label, weighted, tail)

    masks = {"all": np.ones_like(target, dtype=bool), "lt550": target < 550.0, "ge550": target >= 550.0}
    summary, outside, histograms, rows = {}, {}, {}, []
    for subset, _ in SUBSETS:
        summary[subset], outside[subset] = {}, {}
        for label, residual in residuals.items():
            values = residual[masks[subset]]
            counts, _ = np.histogram(values, bins=EDGES)
            outside[subset][label] = int(values.size - counts.sum())
            histograms[(subset, label)] = counts / values.size
            summary[subset][label] = stats(values)
            arm = summary[subset][label]
            # population sd: mse = bias^2 + var, i.e. the histogram's second moment is the band table's MSE
            assert abs(arm["mse"] - (arm["bias_kms"] ** 2 + arm["sd_kms"] ** 2)) < 1e-6
            assert 0.0 < arm["share_within_50_kms"] < arm["share_within_100_kms"] < 1.0
            for low_edge, high_edge, count in zip(EDGES[:-1], EDGES[1:], counts):
                rows.append({"subset": subset, "arm": label, "bin_low_kms": low_edge, "bin_high_kms": high_edge,
                             "count": int(count), "share": float(count / values.size)})
    assert summary["lt550"]["point4_144h"]["pairs"] + summary["ge550"]["point4_144h"]["pairs"] == target.size
    bins = pd.DataFrame(rows)
    bins.to_csv(OUT / "error_distribution.csv", index=False, float_format="%.8f")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), constrained_layout=True, sharex=True)
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    width = 0.28 * BIN_WIDTH  # three bars per 50 km/s bin, in km/s (the x axis is numeric)
    for position, (ax, (subset, subtitle)) in enumerate(zip(axes, SUBSETS)):
        for offset, (label, (_, color)) in enumerate(ARMS.items()):
            arm = summary[subset][label]
            ax.bar(centers + (offset - 1) * width, 100 * histograms[(subset, label)], width=width, color=color,
                   label=f"{label} (median {arm['median_kms']:+,.1f})")
        ax.axvline(0, color="#333333", lw=.9, ls=":")
        ax.set_xlim(EDGES[0] - BIN_WIDTH / 2, EDGES[-1] + BIN_WIDTH / 2)
        ax.set_xticks(np.arange(-400, 401, 100))
        ax.set_xlabel("residual = forecast − truth (km/s), 50 km/s bins", fontsize=8.5)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(labelsize=8.5)
        pairs = summary[subset]["point4_144h"]["pairs"]
        ax.set_title(f"{subtitle}: {pairs:,} pairs ({100 * pairs / target.size:.1f} %)", fontsize=10, loc="left")
        ax.legend(fontsize=7.6, loc="upper left", title="residual median (km/s)", title_fontsize=7.4)
        if position == 0:
            ax.set_ylabel("share of pairs (% per 50 km/s bin)", fontsize=9)
    fig.text(.012, -.045, "Residual = forecast − truth on the same 6,315 folds (leaderboard truth); negative = under-forecast.\n"
                          "The 4× point weight lifts every residual distribution towards positive: the body (< 550 km/s) pays "
                          "for that shift, the tail (≥ 550 km/s) gains.\n"
                          f"Bins cover ±400 km/s; at most {max(o for per_arm in outside.values() for o in per_arm.values())} "
                          "pairs of an arm (0.07 %) fall outside.", fontsize=7.6, color="#555555")
    # Layout guard: bars must be several pixels wide (a width in index units collapses them to slivers on a km/s
    # axis) and the legend, drawn inside the axes, must sit above every bar under it.
    fig.canvas.draw()
    for ax, (subset, _) in zip(axes, SUBSETS):
        heights = 100 * np.vstack([histograms[(subset, label)] for label in ARMS])
        assert np.allclose(np.array([patch.get_height() for patch in ax.patches]), heights.ravel())
        pixels = ax.transData.transform((width, 0.0))[0] - ax.transData.transform((0.0, 0.0))[0]
        assert pixels >= 3.0, f"bars are {pixels:.2f} px wide in the {subset} panel"
        box = ax.get_legend().get_window_extent().transformed(ax.transData.inverted())
        tallest = float(heights[:, (centers > box.x0) & (centers < box.x1)].max())
        assert tallest < box.y0, f"legend covers a {tallest:.1f} % bar in the {subset} panel (legend floor {box.y0:.1f})"
    fig.savefig(OUT / "error_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    receipt_out = {
        "experiment": "e2000fs_calib_point4_144h",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "folds": int(len(origins)),
        "pairs": int(target.size),
        "bins": {"low_kms": float(EDGES[0]), "high_kms": float(EDGES[-1]), "width_kms": BIN_WIDTH,
                 "pairs_outside": outside},
        "stats": summary,
        "verified_against": {"point4_144h": receipt["mse"], "detpool_gpd": 4360.208035478823},
        "inputs": {label: {"file": str(RUNS / f"{name}_v1")} for label, (name, _) in ARMS.items()},
        "outputs": ["error_distribution.csv", "error_distribution.png"],
        "notes": ["Residual is forecast − truth; every arm is scored on the same pairs, so the bin shares are comparable.",
                  "MSE over the ge550 subset is the ≥550 row of the band table, computed from the same archived archives.",
                  "This is a diagnostic on the published window; nothing here selected a hyper-parameter."],
    }
    (OUT / "error_distribution_receipt.json").write_text(json.dumps(receipt_out, indent=1) + "\n")

    for subset, subtitle in SUBSETS:
        table = pd.DataFrame(summary[subset]).T[["pairs", "bias_kms", "sd_kms", "mse", "mae_kms", "median_kms",
                                                 "p05_kms", "p95_kms", "share_within_50_kms", "share_within_100_kms",
                                                 "share_over_200_kms"]]
        print(f"\n{subset} — {subtitle} — {summary[subset]['point4_144h']['pairs']:,} pairs")
        print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
    print()


if __name__ == "__main__":
    main()
