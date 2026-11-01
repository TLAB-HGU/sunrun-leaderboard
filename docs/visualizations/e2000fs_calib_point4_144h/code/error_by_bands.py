"""Where the point4 144 h arm loses and wins: error by target-speed band and by horizon.

`e2000fs_calib_point4_144h` is the paired control arm of the tweedie15 lane: the same code path, mask
(`origin + 144 h <= cutoff`), blocks, rounds and seed, with the plain point4 objective (4x weight on targets >= 550 km/s)
instead of Tweedie deviance. This script scores it against the 2026-10-31 refit of `tweedie15` (one loss away) and the
`detpool_gpd` stack that the calib5 lane builds on, all on the same 6,315 folds, and writes the band/horizon tables.
Nothing is retrained; the archived forecasts are only read.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib_point4_144h/code/error_by_bands.py
"""
from __future__ import annotations

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
ARMS = {"point4_144h": ("point4_144h", "#2f5f98"), "tweedie15": ("tweedie15", "#c2553f"),
        "detpool_gpd": ("detpool_gpd", "#4f8a4f")}
BANDS = [(0, 350, "<350"), (350, 450, "350–450"), (450, 550, "450–550"), (550, 650, "550–650"),
         (650, 800, "650–800"), (800, 10_000, "≥800")]


def load(name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    folder = RUNS / f"{name}_v1"
    single = folder / "forecast.npz"
    if single.exists():
        z = np.load(single)
        return pd.DatetimeIndex(z["origins_ns"], tz="UTC"), z["predictions"].astype(np.float64)
    parts = [np.load(folder / f"{period}_predictions.npz") for period in ("F1", "F2", "F3")]
    origins = pd.DatetimeIndex(np.concatenate([z["origins"] for z in parts]), tz="UTC")
    return origins, np.vstack([z["pred"] for z in parts]).astype(np.float64)


def main() -> None:
    receipt = json.loads((OUT / "leaderboard-result.json").read_text())
    folds = pd.read_parquet(V.pinned("folds"))
    truth = pd.read_parquet(V.pinned("truth"))
    origins = pd.DatetimeIndex(folds["origin_last_input_utc"])
    periods = folds["period"].to_numpy()
    target = truth.set_index(V.KEY)["target_kms"].unstack().loc[origins.strftime("%Y-%m-%dT%H:%M:%SZ")].to_numpy()
    assert target.shape == (len(origins), V.H) and np.isfinite(target).all()

    predictions = {}
    for label, (name, _) in ARMS.items():
        arm_origins, pred = load(name)
        assert arm_origins.equals(origins), f"{name} is not on the same 6,315 folds"
        predictions[label] = pred
    overall = {label: float(((pred - target) ** 2).mean()) for label, pred in predictions.items()}
    # detpool_gpd is archived as float32 (forecast.npz), so its recomputed float64 score may differ in the last digits
    for label, archive, tolerance in (("point4_144h", receipt["mse"], 1e-6), ("detpool_gpd", 4360.208035478823, 1e-3)):
        assert abs(overall[label] - archive) < tolerance, (label, overall[label], archive)

    rows = []
    for low, high, name in BANDS:
        sel = (target >= low) & (target < high)
        row = {"band_kms": name, "pairs": int(sel.sum()), "share_pairs": float(sel.mean())}
        for label, pred in predictions.items():
            row[f"mse_{label}"] = float(((pred[sel] - target[sel]) ** 2).mean())
        row["n_folds"] = int(sel.any(axis=1).sum())
        rows.append(row)
    bands = pd.DataFrame(rows)
    bands.to_csv(OUT / "error_by_band.csv", index=False, float_format="%.6f")

    horizon = pd.DataFrame({"horizon_hours": np.arange(1, V.H + 1)})
    for label, pred in predictions.items():
        horizon[f"mse_{label}"] = ((pred - target) ** 2).mean(axis=0)
    horizon.to_csv(OUT / "error_by_horizon.csv", index=False, float_format="%.6f")

    period_rows = []
    for period in ("F1", "F2", "F3"):
        sel = np.broadcast_to((periods == period)[:, None], target.shape)
        row = {"period": period, "pairs": int(sel.sum())}
        for label, pred in predictions.items():
            row[f"mse_{label}"] = float(((pred - target) ** 2)[sel].mean())
            row[f"mse_{label}_ge550"] = float(((pred - target) ** 2)[sel & (target >= 550)].mean())
        period_rows.append(row)
    periods_table = pd.DataFrame(period_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(len(bands))
    width = 0.26
    for offset, (label, (_, color)) in enumerate(ARMS.items()):
        axes[0].bar(positions + (offset - 1) * width, bands[f"mse_{label}"], width=width, color=color,
                    label=f"{label} (all {overall[label]:,.0f})")
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"{b}\n{100 * s:.0f}% of pairs" for b, s in zip(bands["band_kms"], bands["share_pairs"])],
                            fontsize=8.5)
    axes[0].set_xlabel("target speed band (km/s, share of the 454,680 pairs)", fontsize=9)
    axes[0].set_ylabel("MSE (km/s)², log scale", fontsize=9)
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8, title="overall MSE in the legend", title_fontsize=7.5)
    axes[0].set_title("MSE by target-speed band", fontsize=10.5, loc="left")
    for label, (_, color) in ARMS.items():
        axes[1].plot(horizon["horizon_hours"], horizon[f"mse_{label}"], color=color, lw=1.3, label=label)
    axes[1].set_xlabel("forecast horizon (hours after the origin)", fontsize=9)
    axes[1].set_ylabel("MSE (km/s)²", fontsize=9)
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title("MSE by horizon", fontsize=10.5, loc="left")
    fig.text(.012, -.03, "All three forecasts are scored on the same 6,315 folds with the leaderboard truth. "
                         "point4_144h and tweedie15 share code path, mask, blocks, rounds and seed and differ only in the "
                         "training loss; detpool_gpd is the base of the calib5 stack.", fontsize=7.6, color="#555555")
    fig.savefig(OUT / "error_by_band.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    receipt_out = {
        "experiment": "e2000fs_calib_point4_144h",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "folds": int(len(origins)),
        "pairs": int(target.size),
        "overall_mse": overall,
        "verified_against": {"point4_144h": receipt["mse"], "detpool_gpd": 4360.208035478823},
        "inputs": {label: {"file": str(RUNS / f"{name}_v1"), "forecast_sha256_12": V.sha(
            (RUNS / f"{name}_v1/forecast.npz") if (RUNS / f"{name}_v1/forecast.npz").exists()
            else (RUNS / f"{name}_v1/F1_predictions.npz"))[:12]} for label, (name, _) in ARMS.items()},
        "outputs": ["error_by_band.csv", "error_by_horizon.csv", "error_by_band.png"],
        "notes": ["Bands are target speed (truth), not forecast speed; the same pairs are counted for every arm.",
                  "This is a diagnostic on the published window; nothing here selected a hyper-parameter."],
    }
    (OUT / "error_by_band_receipt.json").write_text(json.dumps(receipt_out, indent=1) + "\n")
    print(bands.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print(periods_table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
