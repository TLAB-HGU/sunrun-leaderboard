#!/usr/bin/env python3
"""Copy and plot the native loss logs of the submitted Feat30 Extreme models."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STUDY = Path("/home/t-lab01/sunrun/analysis/e2000fs_feat30_followups")
RUN = STUDY / "runs/full_20261028_v1/extreme"
STAGE_TITLES = {
    "f1": "F1 model (fit targets < 2026-02-26)",
    "final": "F2 model (fit targets < 2026-05-29)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (stage, title) in zip(axes, STAGE_TITLES.items()):
        paths = sorted((RUN / stage / "extreme_4.0").glob("b??_loss.csv"))
        if len(paths) != 3:
            raise SystemExit(f"{stage}: expected exactly three block loss logs, found {len(paths)}")
        destination = args.out / "loss_history" / stage
        destination.mkdir(parents=True, exist_ok=True)
        colors = plt.cm.viridis(np.linspace(0, 0.9, 3))
        for block, (color, path) in enumerate(zip(colors, paths)):
            shutil.copyfile(path, destination / path.name)
            data = pd.read_csv(path)
            expected = {"round", "train_mse", "diagnostic_validation_mse", "validation_is_held_out"}
            if set(data) != expected or len(data) != 900 or data.validation_is_held_out.any():
                raise SystemExit(f"unexpected native loss contract: {path}")
            label = f"h{block * 24 + 1}–{(block + 1) * 24}"
            ax.plot(data["round"], data.train_mse, color=color, lw=1.3, label=f"{label} train")
            ax.plot(data["round"], data.diagnostic_validation_mse, color=color, lw=1.3,
                    linestyle="--", label=f"{label} validation")
            last = data.iloc[-1]
            summary.append({
                "stage": stage,
                "block": label,
                "rounds": int(last["round"]),
                "train_mse": float(last.train_mse),
                "validation_mse": float(last.diagnostic_validation_mse),
                "validation_is_held_out": False,
            })
        ax.set_title(title)
        ax.set_ylabel("MSE (km/s)$^2$")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("boosting round")
    fig.suptitle("E2000FS Feat30 Extreme (seed 42): native per-round XGBoost MSE\n"
                 "solid = train, dashed = overlapping diagnostic validation")
    fig.tight_layout()
    fig.savefig(args.out / "loss_history.png", dpi=130)
    pd.DataFrame(summary).to_csv(args.out / "loss_summary.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
