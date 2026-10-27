"""Plot E2000FS DirMO-24 per-round train/validation MSE (native XGBoost eval logs copied to ../loss_history)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGES = [("f1", "F1 model (fit targets < 2026-02-26)"), ("final", "F2 model (fit targets < 2026-05-29)")]
BLOCKS = [("b00", "h 1-24"), ("b01", "h 25-48"), ("b02", "h 49-72")]

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for row, (stage, stage_title) in zip(axes, STAGES):
    for ax, (block, block_title) in zip(row, BLOCKS):
        d = pd.read_csv(ROOT / "loss_history" / stage / f"DirMO-24_{block}_s42.csv")
        ax.plot(d["step"], d["train_mse"], color="#1f77b4", lw=1.4, label="train")
        ax.plot(d["step"], d["validation_mse"], color="#ff7f0e", lw=1.4, label="validation (in-sample)")
        ax.axvline(d["step"].iloc[-1], color="#666666", ls=":", lw=1)
        last = d.iloc[-1]
        ax.set_title(f"{stage_title}\n{block_title}: round 900 train {last.train_mse:,.0f} / val {last.validation_mse:,.0f}",
                     fontsize=9)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        print(f"{stage} {block}: round {int(last.step)} train {last.train_mse:.3f} validation {last.validation_mse:.3f}")
    row[0].set_ylabel("MSE (km/s)$^2$")
for ax in axes[1]:
    ax.set_xlabel("boosting round (900 used)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("E2000FS DirMO-24 (seed 42): native per-round XGBoost MSE of the submitted F1 and F2 models\n"
             "validation = origins 2025-12-04..2026-02-22, which lie inside both training windows (diagnostic only, not held-out)",
             fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "loss_history.png", dpi=130)
print("wrote", ROOT / "loss_history.png")
