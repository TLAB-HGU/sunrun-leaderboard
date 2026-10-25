"""Plot reconstructed E1813 per-round train/validation MSE (reads the CSVs written by reconstruct.py)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "e1813_loss_history_long.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
panels = [("all", "Combined (each head on the rows it serves)"), ("short", "Short head (h 1-6)"),
          ("mid", "Mid head (h 7-24)"), ("long", "Long head (fit h 1-72, serves h 25-72)")]
for ax, (head, title) in zip(axes.ravel(), panels):
    for seed, ls in (("0", "-"), ("1", "--"), ("2", ":")):
        for split, color in (("train", "#1f77b4"), ("validation", "#d62728")):
            g = d[(d["split"] == split) & (d["seed"] == seed) & (d["head"] == head)]
            if len(g):
                ax.plot(g["round"], g["mse"], ls, color=color, lw=1.2, label=f"{split} s{seed}")
    if head == "all":
        for split, color in (("train", "#1f77b4"), ("validation", "#d62728")):
            g = d[(d["split"] == split) & (d["seed"] == "mean") & (d["head"] == "all")]
            ax.plot(g["round"], g["mse"], color=color, lw=2.4, alpha=0.5, label=f"{split} seed-mean pred")
    ax.set_title(title, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("MSE (km/s)$^2$")
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("boosting round")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("E1813 direct HUXt XGBoost: per-round MSE, reconstructed post hoc from saved official-fit models\n"
             "train = fit rows (targets < 2026-05-29); validation = leaderboard window Jun-Sep 2026 (already exposed, diagnostic only)",
             fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "loss_history.png", dpi=130)
print("wrote", HERE / "loss_history.png")
