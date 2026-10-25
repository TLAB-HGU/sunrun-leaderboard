"""Plot seed-0 per-epoch train/validation reconstruction loss."""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
curves = json.loads((HERE / "encoder_curves_s0.json").read_text())
receipts_raw = json.loads((HERE / "encoder_receipts.json").read_text())
receipts = {row["fold"]: row for row in receipts_raw.values()}
rows = [row for row in curves if "epoch" in row]
by_fold = collections.defaultdict(list)
for row in rows:
    by_fold[row["fold"]].append(row)

fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=False)
for ax, fold in zip(axes.ravel(), sorted(by_fold)):
    fold_rows = sorted(by_fold[fold], key=lambda row: row["epoch"])
    receipt = receipts[fold]
    epochs = [row["epoch"] for row in fold_rows]
    ax.plot(epochs, [row["train_loss"] for row in fold_rows], color="#1f4e79", lw=1.8, label="train")
    ax.plot(epochs, [row["valid_loss"] for row in fold_rows], color="#c00000", lw=1.8, label="validation")
    ax.axvline(receipt["best_epoch"], color="#c00000", ls=":", lw=1, alpha=.7)
    ax.set_title(
        f'fold {fold} — cutoff {receipt["cutoff_utc"][:10]} '
        f'(train {receipt["train_origins"]}, valid {receipt["validation_origins"]})',
        fontsize=10,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth L1 reconstruction loss")
    ax.grid(alpha=.25)
    ax.text(
        .98, .95,
        f'best epoch {receipt["best_epoch"]}\nbest valid {receipt["best_valid_loss"]:.4f}',
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#cccccc", alpha=.9),
    )
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .76))
fig.suptitle(
    "E1466-style TD-WSA field encoder — seed 0 per-epoch loss "
    "(reproduction, 6 shortened folds)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(HERE / "e1466_epoch_loss_seed0.png", dpi=170)
print("wrote", HERE / "e1466_epoch_loss_seed0.png")
