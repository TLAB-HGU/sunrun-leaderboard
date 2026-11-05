"""Copy and plot the per-round XGBoost loss logs behind the budget-spectral submission.

Every curve is the `evals_result` history that XGBoost wrote while training the archived models: 900 fixed
rounds, no early stopping, no smoothing, no reconstruction. Each stage trained one booster per 24 h block of
the 72 h horizon (b0/b1/b2, 24 outputs per tree) for two paired arms:

  control   frozen point4 weighted L2 (550 km/s targets counted four times, mean-normalized per horizon)
  spectral  the same anchor plus lambda 0.10 radial pseudo-Huber penalty on the unitary FFT of the 24 h error

`weighted_mse` is the point4 objective in (km/s)^2 and is *not* the official unweighted leaderboard MSE;
`spectral_loss` is the FFT penalty term in (km/s)^2. Rows of the main figure are the two official stages
(F1 targets < 2026-02-26, F2 targets < 2026-05-29); the development figure shows the four earlier windows
(D1/D2/D3/C) that were cut before the official grids.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_budget_spectral/code/plot_loss.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
STUDY = ROOT / "research/e2000fs_budget_spectral"
RESULTS = STUDY / "results"
LOSSES = RESULTS / "runs/full_v1/publication/package/releases/e2000fs_budget_spectral/losses"
OUT = Path(__file__).resolve().parents[1]
COLUMNS = ["round", "train_rmse", "train_weighted_mse", "train_spectral_loss", "train_joint_loss",
           "validation_rmse", "validation_weighted_mse", "validation_spectral_loss", "validation_joint_loss"]
OFFICIAL = [("f1", "F1 model — fit targets < 2026-02-26"), ("final", "F2 model — fit targets < 2026-05-29")]
DEV = [("D1", "D1 2024-12-04 → 2025-02-22"), ("D2", "D2 2025-03-04 → 2025-05-22"),
       ("D3", "D3 2025-09-04 → 2025-11-22"), ("C", "C 2025-12-04 → 2026-02-22")]
ARMS = ("control", "spectral")
# The two arms often sit on top of each other, so the control is drawn as a wide line underneath the
# thin spectral line; where they coincide the control still shows as a halo.
ARM_STYLE = {"control": dict(color="#4c72b0", ls="-", lw=2.8, alpha=1., label="control (point4 L2)"),
             "spectral": dict(color="#c44e52", ls="-", lw=1.2, alpha=1., label="spectral (submitted)")}


def load(name):
    return pd.read_csv(LOSSES / f"{name}.csv")


def collect():
    """{stage: {(arm, block): frame}} for every archived boosters, plus the copied log names."""
    frames, names = {}, []
    for stage, _ in OFFICIAL + DEV:
        for arm in ARMS:
            for block in range(3):
                name = f"{stage}_{arm}_b{block}"
                path = LOSSES / f"{name}.csv"
                frame = pd.read_csv(path)
                assert list(frame.columns) == COLUMNS, f"unexpected columns in {path}"
                assert len(frame) == 900 and frame["round"].iloc[0] == 1 and frame["round"].iloc[-1] == 900
                assert np.isfinite(frame[COLUMNS[1:]].to_numpy()).all(), f"non-finite loss in {path}"
                frames[(stage, arm, block)] = frame
                names.append(name)
    return frames, names


def mean_over_blocks(frames, stage, arm, column):
    values = [frames[(stage, arm, block)][column].to_numpy() for block in range(3)]
    assert all(len(v) == 900 for v in values)
    return np.mean(values, axis=0)


def summary(frames):
    rows = []
    for stage, _ in OFFICIAL + DEV:
        for arm in ARMS:
            for block in range(3):
                frame = frames[(stage, arm, block)]
                best = int(frame["validation_weighted_mse"].idxmin())
                rows.append(dict(stage=stage, arm=arm, block=block,
                                 train_weighted_mse_last=float(frame["train_weighted_mse"].iloc[-1]),
                                 validation_weighted_mse_last=float(frame["validation_weighted_mse"].iloc[-1]),
                                 validation_weighted_mse_min=float(frame["validation_weighted_mse"].iloc[best]),
                                 validation_weighted_mse_min_round=int(frame["round"].iloc[best]),
                                 train_spectral_loss_last=float(frame["train_spectral_loss"].iloc[-1]),
                                 validation_spectral_loss_last=float(frame["validation_spectral_loss"].iloc[-1]),
                                 train_joint_loss_last=float(frame["train_joint_loss"].iloc[-1]),
                                 validation_joint_loss_last=float(frame["validation_joint_loss"].iloc[-1]),
                                 rounds=len(frame)))
    return pd.DataFrame(rows)


def plot_official(frames):
    rounds = np.arange(1, 901)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, constrained_layout=True)
    panels = (("train_weighted_mse", "validation_weighted_mse", "point4 weighted MSE (km/s)²"),
              ("train_spectral_loss", "validation_spectral_loss", "spectral penalty term (km/s)²"))
    for row, (stage, title) in enumerate(OFFICIAL):
        for col, (train_column, validation_column, ylabel) in enumerate(panels):
            ax = axes[row, col]
            for arm in ARMS:
                style = ARM_STYLE[arm]
                ax.plot(rounds, mean_over_blocks(frames, stage, arm, train_column),
                        color=style["color"], ls="-", lw=style["lw"], alpha=style["alpha"],
                        label=f"{style['label']} — train")
                ax.plot(rounds, mean_over_blocks(frames, stage, arm, validation_column),
                        color=style["color"], ls="--", lw=style["lw"], alpha=style["alpha"],
                        label=f"{style['label']} — validation")
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            if row == 1:
                ax.set_xlabel("boosting round")
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, loc="upper right", framealpha=.9)
    fig.suptitle("e2000fs_budget_spectral — per-round loss of the submitted spectral arm and its paired control\n"
                 "each line is the mean of the three 24-output boosters (b0/b1/b2); solid train, dashed validation; "
                 "900 fixed rounds, no smoothing", fontsize=11.5)
    fig.savefig(OUT / "loss_history.png", dpi=130)
    plt.close(fig)



def plot_blocks(frames):
    rounds = np.arange(1, 901)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(.05, .8, 3))
    for row, (stage, title) in enumerate(OFFICIAL):
        for block in range(3):
            ax = axes[row, block]
            spectral = frames[(stage, "spectral", block)]
            control = frames[(stage, "control", block)]
            ax.plot(rounds, spectral.train_weighted_mse, color=colors[block], lw=1.5,
                    label="spectral — train")
            ax.plot(rounds, spectral.validation_weighted_mse, color=colors[block], lw=1.5, ls="--",
                    label="spectral — validation")
            ax.plot(rounds, control.train_weighted_mse, color="#8c8c8c", lw=1.1, label="control — train")
            ax.plot(rounds, control.validation_weighted_mse, color="#8c8c8c", lw=1.1, ls="--",
                    label="control — validation")
            hours = f"h{block * 24 + 1}–h{block * 24 + 24}"
            ax.set_title(f"{title.split(' —')[0]} · booster b{block} ({hours})", fontsize=10.5)
            ax.set_ylabel("point4 weighted MSE (km/s)²")
            ax.grid(alpha=.25)
            if row == 1:
                ax.set_xlabel("boosting round")
            if row == 0 and block == 0:
                ax.legend(fontsize=8.5, loc="upper right", framealpha=.9)
    fig.suptitle("Per-booster loss: one 24-output vector tree per block of the 72 h horizon", fontsize=11.5)
    fig.savefig(OUT / "loss_history_blocks.png", dpi=130)
    plt.close(fig)


def plot_development(frames):
    rounds = np.arange(1, 901)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=False, constrained_layout=True)
    for ax, (stage, title) in zip(axes, DEV):
        for arm in ARMS:
            style = ARM_STYLE[arm]
            ax.plot(rounds, mean_over_blocks(frames, stage, arm, "train_weighted_mse"),
                    color=style["color"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{style['label']} — train")
            ax.plot(rounds, mean_over_blocks(frames, stage, arm, "validation_weighted_mse"),
                    color=style["color"], lw=style["lw"], alpha=style["alpha"], ls="--",
                    label=f"{style['label']} — held-out window")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("boosting round")
        ax.grid(alpha=.25)
        if stage == "D1":
            ax.set_ylabel("point4 weighted MSE (km/s)²")
            ax.legend(fontsize=8, loc="upper right", framealpha=.9)
    fig.suptitle("Development windows (fit on mature rows before each window, purge 72 h): the internal check\n"
                 "mean of the three boosters; the spectral arm was not consistently better here, which is why the\n"
                 "internal check failed and neither arm was promoted by it", fontsize=10.5)
    fig.savefig(OUT / "loss_history_development.png", dpi=130)
    plt.close(fig)


def main():
    frames, names = collect()
    (OUT / "loss_history").mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(LOSSES / f"{name}.csv", OUT / "loss_history" / f"{name}.csv")
    out = summary(frames)
    out.to_csv(OUT / "loss_summary.csv", index=False)
    plot_official(frames)
    plot_blocks(frames)
    plot_development(frames)
    files = [p.name for p in (OUT / "loss_history.png", OUT / "loss_history_blocks.png",
                              OUT / "loss_history_development.png", OUT / "loss_summary.csv")]
    (OUT / "loss_receipt.json").write_text(json.dumps(dict(
        source="research/e2000fs_budget_spectral/results/runs/full_v1/publication/package/releases/"
               "e2000fs_budget_spectral/losses",
        method="XGBoost evals_result history written during training; no smoothing or reconstruction; "
               "official stages are 'f1' and 'final', development windows are D1/D2/D3/C",
        columns=COLUMNS,
        note="weighted_mse is the point4 training objective, not the official unweighted leaderboard MSE",
        loss_logs=names,
        outputs=files), indent=2) + "\n")
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
