"""Copy the native per-round XGBoost MSE logs of one submitted model into its docs folder and plot them.

Rows are the F1 and F2 models; columns are the booster roles of the strategy (blocks / horizons, and for
Rectify-type strategies the purged OOF bases, the full base and the residual). Every booster is drawn, train
solid and validation dashed, coloured by block or horizon order. No smoothing, no reconstruction.

    python plot_loss.py --experiment e2000fs_dirmo_12 --out DIR
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STUDY = Path("/home/t-lab01/sunrun/analysis/forecast_strategies_e2000")
E2000 = Path("/home/t-lab01/sunrun/research/e2000/publication")
ORIGINAL = {"e2000_direct_s42": "E1987", "e2000_horizon_conditioned_s42": "E1988"}
STAGE_TITLES = {"f1": "F1 model (fit targets < 2026-02-26)", "final": "F2 model (fit targets < 2026-05-29)"}


def sources(experiment):
    """{stage: [csv paths]} of the logs behind the submitted forecast."""
    if experiment in ORIGINAL:
        root = E2000 / ORIGINAL[experiment]
        return {"f1": sorted((root / "f1").glob("loss_history_s42_*.csv")),
                "final": sorted(root.glob("loss_history_s42_*.csv"))}
    done = json.loads((STUDY / "publication/submission.json").read_text())
    name = next(n for n, r in done.items() if r["experiment"] == experiment)
    import sys
    sys.path.insert(0, str(STUDY))
    import run
    source = run.canonical(name)
    return {s: sorted((STUDY / "outputs/loss_history" / s / source).glob("*.csv")) for s in ("f1", "final")}


def role(path):
    stem = path.stem.removesuffix("_s42")
    if stem.startswith("loss_history_s42_"):
        return "per-horizon models" if re.search(r"_h\d\d$", stem) else "pooled model"
    if stem.startswith("base_") and "_oof" in stem:
        return "base, purged OOF folds"
    if stem.startswith("base_") and "_full" in stem:
        return "base, full fit"
    if stem.startswith("res_"):
        return "residual"
    return "blocks" if re.search(r"_b\d\d$", stem) else "shared model"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    found = sources(args.experiment)
    roles = list(dict.fromkeys(role(p) for s in found.values() for p in s))
    summary = []
    fig, axes = plt.subplots(2, len(roles), figsize=(4.8 * len(roles) + 1.5, 7.5), squeeze=False, sharex=True)
    for row, (stage, paths) in zip(axes, found.items()):
        dest = args.out / "loss_history" / stage
        dest.mkdir(parents=True, exist_ok=True)
        for ax, r in zip(row, roles):
            group = [p for p in paths if role(p) == r]
            colors = plt.cm.viridis(np.linspace(0, .9, max(len(group), 1)))
            last = []
            for c, p in zip(colors, group):
                shutil.copyfile(p, dest / p.name)
                d = pd.read_csv(p)
                ax.plot(d["step"], d["train_mse"], color=c, lw=.9 if len(group) > 8 else 1.4)
                ax.plot(d["step"], d["validation_mse"], color=c, lw=.9 if len(group) > 8 else 1.4, ls="--")
                last.append(d.iloc[-1])
            if last:
                t = np.array([x.train_mse for x in last])
                v = np.array([x.validation_mse for x in last])
                summary.append({"stage": stage, "role": r, "boosters": len(group), "rounds": int(last[0].step),
                                "train_min": t.min(), "train_median": float(np.median(t)), "train_max": t.max(),
                                "validation_min": v.min(), "validation_median": float(np.median(v)),
                                "validation_max": v.max()})
                ax.set_title(f"{STAGE_TITLES[stage]}\n{r}: {len(group)} booster(s), round {int(last[0].step)} "
                             f"median train {np.median(t):,.0f} / val {np.median(v):,.0f}", fontsize=9)
            ax.set_yscale("log")
            ax.grid(alpha=.3)
        row[0].set_ylabel("MSE (km/s)$^2$")
    for ax in axes[1]:
        ax.set_xlabel("boosting round")
    axes[0, 0].plot([], [], color="k", label="train")
    axes[0, 0].plot([], [], color="k", ls="--", label="validation")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{args.label} (seed 42): native per-round XGBoost MSE of the submitted F1 and F2 models "
                 "(colour = block / horizon order)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "loss_history.png", dpi=130)
    pd.DataFrame(summary).to_csv(args.out / "loss_summary.csv", index=False, float_format="%.6g")
    print("wrote", args.out / "loss_history.png", sum(len(v) for v in found.values()), "csv")


if __name__ == "__main__":
    main()
