"""E2000FS DirMO-24 feature importance: refit the submitted F1/F2 block models, then split counts + permutation.

The original boosters were not saved, so the 6 models (stage f1/final x blocks h1-24/h25-48/h49-72) are refit with
the frozen recipe (seed 42) and must reproduce the saved predictions before anything is computed. XGBoost 2.1.4
segfaults on get_score / pred_contribs for multi_output_tree models, so importance is computed by hand:
  split count  internal nodes per feature, read from the booster JSON (summed over the 3 blocks)
  permutation  increase in 72 h MSE on the model's own leaderboard period when a feature is shuffled across origins
Per block the 594 inputs are [90 origin features | 24 hourly speeds | 20 horizon features x 24 horizons]; a horizon
feature's 24 copies are one feature here (counted and shuffled together).

    python feature_importance.py              # fit (or reuse WORK boosters), gate, importance, plot
    python feature_importance.py --plot-only  # redraw feature_importance.png from feature_importance.csv
    python feature_importance.py --pyc-shim DIR  # import experiments.* from compiled caches in DIR (sources lost)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path("/home/t-lab01/sunrun/analysis/forecast_strategies_e2000")
WORK = STUDY / "outputs/feature_importance"
ROOT = Path(__file__).resolve().parents[1]
STAGES = {"f1": ("f1", "F1 model (2026-03..05, 2,137 folds)", 3878.8761443485832),
          "final": ("official", "F2 model (2026-06..09, 2,833 folds)", 3179.592068298927)}
SIGMA, REPEATS, SEED = 24, 5, 20261027
TOP = 25


def kind(name):
    if name.startswith("speed_t-") or name.startswith("v_") or name in ("speed_d24",):
        return "recent speed"
    if name.startswith("ace_") or name == "epam_p1_d6":
        return "ACE plasma / field / particles"
    if name.startswith("rec"):
        return "27-day recurrence"
    if name.startswith("strip_") or name.startswith("ch_") or name.startswith("img_"):
        return "SUVI strips / coronal holes"
    return "WSA-ENLIL"


COLORS = {"recent speed": "#1f77b4", "ACE plasma / field / particles": "#2ca02c", "27-day recurrence": "#9467bd",
          "SUVI strips / coronal holes": "#ff7f0e", "WSA-ENLIL": "#d62728"}


def column_groups(constant, varying, lag, width):
    """Group name of every model input column, in strategies.DirMO.features order."""
    past = [f"speed_t-{lag - 1 - j}" for j in range(lag)]  # data.past_window: origin-lag+1 .. origin
    return np.array(list(constant) + past + [varying[k % len(varying)] for k in range(width * len(varying))])


def split_counts(booster, groups):
    trees = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    counts = np.zeros(len(groups), np.int64)
    for tree in trees:
        left, index = np.asarray(tree["left_children"]), np.asarray(tree["split_indices"])
        np.add.at(counts, index[left != -1], 1)
    return pd.Series(counts, index=groups).groupby(level=0, sort=False).sum(), int(counts.sum())


class Loaded:
    """Saved booster behind the two XGBRegressor methods used here (the sklearn wrapper cannot save with this sklearn)."""

    def __init__(self, path):
        import xgboost as xgb
        self.booster = xgb.Booster(model_file=str(path))

    def predict(self, X):
        return self.booster.inplace_predict(X)

    def get_booster(self):
        return self.booster


def compute(shim=None):
    if shim is not None:  # regular packages in the shim win over the broken source tree's namespace portions
        sys.path.insert(0, str(shim))
    sys.path.insert(0, str(STUDY))
    import data
    import run
    import strategies as S

    train, f1, official, series, _ = data.load(STUDY / "outputs/cache")
    names = json.loads((STUDY / "outputs/cache/names.json").read_text())
    groups = column_groups(names["constant"], names["varying"], data.LAG, SIGMA)
    cfg = S.recipe()
    rows, receipt = [], {}
    for stage, (target, _, lb_mse) in STAGES.items():
        st = run.Stage(stage, train, f1, official, series, WORK / "stage", cfg)
        paths = [WORK / f"{stage}_b{b:02}.json" for b in range(len(S.blocks(SIGMA)))]
        model = S.DirMO(SIGMA, cfg, S.Logger(None), "DirMO-24")
        if all(p.exists() for p in paths):
            model.models = {b: Loaded(p) for b, p in enumerate(paths)}
        else:
            model.fit(st.fit, st.Y, st.valid, st.Yv)
            WORK.mkdir(parents=True, exist_ok=True)
            for b, p in enumerate(paths):
                model.models[b].get_booster().save_model(p)
        inp = st.targets[target]
        pred = np.clip(model.predict(inp), S.PRED_MIN, S.PRED_MAX)
        saved = np.load(STUDY / f"outputs/predictions/{stage}/{target}_DirMO-24.npy")
        truth = data.future_targets(series, st.target_origins)
        diff, mse = float(np.abs(pred - saved).max()), float(np.mean((truth - pred) ** 2))
        print(f"{stage}: refit vs saved max |diff| {diff:.2e} km/s, MSE {mse:.6f} (leaderboard {lb_mse:.6f})", flush=True)
        if diff > 1e-3 or abs(mse - lb_mse) > 1e-6:
            raise SystemExit(f"{stage}: refit does not reproduce the submitted predictions; stopping")
        receipt[stage] = {"max_abs_diff_kms": diff, "mse": mse, "leaderboard_mse": lb_mse, "origins": len(saved),
                          "boosters": [p.name for p in paths]}

        spans = S.blocks(SIGMA)
        X = [model.features(inp, None, s, e) for s, e in spans]
        Yb = [truth[:, s - 1:e] for s, e in spans]
        base = [np.clip(S._predict(model.models[b], X[b], e - s + 1), S.PRED_MIN, S.PRED_MAX) for b, (s, e) in enumerate(spans)]
        base_mse = [float(np.mean((Yb[b] - base[b]) ** 2)) for b in range(len(spans))]
        splits = sum(split_counts(model.models[b].get_booster(), groups)[0] for b in range(len(spans)))
        receipt[stage]["total_splits"] = int(sum(split_counts(model.models[b].get_booster(), groups)[1] for b in range(len(spans))))
        rng = np.random.default_rng(SEED)
        unique = list(dict.fromkeys(groups))
        for i, g in enumerate(unique):
            cols = np.flatnonzero(groups == g)
            deltas = []
            for _ in range(REPEATS):
                perm = rng.permutation(len(inp))
                total = 0.
                for b, (s, e) in enumerate(spans):
                    Xp = X[b].copy()
                    Xp[:, cols] = X[b][perm][:, cols]
                    p = np.clip(S._predict(model.models[b], Xp, e - s + 1), S.PRED_MIN, S.PRED_MAX)
                    total += float(np.mean((Yb[b] - p) ** 2)) - base_mse[b]
                deltas.append(total / len(spans))  # blocks have equal width, so this is the 72 h MSE change
            rows.append({"stage": stage, "feature": g, "kind": kind(g), "split_count": int(splits[g]),
                         "perm_delta_mse_mean": float(np.mean(deltas)), "perm_delta_mse_sd": float(np.std(deltas, ddof=1))})
            if (i + 1) % 20 == 0:
                print(f"{stage}: permuted {i + 1}/{len(unique)} features", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "feature_importance.csv", index=False, float_format="%.6g")
    receipt.update({"repeats": REPEATS, "seed": SEED, "sigma": SIGMA, "recipe": cfg.params()})
    if shim is not None:
        receipt["pyc_shim"] = {str(f.relative_to(shim)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(Path(shim).rglob("*.pyc"))
                               if f.parts[-2] == "e2000" or f.stem == "contracts"}
    (ROOT / "feature_importance_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = pd.read_csv(ROOT / "feature_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for row, (stage, (_, title, _)) in zip(axes, STAGES.items()):
        s = d[d.stage == stage]
        for ax, col, label in ((row[0], "perm_delta_mse_mean", "permutation: increase in 72 h MSE (km/s)$^2$"),
                               (row[1], "split_count", "split count (900 trees x 3 blocks)")):
            top = s.nlargest(TOP, col).iloc[::-1]
            xerr = top.perm_delta_mse_sd if col == "perm_delta_mse_mean" else None
            ax.barh(top.feature, top[col], xerr=xerr, color=[COLORS[k] for k in top.kind], error_kw={"lw": .8})
            ax.set_title(f"{title}\n{label}, top {TOP}", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=.3)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc="lower center", ncol=5, fontsize=9,
               frameon=False)
    fig.suptitle("E2000FS DirMO-24 (seed 42) feature importance: refit of the submitted models (reproduces submitted predictions)\n"
                 "permutation on the already-scored leaderboard period (diagnostic only; 5 shuffles, bar = mean, line = sd); "
                 "correlated features share credit", fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .97))
    fig.savefig(ROOT / "feature_importance.png", dpi=110)
    print("wrote", ROOT / "feature_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--pyc-shim", type=Path)
    args = ap.parse_args()
    if not args.plot_only:
        compute(args.pyc_shim)
    plot()
