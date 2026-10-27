"""E2000FS feature importance for any strategy: refit the submitted F1/F2 models, then split counts + permutation.

Generalises docs/visualizations/e2000fs_dirmo_24/code/feature_importance.py to every strategy of
analysis/forecast_strategies_e2000. Boosters were not saved, so each stage's model (all blocks, and for
Rectify-type strategies the purged OOF bases, the full base and the residual blocks) is refit with the frozen recipe
(seed 42) and must reproduce the saved submitted predictions before anything is computed.
  split count  internal nodes per input group, read from every booster's JSON (summed over all boosters)
  permutation  increase in 72 h MSE on the model's own leaderboard period when one input group is shuffled across
               origins; the whole strategy is re-run, so recursive and residual paths propagate the shuffle
Input groups are the 134 origin-level inputs: 90 origin features, 24 hourly speeds (speed_t-23..speed_t-0) and 20
horizon features (all 72 horizon copies shuffled together). Split counts additionally report model-internal inputs
that are not permutable: lagged trajectory (RecMO), earlier-horizon forecasts (DirRecMO), lead, base forecast
(residual models).

    python feature_importance.py --strategy DirMO-12 --out DIR             # fit, gate, importance, plot
    python feature_importance.py --strategy DirMO-12 --out DIR --plot-only # redraw from DIR/feature_importance.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path("/home/t-lab01/sunrun/analysis/forecast_strategies_e2000")
SUBMISSIONS = STUDY / "publication/submission.json"
STAGES = {"f1": ("f1", "F1 model (2026-03..05, 2,137 folds)", "mse_f1"),
          "final": ("official", "F2 model (2026-06..09, 2,833 folds)", "mse_f2")}
REPEATS, SEED, TOP = 5, 20261027, 25
INTERNAL = "model-internal (not permuted)"


def kind(name):
    if name in ("lagged trajectory", "earlier-horizon forecasts", "lead", "base forecast"):
        return INTERNAL
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
          "SUVI strips / coronal holes": "#ff7f0e", "WSA-ENLIL": "#d62728", INTERNAL: "#7f7f7f"}


def block_groups(S, LAG, model, ctx_names, past_names, varying, s, e):
    """Group name of every input column of one block model, in the strategies.*.features order."""
    horizon = [varying[k % len(varying)] for k in range((e - s + 1) * len(varying))]
    if isinstance(model, S.RecMO):
        return list(ctx_names) + ["lagged trajectory"] * LAG + horizon + ["lead"]
    if isinstance(model, S.DirRecMO):
        return list(ctx_names) + past_names + ["earlier-horizon forecasts"] * (s - 1) + horizon
    return list(ctx_names) + past_names + horizon


def boosters(S, LAG, model, constant, varying):
    """(booster, column groups) for every fitted booster that produces the submitted forecast."""
    past = [f"speed_t-{LAG - 1 - j}" for j in range(LAG)]
    out = []

    def block(m, ctx_names):
        for b, (s, e) in enumerate(S.blocks(m.sigma)):
            if m.shared and b:
                continue
            reg = m.models[0 if m.shared else b]
            groups = block_groups(S, LAG, m, ctx_names, past, varying, s, e)
            if m.shared:  # one model over every block: the horizon slot holds sigma copies
                groups = block_groups(S, LAG, m, ctx_names, past, varying, 1, m.sigma)
            out.append((reg.get_booster(), np.array(groups)))

    if isinstance(model, S.Rectify):
        block(model.base, constant)
        block(model.res, list(constant) + past + ["base forecast"] * S.H)
    else:
        block(model, constant)
    return out


def split_counts(booster, groups):
    if booster.num_features() != len(groups):
        raise SystemExit(f"column map has {len(groups)} entries, booster has {booster.num_features()} features")
    trees = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    counts = np.zeros(len(groups), np.int64)
    for tree in trees:
        left, index = np.asarray(tree["left_children"]), np.asarray(tree["split_indices"])
        np.add.at(counts, index[left != -1], 1)
    return pd.Series(counts, index=groups).groupby(level=0, sort=False).sum()


def permuted(S, inp, part, col, perm):
    ctx, E, past = inp.ctx.copy(), inp.E.copy(), inp.past.copy()
    if part == "ctx":
        ctx[:, col] = inp.ctx[perm, col]
    elif part == "past":
        past[:, col] = inp.past[perm, col]
    else:
        E[:, :, col] = inp.E[perm][:, :, col]
    return S.Inputs(ctx, E, past)


def compute(strategy, out):
    sys.path.insert(0, str(STUDY))
    import data
    import run
    import strategies as S

    record = json.loads(SUBMISSIONS.read_text())[strategy]
    train, f1, official, series, _ = data.load(STUDY / "outputs/cache")
    names = json.loads((STUDY / "outputs/cache/names.json").read_text())
    constant, varying = names["constant"], names["varying"]
    source = run.canonical(strategy)
    cfg = S.recipe()
    rows, receipt = [], {"strategy": strategy, "canonical": source, "experiment": record["experiment"],
                         "config_sha256": record["config_sha256"]}
    work = Path(__file__).resolve().parent / "work" / source
    for stage, (target, _, metric) in STAGES.items():
        t0 = time.time()
        st = run.Stage(stage, train, f1, official, series, work, cfg)
        model = S.build(source, cfg, S.Logger(None))
        if isinstance(model, S.Rectify):
            model.fit(st.fit, st.Y, st.valid, st.Yv, st.origins)
        else:
            model.fit(st.fit, st.Y, st.valid, st.Yv)
        fit_s = time.time() - t0
        inp = st.targets[target]
        pred = np.clip(model.predict(inp), S.PRED_MIN, S.PRED_MAX)
        saved = np.load(STUDY / f"outputs/predictions/{stage}/{target}_{source}.npy")
        truth = data.future_targets(series, st.target_origins)
        diff, base = float(np.abs(pred - saved).max()), float(np.mean((truth - pred) ** 2))
        print(f"{strategy} {stage}: refit {fit_s / 60:.1f} min, max |diff| {diff:.2e} km/s, MSE {base:.6f} "
              f"(leaderboard {record[metric]:.6f})", flush=True)
        if diff > 1e-3 or abs(base - record[metric]) > 1e-6:
            raise SystemExit(f"{stage}: refit does not reproduce the submitted predictions; stopping")
        fitted = boosters(S, data.LAG, model, constant, varying)
        splits = pd.concat([split_counts(b, g) for b, g in fitted]).groupby(level=0, sort=False).sum()
        receipt[stage] = {"max_abs_diff_kms": diff, "mse": base, "leaderboard_mse": record[metric],
                          "origins": len(saved), "boosters": len(fitted), "refit_seconds": fit_s,
                          "total_splits": int(splits.sum())}
        groups = ([("ctx", i, n) for i, n in enumerate(constant)]
                  + [("past", j, f"speed_t-{data.LAG - 1 - j}") for j in range(data.LAG)]
                  + [("E", m, n) for m, n in enumerate(varying)])
        rng = np.random.default_rng(SEED)
        for i, (part, col, name) in enumerate(groups):
            deltas = []
            for _ in range(REPEATS):
                perm = rng.permutation(len(inp))
                p = np.clip(model.predict(permuted(S, inp, part, col, perm)), S.PRED_MIN, S.PRED_MAX)
                deltas.append(float(np.mean((truth - p) ** 2)) - base)
            rows.append({"stage": stage, "feature": name, "kind": kind(name), "split_count": int(splits.get(name, 0)),
                         "perm_delta_mse_mean": float(np.mean(deltas)), "perm_delta_mse_sd": float(np.std(deltas, ddof=1))})
            if (i + 1) % 20 == 0:
                print(f"{strategy} {stage}: permuted {i + 1}/{len(groups)} features", flush=True)
        for name in [g for g in splits.index if kind(g) == INTERNAL]:
            rows.append({"stage": stage, "feature": name, "kind": INTERNAL, "split_count": int(splits[name]),
                         "perm_delta_mse_mean": np.nan, "perm_delta_mse_sd": np.nan})
        receipt[stage]["permutation_seconds"] = time.time() - t0 - fit_s
    pd.DataFrame(rows).to_csv(out / "feature_importance.csv", index=False, float_format="%.6g")
    receipt.update({"repeats": REPEATS, "seed": SEED, "recipe": cfg.params(), "permutation_groups": len(groups)})
    (out / "feature_importance_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def plot(strategy, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = pd.read_csv(out / "feature_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for row, (stage, (_, title, _)) in zip(axes, STAGES.items()):
        s = d[d.stage == stage]
        for ax, col, text in ((row[0], "perm_delta_mse_mean", "permutation: increase in 72 h MSE (km/s)$^2$"),
                              (row[1], "split_count", "split count (all boosters)")):
            top = s.dropna(subset=[col]).nlargest(TOP, col).iloc[::-1]
            xerr = top.perm_delta_mse_sd if col == "perm_delta_mse_mean" else None
            ax.barh(top.feature, top[col], xerr=xerr, color=[COLORS[k] for k in top.kind], error_kw={"lw": .8})
            ax.set_title(f"{title}\n{text}, top {TOP}", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=.3)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc="lower center", ncol=6, fontsize=9,
               frameon=False)
    fig.suptitle(f"{label} (seed 42) feature importance: refit of the submitted models (reproduces submitted predictions)\n"
                 "permutation on the already-scored leaderboard period (diagnostic only; 5 shuffles, bar = mean, line = sd); "
                 "correlated features share credit", fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .97))
    fig.savefig(out / "feature_importance.png", dpi=110)
    print("wrote", out / "feature_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        compute(args.strategy, args.out)
    plot(args.strategy, args.out, args.label or f"E2000FS {args.strategy}")
