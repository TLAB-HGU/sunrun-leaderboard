#!/usr/bin/env python3
"""Verify saved Feat30 Extreme boosters, then compute grouped split and permutation importance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

STUDY = Path("/home/t-lab01/sunrun/analysis/e2000fs_feat30_followups")
RUN = STUDY / "runs/full_20261028_v1/extreme"
SUBMISSION = Path("/home/t-lab01/sunrun/analysis/e2000fs-followup-publication-20261028/submission.json")
EXPERIMENT = "e2000fs_feat30_extreme"
STAGES = {
    "f1": ("f1", "F1 model (2026-03..05, 2,137 folds)", "mse_f1"),
    "final": ("official", "F2 model (2026-06..09, 2,833 folds)", "mse_f2"),
}
REPEATS = 5
SEED = 20261028
TOP = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kind(name: str) -> str:
    if name.startswith("v_") or name == "ace_speed_kms_last":
        return "recent speed"
    if name.startswith("ace_"):
        return "ACE plasma / field / particles"
    if name.startswith("rec"):
        return "27-day recurrence"
    if name.startswith(("strip_", "ch_", "img_")):
        return "SUVI strips / coronal holes"
    return "WSA-ENLIL"


COLORS = {
    "recent speed": "#1f77b4",
    "ACE plasma / field / particles": "#2ca02c",
    "27-day recurrence": "#9467bd",
    "SUVI strips / coronal holes": "#ff7f0e",
    "WSA-ENLIL": "#d62728",
}


def submission_record() -> dict:
    records = json.loads(SUBMISSION.read_text())
    for record in records.values():
        if record.get("experiment") == EXPERIMENT:
            return record
    raise SystemExit(f"missing publication record: {EXPERIMENT}")


def load_models(stage: str) -> list[xgb.Booster]:
    models = []
    for block in range(3):
        path = RUN / stage / "extreme_4.0" / f"b{block:02}.ubj"
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        if sha256(path) != receipt["sha256"] or receipt["columns"] != 237:
            raise SystemExit(f"saved model receipt mismatch: {path}")
        model = xgb.Booster(model_file=str(path))
        model.set_param({"nthread": 3})
        models.append(model)
    return models


def predict(run_module, models, bundle) -> np.ndarray:
    return np.clip(np.hstack([
        model.inplace_predict(run_module.design(bundle, block))
        for block, model in enumerate(models)
    ]), 1.0, 3000.0)


def split_counts(model: xgb.Booster, groups: np.ndarray) -> pd.Series:
    if model.num_features() != len(groups):
        raise SystemExit(f"column map has {len(groups)} entries, booster has {model.num_features()}")
    trees = json.loads(model.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    counts = np.zeros(len(groups), dtype=np.int64)
    for tree in trees:
        left = np.asarray(tree["left_children"])
        indices = np.asarray(tree["split_indices"])
        np.add.at(counts, indices[left != -1], 1)
    return pd.Series(counts, index=groups).groupby(level=0, sort=False).sum()


def compute(out: Path) -> None:
    sys.path.insert(0, str(STUDY))
    import run as run_module  # noqa: E402

    source_manifest, names, data = run_module.load_data()
    record = submission_record()
    selected = source_manifest["selected"]
    constants = [name for name in names["constant"] if name in selected]
    varying = [name for name in names["varying"] if name in selected]
    if len(constants) != 21 or len(varying) != 9 or set(constants + varying) != set(selected):
        raise SystemExit("selected feature contract changed")
    groups = np.asarray(constants + [name for _ in range(24) for name in varying])

    rows = []
    receipt = {
        "experiment": EXPERIMENT,
        "config_sha256": record["config_sha256"],
        "source": "saved submitted boosters",
        "repeats": REPEATS,
        "seed": SEED,
        "selection_disclosure": record["config"]["data"]["feature_selection"],
    }
    for stage, (grid, _, metric) in STAGES.items():
        bundle = data[grid]
        models = load_models(stage)
        base_prediction = predict(run_module, models, bundle)
        frozen = np.load(RUN / f"{stage}_predictions.npz")["pred"]
        max_difference = float(np.abs(base_prediction - frozen).max())
        base_mse = float(np.mean((bundle["Y"] - base_prediction) ** 2))
        if max_difference != 0.0 or abs(base_mse - record[metric]) > 1e-6:
            raise SystemExit(f"{stage}: saved boosters do not reproduce the submitted prediction")

        splits = pd.concat([split_counts(model, groups) for model in models]).groupby(level=0, sort=False).sum()
        receipt[stage] = {
            "origins": len(bundle["origins"]),
            "boosters": len(models),
            "max_abs_diff_kms": max_difference,
            "mse": base_mse,
            "leaderboard_mse": record[metric],
            "total_splits": int(splits.sum()),
            "model_sha256": [sha256(RUN / stage / "extreme_4.0" / f"b{i:02}.ubj") for i in range(3)],
        }
        rng = np.random.default_rng(SEED)
        for name in selected:
            deltas = []
            for _ in range(REPEATS):
                permutation = rng.permutation(len(bundle["origins"]))
                permuted = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in bundle.items()}
                if name in constants:
                    column = constants.index(name)
                    permuted["C"][:, column] = bundle["C"][permutation, column]
                else:
                    column = varying.index(name)
                    permuted["E"][:, :, column] = bundle["E"][permutation, :, column]
                candidate = predict(run_module, models, permuted)
                deltas.append(float(np.mean((bundle["Y"] - candidate) ** 2)) - base_mse)
            rows.append({
                "stage": stage,
                "feature": name,
                "kind": kind(name),
                "split_count": int(splits.get(name, 0)),
                "perm_delta_mse_mean": float(np.mean(deltas)),
                "perm_delta_mse_sd": float(np.std(deltas, ddof=1)),
            })
            print(f"{stage}: {name}", flush=True)

    pd.DataFrame(rows).to_csv(out / "feature_importance.csv", index=False, float_format="%.8g")
    (out / "feature_importance_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def plot(out: Path) -> None:
    data = pd.read_csv(out / "feature_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for row, (stage, (_, title, _)) in zip(axes, STAGES.items()):
        subset = data[data.stage == stage]
        for ax, column, label in (
            (row[0], "perm_delta_mse_mean", "permutation: increase in 72 h MSE (km/s)$^2$"),
            (row[1], "split_count", "split count (three saved boosters)"),
        ):
            top = subset.nlargest(TOP, column).iloc[::-1]
            errors = top.perm_delta_mse_sd if column == "perm_delta_mse_mean" else None
            ax.barh(top.feature, top[column], xerr=errors, color=[COLORS[item] for item in top.kind],
                    error_kw={"lw": 0.8})
            ax.set_title(f"{title}\n{label}, top {TOP}", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.3)
    fig.legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()],
               loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("E2000FS Feat30 Extreme (seed 42) feature importance from saved submitted models\n"
                 "permutation on the already-scored leaderboard periods (diagnostic only; 5 shuffles; "
                 "features were selected post-hoc from official-exposed analysis)", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(out / "feature_importance.png", dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        compute(args.out)
    plot(args.out)


if __name__ == "__main__":
    main()
