"""Permutation importance of the submitted budget-spectral forecast.

The archived boosters are multi-output vector trees, and XGBoost 2.1.4 cannot dump them (json/text dumps are
"not yet implemented" and `get_score` crashes), so native split/gain importance does not exist for this model
family. Instead each feature is shuffled across origins and the whole 72 h forecast is rebuilt, exactly as the
submission does it: three 24-output boosters per stage, predictions clipped to 1–3000 km/s.

ΔMSE is `MSE(shuffled feature) − MSE(submitted forecast)` on the officially scored windows (F1 2,137 origins,
F2 2,833 origins), five shuffles, mean ± standard deviation. The windows are already public and scored, so this
is a diagnostic; nothing here selected features or rounds. Correlated inputs (for example `v_lag0` and
`ace_speed_kms_last`) share their importance.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_budget_spectral/code/feature_importance.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
STUDY = ROOT / "research/e2000fs_budget_spectral"
RESULTS = STUDY / "results"
RUN = RESULTS / "runs/full_v1"
RELEASE = RUN / "publication/package/releases/e2000fs_budget_spectral"
DATA = ROOT / "data-v2/model-inputs/leaderboard-store/e2000fs-lag0"
OUT = Path(__file__).resolve().parents[1]
STAGES = (("f1", "f1", "F1 model (fit targets < 2026-02-26)", "mse_f1"),
          ("final", "official", "F2 model (fit targets < 2026-05-29)", "mse_f2"))
REPEATS = 5
SEED = 42
CLIP = (1.0, 3000.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(stage_grid, truth):
    """(origins, 72) official truth aligned to a grid's origins."""
    origins = pd.DatetimeIndex(stage_grid["origins"], tz="UTC")
    frame = truth[pd.DatetimeIndex(truth["origin_last_input_utc"]).isin(origins)]
    wide = frame.pivot(index="origin_last_input_utc", columns="horizon_hours", values="target_kms")
    wide = wide.reindex(origins.strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert wide.shape == (len(origins), 72) and np.isfinite(wide.to_numpy()).all(), "truth does not cover a grid"
    return origins, wide.to_numpy()


def main():
    config = json.loads((RESULTS / "config.json").read_text())
    receipt = json.loads((RESULTS / "leaderboard-result.json").read_text())
    selected = config["data"]["selected"]
    names = json.loads((DATA / "names.json").read_text())
    truth = pd.read_parquet(next(p for p in config["data"]["input_identity"]
                                 if p.endswith("/truth.parquet")))
    constant = [i for i, n in enumerate(names["constant"]) if n in selected]
    varying = [i for i, n in enumerate(names["varying"]) if n in selected]
    fixed_names = [names["constant"][i] for i in constant]
    lead_names = [names["varying"][i] for i in varying]
    width = len(constant) + 24 * len(varying)
    assert width == config["data"]["columns_per_block"] == 237, f"unexpected block width {width}"
    assert len(fixed_names) + len(lead_names) == len(selected) == 30

    def columns(feature):
        """Column indices of one feature inside a block design matrix."""
        if feature in fixed_names:
            return [fixed_names.index(feature)], "fixed"
        slot = lead_names.index(feature)
        return [len(fixed_names) + hour * len(lead_names) + slot for hour in range(24)], "hourly"

    def design(stage_grid, block):
        values = stage_grid["E"][:, block * 24:(block + 1) * 24, :][:, :, varying]
        values = values.reshape(len(stage_grid["C"]), -1)
        return np.hstack([stage_grid["C"][:, constant], values]).astype(np.float32)

    rng = np.random.default_rng(SEED)
    rows, model_hashes, parity = [], {}, {}
    for stage, grid_name, title, score_key in STAGES:
        grid = dict(np.load(DATA / f"{grid_name}.npz"))
        origins, y = targets(grid, truth)
        archived = np.load(RUN / f"{stage}_spectral_predictions.npz")
        assert (pd.DatetimeIndex(archived["origins"], tz="UTC") == origins).all()
        matrices = [design(grid, block) for block in range(3)]
        models = []
        for block in range(3):
            path = RELEASE / "models" / f"{stage}_spectral_b{block}.ubj"
            digest = sha(path)
            assert digest == config["data"]["model_sha256"][f"models/{path.name}"], f"model changed: {path}"
            model_hashes[str(path.relative_to(ROOT))] = digest
            models.append(xgb.Booster(model_file=str(path)))

        def predict(mats):
            out = np.empty((len(origins), 72))
            for block, (model, matrix) in enumerate(zip(models, mats)):
                out[:, block * 24:(block + 1) * 24] = model.inplace_predict(matrix)
            return np.clip(out, *CLIP)

        base = predict(matrices)
        assert np.array_equal(base, archived["pred"]), f"{stage}: rebuilt design does not reproduce the forecast"
        base_mse = float(np.mean((base - y) ** 2))
        assert abs(base_mse - receipt[score_key]) < 1e-9, f"{stage}: MSE {base_mse!r} != {receipt[score_key]!r}"
        parity[stage] = dict(origins=int(len(origins)), mse=base_mse, expected=receipt[score_key],
                             prediction_parity="exact (bit-identical to the submitted grid)")

        for feature in selected:
            cols, kind = columns(feature)
            deltas = []
            for _ in range(REPEATS):
                shuffled = [matrix.copy() for matrix in matrices]
                order = rng.permutation(len(origins))
                for matrix in shuffled:
                    matrix[:, cols] = matrix[order][:, cols]
                deltas.append(float(np.mean((predict(shuffled) - y) ** 2)) - base_mse)
            deltas = np.array(deltas)
            rows.append(dict(stage=stage, feature=feature, kind=kind, columns=len(cols),
                             delta_mse_mean=float(deltas.mean()), delta_mse_std=float(deltas.std(ddof=1)),
                             repeats=REPEATS))
        print(f"{stage}: {len(origins)} origins, base MSE {base_mse:.6f} km²/s²", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), constrained_layout=True)
    for ax, (stage, _, title, _) in zip(axes, STAGES):
        top = table[table.stage == stage].nlargest(20, "delta_mse_mean").sort_values("delta_mse_mean")
        colors = ["#c44e52" if kind == "fixed" else "#4c72b0" for kind in top["kind"]]
        ax.barh(top["feature"], top["delta_mse_mean"], xerr=top["delta_mse_std"], color=colors,
                error_kw=dict(lw=.9, capsize=2))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ΔMSE when the feature is shuffled across origins (km/s)²")
        ax.grid(alpha=.25, axis="x")
        ax.tick_params(labelsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c44e52"), plt.Rectangle((0, 0), 1, 1, color="#4c72b0")]
    fig.legend(handles, ["fixed-per-origin feature", "hourly feature (24 lead copies)"], loc="lower center",
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("e2000fs_budget_spectral — permutation importance of the submitted spectral forecast\n"
                 f"whole 72 h strategy rebuilt with one feature shuffled across origins; {REPEATS} shuffles, "
                 "mean ± s.d.; officially scored windows, diagnostic only", fontsize=11)
    fig.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close(fig)

    (OUT / "feature_importance_receipt.json").write_text(json.dumps(dict(
        method="permutation across origins; the three 24-output boosters of a stage are re-predicted with the "
               "feature's columns shuffled; predictions clipped to 1-3000 km/s; ΔMSE vs the submitted forecast",
        repeats=REPEATS, seed=SEED, stages={k: v for k, v in parity.items()},
        native_importance="unavailable: XGBoost 2.1.4 cannot dump multi-target trees and get_score crashes",
        model_sha256=model_hashes,
        note="official-exposed window; not used for feature selection or model choice",
        outputs={"feature_importance.csv": sha(OUT / "feature_importance.csv"),
                 "feature_importance.png": sha(OUT / "feature_importance.png")}), indent=2) + "\n")
    print(table.groupby("stage").delta_mse_mean.describe().to_string())
    print("wrote feature_importance.{csv,png} and the receipt")


if __name__ == "__main__":
    main()
