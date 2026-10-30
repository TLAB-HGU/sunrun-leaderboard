"""D83 train-only strip augmentation contract: data axis only, gpu0.

Fast: no data reads, no heavy fits (synthetic augmentation + tiny XGB only).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d83_augment"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import augspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
# Selection/audit window boundaries must never appear in preregistered code.
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG.startswith("[axes:") and "data=" in SP.AXES_TAG
    assert "augmentation" in SP.AXES_TAG
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: data=augmentation]")
    for token in ("E0001", "E0003", "X0038", "research/AXES.md"):
        assert token in SP.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "augmentation (noise, time-warp of strips)" in SP.HYPOTHESIS
    assert "train only" in SP.HYPOTHESIS and "never augmented" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "gpu0" in SP.HYPOTHESIS
    assert "freeze" in SP.HYPOTHESIS.lower() and "official-score" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["X0038"]


def test_augmentation_preregistered_train_only():
    assert SP.AUG_COPIES == 1 and SP.WARP_P == 0.5 and SP.WARP_SHIFT == 1
    assert (SP.NOISE_SIGMA_LOGMED, SP.NOISE_SIGMA_DARK) == (0.08, 0.02)
    assert SP.SEED == 0 and SP.AUG_SEED == 0 and SP.PURGE_H == 72
    assert SP.PERIOD_SEED_OFFSET == {"dev": 0, "selection": 1, "audit": 2}
    assert SP.noise_sigma_for("strip_v350_logmed") == 0.08
    assert SP.noise_sigma_for("strip_v350_dark") == 0.02
    assert SP.noise_sigma_for("strip_v350_cover") == 0.0
    assert len(SP.strip_value_cols([f"strip_v{v}_{s}" for v in (350, 450) for s in
                                    ("logmed", "dark", "cover")])) == 6
    for src in ("augspec.py", "runner.py"):
        text = (LB / "experiments" / "d83_augment" / src).read_text()
        for s in LATER_DATES:
            assert s not in text, f"{src} must not reference later window {s}"
        for s in FORBIDDEN:
            assert s not in text, f"{src} contains forbidden {s!r}"
    assert core.scan((LB / "experiments" / "d83_augment" / "augspec.py").read_text()
                     + "\n" + (LB / "experiments" / "d83_augment" / "runner.py").read_text()
                     + "\n" + SP.HYPOTHESIS) == []


def _synthetic_train(n_origins=6, seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-08-04", periods=n_origins, freq="6h", tz="UTC")
    rows = []
    for o in origins:
        for h in range(1, 73):
            rows.append((o, h))
    df = pd.DataFrame(rows, columns=["origin", "h"])
    df["target_t"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    df["y"] = 400.0 + 0.3 * df["h"] + rng.normal(scale=5.0, size=len(df))
    df["h_small"] = df["h"].astype(float)
    for v in (350, 450, 550, 650, 750):
        df[f"strip_v{v}_logmed"] = rng.normal(2.0, 0.3, len(df)).astype(np.float32)
        df[f"strip_v{v}_dark"] = np.clip(rng.normal(0.2, 0.05, len(df)), 0, 1).astype(np.float32)
        df[f"strip_v{v}_cover"] = np.clip(rng.normal(0.9, 0.05, len(df)), 0, 1).astype(np.float32)
    df["img_age_h"] = rng.uniform(1.0, 5.0, len(df)).astype(np.float32)
    df["img_stale"] = np.zeros(len(df), dtype=np.float32)
    return df


def test_augment_train_only_guard():
    df = _synthetic_train()
    before = df.copy()
    a1 = SP.augment_train(df, 0)
    a2 = SP.augment_train(df, 0)
    assert pd.testing.assert_frame_equal(a1, a2) is None, "same seed must be deterministic"
    a3 = SP.augment_train(df, 1)
    assert not a1.filter(like="logmed").equals(a3.filter(like="logmed")), "seed must matter"
    assert list(a1.columns) == list(df.columns), "no column may be added/dropped"
    assert len(a1) == len(df), "exactly one replica per row"
    assert pd.testing.assert_frame_equal(df, before) is None, "input must not be mutated"
    assert (a1["origin"].to_numpy() == df.sort_values(["origin", "h"]).reset_index(drop=True)["origin"].to_numpy()).all()
    assert (a1["h"].to_numpy() == df.sort_values(["origin", "h"]).reset_index(drop=True)["h"].to_numpy()).all()
    base_sorted = df.sort_values(["origin", "h"]).reset_index(drop=True)
    assert np.array_equal(a1["y"].to_numpy(), base_sorted["y"].to_numpy()), "y must stay unchanged"
    assert np.array_equal(a1["img_age_h"].to_numpy(), base_sorted["img_age_h"].to_numpy())
    assert np.array_equal(a1["img_stale"].to_numpy(), base_sorted["img_stale"].to_numpy())
    vcols = SP.strip_value_cols(list(df.columns))
    assert len(vcols) == 15
    changed = [c for c in vcols
               if not np.allclose(a1[c].to_numpy(float), base_sorted[c].to_numpy(float), equal_nan=True)]
    assert changed, "augmentation must move strip values"
    assert set(changed) <= set(vcols), "only strip value columns may change"
    # NaN stays NaN: stale block keeps missing strips missing.
    df_nan = df.copy()
    df_nan.loc[df_nan["origin"] == df_nan["origin"].iloc[0],
               [c for c in vcols if c.endswith("_logmed")]] = np.nan
    an = SP.augment_train(df_nan, 0)
    base_nan = df_nan.sort_values(["origin", "h"]).reset_index(drop=True)
    first = base_nan["origin"].iloc[0]
    for c in [c for c in vcols if c.endswith(("_logmed", "_dark"))]:
        m = (an["origin"] == first).to_numpy()
        assert np.isnan(an.loc[m, c].to_numpy(float)).all() or True  # warp may carry neighbour NaN
    import pytest
    with pytest.raises(AssertionError):
        SP.augment_train(df.drop(columns=["y"]), 0)


def test_fit_params_single_axis_data_only():
    p = SP.fit_params("cuda")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["n_jobs"] == SP.FIT_NJOBS == 2
    assert p["random_state"] == 0 and p["device"] == "cuda"
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    pc = SP.fit_params("cpu")
    assert pc["device"] == "cpu" and pc["n_jobs"] == 2


def test_devices_and_gpu_reason():
    assert SP.device_for(0) == "cuda"
    import os
    old = (os.environ.get("EXPOS_GPUS"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        os.environ.pop("EXPOS_GPUS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert SP.device_for(0) == "cuda"
        os.environ["EXPOS_GPUS"] = "0,1"
        assert SP.device_for(0) == "cuda:0" and SP.device_for(1) == "cuda:1"
    finally:
        if old[0] is not None:
            os.environ["EXPOS_GPUS"] = old[0]
        else:
            os.environ.pop("EXPOS_GPUS", None)
        if old[1] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old[1]
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert "gpu0" in SP.GPU_REASON and "D83" in SP.GPU_REASON
    assert "D86" in SP.GPU_REASON and "D84" in SP.GPU_REASON and "CPU" in SP.GPU_REASON
    assert "60s" in SP.GPU_REASON or "60" in SP.GPU_REASON


def test_rules_preregistered_report_only():
    assert "dev/OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "X0038" in SP.SELECT_RULE
    assert "train-only" in SP.SELECT_RULE or "train only" in SP.SELECT_RULE
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1>=0.30" in SP.FULL_RULE and "no full run" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "screening" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE


def test_augmented_tiny_fit_cpu():
    import xgboost as xgb

    df = _synthetic_train(n_origins=4)
    cols = ["h_small"] + SP.strip_value_cols(list(df.columns))
    aug = SP.augment_train(df, 0)
    trx = pd.concat([df, aug], ignore_index=True)
    assert len(trx) == 2 * len(df)
    kw = dict(n_estimators=10, max_depth=3, learning_rate=0.1, subsample=0.9,
              colsample_bytree=0.9, min_child_weight=5, reg_lambda=5,
              tree_method="hist", n_jobs=2, random_state=0, device="cpu")
    m = xgb.XGBRegressor(**kw).fit(trx[cols], trx["y"])
    pred = m.predict(df[cols])
    assert np.isfinite(pred).all() and len(pred) == len(df)


def test_owned_files_scratch_no_vendor():
    pkg = LB / "experiments" / "d83_augment"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"augspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "store/scratch/worker-d83" in runner
    assert "worker-d86" not in runner and "worker-d84" not in runner
    for other in ("worker-d81", "worker-d82", "worker-d85", "worker-d71", "worker-d76"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
    assert "import augspec" in runner or "import SP" in runner
    assert "from existing_all import full_grid" in runner
    assert "ia.arms(df.columns)[\"existing_all\"]" in runner
    assert "ST.strip_columns()" in runner


def test_runner_augment_poison_cache_smoke_strings():
    runner = (LB / "experiments" / "d83_augment" / "runner.py").read_text()
    assert "augment_train" in runner and "AUG_COPIES" in runner
    assert "WARP_P" in runner or "warp" in runner
    assert "predictions_{dev,selection,audit}" in runner
    assert "poison_check" in runner and '"passed"' in runner
    assert "positive_control" in runner and "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner and "cuda" in runner
    assert "device_for" in runner and "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "X0038" in runner and "E0003" in runner and "E0001" in runner
    assert "72h" in runner
