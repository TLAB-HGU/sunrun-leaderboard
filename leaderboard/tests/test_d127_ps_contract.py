"""D127 perm-subset contract: 3-seed permutation-stable top-35 on E0001 twin, dev-only.

Fast static + unit checks only: no data reads, no fits beyond small
synthetic CPU fits, no held-window contact. The pilot run itself
(registered separately) proves end-to-end behaviour through expos
run/evaluate/poison.
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d127_permsub"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import pspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d127_permsub" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d127_permsub" / "pspec.py").read_text()

COPY_BANS = ["def speed_features(", "def ace_features(", "def ch_features(",
             "def enlil_features(", "def build(", "def arms(", "def full_grid(",
             "def write("]
LATER_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
               "2026-05-29", "2026-05-31", "2026-09-26")


def test_hypothesis_dual_tag_and_experience():
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: sel=permutation]")
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG == "[axes: sel=permutation]"
    assert SP.AXES_TAG.startswith("[axes:") and "sel=" in SP.AXES_TAG
    assert "permutation" in SP.AXES_TAG
    assert tuple(SP.EXPERIENCE) == ("X0105", "X0135")
    for tok in ("E0114", "X0105", "E0149", "X0135", "E0001"):
        assert tok in SP.HYPOTHESIS, f"hypothesis must cite {tok}"
    assert SP.DIRECTION == "D127" and SP.OWNER == "worker-d127"
    assert "no freeze" in SP.HYPOTHESIS.lower()
    assert "official-score" in SP.HYPOTHESIS
    assert "6h" in SP.HYPOTHESIS and "3h" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS  # 6h F1 is screening only
    assert "held window" in SP.HYPOTHESIS
    assert "gpu1" in SP.HYPOTHESIS
    assert "dev-OOF" in SP.HYPOTHESIS or "dev-only" in SP.HYPOTHESIS


def test_selection_rule_fixed_and_dev_only():
    assert SP.TOP_K == 35 and SP.PER_TRIAL_K == 35
    assert SP.PERM_REPEATS == 3
    assert SP.SEEDS == (0, 1, 2) and SP.N_TRIALS == 3
    assert SP.VAL_LO_UTC == "2025-10-15T00:00:00Z" and SP.VAL_HI_UTC == SP.DEV_END_UTC
    assert SP.check_spec()["top_k"] == 35
    assert "dev-OOF" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE
    assert "E0114" in SP.SELECT_RULE and "X0105" in SP.SELECT_RULE
    assert "stability" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "permutation" in SP.SELECT_RULE
    import pandas as pd
    assert pd.Timestamp(SP.DEV_LO_UTC) < pd.Timestamp(SP.VAL_LO_UTC) < pd.Timestamp(SP.DEV_END_UTC)
    n = 200
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", "2025-11-27 23:00", freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    df = df.sort_values("origin").reset_index(drop=True)
    tr, va = SP.split_for(df, SP.VAL_LO_UTC, SP.VAL_HI_UTC)
    assert (va["origin"] >= pd.Timestamp(SP.DEV_LO_UTC)).all()
    assert (va["origin"] <= pd.Timestamp(SP.DEV_END_UTC)).all()
    assert (tr["target_t"] < pd.Timestamp(SP.VAL_LO_UTC) - pd.Timedelta(hours=72)).all()
    tr2, va2 = SP.split_dev_oof(df)
    assert len(tr2) == len(tr) and len(va2) == len(va)
    import pytest
    with pytest.raises(AssertionError):
        SP.split_for(df, "2025-12-04T00:00:00Z", "2026-02-25T23:00:00Z")  # selection window banned


def _synthetic_reg(n=300, seed=0, n_noise=5):
    import pandas as pd

    rng = np.random.default_rng(seed)
    strong = rng.standard_normal(n) * 50 + 400
    df = pd.DataFrame({"h": rng.integers(1, 73, n).astype(float),
                       "strong": strong,
                       **{f"noise{i}": rng.standard_normal(n) * 50 + 400 for i in range(n_noise)}})
    df["y"] = 400.0 + 0.8 * df["strong"] + 0.3 * df["h"] + rng.standard_normal(n) * 5.0
    return df


def test_permutation_and_stability_deterministic():
    import xgboost as xgb

    cols = ["h", "strong", "noise0", "noise1", "noise2", "noise3", "noise4"]
    df = _synthetic_reg()
    m = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                         reg_lambda=5, tree_method="hist", n_jobs=2, random_state=0)
    m.fit(df[cols], df["y"])
    base, d1 = SP.permutation_deltas(m, df, cols, seed=1)
    base2, d2 = SP.permutation_deltas(m, df, cols, seed=1)
    assert np.isfinite(base) and base > 0
    assert d1 == d2, "permutation deltas must be deterministic"
    assert d1["strong"] > 0 and d1["h"] > 0
    assert d1["strong"] > max(d1[f"noise{i}"] for i in range(5))
    tops = [SP.trial_topk(d1, {c: 1.0 for c in cols}, top_k=3) for _ in range(3)]
    deltas = [dict(d1) for _ in range(3)]
    gains = [{c: 1.0 for c in cols} for _ in range(3)]
    sub, stab, md, mg = SP.stability_select(tops, deltas, gains, top_k=3)
    assert len(sub) == 3 and "h" in sub and "strong" in sub
    assert stab["strong"] == 1.0
    tops2 = list(tops)
    tops2[0] = ["h", "noise0", "noise1"]
    sub2, stab2, _, _ = SP.stability_select(tops2, deltas, gains, top_k=3)
    assert stab2["strong"] < 1.0 and stab2["strong"] > stab2["noise0"]
    assert "strong" in sub2 and "noise0" not in sub2
    assert SP.stability_select(tops, deltas, gains, top_k=3)[0] == sub  # deterministic


def test_trial_topk_keeps_h_and_check_subset_contract():
    base = ["h", "a", "b", "c", "d"] + [f"x{i}" for i in range(100)]
    d = {c: float(i) for i, c in enumerate(base)}
    g = {c: 1.0 for c in base}
    top = SP.trial_topk(d, g)
    assert len(top) == 35 and "h" in top
    good = ["h"] + [f"x{i}" for i in range(34)]
    assert SP.check_subset(good, base) == good
    import pytest
    with pytest.raises(AssertionError):
        SP.check_subset(["a"] + [f"x{i}" for i in range(34)], base)  # missing h
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)], base)  # wrong size
    with pytest.raises(AssertionError):
        SP.check_subset(["h"] + [f"x{i}" for i in range(33)] + ["nope"], base)  # outside base


def test_fit_params_match_e0001_recipe_infra_only():
    p = SP.fit_params("cuda:1", seed=1)
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == SP.FIT_NJOBS == 6
    assert p["random_state"] == 1
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 20
    assert SP.PERIOD_JOBS == 3, "three periods run in parallel"


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert SP.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "1")
    assert SP.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert SP.device_for(0) == "cuda:0"
    assert SP.device_for(1) == "cuda:1"
    assert SP.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1" in SP.FULL_RULE and "0.30" in SP.FULL_RULE
    assert "no full run" in SP.FULL_RULE
    assert "freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "6h F1" in SP.FULL_RULE  # 6h F1 is screening only
    assert "3h" in SP.FULL_RULE  # F1 claimed only at 3h density
    assert "rc==0" in SP.FULL_RULE and "poison" in SP.FULL_RULE


def test_gpu_reason_assigns_gpu1():
    assert "gpu1" in SP.GPU_REASON and "D127" in SP.GPU_REASON
    assert "D124" in SP.GPU_REASON and "gpu0" in SP.GPU_REASON
    assert "D126" in SP.GPU_REASON and "CPU" in SP.GPU_REASON
    assert "60s" in SP.GPU_REASON and "foreground" in SP.GPU_REASON
    assert "lease" in SP.GPU_REASON and "cuda" in SP.GPU_REASON
    assert "--gpu" in SP.GPU_REASON
    assert "store/scratch/worker-d127" in SP.GPU_REASON
    assert "18" in SP.GPU_REASON and "20" in SP.GPU_REASON


def test_scratch_isolation():
    assert SP.SCRATCH_NAME == "worker-d127"
    assert "store/scratch/worker-d127" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d127" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC


def test_single_axis_sel_only_no_feat_change():
    # no strip/image columns: pure E0001 twin, sel change only
    assert "strip" not in RUNNER_SRC.lower()
    assert "suvi" not in RUNNER_SRC.lower()
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "sample_weight" not in RUNNER_SRC
    assert "EPAM_COLS" not in RUNNER_SRC and "EVENT_THRESHOLD" not in RUNNER_SRC
    assert "blend" not in RUNNER_SRC.lower()
    assert SP.check_spec()["axes"] == SP.AXES_REF


def test_owned_files_and_no_vendored():
    pkg = LB / "experiments" / "d127_permsub"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"pspec.py", "runner.py"} <= set(files)
    assert "import pspec" in RUNNER_SRC
    assert "from existing_all import" in RUNNER_SRC
    assert "import info_ablation" in RUNNER_SRC
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
    for name, src in (("pspec.py", SPEC_SRC), ("runner.py", RUNNER_SRC)):
        for tok in ("datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
                    "naive.parquet", "official-predictions", "baseline-replay",
                    "official_check", "official-check", "predictions_official_truth"):
            assert tok not in src, (name, tok)
        for s in LATER_DATES:
            assert s not in src, (name, f"later window {s}")
    assert "noaa-goes" not in RUNNER_SRC.lower()
    assert core.scan(SPEC_SRC + "\n" + RUNNER_SRC + "\n" + SP.HYPOTHESIS) == []


def test_no_foreign_experiment_reads_or_runner_freeze():
    assert "research/expos/runs/E" not in RUNNER_SRC
    assert "RUNS /" not in RUNNER_SRC and "RUNS/" not in RUNNER_SRC
    assert "E0003" not in RUNNER_SRC and "E0114" not in RUNNER_SRC and "E0149" not in RUNNER_SRC
    assert "scorer" not in RUNNER_SRC.lower()
    assert "core.freeze" not in RUNNER_SRC and "official_score(" not in RUNNER_SRC
    # runner itself never claims freeze/score windows
    assert "freeze" not in RUNNER_SRC.lower()
    assert "predictions_official" not in RUNNER_SRC


def test_runner_selection_poison_gpu_cache_and_smoke():
    assert "dev_oof_stability" in RUNNER_SRC
    assert "permutation_deltas" in RUNNER_SRC and "stability_select" in RUNNER_SRC
    assert "trial_topk" in RUNNER_SRC and "check_subset" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and '"passed"' in RUNNER_SRC
    assert "positive_control" in RUNNER_SRC and "prediction_invariant" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC
    assert "ThreadPoolExecutor" in RUNNER_SRC
    assert "device" in RUNNER_SRC and "cuda" in RUNNER_SRC
    assert "for i, n in enumerate(names)" in RUNNER_SRC
    assert "availability_manifest" in RUNNER_SRC
    assert "fcache" in RUNNER_SRC.lower() and "featcache" in RUNNER_SRC
    assert "EXPOS_SMOKE" in RUNNER_SRC
    assert "design_cols" in RUNNER_SRC and "dict.fromkeys" in RUNNER_SRC
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC
    assert "write(te, pred, name)" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d127_permsub").iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"pspec.py", "runner.py"} <= set(files)


def test_subset_unit_fit_smoke_cpu():
    """Seed-stable subset machinery splits on synthetic data on CPU."""
    import xgboost as xgb

    rng = np.random.default_rng(7)
    X = rng.standard_normal((288, 6))
    y = 400.0 + 30.0 * X[:, 0] + rng.standard_normal(288) * 10.0
    p = SP.fit_params("cpu", seed=0)
    assert p["device"] == "cpu"
    m = xgb.XGBRegressor(**dict(p, n_estimators=20)).fit(X, y)
    pred = np.asarray(m.predict(X), dtype=float)
    assert np.isfinite(pred).all() and float(np.std(pred)) > 0
