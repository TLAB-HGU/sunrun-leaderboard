"""D94 bagging contract: hypothesis tags, single-axis outer bootstrap, twin spec, isolation.

Fast static + unit checks only: no data reads, no fits beyond small synthetic
CPU fits, no official contact. The pilot run itself (registered separately)
proves end-to-end behaviour through expos run/evaluate/poison.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d94_bag"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import bagspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core  # noqa: E402

RUNNER_SRC = (LB / "experiments" / "d94_bag" / "runner.py").read_text()
SPEC_SRC = (LB / "experiments" / "d94_bag" / "bagspec.py").read_text()

COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_hypothesis_dual_tag_and_experience():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG == "[axes: ens=bagging]"
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: ens=bagging]")
    assert SP.EXPERIENCE == "X0109" and "X0109" in SP.HYPOTHESIS
    assert "E0117" in SP.HYPOTHESIS
    assert "research/AXES.md" in SP.HYPOTHESIS and "ens" in SP.HYPOTHESIS
    assert "bagging" in SP.HYPOTHESIS and "bootstrap" in SP.HYPOTHESIS
    assert "B=5" in SP.HYPOTHESIS and "mean" in SP.HYPOTHESIS
    assert SP.DIRECTION == "D94" and SP.OWNER == "worker-d94"
    assert "6h" in SP.HYPOTHESIS
    assert "gpu1" in SP.HYPOTHESIS
    assert "screening" in SP.HYPOTHESIS
    assert "No freeze" in SP.HYPOTHESIS or "no freeze" in SP.HYPOTHESIS
    assert "official-score" in SP.HYPOTHESIS


def test_single_axis_bagging_only():
    p = SP.fit_params("cuda")
    assert p["device"].startswith("cuda")
    assert p.get("objective", "reg:squarederror") == "reg:squarederror"
    assert "tweedie_variance_power" not in p
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0001 recipe"
    assert p["n_estimators"] == 600 and p["max_depth"] == 6
    assert p["random_state"] == 0
    assert "base_score" not in p
    assert "sample_weight" not in RUNNER_SRC
    assert SP.N_BAGS == 5 and SP.BOOT_BASE == 1000
    assert type(__import__("xgboost").XGBRegressor(**p)).__name__ == "XGBRegressor"


def test_e0001_twin_rows_purge_budget():
    assert SP.PURGE_HOURS == 72
    assert SP.FREQ_PILOT == "6h" and SP.FREQ_FULL == "3h"
    assert SP.N_ESTIMATORS == 600 and SP.SEED == 0
    assert 'ia.arms(df.columns)["existing_all"]' in RUNNER_SRC
    assert "SP.PURGE_HOURS" in RUNNER_SRC
    assert RUNNER_SRC.count(".fit(") == 1, "exactly one head fit call site (looped over bags)"
    assert SP.check_spec()["bags"] == 5


def test_gpu_budget_lease():
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    assert SP.PERIOD_JOBS == 3, "three periods run in parallel"
    assert SP.FIT_NJOBS <= 2
    assert SP.fit_params("cuda")["device"].startswith("cuda")
    blob = RUNNER_SRC + "\n" + SPEC_SRC
    assert "device_for" in blob and "EXPOS_GPUS" in blob
    assert "SP.device_for" in RUNNER_SRC
    assert "cuda" in blob
    assert "cuda" in RUNNER_SRC
    assert "gpu1" in SP.GPU_REASON and "60s" in SP.GPU_REASON
    assert "D93" in SP.GPU_REASON and "D95" in SP.GPU_REASON and "D94" in SP.GPU_REASON
    assert "foreground" in SP.GPU_REASON


def test_scratch_isolation():
    assert "store/scratch/worker-d94" in RUNNER_SRC
    others = [l for l in RUNNER_SRC.splitlines() if "worker-" in l and "worker-d94" not in l]
    assert others == [], f"scratch leak: {others[:3]}"
    assert "featcache" in RUNNER_SRC and "fcache_key" in RUNNER_SRC
    assert SP.SCRATCH_NAME == "worker-d94"


def test_readonly_admin_no_vendored_strips():
    for s in COPY_BANS:
        assert s not in RUNNER_SRC, f"runner vendors d1-strip code {s!r}"
        assert s not in SPEC_SRC, f"spec vendors d1-strip code {s!r}"
    assert "import strips" not in RUNNER_SRC


def test_no_forbidden_paths_in_registered_code():
    assert core.scan(RUNNER_SRC + "\n" + SPEC_SRC) == []
    assert core.scan(SP.HYPOTHESIS) == []


def test_full_rule_preregistered_no_freeze_claim():
    assert "rc==0" in SP.FULL_RULE and "poison" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1>=0.30" in SP.FULL_RULE
    assert "6h F1" in SP.FULL_RULE
    assert "3h" in SP.FULL_RULE
    assert "No freeze" in SPEC_SRC or "no freeze" in SPEC_SRC
    assert "official_score(" not in RUNNER_SRC and "core.freeze" not in RUNNER_SRC
    assert "official-score" in SP.FULL_RULE


def test_predictions_three_kinds_and_poison():
    assert "predictions_{dev,selection,audit}" in RUNNER_SRC
    assert "write(te, pred, name)" in RUNNER_SRC
    assert "predictions_official" not in RUNNER_SRC
    assert "bagged_predict" in RUNNER_SRC and "bootstrap" in RUNNER_SRC
    assert "mean" in RUNNER_SRC
    assert "poison_check" in RUNNER_SRC and "positive_control" in RUNNER_SRC
    assert "passed" in RUNNER_SRC and "poison.json" in RUNNER_SRC
    assert "SystemExit" in RUNNER_SRC


def test_owned_dir_is_small():
    files = sorted(p.name for p in (LB / "experiments" / "d94_bag").iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"bagspec.py", "runner.py"} <= set(files)


def test_bootstrap_rule_deterministic_and_mean():
    """Preregistered bootstrap rule: fixed seeds, distinct bags, mean aggregation."""
    import numpy as np

    a = SP.bootstrap_indices(100, SP.bag_seed("dev", 0))
    b = SP.bootstrap_indices(100, SP.bag_seed("dev", 0))
    c = SP.bootstrap_indices(100, SP.bag_seed("dev", 1))
    assert np.array_equal(np.asarray(a), np.asarray(b)), "same seed must resample identically"
    assert not np.array_equal(np.asarray(a), np.asarray(c)), "distinct bags must differ"
    assert len({SP.bag_seed(p, i) for p in SP.PERIOD_ORDER for i in range(SP.N_BAGS)}) == 15

    class _Head:
        def __init__(self, v):
            self.v = float(v)

        def predict(self, X):
            import numpy as _np

            return _np.full(len(X), self.v)

    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    m = SP.bag_predict([_Head(1.0), _Head(3.0)], X)
    assert list(m) == [2.0, 2.0, 2.0]


def test_bagged_heads_vary_and_split_synthetic():
    """Outer bags on synthetic data must split, vary, and beat the null (CPU, tiny)."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(7)
    n = 1500
    X = rng.standard_normal((n, 6))
    y = 400.0 + 60.0 * X[:, 0] - 30.0 * X[:, 1] + 8.0 * rng.standard_normal(n)
    cols = [f"f{i}" for i in range(6)]
    import pandas as pd

    df = pd.DataFrame(X, columns=cols)
    p = dict(SP.fit_params("cpu"), n_estimators=50)
    assert p["device"] == "cpu"
    models = []
    for b in range(3):
        idx = np.asarray(SP.bootstrap_indices(n, 1000 + b))
        m = xgb.XGBRegressor(**p).fit(df.iloc[idx][cols], y[idx])
        models.append(m)
    pred = SP.bag_predict(models, df[cols])
    assert np.isfinite(pred).all()
    assert float(np.std(pred)) > 0, "bag mean must vary"
    indiv = [np.asarray(m.predict(df[cols]), dtype=float) for m in models]
    assert any(not np.allclose(indiv[0], q) for q in indiv[1:]), "bags must differ"
    null_mse = float(((y - y.mean()) ** 2).mean())
    assert float(((pred - y) ** 2).mean()) < null_mse
    dump = models[0].get_booster().get_dump(dump_format="text")
    assert sum(t.count("[f") for t in dump) > 0
