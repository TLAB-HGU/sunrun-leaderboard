"""D21 F1-first joint contract: same joint as D20, dev-only F1-first, causal."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d21_f1joint"
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d10_analog"))
sys.path.insert(0, str(PKG))
import joint as JT  # noqa: E402
import f1first as F1  # noqa: E402
import strips as ST  # noqa: E402
import analog as AN  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_weights", "def strip_features", "def select_slots",
             "def analog_long", "def build_state", "def add_state_features",
             "def search_one", "def lam(", "PARKER_WEST", "OVERLAP"]
WINDOW_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                "2026-05-29", "2026-05-31", "2026-09-26")


def test_joint_columns_17_plus_4_new():
    assert len(ST.strip_columns()) == 17
    assert len(AN.analog_columns()) == 4
    extra = JT.extra_columns()
    assert len(extra) == 21
    assert extra[:17] == ST.strip_columns() and extra[17:] == AN.analog_columns()
    assert all(c.startswith(("strip_", "img_")) for c in extra[:17])
    assert all(c.startswith("analog_") for c in extra[17:])


def test_import_only_no_copy():
    for f in ("joint.py", "f1first.py", "runner.py"):
        text = (PKG / f).read_text()
        for ban in COPY_BANS:
            assert ban not in text, f"{f} copies {ban}"
    assert "import strips as ST" in (PKG / "joint.py").read_text()
    assert "import analog as AN" in (PKG / "joint.py").read_text()


def test_pregrid_pinned_six_combos():
    assert tuple(F1.GRID) == ("CH6/l2", "CH6/l1", "CH6V2/l2", "CH6V2/l1",
                              "CH3V3/l2", "CH3V3/l1")
    assert tuple(JT.GRID) == tuple(F1.GRID)
    assert F1.F1_FLOOR == 0.41121495
    assert "0.41121495" in F1.SELECT_RULE


def test_split_config_matches_analog_module():
    for cfg in F1.GRID:
        keys, metric = JT.split_config(cfg)
        kname = cfg.split("/")[0]
        assert keys == list(AN.KEY_SETS[kname]) and metric in AN.METRICS


def test_mse_first_is_argmin_dev_mse():
    mse = {c: 8000.0 - 10 * i for i, c in enumerate(F1.GRID)}
    assert F1.mse_first(mse) == "CH3V3/l1"


def _scores(f1_top="CH3V3/l1", feasible_all=True):
    mse = {c: (7800.0 + 5 * i) for i, c in enumerate(F1.GRID)}
    f1 = {c: 0.45 for c in F1.GRID}
    f1[f1_top] = 0.52
    guard = 7900.0 if feasible_all else 7790.0
    return mse, f1, guard


def test_select_picks_highest_feasible_f1():
    mse, f1, guard = _scores()
    w, info = F1.select_config_f1first(mse, f1, guard)
    assert w == "CH3V3/l1" and info["abstained"] is False


def test_select_guard_blocks_mse_blowup():
    mse = {c: 7800.0 for c in F1.GRID}
    mse["CH3V3/l1"] = 9000.0  # best F1 but over the E0003-level guard
    f1 = {c: 0.45 for c in F1.GRID}
    f1["CH3V3/l1"] = 0.60
    w, info = F1.select_config_f1first(mse, f1, 7850.0)
    assert w == "CH3V3/l2" and w in info["feasible"]
    assert "CH3V3/l1" not in info["feasible"]


def test_select_floor_abstains_to_mse_first():
    mse, f1, guard = _scores()
    f1 = {c: 0.30 for c in F1.GRID}  # all below the 0.4112 floor
    w, info = F1.select_config_f1first(mse, f1, guard)
    assert w == info["mse_first"] and info["abstained"] is True
    assert "abstain" in info["reason"]


def test_select_undefined_f1_falls_back():
    mse, f1, guard = _scores()
    f1 = {c: None for c in F1.GRID}
    w, info = F1.select_config_f1first(mse, f1, guard)
    assert w == info["mse_first"] and info["abstained"] is True


def test_select_rejects_wrong_grid_or_guard():
    mse, f1, guard = _scores()
    bad = dict(mse)
    bad.pop(F1.GRID[0])
    try:
        F1.select_config_f1first(bad, f1, guard)
    except ValueError:
        pass
    else:
        raise AssertionError("must reject off-grid scores")
    try:
        F1.select_config_f1first(mse, f1, np.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("must reject non-finite guard")


def test_select_dev_only_by_signature():
    assert list(inspect.signature(F1.select_config_f1first).parameters) == [
        "dev_mse", "dev_f1", "e0003_dev_mse"]
    assert list(inspect.signature(F1.mse_first).parameters) == ["dev_mse"]
    src = (PKG / "f1first.py").read_text().lower()
    assert "selection" not in src or "selection/audit never" in src


def test_runner_uses_cuda_pilot_purge_poison_altseed():
    src = (PKG / "runner.py").read_text()
    assert '"device": "cuda"' in src or "'device': 'cuda'" in src or '"device":' in src
    assert "--pilot" in src and "--alt-seed" in src
    assert '72h' in src and "poison.json" in src and "selection.json" in src
    assert "import joint as JT" in src and "import f1first as F1" in src
    assert "import strips as ST" in src and "import analog as AN" in src
    assert "Optuna" not in src and "optuna" not in src


def test_no_selection_fed_choice():
    src = (PKG / "runner.py").read_text()
    low = src.lower()
    assert 'scores["selection"]' not in low and "scores['selection']" not in low
    assert "dev window only" in low or "dev-only" in low
    assert "selection_rule" in src or "choice_rule" in src or "choice rule" in low


def test_scratch_is_worker_d21_only():
    src = (PKG / "runner.py").read_text()
    assert "store/scratch/worker-d21" in src
    for other in ("worker-d16", "worker-d17", "worker-d19", "worker-d20", "worker-d10"):
        assert other not in src


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("joint.py", "f1first.py", "runner.py"):
        text = (PKG / name).read_text()
        for p in FORBIDDEN:
            assert p not in text, f"{name} contains {p}"
        for d in WINDOW_DATES:
            assert d not in text, f"{name} must not reference window date {d}"
        assert "2026-06" not in text, name


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"joint.py", "f1first.py", "runner.py"} <= set(files)
def test_hs_weights_train_only_and_bounded():
    # signature is train-targets only: (y, alpha), no window arguments
    src = (PKG / "runner.py").read_text()
    assert "def hs_weights(y, alpha" in src
    # alpha<=0 (repro path) returns None -> callers fit unweighted, E0052-identical
    assert "sample_weight=sw" in src and "hs_weights(ytr, alpha)" in src
    assert "sample_weight=wtr" in src
    assert "if not np.isfinite(alpha) or alpha <= 0:\n        return None" in src
    # weights bounded in [1, 1+alpha], monotone in y, stats from input only
    assert "np.clip((y - med) / iqr, 0.0, 1.0)" in src
    assert "np.median(y)" in src and "np.percentile(y, [75, 25])" in src


def test_hs_slice_is_report_only():
    src = (PKG / "runner.py").read_text()
    assert "def hs_slice_mse(" in src
    assert "never feeds any selection choice" in src
    # selection rule still takes dev MSE/F1 only; slice dict never enters f1first
    assert "select_config_f1first(dev_mse, dev_f1, e0003_mse)" in src
    assert "dev_hs" not in (PKG / "f1first.py").read_text()
    assert '"hs_note": "dev high-speed-slice MSE is report-only' in src


def test_guard_stays_unweighted_e0003_level():
    src = (PKG / "runner.py").read_text()
    assert ("e0003_model = xgb.XGBRegressor(**P).fit(tr_dev[base_cols + strip_cols], "
            'tr_dev["y"])' in src)


def test_runner_scratch_is_overridable_with_d21_default():
    src = (PKG / "runner.py").read_text()
    assert '"--scratch"' in src and '"--hs-weight"' in src
    assert 'default="store/scratch/worker-d21"' in src
    assert "JT.ensure_scratch(a.scratch)" in src
def test_hs_weights_numeric_bounds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d21_runner_mod", PKG / "runner.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    y = np.array([300.0, 350.0, 400.0, 500.0, 600.0, 800.0])
    assert mod.hs_weights(y, 0.0) is None
    assert mod.hs_weights(y, -1.0) is None
    w = mod.hs_weights(y, 1.0)
    assert w.shape == y.shape and np.all(w >= 1.0) and np.all(w <= 2.0)
    assert bool(np.all(np.diff(w) >= 0.0))
    assert w[0] == 1.0
    assert mod.hs_weights(y, 2.0).max() <= 3.0
def test_parallel_finals_mirror_d23_threads():
    src = (PKG / "runner.py").read_text()
    assert "from concurrent.futures import ThreadPoolExecutor" in src
    assert "ThreadPoolExecutor(max_workers=a.jobs)" in src
    assert '"--jobs"' in src and "choices=(1, 3)" in src and "default=3" in src
    assert "FIT_JOBS = 3" in src and "FIT_NJOBS = 6" in src and "18 <= 20" in src
    assert "ProcessPoolExecutor" not in src
    # the 3 period finals go through one pooled worker with FP params + weights
    assert "def _fit_period(job):" in src
    assert "xgb.XGBRegressor(**FP).fit(Xtr, ytr, sample_weight=wtr)" in src
    assert "FP = {**P, " in src and '"n_jobs": FIT_NJOBS' in src
    assert 'fit_inputs.append(("dev"' in src
    assert 'for name in ("selection", "audit"):' in src
    assert "return m, m.predict(Xte), importance_of(m, cols)" in src


def _load_runner_mod(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, PKG / "runner.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_arrays_dtypes_and_shapes():
    mod = _load_runner_mod("d21_runner_arrays")
    n = 10
    tr = pd.DataFrame({"y": np.linspace(300.0, 800.0, n),
                       "f_a": np.arange(n, dtype=float), "f_b": np.ones(n)})
    te = pd.DataFrame({"y": np.linspace(320.0, 780.0, n // 2),
                       "f_a": np.arange(n // 2, dtype=float), "f_b": np.ones(n // 2)})
    Xtr, ytr, wtr, Xte = mod._fit_arrays(tr, te, ["f_a", "f_b"], 0.0)
    assert Xtr.shape == (n, 2) and Xtr.dtype == np.float32 and Xtr.flags["C_CONTIGUOUS"]
    assert ytr.shape == (n,) and ytr.dtype == float and wtr is None
    assert Xte.shape == (n // 2, 2) and Xte.dtype == np.float32
    _, _, wtr1, _ = mod._fit_arrays(tr, te, ["f_a", "f_b"], 1.0)
    assert wtr1 is not None and wtr1.shape == (n,)


def test_fcache_key_versioned_and_deterministic():
    mod = _load_runner_mod("d21_runner_fcache")
    assert mod.FCACHE_VERSION == 1
    p = {"freq": "6h", "window": "dev", "config": "CH6V2/l2"}
    assert mod.fcache_key("tr_cfg", p) == mod.fcache_key("tr_cfg", p)
    assert mod.fcache_key("tr_cfg", p) != mod.fcache_key("te_cfg", p)
    assert mod.fcache_key("tr_cfg", p) != mod.fcache_key("tr_cfg", {**p, "freq": "3h"})
    src = (PKG / "runner.py").read_text()
    assert '"code": {' in src and '"data": {' in src and '"v": FCACHE_VERSION' in src


def test_fcache_roundtrip_and_tamper_miss(tmp_path):
    mod = _load_runner_mod("d21_runner_fcache_rt")
    df = pd.DataFrame({"origin": pd.date_range("2025-08-04", periods=8, freq="h", tz="UTC"),
                       "h": np.arange(1, 9), "y": np.linspace(350.0, 600.0, 8),
                       "strip_a": np.arange(8, dtype=np.float32)})
    key = mod.fcache_key("tr_cfg", {"freq": "6h", "window": "dev", "config": "CH6/l2"})
    assert mod.fcache_get(tmp_path, key) is None
    out = mod.fcache_put(tmp_path, key, df)
    assert len(out) == len(df)
    hit = mod.fcache_get(tmp_path, key)
    assert hit is not None and len(hit) == len(df) and list(hit.columns) == list(df.columns)
    assert mod._origins_sha(hit) == mod._origins_sha(df)
    # tampered sidecar or missing file is a miss, never an error
    (tmp_path / "featcache" / (key + ".json")).write_text('{"key": "wrong"}')
    assert mod.fcache_get(tmp_path, key) is None
    (tmp_path / "featcache" / (key + ".parquet")).unlink()
    assert mod.fcache_get(tmp_path, key) is None
