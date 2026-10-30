"""D31 richmean contract: fresh 3-seed level-mean of the event-rich pipeline, dev-only, causal."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d31_richmean"
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d10_analog"))
sys.path.insert(0, str(LB / "experiments/d21_f1joint"))
sys.path.insert(0, str(PKG))
import joint as JT  # noqa: E402
import f1first as F1  # noqa: E402
import strips as ST  # noqa: E402
import analog as AN  # noqa: E402
import richmean as RM  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_weights", "def strip_features", "def select_slots",
             "def analog_long", "def build_state", "def add_state_features",
             "def search_one", "def lam(", "PARKER_WEST", "OVERLAP"]
WINDOW_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                "2026-05-29", "2026-05-31", "2026-09-26")
STALE_SEEDS = (0, 1, 7, 123, 10, 11, 12, 13, 14, 2, 3, 4)


def test_seeds_are_three_fresh():
    assert tuple(RM.SEEDS) == (21, 22, 23)
    for s in RM.SEEDS:
        assert s not in STALE_SEEDS, f"seed {s} is not fresh"
    assert RM.F1_FLOOR == 0.41121495
    assert "0.41121495" in RM.SELECT_RULE


def test_level_mean_is_arithmetic_mean():
    a = np.array([400.0, 500.0, 600.0])
    b = np.array([410.0, 490.0, 610.0])
    c = np.array([390.0, 510.0, 590.0])
    out = RM.mean_predictions([a, b, c])
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert np.allclose(out, np.array([400.0, 500.0, 600.0]))
    try:
        RM.mean_predictions([a, b])
    except ValueError:
        pass
    else:
        raise AssertionError("must require exactly 3 seed preds")
    try:
        RM.mean_predictions([a, b, np.array([1.0, np.nan, 3.0])])
    except ValueError:
        pass
    else:
        raise AssertionError("must reject non-finite preds")


def test_check_aligned_detects_misalignment():
    f1 = pd.DataFrame({"origin": pd.to_datetime(["2025-08-04"] * 2, utc=True), "h": [1, 2]})
    f2 = pd.DataFrame({"origin": pd.to_datetime(["2025-08-04"] * 2, utc=True), "h": [1, 2]})
    f3 = pd.DataFrame({"origin": pd.to_datetime(["2025-08-04"] * 2, utc=True), "h": [1, 3]})
    assert RM.check_aligned([f1, f2]) is True
    assert RM.check_aligned([f1, f3]) is False


def test_import_only_no_copy():
    for f in ("runner.py", "richmean.py"):
        text = (PKG / f).read_text()
        for ban in COPY_BANS:
            assert ban not in text, f"{f} copies {ban}"
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src
    assert "import analog as AN" in src
    assert "import joint as JT" in src
    assert "import f1first as F1" in src
    assert "import richmean as RM" in src


def test_joint_columns_still_17_plus_4():
    assert len(ST.strip_columns()) == 17
    assert len(AN.analog_columns()) == 4
    assert len(JT.extra_columns()) == 21


def test_pregrid_and_floor_match_d21():
    assert tuple(F1.GRID) == ("CH6/l2", "CH6/l1", "CH6V2/l2", "CH6V2/l1",
                              "CH3V3/l2", "CH3V3/l1")
    assert F1.F1_FLOOR == 0.41121495


def test_runner_uses_cuda_pilot_poison_levelmean():
    src = (PKG / "runner.py").read_text()
    assert '"device": "cuda"' in src or "'device': 'cuda'" in src
    assert "--pilot" in src and "--seeds" in src and "--jobs" in src
    assert "72h" in src and "poison.json" in src and "selection.json" in src
    assert "RM.mean_predictions" in src and "RM.check_aligned" in src
    assert "RM.SEEDS" in src or "RM.SELECT_RULE" in src
    assert "Optuna" not in src and "optuna" not in src


def test_no_selection_fed_choice():
    src = (PKG / "runner.py").read_text()
    low = src.lower()
    assert "dev window only" in low or "dev-only" in low
    assert "selection/audit never" in low
    assert "select_config_f1first(dev_mse, dev_f1, e0003_mse)" in src


def test_scratch_is_worker_d31_only():
    src = (PKG / "runner.py").read_text()
    assert "store/scratch/worker-d31" in src
    for other in ("worker-d21", "worker-d22", "worker-d23", "worker-d28", "worker-d30",
                  "worker-d16", "worker-d17", "worker-d19", "worker-d20", "worker-d10"):
        assert other not in src


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("runner.py", "richmean.py"):
        text = (PKG / name).read_text()
        for p in FORBIDDEN:
            assert p not in text, f"{name} contains {p}"
        for d in WINDOW_DATES:
            assert d not in text, f"{name} must not reference window date {d}"
        assert "2026-06" not in text, name


def test_no_stale_prediction_reads():
    for name in ("runner.py", "richmean.py"):
        text = (PKG / name).read_text()
        assert "runs/E0060" not in text and "runs/E0079" not in text
        assert "official-predictions" not in text
        # fresh refits only: never open another experiment's predictions parquet
        assert "predictions_dev.parquet" not in text.replace("predictions_{dev", "")


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"runner.py", "richmean.py"} <= set(files)


def test_parallel_finals_bounded_threads():
    src = (PKG / "runner.py").read_text()
    assert "from concurrent.futures import ThreadPoolExecutor" in src
    assert "ThreadPoolExecutor(max_workers=a.jobs)" in src
    assert "choices=(1, 3)" in src and "default=3" in src
    assert "FIT_JOBS = 3" in src and "FIT_NJOBS = 6" in src and "18 <= 20" in src
    assert "ProcessPoolExecutor" not in src
    assert "def _fit_period(job):" in src
    assert "FP = {**BASE_PARAMS," in src or "n_jobs" in src


def _load_runner_mod(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, PKG / "runner.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_arrays_dtypes_and_shapes():
    mod = _load_runner_mod("d31_runner_arrays")
    n = 10
    tr = pd.DataFrame({"y": np.linspace(300.0, 800.0, n),
                       "f_a": np.arange(n, dtype=float), "f_b": np.ones(n)})
    te = pd.DataFrame({"y": np.linspace(320.0, 780.0, n // 2),
                       "f_a": np.arange(n // 2, dtype=float), "f_b": np.ones(n // 2)})
    Xtr, ytr, Xte = mod._fit_arrays(tr, te, ["f_a", "f_b"])
    assert Xtr.shape == (n, 2) and Xtr.dtype == np.float32 and Xtr.flags["C_CONTIGUOUS"]
    assert ytr.shape == (n,) and ytr.dtype == float
    assert Xte.shape == (n // 2, 2) and Xte.dtype == np.float32


def test_fcache_key_versioned_and_deterministic():
    mod = _load_runner_mod("d31_runner_fcache")
    assert mod.FCACHE_VERSION == 1
    assert tuple(mod.SEEDS) == (21, 22, 23)
    p = {"freq": "6h", "window": "dev", "config": "CH6V2/l2"}
    assert mod.fcache_key("tr_cfg", p) == mod.fcache_key("tr_cfg", p)
    assert mod.fcache_key("tr_cfg", p) != mod.fcache_key("te_cfg", p)
    assert mod.fcache_key("tr_cfg", p) != mod.fcache_key("tr_cfg", {**p, "freq": "3h"})
    src = (PKG / "runner.py").read_text()
    assert '"code": {' in src and '"data": {' in src and '"v": FCACHE_VERSION' in src


def test_fcache_roundtrip_and_tamper_miss(tmp_path):
    mod = _load_runner_mod("d31_runner_fcache_rt")
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
    (tmp_path / "featcache" / (key + ".json")).write_text('{"key": "wrong"}')
    assert mod.fcache_get(tmp_path, key) is None
    (tmp_path / "featcache" / (key + ".parquet")).unlink(missing_ok=True)
    assert mod.fcache_get(tmp_path, key) is None


def test_runner_rejects_wrong_seeds():
    mod = _load_runner_mod("d31_runner_seeds")
    assert tuple(mod.SEEDS) == tuple(RM.SEEDS) == (21, 22, 23)
    src = (PKG / "runner.py").read_text()
    assert "seeds must be exactly" in src
    assert "need exactly 3 distinct fresh seeds" in src
