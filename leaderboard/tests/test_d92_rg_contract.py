"""D92 regime-gate contract: dual tag, imported regime definition, fixed gate, gpu1, report-only.

Synthetic + static checks only (no GPU, no fits, no network).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d92_reggate"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "d88_regime"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import gatespec as SP  # noqa: E402
import regspec as D88  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core, evaluate  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W", "def unique_event_metrics",
             "def unique_event_report", "def stack_predictions", "def hsplit",
             "def lvlmean", "def assign(", "def origin_frac(", "def full_grid(",
             "def write("]
ALLOWED_IDS = {"E0001", "E0003", "E0163"}


def _blobs():
    return {f.name: (PKG / f.name).read_text() for f in
            [PKG / "gatespec.py", PKG / "runner.py"]}


def test_dual_tag_and_parent_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG == "[axes: ens=regime-gate]"
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes: ens=regime-gate]")
    assert SP.HYPOTHESIS.index("[family=") < SP.HYPOTHESIS.index("[axes:")
    for eid in ("E0001", "E0003", "E0163"):
        assert eid in SP.HYPOTHESIS, eid
    assert "X0144" in SP.HYPOTHESIS and "X0008" in SP.HYPOTHESIS
    assert "D4" in SP.HYPOTHESIS  # form + parents stated distinct from D4
    assert "no learned gate" in SP.HYPOTHESIS
    assert "experiments/d88_regime/runner.py" in SP.HYPOTHESIS
    assert "experiments/d88_regime/regspec.py" in SP.HYPOTHESIS
    assert "experiments/d1_longitude_strips/runner.py" in SP.HYPOTHESIS
    assert "research/AXES.md" in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["X0144"]
    assert SP.check_spec()["gate"] == {"slow": True, "mid": False, "fast": True}


def test_regime_definition_imported_not_copied():
    assert (D88.SLOW_MAX, D88.FAST_MIN) == (420.0, 520.0)
    assert D88.V0_COL == "v_lag0" and D88.REGIME_COL == "regime"
    assert tuple(D88.REGIMES) == ("slow", "mid", "fast")
    blobs = _blobs()
    for name, text in blobs.items():
        assert "import regspec" in text, f"{name} must import the D88 regime definition"
    for line in blobs["gatespec.py"].splitlines() + blobs["runner.py"].splitlines():
        s = line.strip()
        assert not re.match(r"(SLOW_MAX|FAST_MIN)\s*=\s*[\d.]", s), f"vendored threshold: {s}"
    assert SP.route_uses_regime("slow") is True
    assert SP.route_uses_regime("mid") is False
    assert SP.route_uses_regime("fast") is True
    with pytest.raises(KeyError):
        SP.route_uses_regime("quiet")


def test_periods_gpu_and_single_axis():
    assert {k: (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for k, (a, b) in ia.PERIODS.items()} == \
        {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in evaluate.PERIODS.items()}
    assert (SP.PURGE_H, SP.SEED, SP.FREQ_PILOT, SP.FREQ_FULL) == (72, 0, "6h", "3h")
    assert SP.SCRATCH_NAME == "worker-d92"
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 10
    for token in ("D90", "D92", "D91", "gpu0", "gpu1", "CPU", "60s",
                  "worker-d92", "n_jobs<=20", "--prefer-gpu 1"):
        assert token in SP.GPU_REASON, token
    p = SP.fit_params("cuda")
    assert p["device"] == "cuda" and p["n_jobs"] == 2 and p["random_state"] == 0
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v, k
    assert SP.device_for.__code__.co_varnames[:1] == ("slot",)


def test_rules_preregistered_report_only():
    assert "dev only" in SP.SELECT_RULE and "selection/audit" in SP.SELECT_RULE
    assert "never read" in SP.SELECT_RULE and "72h purge" in SP.SELECT_RULE
    assert "no candidate search" in SP.SELECT_RULE
    assert "full 3h" in SP.FULL_RULE and "E0001" in SP.FULL_RULE
    assert "F1>=0.30" in SP.FULL_RULE and "no full run" in SP.FULL_RULE
    assert "E0003" in SP.FULL_RULE and "seed-repro" in SP.FULL_RULE
    assert "screening" in SP.FULL_RULE
    assert "no freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE


class _Head:
    def __init__(self, c):
        self.c = float(c)

    def predict(self, df):
        return np.full(len(df), self.c)


def _frame():
    return pd.DataFrame({"v_lag0": [300.0, 450.0, 600.0, float("nan")],
                         "regime": ["slow", "mid", "fast", "mid"],
                         "x": [1.0, 2.0, 3.0, 4.0]})


def test_gate_routes_tails_to_regime_mid_to_global():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d92_runner", PKG / "runner.py")
    RN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RN)
    heads = {"_global": _Head(0.0), "slow": _Head(1.0),
             "mid": _Head(2.0), "fast": _Head(3.0)}
    out = RN.gate_predict(heads, _frame(), ["x"])
    assert list(out) == [1.0, 0.0, 3.0, 0.0]  # slow/fast regime, mid/NaN-mid global
    assert RN.D88 is D88  # runner shares the imported regime definition


def test_owned_files_scratch_no_vendor():
    files = sorted(p.name for p in PKG.iterdir() if p.is_file() and not p.name.startswith("__"))
    files = [f for f in files if not f.endswith(".pyc")]
    assert len(files) <= 5, files
    assert {"gatespec.py", "runner.py"} <= set(files)
    blobs = _blobs()
    runner = blobs["runner.py"]
    for bad in FORBIDDEN:
        for name, text in blobs.items():
            assert bad not in text, f"{name}: {bad}"
    for bad in COPY_BANS:
        for name, text in blobs.items():
            assert bad not in text, f"{name}: {bad}"
    assert "cuda" in blobs["gatespec.py"] and "EXPOS_GPUS" in blobs["gatespec.py"]  # gpu1 device path lives in the spec
    assert "SP.device_for" in runner and "SP.fit_params" in runner  # runner takes its device from the spec
    assert SP.device_for(0) == "cpu" or "CUDA_VISIBLE_DEVICES" in os.environ or "EXPOS_GPUS" in os.environ  # foreground-CPU fallback reachable
    assert "gpu_lease" not in runner  # lease is handled by expos run --gpu
    assert "expos run" not in runner and "run E0" not in runner
    ids = set(re.findall(r"E0\d{3}", runner + blobs["gatespec.py"]))
    assert ids <= ALLOWED_IDS, ids - ALLOWED_IDS
    assert "store/scratch/worker-d92" in runner
    assert "worker-d88" not in runner  # scratch isolation is worker-d92 only
    assert "72h" in runner and "poison.json" in runner and "selection.json" in runner
    assert "PERIODS = dict(ia.PERIODS)" in runner
    assert "import gatespec" in runner and "import regspec" in runner
    assert "from existing_all import" in runner
    for name in ("freeze", "official_score"):
        assert name not in runner


def test_register_blob_would_not_flag():
    blob = "\n".join(_blobs().values())
    assert core.scan(blob) == []
