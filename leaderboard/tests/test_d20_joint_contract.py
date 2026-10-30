"""D20 joint contract: strips+analog, import-only, dev-only, causal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d20_strip_analog"
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d10_analog"))
sys.path.insert(0, str(PKG))
import joint as JT  # noqa: E402
import analog as AN  # noqa: E402
import strips as ST  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_weights", "def strip_features", "def select_slots",
             "def search_one", "def analog_long", "def build_state", "def scaler_from_predev"]


def test_joint_columns_17_plus_4():
    assert len(ST.strip_columns()) == 17
    assert len(AN.analog_columns()) == 4
    assert AN.analog_columns() == ["analog_best", "analog_mean3", "analog_dist", "analog_spread"]
    extra = JT.joint_extra_columns()
    assert len(extra) == 21 and len(set(extra)) == 21
    assert extra == ST.strip_columns() + AN.analog_columns()
    assert all(c.startswith(("strip_", "img_")) for c in extra[:17])
    assert all(c.startswith("analog_") for c in extra[17:])


def test_import_only_no_copy():
    for f in ("joint.py", "runner.py"):
        text = (PKG / f).read_text()
        assert "import strips" in text or "import analog" in text or "import joint" in text, f
        for ban in COPY_BANS:
            assert ban not in text, f"{f} copies {ban}"


def test_analog_lookup_window_and_keys():
    assert AN.WIN_LO_H == 480 and AN.WIN_HI_H == 840 and AN.K_TOP == 3
    assert set(AN.KEY_SETS) == {"CH6", "CH6V2", "CH3V3"}
    assert set(AN.METRICS) == {"l2", "l1"}
    w, m = AN.select_config({"CH6/l2": 100.0, "CH6V2/l2": 90.0, "CH3V3/l1": 95.0})
    assert (w, m) == ("CH6V2", "l2")


def test_runner_joint_cuda_pilot_purge_poison_curves_seed():
    src = (PKG / "runner.py").read_text()
    assert '"cuda"' in src and "device" in src
    assert "--pilot" in src and '"6h"' in src
    assert "--seed" in src and "random_state" in src
    assert "72h" in src
    assert "poison.json" in src and "pred_invariant" in src
    assert "positive_control" in src
    assert "poison-strip" in src or "strip_rep" in src
    assert "poison-joint" in src or "joint_rep" in src
    assert "importance.json" in src
    assert "get_score" in (PKG / "joint.py").read_text()
    assert "learning_curves.json" in src and "evals_result" in src
    assert "dev_internal_curves" in src
    assert "selection.json" in src and "dev rows only" in src
    assert "manifest.json" in src
    assert "store/scratch/worker-d20" in src
    assert "JT.add_joint" in src or "add_joint" in src
    assert "Optuna" not in src and "optuna" not in src


def test_no_selection_fed_choice():
    src = (PKG / "runner.py").read_text()
    low = src.lower()
    assert "scores[\"selection\"]" not in src and "scores['selection']" not in src
    assert "df_dev" in src or "dev rows only" in src
    assert "dev_scores_for" in src
    assert "select_config" in src
    assert "selection_rule" in src


def test_scratch_isolation():
    for f in ("joint.py", "runner.py"):
        text = (PKG / f).read_text()
        assert "worker-d20" in text, f
        for other in ("worker-d16", "worker-d10", "worker-d21", "worker-d17", "worker-d19"):
            assert other not in text, f"{f} leaks {other}"


def test_no_forbidden_or_official_window():
    for f in ("joint.py", "runner.py"):
        text = (PKG / f).read_text()
        assert not [p for p in FORBIDDEN if p in text], f
        assert "2026-06" not in text, f


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 4, files
    assert {"joint.py", "runner.py"} <= set(files)
