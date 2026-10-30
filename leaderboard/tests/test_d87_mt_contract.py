"""D87 multi-task contract: preregistration guards for the level+event MLP pilot.

Static checks only (no GPU, no runs): double-tag hypothesis, single target
axis, E0124/X0111 citation, frozen auxiliary weight, 2-head model shapes,
file/scratch isolation, forbidden-path scan, and output-contract keys.
"""
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1] / "experiments" / "d87_multitask"
sys.path.insert(0, str(PKG))

import model as MO  # noqa: E402
import spec as SP  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet",
             "folds.parquet", "naive.parquet", "official-predictions",
             "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]


def test_hypothesis_double_tag_leads_and_cites_e0124():
    h = SP.HYPOTHESIS
    assert h.startswith("[family=nn]")
    assert "[axes: target=multi-task]" in h
    assert "E0124" in h and "X0111" in h
    SP.check_tags()


def test_single_axes_change_and_full_rule_preregistered():
    import re
    tags = re.findall(r"\[axes:\s*([^\]]+)\]", SP.HYPOTHESIS)
    assert len(tags) == 1
    axes = [p.split("=")[0].strip().lower() for p in tags[0].split(",") if "=" in p]
    assert axes == ["target"]
    assert "FULL_RULE" in dir(SP)
    for key in ("rc 0", "poison", "converged", "E0001"):
        assert key in SP.FULL_RULE


def test_auxiliary_weight_and_event_threshold_frozen():
    assert SP.LAMBDA_EVT == 0.5
    assert SP.EVENT_THRESH == 500.0
    src = (PKG / "runner.py").read_text()
    assert "SP.LAMBDA_EVT" in src
    assert "SP.EVENT_THRESH" in src


def test_model_has_level_plus_event_heads():
    net = MO.MultiTaskMLP(8)
    x = torch.randn(5, 8)
    lvl, evt = net(x)
    assert lvl.shape == (5,) and evt.shape == (5,)
    assert MO.count_params(net) > 0
    assert len(MO.ARCH["heads"]) == 2


def test_file_and_scratch_isolation():
    files = sorted(p.name for p in PKG.iterdir() if p.is_file())
    assert len(files) <= 5
    assert set(files) <= {"spec.py", "features.py", "model.py", "runner.py"}
    assert SP.SCRATCH.name == "worker-d87"
    blob = "".join((PKG / f).read_text() for f in files)
    assert "worker-d53" not in blob and "worker-d88" not in blob and "worker-d89" not in blob
    for f in files:
        assert "store/scratch/worker-d87" in blob or f != "spec.py"


def test_no_forbidden_paths_in_code():
    blob = "".join((PKG / f).read_text(errors="ignore") for f in PKG.iterdir() if f.is_file())
    assert [p for p in FORBIDDEN if p in blob] == []


def test_output_contract_keys_present_in_runner():
    src = (PKG / "runner.py").read_text()
    for key in ("predictions_{dev,selection,audit}", "poison.json", "learning_curves.json",
                '"passed"', "6h", "dev-train-tail-OOF-only", "cuda"):
        assert key in src
