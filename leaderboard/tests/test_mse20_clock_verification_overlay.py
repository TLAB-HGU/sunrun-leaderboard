"""Proof tests for the verify-v1 overlay. No fits, no scores, CPU-only tmp dirs.

Passes with or without the clock plugin loaded. The two subprocess tests
run ONLY the targeted legacy protocol tests (4 synthetic claim tests + 5
deadline boundary params) with ``-p <clock plugin>``; everything else is
in-process and synthetic.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "experiments.mse20_hss_phase_verify_v1.clock_plugin"

from experiments.mse20_ch_dynamics import protocol as legacy
from experiments.mse20_hss_phase_ff1 import protocol as ff1
from experiments.mse20_hss_phase_verify_v1 import clock_plugin, contract
from experiments.mse20_hss_phase_verify_v1 import evaluate as overlay_evaluate
from experiments.mse20_hss_phase_verify_v1 import finish as overlay_finish
from experiments.mse20_hss_phase_verify_v1 import prepare as overlay_prepare
from experiments.mse20_hss_phase_verify_v1 import regression as overlay_regression


def _cpu_env():
    return dict(
        os.environ, CUDA_VISIBLE_DEVICES="", MSE20_TEST_DEVICE="cpu",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
    )


def _run_legacy(node_ids):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", PLUGIN, *node_ids],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=_cpu_env(), timeout=600,
    )


def test_plugin_repairs_synthetic_claim_tests():
    result = _run_legacy(list(contract.FAILED_NODE_IDS))
    assert result.returncode == 0, result.stdout[-3000:]
    assert "4 passed" in result.stdout


def test_plugin_preserves_explicit_now_boundaries():
    node = "tests/test_mse20_ch_dynamics_protocol.py::test_deadline_boundaries"
    result = _run_legacy([node])
    assert result.returncode == 0, result.stdout[-3000:]
    assert "5 passed" in result.stdout


def _synthetic_runtime(tmp_path, stop="2026-10-04T06:26:29Z"):
    legacy.write_new(
        tmp_path / "runtime.json",
        {"search_stop_utc": stop, "optimization_stop_utc": stop,
         "deadline_utc": "2026-10-04T07:41:29Z"},
    )


def test_original_deadline_rejects_real_expiry_and_reserve(tmp_path):
    """Production deadline, real wall clock, no plugin involved."""
    _synthetic_runtime(tmp_path)
    for stage in ("search", "optimization", "final"):
        with pytest.raises(TimeoutError):
            legacy.deadline(tmp_path, stage, 0)
    # Explicit now: just inside the reserve passes, exactly at it fails.
    assert legacy.deadline(tmp_path, "search", 900, "2026-10-04T06:11:28Z") > 900
    with pytest.raises(TimeoutError):
        legacy.deadline(tmp_path, "search", 900, "2026-10-04T06:11:29Z")
    with pytest.raises(TimeoutError):
        legacy.deadline(tmp_path, "optimization", 0, "2026-10-04T06:26:29Z")
    assert legacy.deadline(tmp_path, "final", 0, "2026-10-04T07:41:28Z") > 0
    with pytest.raises(TimeoutError):
        legacy.deadline(tmp_path, "final", 0, "2026-10-04T07:41:29Z")


def test_no_global_clock_freeze_and_narrow_scope():
    before_deadline, before_now = legacy.deadline, pd.Timestamp.now.__func__
    import importlib

    importlib.reload(clock_plugin)
    assert legacy.deadline is before_deadline
    assert pd.Timestamp.now.__func__ is before_now
    assert clock_plugin.TARGET_ATTR == "fake_run"
    assert set(clock_plugin.TARGET_MODULES) == {
        "test_mse20_ch_dynamics_protocol", "tests.test_mse20_ch_dynamics_protocol"
    }
    # Collection hook is a no-op without the named module under test.
    clock_plugin.pytest_collection_modifyitems(session=None, config=None, items=[])
    source = inspect.getsource(clock_plugin.pytest_collection_modifyitems)
    assert "setattr" not in source


def test_other_modules_and_fixtures_unchanged():
    text = (ROOT / "tests/test_mse20_ch_dynamics_protocol.py").read_text()
    block = text.split("def fake_run", 1)[1].split("\ndef ", 1)[0]
    assert "utcnow" in block and "deadline(" not in block and "p.deadline" not in block
    protocol_text = (ROOT / "experiments/mse20_ch_dynamics/protocol.py").read_text()
    assert 'pd.Timestamp.now(tz="UTC") if now is None' in protocol_text
    assert legacy.deadline.__module__ == "experiments.mse20_ch_dynamics.protocol"


def test_contract_suite_gates_and_closures():
    tests = contract.suite_tests()
    assert all(not t.startswith("tests/test_vjepa") for t in tests)
    assert "tests/test_mse20_clock_verification_overlay.py" in tests
    assert "tests/test_mse20_ch_dynamics_protocol.py" in tests
    assert contract.TOTAL_MAX == 3289.0890348547073
    assert contract.CAPS["dev_claimed_attempts"] == 28
    assert contract.CAPS["dev_backbone_claims"] == 42
    from experiments.mse20_hss_phase_ff1 import evaluate as ff1_evaluate
    from experiments.mse20_hss_phase_ff1 import protocol as ff1

    assert contract.MODEL_SOURCE_NAMES is ff1_evaluate.MODEL_SOURCE_NAMES
    assert contract.CAPS is ff1.CAPS
    assert "experiments/mse20_hss_phase_ff1/model.py" in contract.MODEL_SOURCE_NAMES


def test_prepare_binds_without_fitting_or_scoring(tmp_path):
    original = contract.ORIGINAL_RUN
    before = (original / "regression-summary.json").read_bytes()
    overlay = overlay_prepare.prepare(tmp_path / "proof", original)
    assert overlay["scientific_attempts_added"] == 0
    assert overlay["model_refits"] == 0
    assert overlay["original_failed_regression"]["passed"] is False
    assert overlay["counts"]["dev_claimed_attempts"] == 28
    assert overlay["counts"]["dev_backbone_claims"] == 42
    assert (original / "regression-summary.json").read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "proof").iterdir()) == ["verification-overlay.json"]
    with pytest.raises(FileExistsError):
        overlay_prepare.prepare(tmp_path / "proof", original)


def test_regression_and_finish_refuse_unsafe_targets(tmp_path):
    original = contract.ORIGINAL_RUN
    with pytest.raises(ValueError, match="NEW run"):
        overlay_regression.regression(original, original)
    overlay_prepare.prepare(tmp_path / "proof", original)
    with pytest.raises(ValueError, match="regression"):
        overlay_finish.finish(tmp_path / "proof", original)
    assert overlay_evaluate.main(
        ["--original-run", str(original), "--run-dir", str(tmp_path / "proof")]
    ) == 2


def test_overlay_binding_records_failed_evidence(tmp_path):
    original = contract.ORIGINAL_RUN
    overlay = overlay_prepare.prepare(tmp_path / "proof", original)
    failed = json.loads((original / "regression-summary.json").read_text())
    assert overlay["original_failed_regression"]["exit_code"] == failed["exit_code"] == 1
    assert overlay["original_failed_regression"]["log"] == failed["files"]["log"]

def _build_synthetic_proof(tmp_path, dirname="proof"):
    """Valid-case proof parts: real registration sources, live overlay closure,
    refs to real suite files, exact command, non-empty log. NOT written."""
    original = contract.ORIGINAL_RUN
    registration = ff1.read_json(original / "registered.json")
    run_dir = tmp_path / dirname
    run_dir.mkdir()
    log = run_dir / "regression-synthetic.log"
    log.write_text("synthetic proof log\n")
    tests = contract.suite_tests()
    files = {"log": ff1.file_ref(log)}
    for name in tests:
        files[name] = ff1.file_ref(ff1.ROOT / name)
    proof = {
        "suite": contract.REGRESSION_SUITE,
        "passed": True,
        "exit_code": 0,
        "model_sources": registration["sources"],
        "overlay_sources": contract.overlay_refs(),
        "files": files,
        "command": [str(ff1.MODEL_PYTHON), "-m", "pytest", "-q",
                    "-p", contract.PLUGIN, *tests],
        "log": files["log"],
        "clock_plugin": contract.PLUGIN,
    }
    return run_dir, original, registration, proof


def test_proof_accepts_valid_synthetic_binding(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path)
    ff1.write_new(run_dir / "regression-summary.json", proof)
    checked = overlay_evaluate._verify_proof(run_dir, original, registration)
    assert checked["passed"] is True and checked["exit_code"] == 0


def test_proof_rejects_pruned_inventory(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path)
    pruned = next(k for k in proof["files"] if k != "log")
    del proof["files"][pruned]
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="test inventory incomplete"):
        overlay_evaluate._verify_proof(run_dir, original, registration)


def test_proof_rejects_changed_command_and_plugin(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path, "p1")
    proof["command"] = [c for c in proof["command"] if c != "-p"]
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="missing plugin"):
        overlay_evaluate._verify_proof(run_dir, original, registration)
    run_dir, _, _, proof = _build_synthetic_proof(tmp_path, "p2")
    proof["command"] = [*proof["command"], "--tb=short"]
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="command differs"):
        overlay_evaluate._verify_proof(run_dir, original, registration)
    run_dir, _, _, proof = _build_synthetic_proof(tmp_path, "p3")
    proof["clock_plugin"] = "experiments.other.clock_plugin"
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="without the clock plugin"):
        overlay_evaluate._verify_proof(run_dir, original, registration)


def test_proof_rejects_swapped_or_empty_log(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path, "p1")
    other = tmp_path / "other.log"
    other.write_text("unrelated log\n")
    proof["log"] = ff1.file_ref(other)
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="execution log"):
        overlay_evaluate._verify_proof(run_dir, original, registration)
    run_dir, _, _, proof = _build_synthetic_proof(tmp_path, "p2")
    empty = run_dir / "regression-synthetic.log"
    empty.write_text("")
    ref = ff1.file_ref(empty)
    proof["files"]["log"] = ref
    proof["log"] = ref
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="empty proof log"):
        overlay_evaluate._verify_proof(run_dir, original, registration)


def test_proof_rejects_changed_source_closures(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path, "p1")
    proof["model_sources"] = dict(proof["model_sources"])
    proof["model_sources"]["experiments/mse20_hss_phase_ff1/model.py"] = {
        "path": "/tmp/swapped.py", "sha256": "0" * 64}
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="changed sources"):
        overlay_evaluate._verify_proof(run_dir, original, registration)
    run_dir, _, _, proof = _build_synthetic_proof(tmp_path, "p2")
    proof["overlay_sources"] = {}
    ff1.write_new(run_dir / "regression-summary.json", proof)
    with pytest.raises(Exception, match="source closure differs"):
        overlay_evaluate._verify_proof(run_dir, original, registration)


def test_proof_rejects_scientific_state(tmp_path):
    run_dir, original, registration, proof = _build_synthetic_proof(tmp_path)
    ff1.write_new(run_dir / "regression-summary.json", proof)
    (run_dir / "attempts").mkdir()
    with pytest.raises(Exception, match="scientific state"):
        overlay_evaluate._verify_proof(run_dir, original, registration)


def _failed_log_text(node_ids, traces=4):
    lines = []
    for i, node in enumerate(node_ids):
        marker = "TimeoutError: search exhausted" if i < traces else "AssertionError: synthetic"
        lines.append(f"FAILED {node} - {marker}\n")
    return "".join(lines)


def _fake_failed_original(tmp_path, name, log_text):
    orig = tmp_path / name
    orig.mkdir()
    (orig / "registered.json").write_text("{}")
    log = tmp_path / f"{name}.log"
    log.write_text(log_text)
    ref = ff1.file_ref(log)
    ff1.write_new(orig / "regression-summary.json", {
        "passed": False, "exit_code": 1, "files": {"log": ref}, "log": ref})
    return orig


def test_prepare_rejects_unrelated_failed_nodes(tmp_path):
    nodes = (*contract.FAILED_NODE_IDS[:3], "tests/test_other.py::test_unrelated")
    orig = _fake_failed_original(tmp_path, "orig-bad", _failed_log_text(nodes))
    with pytest.raises(ValueError, match="four diagnosed"):
        overlay_prepare.prepare(tmp_path / "new-bad", orig)
    short = _fake_failed_original(tmp_path, "orig-short", _failed_log_text(contract.FAILED_NODE_IDS, traces=3))
    with pytest.raises(ValueError, match="four diagnosed"):
        overlay_prepare.prepare(tmp_path / "new-short", short)
    # Positive control: exact nodes + 4 traces passes the node guard and
    # fails later (fake original has no science files), never on node identity.
    exact = _fake_failed_original(tmp_path, "orig-exact", _failed_log_text(contract.FAILED_NODE_IDS))
    with pytest.raises(Exception) as exc:
        overlay_prepare.prepare(tmp_path / "new-exact", exact)
    assert "four diagnosed" not in str(exc.value)


def test_prepare_rejects_swapped_and_rotated_log(tmp_path):
    orig = tmp_path / "orig-swap"
    orig.mkdir()
    (orig / "registered.json").write_text("{}")
    log = tmp_path / "real.log"
    log.write_text(_failed_log_text(contract.FAILED_NODE_IDS))
    other = tmp_path / "other.log"
    other.write_text("unrelated\n")
    ff1.write_new(orig / "regression-summary.json", {
        "passed": False, "exit_code": 1,
        "files": {"log": ff1.file_ref(log)}, "log": ff1.file_ref(other)})
    with pytest.raises(ValueError, match="log binding differs"):
        overlay_prepare.prepare(tmp_path / "new-swap", orig)
    rotated = tmp_path / "orig-rot"
    rotated.mkdir()
    (rotated / "registered.json").write_text("{}")
    rlog = tmp_path / "rot.log"
    rlog.write_text(_failed_log_text(contract.FAILED_NODE_IDS))
    stale = ff1.file_ref(rlog)
    rlog.write_text(rlog.read_text() + "appended after ref\n")
    ff1.write_new(rotated / "regression-summary.json", {
        "passed": False, "exit_code": 1, "files": {"log": stale}, "log": stale})
    # A rotated log dies at ref verification before the rotation compare.
    with pytest.raises(ValueError, match="stale artifact"):
        overlay_prepare.prepare(tmp_path / "new-rot", rotated)


def test_finish_calls_fresh_proof_before_scoring(tmp_path, monkeypatch):
    original = contract.ORIGINAL_RUN
    run_dir = tmp_path / "proof"
    overlay_prepare.prepare(run_dir, original)
    _, _, _, proof = _build_synthetic_proof(tmp_path, "staged")
    staged = run_dir / "regression-summary.json"
    ff1.write_new(staged, proof)
    calls = []

    def spy(rd, o, reg):
        calls.append((Path(rd).resolve(), Path(o).resolve()))
        raise RuntimeError("synthetic proof veto")

    monkeypatch.setattr(overlay_evaluate, "_verify_proof", spy)
    with pytest.raises(RuntimeError, match="synthetic proof veto"):
        overlay_finish.finish(run_dir, original)
    assert calls == [(run_dir.resolve(), original.resolve())]
    assert list(run_dir.glob("final-score-*.json")) == []
    assert not (run_dir / "final-manifest.json").exists()


def test_clock_order_and_deadline_gates():
    with pytest.raises(ValueError, match="nonmonotonic"):
        overlay_evaluate.clock(
            ["2026-10-04T07:00:00Z", "2026-10-04T06:00:00Z"], "synthetic order")
    with pytest.raises(ValueError, match="future-dated"):
        overlay_evaluate.clock(
            ["2026-10-04T06:00:00Z", "2999-01-01T00:00:00Z"], "synthetic future")
    assert ff1.deadline(contract.ORIGINAL_RUN, "final") == 3600.0


def test_structural_wiring_for_full_claim_guards():
    """Structural call-proof only (complement, not replacement).

    Full global-winner/fit-claim/replay/score fixtures need the immutable
    model plus the official benchmark, so a synthetic full-claim mock is
    impractical here (gap recorded in the plan). This pins the restored
    call sites and their order instead; the behavioral tamper tests above
    exercise the proof/prepare/finish/clock helpers at runtime.
    """
    import inspect

    module_src = inspect.getsource(overlay_evaluate)
    verify_src = inspect.getsource(overlay_evaluate.verify)
    for site in (
        "selection_bridge.verify_selection",
        "global-dev-selection.json",
        "final-fit-claim.json",
        "finalize.model_inventory",
        "verify_backbone_inventory",
        "require_legacy_complete",
    ):
        assert site in module_src, site
    for site in (
        "verify_registration",
        "official_benchmark",
        "score_pair",
        "ff1.deadline",
        "global winner before final fit",
        "selection/refit freeze",
        "final inference reserve",
        "fresh post-fit proof regression",
        "final inference/replay clock",
        "final repeated scoring",
        "repeat score order",
    ):
        assert site in verify_src, site
    finish_src = inspect.getsource(overlay_finish.finish)
    assert finish_src.index("_require_fresh_regression") < finish_src.index("score_pair")
    regression_src = inspect.getsource(overlay_regression.regression)
    for site in ('"-p"', "contract.PLUGIN", "contract.suite_tests()", "overlay_sources"):
        assert site in regression_src, site
    prepare_src = inspect.getsource(overlay_prepare.prepare)
    for site in ("FAILED_NODE_IDS", "TimeoutError: search exhausted"):
        assert site in prepare_src, site
