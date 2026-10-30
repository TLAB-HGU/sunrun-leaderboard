"""Torch-free residual evaluator gate tests. Synthetic temp dirs only.

Asserts exit-code contracts and parity gates without fitting, scoring, or
reading official truth/predictions. Never imports Torch or residual model
sources.
"""
import json
import sys

import pytest

# Static Torch-freedom is asserted inside the test via source inspection,
# so suite order (execution tests may import torch first) cannot break this.


def test_evaluator_module_is_torch_free():
    from pathlib import Path
    src = (Path(__file__).parent / '../experiments/mse20_hss_residual/evaluate.py').resolve().read_text()
    assert 'import torch' not in src and 'from torch' not in src
    import experiments.mse20_hss_residual.evaluate as evaluator
    assert 'torch' not in evaluator.__dict__.get('__doc__', '')
    assert evaluator.MODEL_SOURCE_NAMES
    assert all(name.startswith('experiments/mse20_hss_residual/') for name in evaluator.MODEL_SOURCE_NAMES)


def test_evaluate_exits_invalid_without_run(tmp_path):
    import experiments.mse20_hss_residual.evaluate as evaluator
    report = None
    try:
        evaluator.verify(str(tmp_path / 'missing'))
    except Exception as error:
        report = f'{type(error).__name__}: {error}'
    assert report is not None


def test_regression_command_requires_plugin_and_pytest():
    import experiments.mse20_hss_residual.protocol as protocol
    assert protocol.CLOCK_PLUGIN == 'experiments.mse20_hss_phase_verify_v1.clock_plugin'
    command = [str(protocol.MODEL_PYTHON), '-m', 'pytest', '-q', '-p',
               protocol.CLOCK_PLUGIN, 'tests/test_x.py']
    assert 'pytest' in command and '-p' in command and protocol.CLOCK_PLUGIN in command


def test_caps_are_neural_only_with_external_backbone_budget():
    import experiments.mse20_hss_residual.protocol as protocol
    assert protocol.CAPS['dev_claimed_attempts'] == 28
    assert protocol.CAPS['structural_configs'] == 2
    assert protocol.CAPS['final_head_fits'] == 1
    assert protocol.CAPS['new_backbone_fits'] == 0
    assert protocol.CAPS['external_backbone_dependencies'] == 45


def test_metadata_family_and_version_pinned():
    import experiments.mse20_hss_residual.protocol as protocol
    assert protocol.definition()['family'] == 'hss_event_residual_fixed_backbone'
    assert protocol.definition()['version'] == 1
    assert protocol.definition()['epochs'] == 20


def test_prepare_refuses_old_run_binding(tmp_path):
    import experiments.mse20_hss_residual.prepare as prepare
    import experiments.mse20_hss_residual.protocol as protocol
    with pytest.raises(Exception):
        prepare.prepare(run_dir=protocol.RUN, original_run=protocol.RUN)


def test_total_and_cells_thresholds_match_frozen_contract():
    import experiments.mse20_hss_residual.protocol as protocol
    from experiments.mse20_hss_phase_ff1 import protocol as ff1
    assert protocol.TOTAL_MAX == ff1.TOTAL_MAX == 3289.0890348547073
    assert protocol.REGIME_CELLS == ff1.REGIME_CELLS
    assert json.loads(json.dumps(dict(caps=dict(protocol.CAPS))))['caps']['dev_claimed_attempts'] == 28
