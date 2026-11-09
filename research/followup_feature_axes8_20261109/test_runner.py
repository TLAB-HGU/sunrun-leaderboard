"""Dynamic feature unions, frozen science, and resource-gated launch contracts."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

import baseline
import core
import host_scheduler
import scheduler_support as support

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def inventory():
    from basic_features import describe
    from strip_features import describe as strip_describe
    from external_features import describe as external_describe
    specs = {axis: describe(axis) for axis in range(1, 5)}
    specs[5] = strip_describe()
    for axis in (6, 7, 8):
        spec = external_describe(axis)
        specs[axis] = {'constant': spec['features'], 'varying': []}
    candidates = [{'name': name, 'kind': 'constant', 'status': 'eligible'}
                  for name in ('v_lag24', 'v_lag48', 'v_lag72')]
    for spec in specs.values():
        for kind in ('constant', 'varying'):
            candidates.extend({'name': name, 'kind': kind, 'status': 'eligible'} for name in spec[kind])
    return {'parent_features': baseline.PARENT_NAMES, 'candidates': candidates}, specs


@pytest.mark.parametrize('axis', range(1, 9))
@pytest.mark.parametrize('base', [[], ['v_lag24', 'v_lag48', 'v_lag72']])
def test_dynamic_axis_union_and_expansion(inventory, axis, base):
    catalog, specs = inventory
    spec = specs[axis]
    adds = base + list(spec['constant']) + list(spec['varying'])
    config = core.normalize_config({'seed': 43, 'add_features': adds}, catalog)
    names = core.feature_names(config, catalog)
    assert sum(map(len, names.values())) == 28 + len(adds)
    for kind in ('constant', 'varying'):
        assert names[kind][:len(baseline.PARENT_NAMES[kind])] == baseline.PARENT_NAMES[kind]
    rng = np.random.default_rng(7)
    grid = {'C': rng.normal(size=(3, len(names['constant']))),
            'E': rng.normal(size=(3, 72, len(names['varying'])))}
    parent = {'C': grid['C'][:, :20], 'E': grid['E'][:, :, :8]}
    for start in (0, 24, 48):
        matrix = core.design(grid, start, 24)
        assert matrix.shape == (3, 212 + len(base) + len(spec['constant']) + 24 * len(spec['varying']))
        assert len(core.expanded_names(names, start, 24)) == matrix.shape[1]
        np.testing.assert_array_equal(matrix[:, :212], core.design(parent, start, 24))
    recipe = {'experiment': f'axis{axis}_s43', 'seed': 43,
              'config': {'seed': 43, 'add_features': adds}}
    with patch.object(core, 'read_inventory', return_value=catalog):
        assert support.expected_config(recipe)['add_features'] == adds


def test_scientific_functions_and_baseline_identical():
    previous = ROOT.parent / 'lag24_48_72_confirm_20261109'
    for filename in ('baseline.py', 'competition.py', 'replay.py'):
        assert (ROOT / filename).read_bytes() == (previous / filename).read_bytes()
    def functions(path):
        return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name != 'main'}
    assert functions(ROOT / 'core.py') == functions(previous / 'core.py')


def test_resource_limits():
    assert host_scheduler.capacity([{'threads': 4}] * 3, 11, 12)
    assert not host_scheduler.capacity([{'threads': 4}] * 6, 100, 0)
    assert not host_scheduler.capacity([{'threads': 22}], 100, 0)
    assert not host_scheduler.capacity([], 10.99, 0)
    assert not host_scheduler.capacity([], 100, 21)


def test_launch_per_axis_input_cpu_and_single_architecture(tmp_path, inventory):
    catalog, specs = inventory
    for name in ('core.py', 'baseline.py', 'competition.py', 'inventory.json'):
        shutil.copy2(ROOT / name, tmp_path / name)
    data = tmp_path / 'axis2.parquet'
    data.write_bytes(b'identity fixture only')
    recipe = {'experiment': 'axis2_s42', 'seed': 42, 'data': str(data),
              'config': {'seed': 42, 'add_features': list(specs[2]['varying'])}}
    assignment = {'host': 'golab2', 'python': 'python', 'data': '/unused', 'observations': '/observations'}
    with patch.object(core, 'read_inventory', return_value=catalog), \
            patch.object(support, 'session_alive', return_value=False), \
            patch.object(support, 'session_id', return_value='$123'), \
            patch.object(support.subprocess, 'run'):
        launch = support.launch(tmp_path, assignment, recipe)
    out = tmp_path / 'runs' / recipe['experiment']
    assert launch['input_identity'] == support.identity(data)
    assert launch['command'][launch['command'].index('--data') + 1] == str(data)
    assert launch['threads'] == 4
    wrapper = (out / 'experiment.py').read_text()
    assert 'SUPPORTED' not in wrapper
    compile(wrapper, 'experiment.py', 'exec')
    shell = (out / 'run.sh').read_text()
    assert 'CUDA_VISIBLE_DEVICES=' in shell
    assert '"unit": 0, "units": 6' in shell
    subprocess.run(['bash', '-n', str(out / 'run.sh')], check=True)


def test_preflight_rejects_input_and_assignment_drift(tmp_path, inventory):
    catalog, _ = inventory
    data = tmp_path / 'axis.parquet'
    data.write_bytes(b'unchanged')
    recipe = {'experiment': 'axis1_s42', 'config': {'seed': 42, 'add_features': ['v_lag24']}, 'data': str(data)}
    assignment = {'host': 'golab2', 'experiments': [recipe]}
    filenames = ('core.py', 'baseline.py', 'competition.py', 'inventory.json', 'replay.py',
                 'scheduler_support.py', 'host_scheduler.py', 'assignment.json')
    for name in filenames:
        (tmp_path / name).write_text('{}')
    gate = {'passed': True, 'golab2': {
        'input_identities': {str(data): support.identity(data)},
        'source_identity': {name: support.identity(tmp_path / name) for name in filenames}}}
    (tmp_path / 'preflight.json').write_text(json.dumps(gate))
    with patch.object(core, 'read_inventory', return_value=catalog):
        host_scheduler.validate_preflight(tmp_path, assignment)
        data.write_bytes(b'changed input')
        with pytest.raises(ValueError, match='per-axis input changed'):
            host_scheduler.validate_preflight(tmp_path, assignment)
        gate['golab2']['input_identities'][str(data)] = support.identity(data)
        (tmp_path / 'preflight.json').write_text(json.dumps(gate))
        (tmp_path / 'assignment.json').write_text('{"tampered": true}')
        with pytest.raises(ValueError, match='source or assignment changed'):
            host_scheduler.validate_preflight(tmp_path, assignment)


def test_zero_exit_alone_cannot_complete(tmp_path, inventory):
    catalog, _ = inventory
    recipe = {'experiment': 'axis1_s42', 'config': {'seed': 42, 'add_features': []}}
    out = tmp_path / 'runs' / recipe['experiment']
    out.mkdir(parents=True)
    (out / 'train.exit').write_text('0\n')
    with patch.object(core, 'read_inventory', return_value=catalog):
        with pytest.raises(FileNotFoundError):
            support.validate_output(tmp_path, {}, recipe)
