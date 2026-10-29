"""Data-free checks for the reproduction pipelines (run: python -m pytest reproduce/tests)."""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))


def load(name, relative):
    path = REPRO / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def pw4():
    return load('pw4_train_predict', 'feat30_pointweight4/train_predict.py')


@pytest.fixture(scope='module')
def e1604():
    return load('e1604_run', 'e1604_hsspeak25/run.py')


def test_point_weights_upweight_high_targets_and_normalize(pw4):
    y = np.array([100., 549.9, 550., 700.])
    w = pw4.point_weights(y)
    assert w.mean() == pytest.approx(1.0)
    assert w[2] / w[0] == pytest.approx(4.0) and w[3] == w[2] and w[1] == w[0]


def test_point_weights_reject_nonfinite(pw4):
    with pytest.raises(ValueError):
        pw4.point_weights(np.array([1., np.nan]))


def test_design_selects_constant_and_varying_blocks(pw4):
    names = {'constant': ['a', 'b', 'c'], 'varying': ['u', 'v']}
    bundle = {'C': np.arange(6.).reshape(2, 3), 'E': np.arange(2 * 72 * 2.).reshape(2, 72, 2)}
    x = pw4.design(bundle, names, {'b', 'v'}, 24)
    assert x.shape == (2, 1 + 24) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:, 0], bundle['C'][:, 1])
    np.testing.assert_array_equal(x[:, 1:], bundle['E'][:, 24:48, 1])


def test_frozen_config_and_scorer_hash(pw4):
    frozen = json.loads((REPRO / 'feat30_pointweight4/config/frozen.json').read_text())
    assert len(set(frozen['selected_groups'])) == 30 and frozen['params']['n_estimators'] == 400
    scorer = REPRO / 'feat30_pointweight4/scorer/scoring.py'
    assert hashlib.sha256(scorer.read_bytes()).hexdigest() == frozen['scorer_sha256']


def test_bump_only_from_lead_25_and_never_below_base(e1604):
    base = np.full((3, 4, 72), 400.)
    peak, hour = np.full((3, 4), 700.), np.full((3, 4), 60.)
    out = e1604.bump(base, peak, hour)
    assert (out[..., :24] == 400.).all() and (out >= base).all()
    assert out[..., 59].max() == pytest.approx(400. + 0.15 * 300.)  # bump centre, height 0.15 * (peak - path min)


def test_features_has_29_columns_and_is_causal(e1604):
    index = pd.date_range('2024-01-01', periods=24 * 40, freq='h', tz='UTC')
    ace = pd.Series(400. + 50. * np.sin(np.arange(len(index)) / 20.), index=index)
    o = index[24 * 30:24 * 30 + 5]
    path, hx = np.full((5, 72), 420.), np.full((5, 72), 410.)
    x = e1604.features(o, path, hx, ace)
    assert x.shape == (5, 29)
    # changing ACE after the origin must not change features that use only the past (columns 3..11 and 12..28)
    later = ace.copy()
    later[later.index > o[0]] += 1000.
    assert np.allclose(e1604.features(o[:1], path[:1], hx[:1], later)[:, 3:12], x[:1, 3:12])


def test_no_absolute_user_paths_or_secrets():
    pattern = re.compile(r'/home/|t-lab01|hf_[A-Za-z0-9]{20,}|WANDB_API_KEY')
    for path in REPRO.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.md', '.sh', '.txt'} and path.name != 'test_reproduce.py':
            assert not pattern.search(path.read_text()), path


def test_manifest_is_well_formed():
    manifest = json.loads((REPRO / 'data_manifest.json').read_text())
    assert manifest['files']
    for entry in manifest['files']:
        assert re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) and entry['source'] in {'repo', 'hf', 'lb'}
        assert entry['group'] in {'core', 'reference'} and not entry['path'].startswith('/')
        if entry['source'] == 'repo':
            assert (REPRO.parent / entry['repo_path']).is_file()
