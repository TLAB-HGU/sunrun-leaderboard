import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transfer_run_test', HERE / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
m = r.m


def test_center_causal_and_units():
    names = dict(constant=['v_lag0', 'density', 'ace_speed_kms_last'], varying=['rec27', 'enlil_kp'])
    mask = m.speed_mask(names, ['v_lag0', 'density', 'ace_speed_kms_last', 'rec27', 'enlil_kp'])
    block = np.full((2, len(mask)), 500., dtype=float)
    past = np.array([[400.] * 24, [600.] * 24])
    transformed, anchor = m.center([block], past, mask)
    np.testing.assert_array_equal(transformed[0][:, ~mask], block[:, ~mask])
    np.testing.assert_array_equal(transformed[0][:, mask] + anchor[:, None], block[:, mask])
    np.testing.assert_array_equal(block, 500.)
    assert mask.sum() == 26


def test_training_only_policy_and_maturity():
    oo = pd.date_range('2022-01-01', periods=800, freq='D', tz='UTC')
    indices = m.mature_indices(oo, oo[500])
    assert indices[-1] == 496  # exactly72h is excluded
    recent, _ = m.training_policy(oo, indices, 'recent_window')
    assert recent[0] == 131 and recent[-1] == 496
    ix, weights = m.training_policy(oo, indices, 'recency')
    np.testing.assert_array_equal(ix, indices)
    assert weights[-1] > weights[0] and weights.mean() == pytest.approx(1.)
    _, again = m.training_policy(oo[:600], indices, 'recency')
    np.testing.assert_array_equal(weights, again)


def test_fail_closed_gates():
    base = dict(mse=100., mse_lt550=100., mse_ge550=100.)
    good = dict(mse=98., mse_lt550=99., mse_ge550=94.)
    baseline = {s: base for s in ['dev0', 'dev1', 'dev2', 'pooled']}
    candidate = {s: good.copy() for s in baseline}
    assert m.development_gate(candidate, baseline)
    candidate['dev1']['mse_lt550'] = 100.01
    assert not m.development_gate(candidate, baseline)
    assert not m.gate(dict(good, mse_ge550=None), base, True)
    assert not m.gate(dict(good, mse=float('nan')), base, True)
    assert not m.gate(dict(good, mse_ge550=96.), base, True)


def test_source_and_checkpoint_tampering(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('original')
    hashes = {str(source): r.d.digest(source)}
    r.verify_hashes(hashes)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Frozen file'):
        r.verify_hashes(hashes)
    r.d.write(tmp_path / 'receipt.json', dict(context={'lane': 'baseline'}, sha256={}))
    with pytest.raises(ValueError, match='identity'):
        r.checkpoint(tmp_path, {'lane': 'center'})
    r.d.write(tmp_path / 'receipt.json', dict(context={}, sha256={'source.py': hashes[str(source)]}))
    with pytest.raises(ValueError, match='Frozen file'):
        r.checkpoint(tmp_path, {})
