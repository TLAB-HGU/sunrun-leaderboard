import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import launch, runner
from .inputs import atomic_json, reference
from .model import ModelConfig


def fixture_data(tmp_path):
    origins = pd.DatetimeIndex(['2026-02-01T00:00Z', '2026-02-02T00:00Z',
                               '2026-03-04T00:00Z', '2026-03-05T00:00Z'])
    def frame(oo):
        return pd.DataFrame({'origin': np.repeat(oo, 72), 'h': np.tile(np.arange(1, 73), len(oo)),
                             'v_lag0': np.repeat(np.arange(len(oo), dtype=float) + 400, 72)})
    tr = frame(origins)
    labels = tr.copy()
    labels['target_t'] = labels.origin + pd.to_timedelta(labels.h, unit='h')
    labels['y'] = labels.v_lag0 + labels.h * .1
    # Independent missing target cell: keep its feature row.
    labels = labels.drop(index=5)
    official = frame(pd.date_range('2026-05-31T23:00Z', '2026-09-26T23:00Z', freq='h'))
    refs = {}
    for name, df in [('train_features', tr), ('labels', labels), ('official_features', official)]:
        p = tmp_path / (name + '.parquet')
        df.to_parquet(p, index=False)
        refs[name] = reference(p)
    f = tmp_path / 'features.json'
    atomic_json(f, {'features': ['h', 'v_lag0']})
    refs['feature_names'] = reference(f)
    return {'artifacts': refs, 'source_receipts': [{'synthetic': True}]}


def test_full_synthetic_candidate_and_pooled_pipeline(tmp_path, monkeypatch):
    spec = fixture_data(tmp_path)
    # Synthetic fixture has no external source. Production always uses input_audit.
    monkeypatch.setattr(runner, 'input_audit', lambda s: {'passed': True})
    config = ModelConfig(n_estimators=2, min_child_weight=1, n_jobs=1)
    pooled = runner.run(spec, tmp_path / 'pooled', arm='pooled', config=config)
    candidate = runner.run(spec, tmp_path / 'candidate', arm='multioutput', config=config)
    assert pooled['observed_cells'] == candidate['observed_cells'] == 144
    result = pd.read_parquet(tmp_path / 'candidate/predictions_official_s42.parquet')
    assert len(result) == 203976
    assert np.isfinite(result.pred_kms).all()
    assert len(list((tmp_path / 'candidate/internal').glob('loss_history_s42_h*.csv'))) == 72
    assert len(list((tmp_path / 'candidate/official').glob('loss_history_s42_h*.csv'))) == 72
    receipt = json.loads((tmp_path / 'candidate/confirmation.json').read_text())
    assert receipt['checkpoint_replay_exact']
    assert receipt['validation_role'] == 'overlapping_diagnostic_not_selection'
    for stage in ('internal', 'official'):
        assert (tmp_path / 'candidate' / stage / 'loss_history.png').is_file()


def test_real_preflight_rejects_before_training(tmp_path):
    with pytest.raises(ValueError, match='preflight failed'):
        runner.run({'artifacts': {}}, tmp_path / 'not_created', arm='multioutput')
    assert not (tmp_path / 'not_created').exists()


def test_registration_cannot_bypass_failed_input_audit(tmp_path, monkeypatch):
    p = tmp_path / 'inputs.json'
    atomic_json(p, {'artifacts': {}})
    monkeypatch.setattr(launch, 'protected', lambda: None)
    with pytest.raises(ValueError, match='"passed": false'):
        launch.register(p, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def test_swapped_registration_ids_rejected_by_command(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    spec = {'artifacts': {}}
    atomic_json(bundle / 'inputs.json', spec)
    atomic_json(bundle / 'intent.json', {'code_hashes': launch.code_hashes(),
                'input_sha256': launch.digest(spec), 'seeds': [42], 'horizons': 72, 'xgboost_n_jobs': -1})
    atomic_json(bundle / 'registrations.json', {'pooled': 'E9000', 'multioutput': 'E9001'})
    core = SimpleNamespace(state=lambda: {'experiments': {
        'E9001': {'cmd': launch.execution_command(bundle, 'pooled')}}})
    monkeypatch.setattr(launch, 'protected', lambda: core)
    with pytest.raises(ValueError, match='command/arm mismatch'):
        launch.verify(bundle, 'multioutput')


def test_official_eval_rejects_unrelated_baseline_execution(tmp_path, monkeypatch):
    spec = {'artifacts': {}}
    ids = {'pooled': 'E9000', 'multioutput': 'E9001'}
    core = SimpleNamespace(RUNS=tmp_path)
    monkeypatch.setattr(launch, 'verify', lambda bundle, arm: (core, spec, ids, {}))
    atomic_json(tmp_path / 'E9000/execution.json', {'status': 'completed',
                'identity': {'input_sha256': 'unrelated', 'code_hashes': {}, 'arm': 'pooled'}})
    with pytest.raises(ValueError, match='execution identity mismatch'):
        launch.evaluate_once(tmp_path / 'bundle')
