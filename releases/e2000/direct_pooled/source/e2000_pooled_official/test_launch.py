import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.e2000_parallel6.test_runner import fixture_data
from experiments.e2000_njobs_all.model import ModelConfig
from experiments.e2000_njobs_all.inputs import atomic_json
from . import launch


def test_final_pooled_grid_losses_and_checkpoint_replay(tmp_path, monkeypatch):
    spec = fixture_data(tmp_path)
    out = tmp_path / 'result'
    config = ModelConfig(n_estimators=2, min_child_weight=1, n_jobs=1)
    result = launch.final_fit(spec, out, config)
    pred = pd.read_parquet(out / 'predictions_official_s42.parquet')
    assert len(pred) == 203976
    assert np.isfinite(pred.pred_kms).all()
    assert result['checkpoint_replay_exact'] and result['model_count'] == 1
    losses = pd.read_csv(out / 'official/loss_history_s42_pooled.csv')
    assert len(losses) == 2
    assert np.isfinite(losses[['train_mse', 'validation_mse']]).all().all()
    assert (out / 'official/loss_history.png').exists()
    # A committed receipt must load, never refit.
    monkeypatch.setattr('xgboost.XGBRegressor.fit', lambda *a, **k: pytest.fail('unexpected refit'))
    launch.final_fit(spec, out, config)


def test_official_rejects_failed_native_run(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'verify', lambda b: (None, {}, 'E9999', {'runs': [{'rc': 1}]}))
    with pytest.raises(ValueError, match='successful terminal'):
        launch.evaluate_once(tmp_path)


def test_official_once_reuses_record(tmp_path, monkeypatch):
    eid = 'E9999'
    score = {'summary': {'mse_seedmean': 1.0}}
    core = SimpleNamespace(RUNS=tmp_path, state=lambda: {'experiments': {eid: {'lbeval': [score]}}})
    monkeypatch.setattr(launch, 'verify', lambda b: (core, {}, eid, {'runs': [{'rc': 0, 'outputs': {}}]}))
    atomic_json(tmp_path / eid / 'execution.json', {'status': 'completed', 'identity': launch.identity({}), 'outputs': []})
    atomic_json(tmp_path / 'E1983/internal.json', {'mse': 5104.0})
    monkeypatch.setattr('expos.lbeval.lb_eval', lambda *a, **k: pytest.fail('duplicate official scoring'))
    assert launch.evaluate_once(tmp_path)['official'] == score
