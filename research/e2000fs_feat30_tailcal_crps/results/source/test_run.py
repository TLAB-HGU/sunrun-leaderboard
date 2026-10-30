import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('distribution_runner_test', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_current_hpo_uses_completed_trials_in_unfinished_studies(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'HPO', tmp_path)
    for index in range(1, 4):
        folder = tmp_path / 'runs/main' / f'study{index}'
        folder.mkdir(parents=True)
        with sqlite3.connect(folder / 'optuna.sqlite3') as conn:
            conn.executescript('CREATE TABLE trials(trial_id INTEGER,number INTEGER,state TEXT,datetime_complete TEXT);'
                               'CREATE TABLE trial_values(trial_id INTEGER,objective INTEGER,value REAL);'
                               'CREATE TABLE trial_params(trial_id INTEGER,param_name TEXT,param_value REAL,distribution_json TEXT);')
            for tid, state, value in [(1, 'COMPLETE', 10. + index), (2, 'RUNNING', 0.), (3, 'FAIL', -1.)]:
                conn.execute('INSERT INTO trials VALUES(?,?,?,?)', (tid, tid, state, '2026-10-28'))
                conn.execute('INSERT INTO trial_values VALUES(?,?,?)', (tid, 0, value))
            conn.execute('INSERT INTO trial_params VALUES(?,?,?,?)', (1, 'n_estimators', 400., json.dumps({'name': 'IntDistribution'})))
        run.write(folder / 'trials/0001/result.json', {'value': 10. + index, 'params': {'n_estimators': 400}})
    winner = run.current_hpo()
    assert winner['best']['study'] == 1
    assert winner['best']['value'] == 11.
    assert isinstance(winner['best']['params']['n_estimators'], int)
    assert winner['counts']['1']['RUNNING'] == 1


def test_hpo_receipt_must_match_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'HPO', tmp_path)
    folder = tmp_path / 'runs/main/study1'
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / 'optuna.sqlite3') as conn:
        conn.executescript("CREATE TABLE trials(trial_id,number,state,datetime_complete);"
                           "CREATE TABLE trial_values(trial_id,objective,value);"
                           "CREATE TABLE trial_params(trial_id,param_name,param_value,distribution_json);"
                           "INSERT INTO trials VALUES(1,0,'COMPLETE','today');"
                           "INSERT INTO trial_values VALUES(1,0,1.0);")
    run.write(folder / 'trials/0000/result.json', {'params': {}, 'value': 2.})
    with pytest.raises(ValueError, match='receipt'):
        run.current_hpo()


def test_extreme_control_weights_only_its_24h_block():
    y = np.full((3, 72), 400.)
    y[0, 0] = 600.
    y[1, 30] = 600.
    w0 = run.block_extreme_weights(y, 0)
    w1 = run.block_extreme_weights(y, 1)
    assert w0[0] / w0[1] == 4
    assert w1[1] / w1[0] == 4
    np.testing.assert_allclose(w0.mean(), 1)
    np.testing.assert_allclose(run.block_extreme_weights(y, 2), 1)


def test_manifest_rejects_tampering_and_smoke_mismatch(tmp_path):
    source = tmp_path / 'source'
    source.write_text('original')
    trial = tmp_path / 'trial'
    trial.write_text('frozen trial')
    run.write(tmp_path / 'manifest.json', dict(smoke=False, inherited_hashes={},
              source_sha256={str(source): run.digest(source)},
              hpo={'best': {'result_path': str(trial), 'result_sha256': run.digest(trial)}}))
    run.validate_manifest(tmp_path, False)
    with pytest.raises(ValueError, match='Smoke'):
        run.validate_manifest(tmp_path, True)
    source.write_text('modified')
    with pytest.raises(ValueError, match='changed'):
        run.validate_manifest(tmp_path, False)


def test_pair_count_weighted_mse_decomposition():
    class Evaluation:
        @staticmethod
        def metrics(y, p, origins):
            high = y >= 550
            return {'mse': np.mean((p-y)**2), 'mse_ge550': np.mean((p[high]-y[high])**2)}
    y = np.array([[400., 500., 550., 700.]])
    p = np.array([[420., 490., 500., 620.]])
    score = run.point_metrics(Evaluation, y, p, None)
    assert score['n_pairs_lt550'] == 2
    assert score['mse'] == .5 * score['mse_lt550'] + .5 * score['mse_ge550']


def test_resume_rejects_other_horizon_even_with_same_manifest(tmp_path):
    origins = pd.date_range('2025-01-01', periods=3, freq='3h', tz='UTC')
    checkpoint = tmp_path / 'internal/l2/h01/checkpoint.json'
    run.write(checkpoint, {'manifest_sha256': 'same', 'horizon': 2})
    with pytest.raises(ValueError, match='context mismatch'):
        run.train_stage(tmp_path, 'controls', 'internal', np.array([0, 1]), np.array([2]),
                        origins, np.full((3, 72), 400.), [np.zeros((3, 2))] * 3,
                        [np.zeros((1, 2))] * 3, origins[2:], {}, 'same', True)


@pytest.mark.parametrize('name', ['../escape', 'space name', 'a/b', 'a;whoami', ''])
def test_invalid_run_ids(name):
    with pytest.raises(ValueError):
        run.run_directory(name)
