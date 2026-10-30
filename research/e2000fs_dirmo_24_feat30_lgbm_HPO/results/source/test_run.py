import importlib.util
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('hpo', Path(__file__).with_name('run.py'))
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)


def test_temporal_split_has_strict_72h_gap():
    origins = pd.date_range('2025-11-29', '2026-02-24', freq='h', tz='UTC')
    fit, val = hpo.temporal_split(origins)
    assert origins[fit].max() == pd.Timestamp('2025-11-30T23:00:00Z')
    assert origins[val].min() == pd.Timestamp('2025-12-04T00:00:00Z')
    assert origins[val].max() == pd.Timestamp('2026-02-22T23:00:00Z')
    assert not np.intersect1d(fit, val).size


def test_studies_are_isolated_and_resume(tmp_path):
    d1, d2 = tmp_path / 'a', tmp_path / 'b'
    d1.mkdir(); d2.mkdir()
    first = hpo.open_study(0, d1)
    first.optimize(lambda t: t.suggest_float('x', 0, 1), n_trials=1)
    reopened = hpo.open_study(0, d1)
    assert len(reopened.trials) == 1
    assert len(hpo.open_study(1, d2).trials) == 0
    pending = reopened.ask()
    reopened.tell(pending.number, state=optuna.trial.TrialState.FAIL)
    assert len(reopened.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 1


def test_selection_uses_validation_mse_and_stable_tiebreak():
    winners = [dict(value=3, study=1, trial=0), dict(value=2, study=3, trial=1), dict(value=2, study=2, trial=9)]
    assert hpo.select_best(winners) == winners[2]


def test_baseline_is_valid_search_point():
    manifest = hpo.read(hpo.BASE / 'logs/full_01/manifest.json')
    fixed = {k: manifest['params'].get(k, 0.) for k in hpo.SEARCH_KEYS}
    result = hpo.parameters(optuna.trial.FixedTrial(fixed))
    assert result['num_leaves'] <= 2**result['max_depth']
    assert hpo.full_params(result)['n_jobs'] == 4


def test_actual_frozen_feature_and_target_contract():
    manifest, names, origins, y, blocks = hpo.load_data()
    assert len(manifest['selected']) == 30
    assert all(x.shape == (len(origins), 237) for x in blocks)
    assert y.shape == (len(origins), 72)
    fit, val = hpo.temporal_split(origins)
    assert np.isfinite(y[fit]).all() and np.isfinite(y[val]).all()


def test_actual_smoke_resume_does_not_repeat_completed_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(hpo, 'HERE', tmp_path)
    hpo.run_study(0, smoke=True)
    directory = tmp_path / 'runs/smoke/study1'
    first = hpo.read(directory / 'best.json')
    history = directory / 'trials/0000/h72_loss.csv'
    modified = history.stat().st_mtime_ns
    hpo.run_study(0, smoke=True)
    study = hpo.open_study(0, directory)
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 1
    assert hpo.read(directory / 'best.json') == first
    assert history.stat().st_mtime_ns == modified
