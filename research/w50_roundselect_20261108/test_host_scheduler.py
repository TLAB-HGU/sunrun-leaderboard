"""Completed-artifact regression tests; no training or scheduler processes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def scheduler():
    spec = importlib.util.spec_from_file_location('roundselect_scheduler_test', Path(__file__).with_name('host_scheduler.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_run(tmp_path):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    write('complete.json', {'status': 'complete', 'exit_code': 0, 'fresh_training': True,
                           'seed': 42, 'total_s': 90.})
    score = {scope: {'n_replaced': 0, **dict.fromkeys(('s', 'rmse', 'trmse', 'timse', 'mae'), 90.)}
             for scope in ('total', 'f1', 'f2', 'f3')}
    for name in ('submission.json', 'local_score.json'):
        write(name, {'competition': score})
    selected = [31, 47, 59]
    for period, origins_count in (('F1', 2089), ('F2', 2113), ('F3', 2113)):
        write(f'{period}_metadata.json', {
            'seed': 42, 'prior_rounds': 900, 'smoke': False, 'fresh_training': True,
            'reused_model_or_predictions': False, 'query_origins': origins_count,
            'selected_rounds': selected, 'feature_count': 28, 'expanded_feature_count': 212,
            'control': {'passed': True, 'max_absolute_difference': 0.}})
        selection = []
        for selected_count in selected:
            curve = [2.] * 900
            curve[selected_count - 1] = 1.
            selection.append({'native_rmse': curve, 'full_rounds': selected_count})
        write(f'{period}_models/tree_selection.json', selection)
        np.savez(tmp_path / f'{period}_predictions.npz',
                 pred=np.full((origins_count, 72), 400.), origins=np.arange(origins_count))
        for stage in ('prior', 'full'):
            for block in range(3):
                path = tmp_path / f'{period}_models' / stage / f'b{block}.ubj'
                path.parent.mkdir(parents=True, exist_ok=True)
                # This validator checks existence/size; model replay is a separate gate.
                path.write_bytes(b'synthetic artifact')
    write('provenance.json', {'fresh_training': True, 'reused_model_or_predictions': False})
    for name in ('predictions.parquet', 'feature_contract.json', 'train.log'):
        (tmp_path / name).write_text('synthetic artifact')
    return tmp_path


def test_completed_run_accepts_official_dimensions_not_selected_rounds(scheduler, completed_run):
    assert scheduler.validate_finished(completed_run, 42) == (True, None)


@pytest.mark.parametrize('period', ['F1', 'F2', 'F3'])
@pytest.mark.parametrize('shape_kind', ['selected_rounds', 'missing_origin', 'missing_horizon'])
def test_completed_run_rejects_incorrect_forecast_dimensions(scheduler, completed_run, period, shape_kind):
    count = 2089 if period == 'F1' else 2113
    pred_shape = (59, 72) if shape_kind == 'selected_rounds' else (count, 71 if shape_kind == 'missing_horizon' else 72)
    origin_count = 59 if shape_kind == 'selected_rounds' else count - (shape_kind == 'missing_origin')
    np.savez(completed_run / f'{period}_predictions.npz',
             pred=np.full(pred_shape, 400.), origins=np.arange(origin_count))
    assert scheduler.validate_finished(completed_run, 42) == (
        False, f'{period}: invalid forecast dimensions or values')
