import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('v2_evaluation_test', Path(__file__).with_name('evaluation.py'))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def statistic(**changes):
    result = dict(n_origins=10, mse=100., unique_tp=2, unique_fp=1, unique_fn=1,
                  unique_csi=.5, unique_peak_mae=10., unique_ramp6_mae=20., unique_timing_mae_h=2.)
    result.update(changes)
    return result


def test_overlapping_windows_deduplicate_absolute_peak():
    origins = pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC')
    series = np.full(73, 400.)
    series[29:32] = [560., 650., 580.]
    y = np.vstack([series[:72], series[1:]])
    p = y.copy()
    p[0, 30] = 610.
    p[1, 29] = 630.
    m = ev.metrics(y, p, origins)
    assert m['tp'] == 2
    assert m['unique_tp'] == 1
    assert m['unique_peak_mae'] == 30.
    assert m['unique_ramp6_mae'] == 30.
    assert m['unique_timing_mae_h'] == 0.
    assert m['unique_events']['matches'][0]['pred']['windows'] == 2


def test_matching_is_one_to_one_and_tolerance_limited():
    y = np.full((1, 72), 400.)
    p = y.copy()
    y[0, 30] = 650.
    p[0, [22, 38]] = 650.
    m = ev.metrics(y, p, pd.date_range('2026-01-01', periods=1, tz='UTC'))
    assert (m['unique_tp'], m['unique_fp'], m['unique_fn']) == (1, 1, 0)
    assert m['unique_timing_mae_h'] == 8.
    p[:] = 400.
    p[0, 44] = 650.
    m = ev.metrics(y, p, pd.date_range('2026-01-01', periods=1, tz='UTC'))
    assert m['unique_tp'] == 0
    assert m['unique_peak_mae'] is None
    assert ev.selection_rank(m, statistic()) is None


def test_selection_guard_and_baseline_retention():
    parent = statistic()
    assert ev.selection_rank(statistic(mse=103.), parent) is not None
    assert ev.selection_rank(statistic(mse=103.0001), parent) is None
    assert ev.choose_candidate({'parent': parent, 'worse': statistic(unique_peak_mae=11.)}) == 'parent'
    assert ev.choose_candidate({'parent': parent, 'equal': parent.copy()}) == 'parent'
    assert ev.choose_candidate({'parent': parent, 'better': statistic(unique_peak_mae=9.)}) == 'better'
    assert ev.choose_candidate({'parent': statistic(unique_tp=0), 'better': parent}) == 'parent'
    assert ev.selection_rank(statistic(unique_ramp6_mae=None), parent) is None


def test_aggregate_weights_and_empty_matching_fold():
    a = statistic(n_origins=10, unique_tp=1, mse=100., unique_peak_mae=10.)
    b = statistic(n_origins=30, unique_tp=3, mse=200., unique_peak_mae=30.)
    m = ev.aggregate_folds([a, b])
    assert m['mse'] == 175.
    assert m['unique_peak_mae'] == 25.
    assert m['unique_tp'] == 4
    assert m['unique_csi'] == .5
    empty = statistic(unique_tp=0, unique_peak_mae=None, unique_ramp6_mae=None, unique_timing_mae_h=None)
    assert ev.aggregate_folds([a, empty])['unique_peak_mae'] == 10.
    with pytest.raises(ValueError):
        ev.aggregate_folds([])
