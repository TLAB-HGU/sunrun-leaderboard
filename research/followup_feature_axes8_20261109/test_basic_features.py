"""Formula, morphology parity, and causal-window contract checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basic_features import SHAPE_FEATURES, build, describe


def test_arithmetic_and_input_preservation():
    frame = pd.DataFrame({'v_lag0': [460., 500.], 'v_lag24': [430., np.nan],
                          'v_lag48': [410., 430.], 'v_lag72': [400., 410.],
                          'rec26': [1., 1.], 'rec27': [2., np.nan],
                          'rec28': [3., 3.], 'rec_mean': [420., 450.],
                          'y': [900., -900.]}, index=[7, 7])
    original = frame.copy(deep=True)
    first = build(1, frame)
    np.testing.assert_allclose(first.iloc[0], [20., 10., 10.])
    assert np.isnan(first.iloc[1, 0])
    assert first.iloc[1, 1] == 20.
    np.testing.assert_allclose(build(2, frame).iloc[0], [np.sqrt(2 / 3)])
    assert np.isnan(build(2, frame).iloc[1, 0])
    np.testing.assert_allclose(build(3, frame).iloc[0], [-40., -10.])
    pd.testing.assert_frame_equal(frame, original)
    assert first.index.equals(frame.index)
    pd.testing.assert_frame_equal(first, build(1, frame.assign(y=0.)))
    for axis in (1, 2, 3):
        assert tuple(build(axis, frame).columns) == describe(axis)['constant'] + describe(axis)['varying']


@pytest.fixture
def history():
    t = np.arange(200)
    return pd.DataFrame({'origin': pd.date_range('2024-01-01', periods=200, freq='h', tz='UTC'),
                         'v_lag0': 400 + t * 1.2 + 30 * np.sin(t / 15.), 'h': 1})


def test_morphology_matches_existing_math(history):
    source = Path(__file__).parents[1] / 'post_ablation_five_axes_20261108/feature_models.py'
    spec = importlib.util.spec_from_file_location('previous_feature_models', source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    frame = history.iloc[[150, 71, 150]][['origin']].copy()
    actual = build(4, frame, history)
    for i, hour in enumerate((150, 71, 150)):
        bins = history.v_lag0.iloc[hour - 71:hour + 1].to_numpy().reshape(12, 6).mean(1)
        np.testing.assert_allclose(actual.iloc[i], previous._shape_features(bins), rtol=0, atol=0)
    assert tuple(actual.columns) == SHAPE_FEATURES
    assert actual.index.equals(frame.index)


def test_future_targets_and_other_horizons_cannot_affect_morphology(history):
    frame = history.iloc[[100, 80]][['origin']].assign(y=1.)
    original = history.copy(deep=True)
    reference = build(4, frame, history)
    poisoned = history.copy()
    poisoned.loc[poisoned.index > 100, 'v_lag0'] = -1e9
    poisoned = pd.concat([poisoned, history.assign(h=2, v_lag0=-1e8)])
    pd.testing.assert_frame_equal(reference, build(4, frame.assign(y=-1e20), poisoned))
    pd.testing.assert_frame_equal(history, original)
    # Also prove earlier outputs ignore rows between earlier and latest query.
    poisoned = history.copy()
    poisoned.loc[poisoned.index > 80, 'v_lag0'] = 1e9
    np.testing.assert_array_equal(reference.iloc[1], build(4, frame, poisoned).iloc[1])


def test_missing_hours_low_variance_and_duplicates(history):
    frame = history.iloc[[100]][['origin']]
    assert build(4, frame, history.assign(v_lag0=400.)).isna().all().all()
    assert build(4, frame, history.drop(index=range(30, 90))).isna().all().all()
    agree = pd.concat([history, history.iloc[[80]]])
    pd.testing.assert_frame_equal(build(4, frame, history), build(4, frame, agree))
    conflict = pd.concat([history, history.iloc[[80]].assign(v_lag0=1.)])
    with pytest.raises(ValueError, match='Duplicate origins'):
        build(4, frame, conflict)
    # Reindexing an absent hour is exactly equivalent to an explicit missing value.
    missing = history.copy()
    missing.loc[50, 'v_lag0'] = np.nan
    pd.testing.assert_frame_equal(build(4, frame, missing), build(4, frame, history.drop(index=50)))


def test_bad_inputs(history):
    with pytest.raises(ValueError, match='requires separate'):
        build(4, history)
    with pytest.raises(ValueError, match='Missing causal'):
        build(1, history)
    with pytest.raises(ValueError, match='Unsupported'):
        describe(5)
    with pytest.raises(ValueError, match='exact UTC hours'):
        build(4, history.assign(origin=history.origin + pd.Timedelta(minutes=1)), history)
    with pytest.raises(ValueError, match='Infinite'):
        build(4, history, history.assign(v_lag0=np.inf))
