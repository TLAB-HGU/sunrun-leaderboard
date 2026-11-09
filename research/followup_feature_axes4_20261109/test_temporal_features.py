import numpy as np
import pandas as pd
import pytest

import temporal_features as features


def source_frame(n=1100):
    rng = np.random.default_rng(817)
    return pd.DataFrame({
        'origin': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        'v_lag0': 450 + rng.normal(0, 40, n),
        features.CH_SOURCE: np.arange(n, dtype=float) ** 2,
    })


@pytest.mark.parametrize('axis', (3, 4))
def test_future_append_and_targets_cannot_change_past(axis):
    frame = source_frame()
    past = frame.iloc[:1000].copy()
    expected = features.build(past, axis)
    frame['y'] = np.arange(len(frame)) * 1e9
    actual = features.build(frame, axis).iloc[:1000]
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(past, source_frame().iloc[:1000])


def test_recurrence_recovers_known_past_only_offset():
    frame = source_frame()
    origin = 1000
    # A copy of the older 27-day +3h window creates one exact phase match.
    frame.loc[origin - 71:origin, 'v_lag0'] = frame.loc[
        origin - 71 - 651:origin - 651, 'v_lag0'].to_numpy()
    result = features.build(frame, 3).iloc[origin]
    assert result.rec27_past72_corr == pytest.approx(1.)
    assert result.rec27_past72_offset_h == 3.


def test_recurrence_nominal_tie_and_constant_windows():
    frame = source_frame()
    frame['v_lag0'] = np.arange(len(frame), dtype=float)
    result = features.build(frame, 3).iloc[-1]
    for day in (26, 27, 28):
        assert result[f'rec{day}_past72_corr'] == pytest.approx(1.)
        assert result[f'rec{day}_past72_offset_h'] == 0.
    frame['v_lag0'] = 450.
    assert features.build(frame, 3).isna().all().all()


def test_recurrence_requires_54_paired_samples():
    frame = source_frame()
    end = len(frame) - 1
    frame.loc[end - 71:end - 53, 'v_lag0'] = np.nan
    assert features.build(frame, 3).iloc[-1].isna().all()
    frame.loc[end - 53, 'v_lag0'] = 450.
    assert features.build(frame, 3).iloc[-1].notna().all()


def test_ch_differences_use_exact_hours_and_keep_missingness():
    frame = source_frame(80)
    result = features.build(frame, 4)
    assert result.iloc[60, 0] == 60 ** 2 - 36 ** 2
    assert result.iloc[60, 1] == 60 ** 2 - 12 ** 2
    assert result.iloc[:24, 0].isna().all()
    assert result.iloc[:48, 1].isna().all()
    frame = frame.drop(index=36)
    result = features.build(frame, 4)
    assert np.isnan(result.loc[frame.loc[60, 'origin']].iloc[0])
    assert result.loc[frame.loc[60, 'origin']].iloc[1] == 60 ** 2 - 12 ** 2


def test_output_preserves_unsorted_origin_order_and_duplicate_values():
    frame = source_frame(80).iloc[::-1]
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    result = features.build(frame, 4)
    assert result.index.equals(pd.DatetimeIndex(frame.origin, name='origin_utc'))
    assert result.iloc[0].equals(result.iloc[-1])
    frame.loc[len(frame) - 1, features.CH_SOURCE] += 1
    with pytest.raises(ValueError, match='Duplicate origins'):
        features.build(frame, 4)


@pytest.mark.parametrize('axis', (3, 4))
def test_invalid_time_and_infinity_rejected(axis):
    frame = source_frame(2)
    frame.loc[0, 'origin'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='exact UTC hours'):
        features.build(frame, axis)
    frame = source_frame(2)
    frame.loc[0, features.describe(axis)['sources'][1]] = np.inf
    with pytest.raises(ValueError, match='Infinite'):
        features.build(frame, axis)


@pytest.mark.parametrize('axis', (3, 4))
def test_empty_history(axis):
    result = features.build(source_frame(0), axis)
    assert result.empty
    assert tuple(result.columns) == features.describe(axis)['constant']
    assert result.index.name == 'origin_utc'
