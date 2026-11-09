import numpy as np
import pandas as pd
import pytest

from strip_features import GRID, NAMES, SPECS, build, prepare_history


def raw_frames():
    data = pd.DataFrame({'origin': pd.to_datetime(['2024-01-01T00:00Z', '2024-01-01T02:00Z']),
                         'img_age_h': [0.5, 0.25]})
    for k, name in enumerate(GRID):
        data[name] = [float(k), float(k + 100)]
    return data


def reference(grid, age, h, speed):
    center = .55 * (147858620.7 / speed / 3600 - (h + age))
    weights = np.array([max(0, min(center + 5, -90 + 15*(i+1))
                                 - max(center - 5, -90 + 15*i))/10 for i in range(12)])
    return (grid * weights).sum()/weights.sum() if weights.sum() > 0 else np.nan


def test_projection_staleness_order_and_no_mutation():
    raw = raw_frames()
    original = raw.copy(deep=True)
    query = pd.DataFrame({'origin': pd.to_datetime(['2024-01-01T01:00Z', '2024-01-01T00:00Z']),
                          'h': [48, 1], 'y': [999999, -999999]}, index=[9, 2])
    query_original = query.copy(deep=True)
    result = build(query, prepare_history(raw))
    for row, age in enumerate([1.5, .5]):
        for col, (speed, stat) in enumerate(SPECS):
            grid = np.arange(12, dtype=float) + (12 if stat == 'dark' else 0)
            assert result.iloc[row, col] == pytest.approx(reference(grid, age, query.h.iloc[row], speed))
    assert result.index.tolist() == [9, 2]
    assert result.columns.tolist() == list(NAMES)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(query, query_original)
    poisoned = raw.copy()
    poisoned.loc[1, list(GRID)] = -1e30
    pd.testing.assert_frame_equal(result, build(query, prepare_history(poisoned)))


def test_whole_partial_frame_missing_and_off_disk():
    raw = raw_frames()
    raw.loc[1, GRID[0]] = np.nan
    query = pd.DataFrame({'origin': pd.to_datetime(['2023-12-31T23:00Z', '2024-01-01T02:00Z',
                                                  '2024-02-01T00:00Z']), 'h': [1, 48, 1]})
    out = build(query, prepare_history(raw))
    assert out.iloc[0].isna().all()
    assert out.iloc[1, [0, 1, 3]].isna().all()  # Preserve original zero-weight NaN propagation.
    assert np.isfinite(out.iloc[1, 2])
    assert out.iloc[2].isna().all()
    raw['img_age_h'] = np.nan
    assert build(query, prepare_history(raw)).isna().all().all()


def test_duplicate_and_invalid_timestamp_rejection():
    raw = raw_frames()
    assert len(prepare_history(pd.concat([raw, raw]))) == 2
    conflicting = raw.iloc[[0]].copy()
    conflicting[GRID[1]] += 1
    with pytest.raises(ValueError, match='Conflicting'):
        prepare_history(pd.concat([raw, conflicting]))
    raw.loc[0, 'img_age_h'] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        prepare_history(raw)
