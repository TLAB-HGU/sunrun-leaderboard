import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('axes4_minute_features', Path(__file__).with_name('minute_features.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
O = pd.Timestamp('2026-01-01T01:00Z')


def samples(offsets, speeds=None, density=None, status=None):
    n = len(offsets)
    return pd.DataFrame({'S': np.zeros(n) if status is None else status,
                         'spd': np.full(n, 400.) if speeds is None else speeds,
                         'dens': np.full(n, 5.) if density is None else density},
                        index=O + pd.to_timedelta(offsets, unit='min'))


def test_full_support_and_empty_window():
    frame = samples(np.arange(-59, -3))
    got = m.build([O, O + pd.Timedelta(days=1), O], frame, 1)
    np.testing.assert_allclose(got.iloc[0], [1., 0., 4.])
    np.testing.assert_allclose(got.iloc[1], [0., 56., np.nan], equal_nan=True)
    pd.testing.assert_series_equal(got.iloc[0], got.iloc[2])
    assert got.index.name == 'origin_utc'
    assert m.describe(1)['varying'] == []


def test_boundaries_status_and_internal_gap():
    # Both outside-window rows must have no effect; 49 missing interior slots.
    frame = samples([-60, -59, -9, -4, -3], [999., 400., 450., 500., 999.])
    np.testing.assert_allclose(m.build([O], frame, 1).iloc[0], [3 / 56, 49, 4])
    frame.loc[O - pd.Timedelta(minutes=4), 'S'] = 1
    frame.loc[O - pd.Timedelta(minutes=9), 'spd'] = -9999.9
    np.testing.assert_allclose(m.build([O], frame, 1).iloc[0], [1 / 56, 55, 59])


def test_correlation_and_population_change_covariance():
    x = np.arange(56, dtype=float) ** 2 + 400
    frame = samples(np.arange(-59, -3), x, 2 * x + 5)
    got = m.build([O], frame, 2).iloc[0]
    np.testing.assert_allclose(got, [1., 2 * np.var(np.diff(x), ddof=0)])
    constant = m.build([O], samples(np.arange(-59, -3)), 2).iloc[0]
    assert np.isnan(constant.iloc[0]) and constant.iloc[1] == 0


def test_minimum_support_and_invalid_density():
    offsets = np.arange(-59, -30)  # 29 points: enough differences, insufficient correlation.
    x = np.arange(29, dtype=float) ** 2 + 400
    got = m.build([O], samples(offsets, x, x / 10), 2).iloc[0]
    assert np.isnan(got.iloc[0]) and np.isfinite(got.iloc[1])
    frame = samples(np.arange(-59, -3), np.arange(56.) + 400, np.arange(56.) + 1)
    frame.iloc[:37, frame.columns.get_loc('dens')] = -9999.9
    assert m.build([O], frame, 2).isna().all().all()
    # Axis 1 does not reject speed because the density is missing.
    assert m.build([O], frame, 1).iloc[0, 0] == 1


@pytest.mark.parametrize('count,corr_ok,cov_ok', [(20, False, False), (21, False, True),
                                               (29, False, True), (30, True, True)])
def test_exact_minimum_pair_counts(count, corr_ok, cov_ok):
    offsets = np.arange(-59, -59 + count)
    x = np.arange(count, dtype=float) ** 2 + 400
    got = m.build([O], samples(offsets, x, x / 10), 2).iloc[0]
    assert bool(np.isfinite(got.iloc[0])) is corr_ok
    assert bool(np.isfinite(got.iloc[1])) is cov_ok


def test_covariance_never_bridges_missing_slots():
    # 28 nonadjacent valid pairs: no one-minute differences exist.
    offsets = np.arange(-59, -3, 2)
    x = np.arange(len(offsets), dtype=float) ** 2 + 400
    assert m.build([O], samples(offsets, x, 2 * x), 2).isna().all().all()
    frame = samples(np.arange(-59, -3), np.arange(56.) ** 2 + 400,
                    np.arange(56.) ** 3 + 10)
    frame.iloc[20:25, 0] = 9
    valid = frame.S.to_numpy() == 0
    adjacent = valid[:-1] & valid[1:]
    dx, dy = np.diff(frame.spd)[adjacent], np.diff(frame.dens)[adjacent]
    expected = np.mean((dx - dx.mean()) * (dy - dy.mean()))
    assert m.build([O], frame, 2).iloc[0, 1] == pytest.approx(expected)


@pytest.mark.parametrize('axis', [1, 2])
def test_future_append_invariance_and_duplicate_conflicts(axis):
    x = np.arange(56.) ** 2 + 400
    frame = samples(np.arange(-59, -3), x, x / 10)
    future = samples([-3, 0, 1, 200], [1e9] * 4, [1e9] * 4)
    pd.testing.assert_frame_equal(m.build([O], frame, axis),
                                  m.build([O], pd.concat([frame, future]), axis))
    pd.testing.assert_frame_equal(m.build([O], frame, axis),
                                  m.build([O], pd.concat([frame, frame.iloc[:1]]), axis))
    conflict = frame.iloc[:1].copy()
    conflict.spd += 1
    with pytest.raises(ValueError, match='Conflicting ACE'):
        m.build([O], pd.concat([frame, conflict]), axis)


def test_invalid_time_alignment_rejected():
    frame = samples([-4])
    with pytest.raises(ValueError, match='minute-aligned'):
        m.build([O + pd.Timedelta(seconds=1)], frame, 1)
    frame.index += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='minute-aligned'):
        m.build([O], frame, 1)


def test_loader_reads_density_retains_markers_and_does_not_fill(tmp_path):
    (tmp_path / '20260101.txt').write_text(
        '# header\n2026 01 01 0001 61041 60 0 5.5 450.0 8.35e4\n'
        '2026 01 01 0002 61041 120 9 -9999.9 -9999.9 -1e5\n')
    frame = m.load_minutes(tmp_path, '2026-01-01', '2026-01-02')
    assert len(frame) == 2 and list(frame.columns) == ['S', 'spd', 'dens']
    assert frame.iloc[0].dens == 5.5 and frame.iloc[1].spd == -9999.9
    assert m.load_minutes(tmp_path, '2026-01-03', '2026-01-04').empty


def test_loader_rejects_malformed_source(tmp_path):
    (tmp_path / '20260101.txt').write_text('malformed row\n')
    with pytest.raises(ValueError, match='Malformed ACE'):
        m.load_minutes(tmp_path, '2026-01-01', '2026-01-01')
