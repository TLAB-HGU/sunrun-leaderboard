"""Causal windows and production-strip equivalence, independent of real data."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('feat30_tcn_data_tests', Path(__file__).with_name('data.py'))
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def test_hourly_inclusive_window_and_microsecond_origins():
    hours = pd.date_range('2025-01-01', periods=200, freq='h', tz='UTC')
    origins = hours[[119, 150]].as_unit('us')
    positions = data.window_positions(data.ns(hours), origins)
    np.testing.assert_array_equal(positions, [119, 150])
    for position, origin in zip(positions, origins):
        window = hours[position - 119:position + 1]
        assert len(window) == 120
        assert window[0] == origin - pd.Timedelta(hours=119)
        assert window[-1] == origin
    with pytest.raises(ValueError, match='complete 120-hour'):
        data.window_positions(data.ns(hours), hours[:1])
    with pytest.raises(ValueError, match='complete hourly'):
        data.window_positions(data.ns(hours.delete(40)), origins)


def test_future_poison_invariant_for_every_channel_with_past_positive_control():
    hours = pd.date_range('2025-01-01', periods=160, freq='h', tz='UTC')
    original = np.arange(160 * 30, dtype=np.float32).reshape(160, 30)
    original[30:35] = np.nan
    origin = hours[130]
    pos = data.window_positions(data.ns(hours), [origin])[0]
    expected = data.causal_prefill(original)[pos - 119:pos + 1]
    for col in range(30):
        poisoned = original.copy()
        poisoned[pos + 1:, col] = -1e8
        got = data.causal_prefill(poisoned)[pos - 119:pos + 1]
        np.testing.assert_array_equal(got, expected)
        poisoned[pos, col] = -1e8
        assert data.causal_prefill(poisoned)[pos, col] != expected[-1, col]


def test_prefill_never_backfills_leading_missing():
    x = np.array([[np.nan, 1.], [np.nan, np.nan], [3., 4.]])
    actual = data.causal_prefill(x)
    assert np.isnan(actual[:2, 0]).all()
    np.testing.assert_array_equal(actual[:, 1], [1., 1., 4.])
    with pytest.raises(ValueError, match='finite-or-NaN'):
        data.causal_prefill([[np.inf]])


def test_future_channels_keep_selected_order_and_all_horizons():
    values = np.arange(2 * 72 * 4).reshape(2, 72, 4)
    future, names = data.future_features({'E': values}, {'varying': ['a', 'b', 'c', 'd']}, ['d', 'static', 'a'])
    assert names == ['d', 'a']
    np.testing.assert_array_equal(future, values[:, :, [3, 0]])


def _strip_fixture(tmp_path):
    spec = importlib.util.spec_from_file_location('original_strip_for_test', data.ARCHIVE / 'experiments/d1_longitude_strips/strips.py')
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    hours = pd.date_range('2025-01-01', periods=5, freq='h', tz='UTC')
    paths = []
    for i in range(5):
        path = tmp_path / f'image{i}.nc'
        path.write_bytes(b'present')
        paths.append(str(path))
    meta = pd.DataFrame(dict(slot=hours, obs_end=hours - pd.Timedelta(minutes=2),
                             s3_modified=hours + pd.Timedelta(minutes=20), status='ok', nc_path=paths))
    block = pd.DataFrame(index=hours)
    for lon in range(12):
        for stat in st.STAT_SUFFIX.values():
            block[f'lon{lon:02}_lat1_{stat}'] = .8 if stat == 'valid_fraction' else .1 * lon + np.arange(5)
    return st, meta, block


def test_vectorized_selection_and_h1_match_original_production(tmp_path):
    st, meta, block = _strip_fixture(tmp_path)
    origins = pd.date_range('2025-01-01T00:00Z', periods=15, freq='h')
    actual_ids, actual_age = data.select_strip_slots(origins, meta, block, st.resolve_nc)
    reference = st.select_slots(origins, meta, block)
    np.testing.assert_array_equal(actual_ids, reference.slot_idx)
    np.testing.assert_allclose(actual_age, reference.img_age_h, equal_nan=True)
    selected = ['strip_v350_logmed', 'strip_v450_logmed', 'strip_v750_logmed', 'strip_v550_dark', 'strip_v650_dark', 'img_age_h']
    actual = data.strip_h1(origins, meta, block, selected, st)
    original, _ = st.strip_features(origins, meta, block)
    expected = original.loc[original.h.eq(1), selected].to_numpy()
    np.testing.assert_allclose(actual[selected], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_unavailable_or_future_image_cannot_change_existing_inputs(tmp_path):
    st, meta, block = _strip_fixture(tmp_path)
    origins = pd.DatetimeIndex([meta.slot.iloc[2] + pd.Timedelta(hours=1)])
    selected = ['strip_v450_logmed']
    expected = data.strip_h1(origins, meta, block, selected, st)
    poisoned = block.copy()
    poisoned.loc[poisoned.index > origins[0] - pd.Timedelta(hours=1)] = 1e8
    np.testing.assert_allclose(data.strip_h1(origins, meta, poisoned, selected, st), expected)
    # Even an older image is unavailable until its publication clock passes.
    late = meta.copy()
    late.loc[2, 's3_modified'] = origins[0] + pd.Timedelta(hours=1)
    ids, _ = data.select_strip_slots(origins, late, block, st.resolve_nc)
    assert ids[0] == 1
    poisoned.iloc[2] = 1e8
    assert not np.allclose(data.strip_h1(origins, meta, poisoned, selected, st), expected)
