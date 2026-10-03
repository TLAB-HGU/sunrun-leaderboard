import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_frozen_fusion import images


def inventory():
    times = pd.date_range('2024-01-01', periods=60, freq='h', tz='UTC')
    return pd.DataFrame({'slot': times, 'obs_end': times-pd.Timedelta(minutes=1),
                         'path': [f'image{i}' for i in range(60)]})


def test_future_image_perturbation_and_deletion_invariance():
    frame = inventory()
    origin = frame.slot.iloc[48]
    expected = images.select_clip(frame, origin)
    frame.loc[frame.slot > origin, 'path'] = 'future-perturbation'
    pd.testing.assert_frame_equal(expected, images.select_clip(frame, origin))
    pd.testing.assert_frame_equal(expected, images.select_clip(frame[frame.slot <= origin], origin))
    assert (expected.obs_end <= expected.requested_time).all()
    assert (expected.requested_time <= origin).all()
    assert expected.requested_time.iloc[0] == origin-pd.Timedelta(hours=45)


def test_bounded_past_fill_no_future_and_single():
    frame = inventory().iloc[[0, 10]].copy()
    origin = frame.slot.iloc[0]+pd.Timedelta(hours=6)
    result = images.select_clip(frame, origin, 'single')
    assert result.missing.all()  # actual observation age is 6h1m
    origin -= pd.Timedelta(hours=1)
    result = images.select_clip(frame, origin, 'single')
    assert result.path.nunique() == 1
    assert result.filled.all()
    assert np.allclose(result.age_hours, 5+1/60)


def test_geometry_north_center_radius_and_quality():
    rad = np.zeros((101, 101))
    rad[70, 50] = 100  # positive solar Y must appear north/up
    dqf = np.zeros_like(rad)
    header = {'CRPIX1': 51, 'CRPIX2': 51, 'RSUN': 40, 'CROTA': 0}
    mapped, valid, geometry = images.canonical_radiance(rad, dqf, header, size=101)
    yy, xx = np.unravel_index(mapped.argmax(), mapped.shape)
    assert yy < 50 and xx == 50
    assert geometry['center'] == [50, 50]
    assert geometry['radius'] == pytest.approx(40.4)
    assert not valid[0, 0]
    dqf[70, 50] = 1
    mapped, _, _ = images.canonical_radiance(rad, dqf, header, size=101)
    assert mapped.max() == 0
    header['CROTA'] = 90
    rad[:] = 0
    rad[50, 70] = 100  # +X rotates into north
    mapped, _, _ = images.canonical_radiance(rad, np.zeros_like(rad), header, size=101)
    assert np.unravel_index(mapped.argmax(), mapped.shape)[0] < 50


def test_train_only_transform_and_finite_clip(monkeypatch):
    frame = inventory()
    cutoff = frame.slot.iloc[20]
    seen = []
    def reader(path):
        seen.append(path)
        value = int(path.removeprefix('image'))+1
        return np.full((10, 10), value), np.zeros((10, 10)), {
            'CRPIX1': 5.5, 'CRPIX2': 5.5, 'RSUN': 4}
    monkeypatch.setattr(images, 'read_suvi_frame', reader)
    transform = images.fit_intensity(frame, cutoff, max_frames=4)
    assert all(int(path.removeprefix('image')) <= 20 for path in seen)
    changed = frame.copy()
    changed.loc[changed.slot > cutoff, 'path'] = 'invalid-future'
    assert transform == images.fit_intensity(changed, cutoff, max_frames=4)
    selected = images.select_clip(frame.iloc[:21], cutoff)
    tensor, stats = images.build_clip(selected, transform, size=32)
    assert tensor.shape == (3, 16, 32, 32)
    assert np.isfinite(tensor).all()
    assert np.array_equal(tensor[0], tensor[1])
    assert stats['missing_fraction'] > 0
    assert np.count_nonzero(tensor[:, selected.missing.to_numpy()]) == 0


def test_reject_future_frame(monkeypatch):
    selection = images.select_clip(inventory(), '2024-01-03', 'single')
    selection.loc[0, 'obs_end'] = pd.Timestamp('2025-01-01', tz='UTC')
    with pytest.raises(ValueError, match='future'):
        images.build_clip(selection, images.IntensityTransform(0, 1, '2024-01-01'), 32)


def test_prepared_inventory_matches_dataframe_and_rejects_bad_end():
    frame = inventory()
    origin = frame.slot.iloc[48]
    pd.testing.assert_frame_equal(images.select_clip(frame, origin),
                                  images.select_clip(images.prepare_inventory(frame), origin))
    frame.loc[0, 'obs_end'] = frame.slot.iloc[0]+pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='observation end'):
        images.prepare_inventory(frame)


def test_inventory_preserves_raw_files_but_excludes_invalid_geometry(tmp_path):
    metadata = tmp_path / 'metadata' / 'year=2024'
    metadata.mkdir(parents=True)
    raw = tmp_path / 'raw' / 'g18' / '2024' / '001'
    raw.mkdir(parents=True)
    slots = pd.date_range('2024-01-01T01:00Z', periods=7, freq='h')
    rows = []
    for i, slot in enumerate(slots):
        name = f'frame{i}'
        (raw / f'{name}.nc').write_bytes(b'raw file exists')
        rows.append({'slot': slot, 'obs_end': slot-pd.Timedelta(minutes=1),
                     'sat': 'G18', 'channel': 'Fe195', 'status': 'ok',
                     'filename': f'{name}.fits', 'rsun': 400., 'crpix1': 640.,
                     'crpix2': 640., 'crota': 0.})
    rows[1]['rsun'] = -999.
    rows[2]['rsun'] = float('nan')
    rows[3]['crpix1'] = float('inf')
    rows[4]['crpix2'] = float('nan')
    rows[5]['crota'] = float('inf')
    rows[6]['rsun'] = 0.
    pd.DataFrame(rows).to_parquet(metadata / 'part.parquet')
    result = images.inventory_sources(metadata.parent, tmp_path / 'raw')
    assert result.raw_available.all()
    assert result.raw_path.notna().all()
    assert result.invalid_geometry.tolist() == [False] + [True]*6
    assert result.available.tolist() == [True] + [False]*6
    assert result.path.iloc[1:].isna().all()
    filled = images.select_clip(images.prepare_inventory(result), slots[1], 'single')
    assert filled.filled.all() and not filled.missing.any()
    assert filled.path.eq(result.path.iloc[0]).all()
    missing = images.select_clip(images.prepare_inventory(result), slots[-1], 'single')
    assert missing.missing.all()
