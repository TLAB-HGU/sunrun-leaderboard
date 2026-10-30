import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.datav2.paths import DatasetPaths, pin_snapshot, sha256
from experiments.datav2.recipe import Features, source_guard
from experiments.datav2.sources import ace_features, enlil_runs


def hourly(frame):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.pop('timestamp_utc'), utc=True).dt.floor('h')
    return frame.loc[~frame.index.duplicated(keep='last')].sort_index()


def publish_ready(root, stage):
    files = {str(p.relative_to(root)): sha256(p) for p in DatasetPaths(root).parquet_files() if p.is_file()}
    files.update({str(Path('suvi-meta') / p.relative_to(stage)): sha256(p)
                  for p in stage.glob('year=*/part.parquet')})
    (root / 'CHAMPION_READY.json').write_text(json.dumps({'files': files}))


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'v2'
    stage = tmp_path / 'stage'
    times = pd.date_range('2026-01-01', periods=96, freq='h', tz='UTC')
    raw = pd.DataFrame({'timestamp_utc': times})
    for name in ('speed_kms', 'density_cm3', 'temperature_k', 'bt_nt', 'bz_gsm_nt', 'by_gsm_nt',
                 'epam_e_channel1', 'epam_e_channel2', 'epam_p_channel1', 'epam_p_channel3',
                 'epam_p_channel5', 'sis_gt10_flux'):
        raw['ace_' + name] = np.arange(96) + 10.
    files = {'ch/ace.parquet': pd.DataFrame({'timestamp_utc': times, 'filled_speed_kms': 400., 'was_missing': False}),
             'ch/ch-hourly.parquet': pd.DataFrame({'meridian_area': 1.}, index=times),
             'suvi-fusion/suvi-hourly-v2.parquet': pd.DataFrame({f'lon{i:02d}_lat1_{stat}': 1. for i in range(12) for stat in ('log_median', 'dark_fraction', 'valid_fraction')}, index=times),
             'ace-snapshot/year=2026/part.parquet': raw}
    for name, frame in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    meta = stage / 'year=2026/part.parquet'
    meta.parent.mkdir(parents=True)
    pd.DataFrame({'slot': times, 'obs_end': times, 's3_modified': times, 'status': 'missing', 'nc_path': ''}).to_parquet(meta)
    donki = root / 'donki/WSAEnlilSimulations_2026-01.json'
    donki.parent.mkdir()
    donki.write_text(json.dumps([{'modelCompletionTime': '2026-01-01T00:00Z',
                                  'estimatedShockArrivalTime': '2026-01-02T00:00Z', 'kp_90': 4}]))
    publish_ready(root, stage)
    return root, stage


def test_snapshot_pins_mutable_inputs_and_detects_tampering(inputs, tmp_path):
    root, stage = inputs
    pinned = pin_snapshot(tmp_path / 'snapshot', root, stage)
    original = pinned.meta_files()[0].read_bytes()
    (stage / 'year=2026/part.parquet').write_bytes(b'new live stage')
    assert pinned.meta_files()[0].read_bytes() == original
    assert pin_snapshot(pinned.root, root, stage).version == pinned.version
    pinned.ace.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        pinned.verify()


def test_missing_v2_does_not_fall_back(inputs, tmp_path):
    root, stage = inputs
    (root / 'ch/ace.parquet').unlink()
    with pytest.raises(FileNotFoundError):
        pin_snapshot(tmp_path / 'snapshot', root, stage)
    assert not (tmp_path / 'snapshot').exists()


def test_guard_rejects_predictions_and_filters_both_timestamp_forms(inputs):
    root, _ = inputs
    paths = DatasetPaths(root)
    cap = pd.Timestamp('2026-01-02', tz='UTC')
    original = pd.read_parquet
    with source_guard(cap, {}, [paths.ace, paths.ch]):
        assert pd.read_parquet(paths.ace).timestamp_utc.max() == cap
        assert pd.read_parquet(paths.ch).index.max() == cap
        with pytest.raises(ValueError, match='non-source'):
            pd.read_parquet(root / 'predictions_official_s0.parquet')
    assert pd.read_parquet is original
    with pytest.raises(ValueError, match='explicit pinned'):
        with source_guard(cap, {}):
            pass


def test_raw_poison_uses_v2_path_and_two_hour_delay(inputs, tmp_path):
    root, stage = inputs
    paths = pin_snapshot(tmp_path / 'snapshot', root, stage)
    data = Features.__new__(Features)
    data.paths, data.allowed_paths, data.access = paths, paths.parquet_files(), {}
    data.raw_paths = {p.resolve() for p in paths.raw_files()}
    data.L = SimpleNamespace(ia=SimpleNamespace(hourly=hourly))
    data.cap = pd.Timestamp('2026-01-04T23:00Z')
    origin = pd.Timestamp('2026-01-03T00:00Z')
    data.af = ace_features(pd.DatetimeIndex([origin - pd.Timedelta(hours=2)]), paths, hourly)
    data.af.index += pd.Timedelta(hours=2)
    assert data.raw_ace_poison(origin) == {'raw_ace_future_invariant': True, 'raw_ace_positive_control': True}


def test_donki_single_archive_deduplicates_and_keeps_utc(inputs):
    root, _ = inputs
    files = DatasetPaths(root).donki_files()
    runs = enlil_runs(files + files)
    assert len(runs) == 1
    assert str(runs.done.dt.tz) == 'UTC'
    assert runs.kp.iloc[0] == 4


def test_full_features_never_open_legacy_json_and_preserve_cutoff(inputs, tmp_path, monkeypatch):
    import builtins
    root, stage = inputs
    paths = pin_snapshot(tmp_path / 'snapshot', root, stage)
    original = builtins.open
    reads = []
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and 'WSAEnlilSimulations_' in str(file):
            reads.append(str(file))
            assert Path(file).resolve() in paths.donki_files()
        return original(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    data = Features(pd.Timestamp('2026-01-04T23:00Z'), paths)
    assert all(Path(p).is_relative_to(paths.root) for p in data.access)
    assert data.labels.index.max() < pd.Timestamp('2026-05-29', tz='UTC')
    assert data.ace.index.min() == pd.Timestamp('2026-01-01T02:00Z')
    assert len(data.runs) == 1
    assert data.poison(pd.Timestamp('2026-01-03T00:00Z'))['passed']


def test_missing_suvi_cache_slot_is_aligned_not_shifted(inputs, tmp_path):
    from experiments.datav2.sources import load_inputs
    root, stage = inputs
    paths = DatasetPaths(root)
    cache = pd.read_parquet(paths.suvi)
    missing = cache.index[1]
    cache.iloc[2] = 3.
    cache.drop(missing).to_parquet(paths.suvi)
    publish_ready(root, stage)
    paths = pin_snapshot(tmp_path / 'snapshot', root, stage)
    meta, block = load_inputs(paths)
    assert block.index.equals(pd.DatetimeIndex(meta.slot))
    assert block.iloc[1].isna().all()
    assert (block.iloc[2] == 3.).all()


def test_image_deletion_after_pin_cannot_change_features(inputs, tmp_path):
    root, stage = inputs
    image = tmp_path / 'externalimage.nc'
    image.write_bytes(b'present')
    path = stage / 'year=2026/part.parquet'
    meta = pd.read_parquet(path)
    meta['nc_path'], meta['status'] = str(image), 'ok'
    meta.to_parquet(path)
    publish_ready(root, stage)
    snapshot = pin_snapshot(tmp_path / 'snapshot', root, stage)
    cap = pd.Timestamp('2026-01-04T23:00Z')
    origins = pd.DatetimeIndex([pd.Timestamp('2026-01-03T00:00Z')])
    before = Features(cap, snapshot).frame(origins)
    assert (before.img_stale == 0).all()
    image.unlink()
    snapshot.verify()
    after = Features(cap, snapshot).frame(origins)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


@pytest.mark.parametrize('failure', ['missing', 'stale'])
def test_failed_generation_cannot_pin(inputs, tmp_path, failure):
    root, stage = inputs
    if failure == 'missing':
        (root / 'CHAMPION_READY.json').unlink()
    else:
        frame = pd.read_parquet(root / 'ch/ace.parquet')
        frame['filled_speed_kms'] += 1
        frame.to_parquet(root / 'ch/ace.parquet')
    with pytest.raises((FileNotFoundError, ValueError)):
        pin_snapshot(tmp_path / 'snapshot', root, stage)
    assert not (tmp_path / 'snapshot').exists()


def test_official_runner_rejects_inherited_smoke(monkeypatch):
    from experiments.datav2.runner import run
    monkeypatch.setenv('EXPOS_SMOKE', '1')
    with pytest.raises(RuntimeError, match='rejects EXPOS_SMOKE'):
        run(SimpleNamespace())
