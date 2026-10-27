from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import prepare


def test_source_root_prefers_complete_archive_over_partial_live(tmp_path, monkeypatch):
    lb, archive = tmp_path / 'leaderboard', tmp_path / 'archive/store'
    (lb / 'store').mkdir(parents=True)
    for relative in prepare.RELATIVE_INPUTS:
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith('snapshot'):
            path.mkdir()
        else:
            path.write_bytes(b'fixture')
    monkeypatch.setattr(prepare, 'LB', lb)
    monkeypatch.setattr(prepare, 'ARCHIVE', archive)
    assert prepare.source_root() == archive.resolve()
    with pytest.raises(FileNotFoundError):
        prepare.source_root(lb / 'store')


def test_overlap_checks_every_preserved_row_and_nan_mask():
    origins = pd.date_range('2026-02-01', periods=1, tz='UTC')
    full = pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.arange(1, 73),
                         'a': np.arange(72, dtype=float)})
    labels = full.iloc[[0, 5, 71]].copy()
    result = prepare.compare_overlap(labels, full, ['h', 'a'])
    assert result['passed'] and result['observed_rows'] == 3
    with pytest.raises(ValueError, match='lost preserved rows'):
        prepare.compare_overlap(labels, full.iloc[:-1], ['h', 'a'])
    full.loc[5, 'a'] = np.nan
    with pytest.raises(ValueError, match='reconstruction differs'):
        prepare.compare_overlap(labels, full, ['h', 'a'])


def test_relocated_builder_restores_globals_after_exception(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    fake = SimpleNamespace(
        HR=SimpleNamespace(LB=old, OUT=old), ia=SimpleNamespace(LB=old, OUT=old),
        A=SimpleNamespace(LB=old), ST=SimpleNamespace(SUVI_CACHE=old, SUNIMG=old, META_GLOB='old', resolve_nc=str),
        P2=SimpleNamespace(CH=old), L9=SimpleNamespace(CH=old, META_GLOB='old'),
        LE=SimpleNamespace(LIVE=old, RUNS='old-runs', _TABLE={'old': 1}, enlil_runs=lambda: 'new-runs'))
    monkeypatch.setattr(prepare.importlib, 'import_module', lambda _: fake)
    root = tmp_path / 'relocated/store'
    with pytest.raises(RuntimeError, match='probe'):
        with prepare.relocated_builder(root) as builder:
            assert builder.HR.LB == root.parent
            assert builder.LE.RUNS == 'new-runs' and builder.LE._TABLE == {}
            assert builder.ST.SUNIMG == prepare.ROOT / 'data-v2/suvi'
            raise RuntimeError('probe')
    assert fake.HR.LB == old and fake.ST.SUNIMG == old
    assert fake.ST.resolve_nc is str
    assert fake.LE.RUNS == 'old-runs' and fake.LE._TABLE == {'old': 1}


def test_source_hash_mismatch_refuses_rebuild(tmp_path):
    first = tmp_path / next(iter(prepare.EXPECTED))
    first.parent.mkdir(parents=True)
    first.write_bytes(b'wrong-generation')
    with pytest.raises(ValueError, match='frozen source mismatch'):
        prepare.verify_sources(tmp_path)


def test_image_paths_relocate_exact_original_prefix():
    expected = str(prepare.ROOT / 'data-v2/suvi/data/g18/example.nc')
    assert prepare.relocated_nc('data/g18/example.nc') == expected
    assert prepare.relocated_nc(str(prepare.ROOT / 'sun-img/data/g18/example.nc')) == expected
    assert prepare.relocated_nc('/unrelated/example.nc') == '/unrelated/example.nc'
    assert prepare.relocated_nc('') == ''
