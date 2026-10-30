"""Synthetic-only teacher tests on a fake old-run tree; never touches real data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_hss_phase_ff1 import coverage as ff1coverage
from experiments.mse20_hss_residual import teacher as t


@pytest.fixture
def fake_old(monkeypatch, tmp_path):
    root = tmp_path / 'hss-phase-ff1'
    (root / 'backbones-ff1').mkdir(parents=True)
    monkeypatch.setattr(t, 'OLD_FF1_RUN', root)
    monkeypatch.setattr(t, '_FIXED_CONFIG_OVERRIDE', {'id': 'synth-backbone'})
    monkeypatch.setattr(t, '_SOURCES_OVERRIDE', {'/synth/mod.py': '0' * 64})
    (root / 'registered.json').write_text(json.dumps({'sources': {}}))
    return root


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path):
    return dict(path=str(Path(path).resolve()), sha256=_sha(path))


def _write_model(root, fit, label_max=None):
    model_dir = root / f'model-{fit.strftime("%Y%m%d%H%M")}'
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(config={'id': 'synth-backbone'}, cutoff=fit.isoformat(),
                train_label_max=(label_max or fit).isoformat())
    (model_dir / 'metadata.json').write_text(json.dumps(meta, sort_keys=True))
    (model_dir / 'metadata.sha256').write_text(_sha(model_dir / 'metadata.json'))
    return model_dir


def add_claim(root, head, fit, role, origins, scope='dev', status='ok', config=None, provenance_max=None):
    head = pd.Timestamp(head)
    directory = root / 'backbones-ff1' / ('cutoff-' + head.strftime('%Y%m%dT%H%M%SZ')) / role
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = _write_model(root, pd.Timestamp(fit))
    frame = pd.DataFrame(dict(
        origin_last_input_utc=np.repeat([v.isoformat() for v in origins], 72),
        horizon_hours=np.tile(np.arange(1, 73), len(origins)),
        pred_kms=np.full(72 * len(origins), 400.)))
    frame.to_parquet(directory / 'forecast.parquet', index=False)
    provenance = dict(train_label_max=pd.Timestamp(provenance_max or fit).isoformat())
    (directory / 'training-provenance.json').write_text(json.dumps(provenance, sort_keys=True))
    identity = dict(role=role, scope=scope, head_cutoff=head.isoformat(),
                    cutoff=pd.Timestamp(fit).isoformat(),
                    origins=[pd.Timestamp(v).isoformat() for v in origins],
                    config={'id': 'synth-backbone'} if config is None else config,
                    sources={'/synth/mod.py': '0' * 64},
                    coverage_policy=dict(ff1coverage.COVERAGE_POLICY),
                    original_config_registration=_ref(root / 'registered.json'))
    (directory / 'claim.json').write_text(json.dumps(
        dict(identity=identity, registration=_ref(root / 'registered.json')), sort_keys=True))
    result = dict(status=status, role=role, cutoff=pd.Timestamp(fit).isoformat(),
                  head_cutoff=head.isoformat(), model_dir=str(model_dir.resolve()),
                  model_files=[], coverage_policy=dict(ff1coverage.COVERAGE_POLICY),
                  claim=_ref(directory / 'claim.json'),
                  forecast=_ref(directory / 'forecast.parquet'),
                  training_provenance=_ref(directory / 'training-provenance.json'))
    (directory / 'result.json').write_text(json.dumps(result, sort_keys=True))
    return directory


def grid(start, n):
    return pd.date_range(pd.Timestamp(start), periods=n, freq='6h', tz='UTC')


def test_pool_selects_nearest_preceding_fit(fake_old):
    fit0, fit1 = '2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z'
    head = '2024-04-01T00:00:00Z'
    origins0 = grid('2024-01-02T00:00:00Z', 10)
    origins1 = grid('2024-02-02T00:00:00Z', 10)
    add_claim(fake_old, head, fit0, 'oof0', origins0)
    add_claim(fake_old, head, fit1, 'oof1', origins1)
    pool, assignment = t.calibration_pool(head, 'dev')
    assert list(pool) == sorted(set(origins0) | set(origins1))
    by_origin = {pd.Timestamp(o).isoformat(): r for o, r in zip(pool, assignment)}
    assert by_origin[origins0[0].isoformat()]['fit_cutoff'] == pd.Timestamp(fit0).isoformat()
    assert by_origin[origins1[0].isoformat()]['fit_cutoff'] == pd.Timestamp(fit1).isoformat()


def test_dedup_prefers_nearest_fit_and_sorted_path(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 4)
    add_claim(fake_old, head, '2024-01-01T00:00:00Z', 'oof0', origins)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof1', origins)
    _, assignment = t.calibration_pool(head, 'dev')
    assert {r['fit_cutoff'] for r in assignment} == {pd.Timestamp('2024-02-01T00:00:00Z').isoformat()}


def test_causality_history_grid_and_future_head_filters(fake_old):
    head = '2024-04-01T00:00:00Z'
    fit = '2024-01-01T00:00:00Z'
    good = grid('2024-01-02T00:00:00Z', 4)
    off_grid = pd.DatetimeIndex([good[0] + pd.Timedelta(hours=3)])
    at_fit = pd.DatetimeIndex([pd.Timestamp(fit)])
    too_late = pd.DatetimeIndex([pd.Timestamp(head) - pd.Timedelta(hours=95)])
    ancient = pd.DatetimeIndex([pd.Timestamp(head) - pd.Timedelta(days=731)])
    add_claim(fake_old, head, fit, 'oof0', good.append(off_grid).append(at_fit).append(too_late).append(ancient))
    pool, _ = t.calibration_pool(head, 'dev')
    assert list(pool) == list(good)
    # Future heads never contribute.
    add_claim(fake_old, '2024-06-01T00:00:00Z', '2024-05-01T00:00:00Z', 'oof0', grid('2024-05-02T00:00:00Z', 3))
    pool2, _ = t.calibration_pool(head, 'dev')
    assert list(pool2) == list(good)


def test_scope_excludes_deployment_and_final(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 3)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins)
    add_claim(fake_old, head, head, 'deployment', grid('2024-04-02T00:00:00Z', 3))
    add_claim(fake_old, '2026-05-31T23:00:00Z', '2026-04-01T00:00:00Z', 'oof0',
              grid('2026-04-02T00:00:00Z', 3), scope='final')
    pool, _ = t.calibration_pool(head, 'dev')
    assert list(pool) == list(origins)
    assert all(Path(r['claim_path']).parent.name != 'deployment' for r in t.inventory('dev'))
    final_pool, _ = t.calibration_pool('2026-05-31T23:00:00Z', 'final')
    # The 2024 dev origins aged out of 730-day history; only final OOF remains.
    assert list(final_pool) == list(grid('2026-04-02T00:00:00Z', 3))


def test_cap_and_hash_stability(fake_old, monkeypatch):
    head = '2024-04-01T00:00:00Z'
    add_claim(fake_old, head, '2024-01-01T00:00:00Z', 'oof0', grid('2024-01-02T00:00:00Z', 30))
    monkeypatch.setitem(t.TEACHER_POLICY, 'max_train_origins', 7)
    try:
        pool, _ = t.calibration_pool(head, 'dev')
        assert len(pool) == 7
    finally:
        monkeypatch.setitem(t.TEACHER_POLICY, 'max_train_origins', 4096)
    assert t.cache_key(head, 'dev') == t.cache_key(head, 'dev')


def test_rejections(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 2)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins, status='failed')
    with pytest.raises(ValueError):
        t.calibration_pool(head, 'dev')
    (fake_old / 'backbones-ff1' / ('cutoff-' + pd.Timestamp(head).strftime('%Y%m%dT%H%M%SZ'))).rename(
        fake_old / 'backbones-ff1' / 'cutoff-old')
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins, config={'id': 'other'})
    with pytest.raises(ValueError):
        t.calibration_pool(head, 'dev')


def test_no_old_writes_and_merge_roundtrip(fake_old, tmp_path):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 3)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins)
    before = {str(p): _sha(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    pool, assignment = t.calibration_pool(head, 'dev')
    merged, ref = t.merged_forecasts(pool, assignment, out_path=tmp_path / 'new' / 'oof.parquet')
    assert ref is not None and len(merged) == 72 * len(pool)
    after = {str(p): _sha(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    assert before == after


def test_deployment_same_cutoff_only(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-04-02T00:00:00Z', 2)
    add_claim(fake_old, head, head, 'deployment', origins)
    forecast, record = t.deployment_backbone(head)
    assert len(forecast) == 72 * len(origins)
    assert record['role'] == 'deployment' and record['cutoff'] == pd.Timestamp(head).isoformat()
    with pytest.raises(ValueError):
        t.deployment_backbone('2024-05-01T00:00:00Z')
