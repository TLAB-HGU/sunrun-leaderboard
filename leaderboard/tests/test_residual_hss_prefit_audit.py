"""Synthetic-only prefit-audit tests; no real data, no GPU, CPU only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_hss_phase_ff1 import coverage as ff1coverage
from experiments.mse20_hss_residual import prefit_audit as audit
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


def _write_model(root, fit):
    model_dir = root / ('model-' + pd.Timestamp(fit).strftime('%Y%m%d%H%M'))
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(config={'id': 'synth-backbone'}, cutoff=pd.Timestamp(fit).isoformat(),
                train_label_max=pd.Timestamp(fit).isoformat())
    (model_dir / 'metadata.json').write_text(json.dumps(meta, sort_keys=True))
    (model_dir / 'metadata.sha256').write_text(_sha(model_dir / 'metadata.json'))
    return model_dir


def add_claim(root, head, fit, role, origins, scope='dev', status='ok'):
    head = pd.Timestamp(head)
    directory = root / 'backbones-ff1' / ('cutoff-' + head.strftime('%Y%m%dT%H%M%SZ')) / role
    directory.mkdir(parents=True, exist_ok=True)
    model_dir = _write_model(root, fit)
    frame = pd.DataFrame(dict(
        origin_last_input_utc=np.repeat([pd.Timestamp(v).isoformat() for v in origins], 72),
        horizon_hours=np.tile(np.arange(1, 73), len(origins)),
        pred_kms=np.full(72 * len(origins), 400.)))
    frame.to_parquet(directory / 'forecast.parquet', index=False)
    (directory / 'training-provenance.json').write_text(
        json.dumps(dict(train_label_max=pd.Timestamp(fit).isoformat()), sort_keys=True))
    identity = dict(role=role, scope=scope, head_cutoff=head.isoformat(),
                    cutoff=pd.Timestamp(fit).isoformat(),
                    origins=[pd.Timestamp(v).isoformat() for v in origins],
                    config={'id': 'synth-backbone'}, sources={'/synth/mod.py': '0' * 64},
                    coverage_policy=dict(ff1coverage.COVERAGE_POLICY),
                    original_config_registration=_ref(root / 'registered.json'))
    (directory / 'claim.json').write_text(json.dumps(
        dict(identity=identity, registration=_ref(root / 'registered.json')), sort_keys=True))
    (directory / 'result.json').write_text(json.dumps(dict(
        status=status, role=role, cutoff=pd.Timestamp(fit).isoformat(),
        head_cutoff=head.isoformat(), model_dir=str(model_dir.resolve()), model_files=[],
        coverage_policy=dict(ff1coverage.COVERAGE_POLICY),
        claim=_ref(directory / 'claim.json'), forecast=_ref(directory / 'forecast.parquet'),
        training_provenance=_ref(directory / 'training-provenance.json')), sort_keys=True))
    return directory


def grid(start, n):
    return pd.date_range(pd.Timestamp(start), periods=n, freq='6h', tz='UTC')


def test_independent_recount_matches_teacher(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins0 = grid('2024-01-02T00:00:00Z', 10)
    origins1 = grid('2024-02-02T00:00:00Z', 10)
    add_claim(fake_old, head, '2024-01-01T00:00:00Z', 'oof0', origins0)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof1', origins1)
    claims, rejected = audit._claim_index(t._old_root())
    assert not rejected
    pool, _ = t.calibration_pool(head, 'dev')
    assert list(audit.independent_pool(head, claims)) == list(pool)
    assert list(pool) == sorted(set(origins0) | set(origins1))


def test_independent_recount_excludes_deployment_future_and_failed(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 4)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins)
    add_claim(fake_old, head, head, 'deployment', grid('2024-04-02T00:00:00Z', 3))
    add_claim(fake_old, '2024-06-01T00:00:00Z', '2024-05-01T00:00:00Z', 'oof0',
              grid('2024-05-02T00:00:00Z', 3))
    claims, _ = audit._claim_index(t._old_root())
    assert list(audit.independent_pool(head, claims)) == list(origins)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof1', origins, status='failed')
    claims2, rejected = audit._claim_index(t._old_root())
    assert rejected and list(audit.independent_pool(head, claims2)) == list(origins)


def test_horizon_checker_reports_missing_and_bad():
    origins = grid('2024-02-02T00:00:00Z', 3)
    full = pd.DataFrame(dict(
        origin_last_input_utc=np.repeat([v.isoformat() for v in origins], 72),
        horizon_hours=np.tile(np.arange(1, 73), len(origins))))
    assert audit.check_forecast_horizons(full, origins) == ([], [])
    dropped = full[~((full['origin_last_input_utc'] == origins[1].isoformat())
                     & (full['horizon_hours'] == 72))]
    missing, bad = audit.check_forecast_horizons(dropped, origins)
    assert missing == [] and bad == [origins[1].isoformat()]
    gone = full[full['origin_last_input_utc'] != origins[2].isoformat()]
    missing, _ = audit.check_forecast_horizons(gone, origins)
    assert missing == [origins[2].isoformat()]


def test_synthetic_cpu_forward_shapes_finite_and_consistent():
    torch = pytest.importorskip('torch')
    from experiments.mse20_hss_residual import model as m
    for config in m.candidate_configs():
        torch.manual_seed(7)
        model = m.ResidualModel(config, 8)
        model.eval()
        x = torch.randn(3, 8)
        current = torch.tensor([380., 420., 500.])
        base = torch.full((3, 72), 430.)
        with torch.no_grad():
            out = model(x, current, base)
            repeat = model(x, current, base)
        assert out['mse'].shape == (3, 72) and out['scenarios'].shape == (3, 21, 72)
        assert torch.isfinite(out['mse']).all() and torch.isfinite(out['probabilities']).all()
        assert torch.equal(out['mse'], repeat['mse'])
        expected = (out['probabilities'][:, :, None] * out['scenarios']).sum(1)
        assert torch.allclose(out['mse'], expected, rtol=1e-5, atol=1e-5)
        assert float(out['mse'].min()) >= 1 and float(out['mse'].max()) <= 3000


def test_cuda_evidence_never_skips_when_unsupported():
    evidence = audit.cuda_parity_evidence()
    assert set(('passed', 'devices', 'errors')) <= set(evidence)
    assert evidence['devices'], 'each fixed config must be reported'
    assert {v['config_id'] for v in evidence['devices']} == {'residual_rise', 'residual_rise_decline'}
    import torch
    if not torch.cuda.is_available() or not torch.cuda.device_count():
        assert evidence['passed'] is False
        assert evidence['errors'], 'unsupported CUDA is failure, not skip'


def test_cli_writes_new_json_only_and_refuses_overwrite(monkeypatch, tmp_path, fake_old):
    before = {str(p): _sha(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    monkeypatch.setattr(audit, 'audit_data',
                        lambda: dict(passed=True, synthetic=True, errors=[]))
    monkeypatch.setattr(audit, 'cuda_parity_evidence',
                        lambda: dict(passed=True, synthetic=True, devices=[], errors=[]))
    target = tmp_path / 'new' / 'prefit.json'
    assert audit.main(['--output', str(target)]) == 0
    report = json.loads(target.read_text())
    assert set(('data', 'cuda')) <= set(report) and report['passed'] is True
    with pytest.raises(ValueError):
        audit.main(['--output', str(target)])
    after = {str(p): _sha(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    assert before == after


def test_independent_assignment_matches_teacher_with_tiebreak(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 4)
    add_claim(fake_old, head, '2024-01-01T00:00:00Z', 'oof0', origins)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof1', origins)
    claims, _ = audit._claim_index(t._old_root())
    pool, assignment = t.calibration_pool(head, 'dev')
    expected = audit.independent_assignment(head, claims)
    assert len(expected) == len(pool)
    for origin, record in zip(pool, assignment):
        key = pd.Timestamp(origin).isoformat()
        assert expected[key]['fit_cutoff'] == pd.Timestamp(record['fit_cutoff']).isoformat()
        assert Path(expected[key]['claim_path']).resolve() == Path(record['claim_path']).resolve()
    # Nearest preceding actual fit wins for every shared origin.
    assert {v['fit_cutoff'] for v in expected.values()} == {
        pd.Timestamp('2024-02-01T00:00:00Z').isoformat()}


def test_independent_assignment_sorted_path_tiebreak(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 3)
    fit = '2024-02-01T00:00:00Z'
    dir0 = add_claim(fake_old, head, fit, 'oof0', origins)
    dir1 = add_claim(fake_old, head, fit, 'oof1', origins)
    claims, _ = audit._claim_index(t._old_root())
    expected = audit.independent_assignment(head, claims)
    winner = str((dir0 / 'claim.json').resolve()) if str(
        (dir0 / 'claim.json').resolve()) < str((dir1 / 'claim.json').resolve()) else str(
        (dir1 / 'claim.json').resolve())
    assert {v['claim_path'] for v in expected.values()} == {winner}
    _, assignment = t.calibration_pool(head, 'dev')
    assert {str(Path(r['claim_path']).resolve()) for r in assignment} == {winner}


def test_ages_from_assignment_use_actual_fit_cutoff(fake_old):
    head = '2024-04-01T00:00:00Z'
    origins = grid('2024-02-02T00:00:00Z', 3)
    add_claim(fake_old, head, '2024-02-01T00:00:00Z', 'oof0', origins)
    claims, _ = audit._claim_index(t._old_root())
    pool, assignment = t.calibration_pool(head, 'dev')
    ages = audit.ages_from_assignment(pool, assignment)
    want = np.array([(pd.Timestamp(o) - pd.Timestamp('2024-02-01T00:00:00Z')).total_seconds() / 3600
                     for o in pool], dtype=np.float32)
    np.testing.assert_allclose(ages, want, rtol=0, atol=1e-6)
    assert ages.shape == (len(pool),) and np.isfinite(ages).all() and (ages >= 0).all()


def test_in_memory_merge_writes_no_cache(fake_old, tmp_path):
    origins = grid('2024-02-02T00:00:00Z', 3)
    add_claim(fake_old, '2024-04-01T00:00:00Z', '2024-02-01T00:00:00Z', 'oof0', origins)
    before = {str(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    pool, assignment = t.calibration_pool('2024-04-01T00:00:00Z', 'dev')
    merged, ref = t.merged_forecasts(pool, assignment, out_path=None)
    assert ref is None and len(merged) == 72 * len(pool)
    after = {str(p) for p in sorted(fake_old.rglob('*')) if p.is_file()}
    assert before == after


def test_source_binding_covers_recipe_and_detects_drift():
    binding = audit.snapshot_sources()
    assert len(binding) == len(audit.AUDIT_SOURCE_NAMES) == 5
    assert all(Path(k).name in ('prefit_audit.py', 'dev.py', 'model.py', 'teacher.py', 'protocol.py')
               for k in binding)
    assert all(len(v) == 64 for v in binding.values())
    altered = dict(binding)
    first = next(iter(altered))
    altered[first] = '0' * 64
    assert audit.sources_drifted(binding, altered) is True
    assert audit.sources_drifted(binding, dict(binding)) is False


def test_feature_finite_mask_semantics():
    origins = grid('2024-02-02T00:00:00Z', 2)
    good = dict(values=np.ones((2, 9), dtype=np.float32),
                current=np.full(2, 400., dtype=np.float32),
                base=np.full((2, 72), 430., dtype=np.float32))
    report = audit.check_feature_finite_mask(good, origins)
    assert report['passed'] and report['rows_match'] and report['current_finite'] and report['base_finite']
    masked = dict(values=np.array([[1., np.nan] + [0.] * 7, [2.] * 9], dtype=np.float32),
                  current=np.full(2, 400., dtype=np.float32),
                  base=np.full((2, 72), 430., dtype=np.float32))
    masked_report = audit.check_feature_finite_mask(masked, origins)
    assert masked_report['values_finite'] is False and masked_report['passed'] is True
    bad = dict(values=np.ones((2, 9), dtype=np.float32),
               current=np.array([400., np.nan], dtype=np.float32),
               base=np.full((2, 72), 430., dtype=np.float32))
    assert audit.check_feature_finite_mask(bad, origins)['passed'] is False
    dropped = dict(values=np.ones((1, 9), dtype=np.float32),
                   current=np.full(1, 400., dtype=np.float32),
                   base=np.full((1, 72), 430., dtype=np.float32))
    assert audit.check_feature_finite_mask(dropped, origins)['passed'] is False


def test_label_hashes_cover_full96_and_class_ids():
    from experiments.mse20_hss_residual import model as m
    idx = pd.date_range('2024-01-01T00:00:00Z', periods=720, freq='h', tz='UTC')
    frames = dict(speed=pd.Series(np.where(np.arange(720) < 500, 400., 700.), index=idx),
                  observed=pd.Series(True, index=idx))
    onsets = m.state.onset_times(frames['speed'], frames['observed'])
    origin = onsets[0] - pd.Timedelta(hours=6)
    labels = m.make_labels(frames, pd.DatetimeIndex([origin]), origin + pd.Timedelta(hours=96))
    hashes = m.label_hashes(labels)
    assert set(('profile96', 'arrival', 'tau_hours')) <= set(hashes)
    assert labels['profile96'].shape == (1, 96) and labels['arrival'].shape == (1,)


def test_cpu_map_is_joint_argmax_on_clipped_scenarios():
    torch = pytest.importorskip('torch')
    from experiments.mse20_hss_residual import model as m
    torch.manual_seed(11)
    for config in m.candidate_configs():
        torch.manual_seed(11)
        model = m.ResidualModel(config, 8)
        fake_x = torch.randn(3, 8)
        current = torch.tensor([380., 420., 500.])
        base = torch.full((3, 72), 430.)
        report = audit._cpu_map_report(model, fake_x, current, base)
        for key in ('shapes_ok', 'finite_ok', 'deterministic', 'expectation_close',
                    'mode_match', 'peak_is_map', 'scenarios_clipped', 'bounds_ok',
                    'probabilities_normalized'):
            assert report[key] is True, (config['id'], key)


def test_cuda_evidence_reports_map_and_triple_tolerance():
    evidence = audit.cuda_parity_evidence()
    assert evidence['tolerance'] == pytest.approx(1e-3)
    for entry in evidence['devices']:
        assert set(('mode_match', 'peak_is_map', 'scenarios_clipped',
                    'probabilities_normalized')) <= set(entry)
        assert entry['mode_match'] is True and entry['peak_is_map'] is True
