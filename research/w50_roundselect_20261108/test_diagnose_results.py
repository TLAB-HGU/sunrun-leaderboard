import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def diagnosis():
    spec = importlib.util.spec_from_file_location('diagnosis_test', Path(__file__).with_name('diagnose_results.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_gate_blocks_all_truth_access_and_outputs(diagnosis, tmp_path, monkeypatch):
    events = []

    def gate(root):
        events.append(('completed_runs', root))
        raise ValueError('training still live')

    def load(path, name):
        assert path.name == 'analyze_results.py', 'No evaluator import before completion gate'
        return SimpleNamespace(completed_runs=gate)

    def forbidden(*args, **kwargs):
        pytest.fail('No data access before completion gate')

    monkeypatch.setattr(diagnosis, 'module', load)
    monkeypatch.setattr(diagnosis.pd, 'read_parquet', forbidden)
    with pytest.raises(ValueError, match='training still live'):
        diagnosis.main(['--root', str(tmp_path), '--data', str(tmp_path / 'unused.parquet')])
    assert events == [('completed_runs', tmp_path)]
    assert not (tmp_path / 'diagnosis.json').exists()
    assert not (tmp_path / 'diagnosis.md').exists()


def test_synthetic_official_parity_and_additive_energy_partitions(diagnosis):
    scorer = diagnosis.module(diagnosis.REPO / 'leaderboard/scorer/competition.py', 'diagnosis_synthetic_scorer')
    kernel = diagnosis.module(diagnosis.ROOT.parent / 'post_ablation_five_axes_20261108/analysis_kernel.py',
                              'diagnosis_synthetic_kernel')
    rng = np.random.default_rng(719)
    observations = pd.DataFrame({'timestamp_utc': pd.date_range('2020-01-01', periods=24 * 25, freq='h', tz='UTC'),
                                 'speed_kms': rng.uniform(250, 850, 24 * 25)})
    observations.loc[[40, 45, 170, 250], 'speed_kms'] = np.nan
    context = scorer.prepare_context(observations, periods={
        'F1': ('2020-01-02', '2020-01-08'), 'F2': ('2020-01-09', '2020-01-15'),
        'F3': ('2020-01-16', '2020-01-22')})
    truth = context['truth']
    control = np.nan_to_num(truth, nan=400.) + rng.normal(10, 30, truth.shape)
    treatment = np.nan_to_num(truth, nan=400.) + rng.normal(-5, 20, truth.shape)
    weights = kernel.metric_weights(truth, np.arange(len(truth)))
    masks = {'all': True}
    leads = np.arange(1, 73)[None, :]
    for low, high in [(1, 6), (7, 24), (25, 48), (49, 72)]:
        masks[f'{low}_{high}'] = (leads >= low) & (leads <= high)
    summary = diagnosis.paired_strata(control, treatment, truth, weights, masks)
    for pred in (control, treatment):
        official = scorer.score_competition(diagnosis.frame(context['origins'], pred), context)
        diagnosis.assert_official_parity(kernel, pred, context, official)
        energy = diagnosis.energy(diagnosis.squared_errors(pred, truth), weights, True)
        assert energy['timse'] == pytest.approx(official['total']['timse'], abs=1e-8)
        assert energy['mse'] == pytest.approx(np.mean([x['rmse'] ** 2 for x in official['total']['by_lead']]), abs=1e-8)
    for metric in ('mse', 'timse'):
        summed = sum(summary[name]['energy_difference_treatment_minus_control'][metric]
                     for name in masks if name != 'all')
        assert summed == pytest.approx(summary['all']['energy_difference_treatment_minus_control'][metric], abs=1e-8)
    corrupted = scorer.score_competition(diagnosis.frame(context['origins'], treatment), context)
    corrupted['total']['s'] += 1
    with pytest.raises(AssertionError):
        diagnosis.assert_official_parity(kernel, treatment, context, corrupted)


@pytest.mark.parametrize('separate_receipt', [True, False])
def test_reconstruction_checks_saved_no_offset_and_final(diagnosis, tmp_path, separate_receipt):
    for i, period in enumerate(diagnosis.PERIODS):
        origins = pd.date_range(f'2020-0{i + 1}-01', periods=2, freq='h', tz='UTC').asi8
        raw = np.tile(np.linspace(-10, 3020, 72), (2, 1))
        offset = np.linspace(-20, 20, 72)
        component = np.clip(raw, 1., 3000.)
        final = np.clip(component + offset, 1., 3000.)
        state = {'offset': offset.tolist(), 'window': 336, 'alpha': .25, 'scope': 'horizon',
                 'applied': True, 'offset_rows': 336}
        (tmp_path / f'{period}_metadata.json').write_text(json.dumps({'config': {
            'clip_component': True, 'clip_final': True, 'offset': True, 'offset_scope': 'horizon',
            'alpha': .25, 'offset_window': 336}, 'frozen_state': state}))
        if separate_receipt:
            directory = tmp_path / f'{period}_models'
            directory.mkdir()
            (directory / 'frozen_calibration.json').write_text(json.dumps(state))
        np.savez(tmp_path / f'{period}_raw_predictions.npz', raw=raw, component=component,
                 preclip=component + offset, offset=offset, origins=origins)
        np.savez(tmp_path / f'{period}_no_offset_predictions.npz', pred=component, origins=origins)
        np.savez(tmp_path / f'{period}_predictions.npz', pred=final, origins=origins)
    frames, receipts = diagnosis.reconstruct(tmp_path)
    assert len(frames['final']) == 3 * 2 * 72
    assert all(receipt['exact_replay'] for receipt in receipts.values())
    assert all(receipt['offset_receipt_source'] == 'metadata.frozen_state' for receipt in receipts.values())
    assert all(len(receipt['validated_offset_receipts']) == (2 if separate_receipt else 1)
               for receipt in receipts.values())
    if separate_receipt:
        receipt_path = tmp_path / 'F1_models/frozen_calibration.json'
        original = receipt_path.read_text()
        invalid = json.loads(original)
        invalid['offset'][0] += 1
        receipt_path.write_text(json.dumps(invalid))
        with pytest.raises(AssertionError):
            diagnosis.reconstruct(tmp_path)
        receipt_path.write_text(original)
    else:
        assert not (tmp_path / 'F1_models').exists()
    metadata_path = tmp_path / 'F1_metadata.json'
    original = metadata_path.read_text()
    for key, value in [('window', 335), ('alpha', .5), ('scope', 'global'), ('applied', False), ('offset_rows', 335)]:
        invalid = json.loads(original)
        invalid['frozen_state'][key] = value
        metadata_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match='Unexpected frozen offset state'):
            diagnosis.reconstruct(tmp_path)
    metadata_path.write_text(original)
    path = tmp_path / 'F3_no_offset_predictions.npz'
    with np.load(path) as saved:
        pred, origins = saved['pred'].copy(), saved['origins'].copy()
    np.savez(path, pred=pred + 1, origins=origins)
    with pytest.raises(AssertionError):
        diagnosis.reconstruct(tmp_path)


def test_timing_keeps_same_central_support_and_48_observation_gate(diagnosis):
    truth = np.tile(np.arange(72.) + 300, (3, 1))
    prediction = truth.copy()
    truth[1, 48:] = np.nan
    truth[2, 47:] = np.nan
    result = diagnosis.timing(prediction, truth)
    assert result['fixed_leads'] == [13, 60]
    assert result['peak_range_origins'] == 2
    assert result['median_absolute_peak_shift_hours'] == 0
    assert result['median_range_ratio'] == 1
    shifts = result['shifted_prediction_minus_truth']
    assert set(shifts) == {'-12', '-6', '0', '6', '12'}
    assert len({row['pairs'] for row in shifts.values()}) == 1
    for shift, row in shifts.items():
        assert row['mae'] == abs(int(shift))


def test_fixed_strata_boundaries_and_missing_inputs(diagnosis, monkeypatch):
    origins = pd.date_range('2020-01-02', periods=6, freq='6h', tz='UTC')
    speed = np.array([399., 400., 550., 700., 500., np.nan])
    delta = np.array([-50., 0., 50., -49., 49., np.nan])
    inputs = pd.DataFrame({'origin': origins, 'v_lag0': speed, 'v_lag24': speed - delta,
                           'img_age_h': [24., 25., np.nan, 0., 0., 0.],
                           'enlil_n_pending': [0., 1., np.nan, 0., 0., 0.]})
    monkeypatch.setattr(diagnosis.pd, 'read_parquet', lambda *a, **k: inputs.copy())
    context = {'origins': origins, 'truth': np.tile(speed[:, None], (1, 72)),
               'period': np.array(['F1', 'F1', 'F2', 'F2', 'F3', 'F3'])}
    observations = pd.DataFrame({'timestamp_utc': pd.date_range('2020-01-01', periods=72, freq='h', tz='UTC'),
                                 'speed_kms': 400.})
    masks = diagnosis.strata(context, observations, 'unused')
    assert masks['origin_speed_lt400'][:, 0].tolist() == [True, False, False, False, False, False]
    assert masks['origin_speed_400_550'][:, 0].tolist() == [False, True, False, False, True, False]
    assert masks['origin_speed_550_700'][2, 0]
    assert masks['origin_speed_ge700'][3, 0]
    assert masks['origin_delta24_fall'][0, 0]
    assert masks['origin_delta24_rise'][2, 0]
    assert masks['origin_delta24_stable'][:, 0].tolist() == [False, True, False, True, True, False]
    assert masks['image_age_gt24_False'][0, 0]
    assert masks['image_age_gt24_True'][1, 0]
    assert masks['image_age_gt24_missing'][2, 0]
    assert not masks['image_age_gt24_False'][2, 0]
    for family in ('origin_speed_', 'origin_delta24_', 'image_age_gt24_', 'pending_enlil_'):
        partition = sum(mask.astype(int) for name, mask in masks.items() if name.startswith(family))
        np.testing.assert_array_equal(partition, np.ones((6, 1), int))
