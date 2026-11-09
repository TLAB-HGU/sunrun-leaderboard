"""Gated, post-fit descriptive diagnostics; never select or correct forecasts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PERIODS = ('F1', 'F2', 'F3')
SEEDS = (42, 43, 44)
LIMITS = [
    'Exposed benchmark: repeated leaderboard selection prevents unseen-future inference.',
    'Descriptive matched-seed differences are not causal attribution or event labels.',
    'Timing diagnostics use future truth; no shift is chosen or applied and no correction is fitted.',
    'Regional MSE/TIMSE energies are additive within a partition; S is nonlinear and is not additive.',
    'Strata overlap across families; do not sum energies across different families.',
    'No-offset forecasts isolate saved pre-offset output algebra, not a separately trained intervention.',
    'Input issue-time availability and historical byte-identity limitations remain unresolved.',
]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def frame(origins, pred):
    origins = pd.to_datetime(origins, utc=True)
    return pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72),
                         'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                         'pred_kms': pred.ravel()})


def reconstruct(run):
    """Replay only the stored clipping/offset algebra, without fitting anything."""
    frames = {'final': [], 'no_offset': []}
    receipts = {}
    for period in PERIODS:
        metadata = json.loads((run / f'{period}_metadata.json').read_text())
        config = metadata['config']
        if (not config['clip_component'] or not config['clip_final'] or not config['offset']
                or config['offset_scope'] != 'horizon' or config['alpha'] != .25
                or config['offset_window'] != 336):
            raise ValueError('Unexpected frozen parent clipping/offset contract')
        state = metadata['frozen_state']
        state_path = run / f'{period}_models/frozen_calibration.json'
        states = {'metadata.frozen_state': state}
        if state_path.exists():
            states['models/frozen_calibration.json'] = json.loads(state_path.read_text())
        for source, receipt in states.items():
            if (receipt['window'] != 336 or receipt['alpha'] != .25
                    or receipt['scope'] != 'horizon' or receipt['applied'] is not True
                    or receipt['offset_rows'] != 336):
                raise ValueError(f'Unexpected frozen offset state: {source}')
            np.testing.assert_array_equal(receipt['offset'], state['offset'])
        with np.load(run / f'{period}_raw_predictions.npz', allow_pickle=False) as saved:
            raw, origins, offset = saved['raw'], saved['origins'], saved['offset']
            component = np.clip(raw, 1., 3000.)
            np.testing.assert_array_equal(offset, np.asarray(state['offset']))
            np.testing.assert_array_equal(saved['component'], component)
            np.testing.assert_array_equal(saved['preclip'], component + offset[None, :])
        replayed = {'no_offset': component, 'final': np.clip(component + offset[None, :], 1., 3000.)}
        for variant, prediction in replayed.items():
            suffix = 'predictions' if variant == 'final' else 'no_offset_predictions'
            with np.load(run / f'{period}_{suffix}.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['origins'], origins)
                np.testing.assert_array_equal(saved['pred'], prediction)
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite reconstructed forecasts')
            frames[variant].append(frame(origins, prediction))
        receipts[period] = {'exact_replay': True, 'offset_receipt_source': 'metadata.frozen_state',
                            'validated_offset_receipts': list(states),
                            'offset_mean': float(offset.mean()),
                            'offset_min': float(offset.min()), 'offset_max': float(offset.max())}
    return {key: pd.concat(parts, ignore_index=True) for key, parts in frames.items()}, receipts


def error_summary(prediction, truth, mask):
    selected = np.broadcast_to(mask, truth.shape) & np.isfinite(truth)
    error = (prediction - truth)[selected]
    return {'pairs': int(selected.sum()), 'bias': float(error.mean()) if error.size else None,
            'mae': float(np.abs(error).mean()) if error.size else None,
            'rmse': float(np.sqrt(np.mean(error ** 2))) if error.size else None}


def squared_errors(prediction, truth):
    return np.where(np.isfinite(truth), (prediction - truth) ** 2, 0.)


def energy(squared, weights, mask):
    """Global per-lead-normalized energy allocation, never a regional S."""
    selected = np.broadcast_to(mask, squared.shape)
    return {key: float(np.sum(squared * weight * selected) / squared.shape[1])
            for key, weight in zip(('mse', 'timse'), weights, strict=True)}


def paired_strata(control, treatment, truth, weights, masks):
    a, b = squared_errors(control, truth), squared_errors(treatment, truth)
    result = {}
    for name, mask in masks.items():
        ae, be = energy(a, weights, mask), energy(b, weights, mask)
        result[name] = {'control': error_summary(control, truth, mask),
                        'treatment': error_summary(treatment, truth, mask),
                        'control_energy': ae, 'treatment_energy': be,
                        'energy_difference_treatment_minus_control': {k: be[k] - ae[k] for k in ae}}
    return result


def assert_official_parity(kernel, array, context, official):
    for scope in ('total', 'f1', 'f2', 'f3'):
        selected = np.ones(len(array), bool) if scope == 'total' else context['period'] == scope.upper()
        truth = context['truth'][selected]
        weights = kernel.metric_weights(truth, np.arange(len(truth)))
        exact = kernel.scores_from_weights(squared_errors(array[selected], truth)[None], *weights)
        if official[scope]['n_replaced']:
            raise ValueError('Official scorer replaced forecasts')
        for metric in ('s', 'rmse', 'trmse', 'timse'):
            np.testing.assert_allclose(exact[metric][0], official[scope][metric], rtol=0, atol=1e-8)


def strata(context, observations, data):
    truth, origins = context['truth'], context['origins']
    columns = ['origin', 'v_lag0', 'v_lag24', 'img_age_h', 'enlil_n_pending']
    inputs = pd.read_parquet(data, columns=columns, filters=[('set', '==', 'query'), ('h', '==', 1)])
    inputs['origin'] = pd.to_datetime(inputs.origin, utc=True)
    if inputs.origin.duplicated().any():
        raise ValueError('Duplicated query feature origins')
    inputs = inputs.set_index('origin')
    if not origins.isin(inputs.index).all():
        raise ValueError('Query input features do not cover official origins')
    inputs = inputs.reindex(origins)
    masks = {'all': np.ones(truth.shape, bool)}
    leads = np.arange(1, 73)[None, :]
    for low, high in ((1, 6), (7, 24), (25, 48), (49, 72)):
        masks[f'lead_{low}_{high}'] = (leads >= low) & (leads <= high)
    for name, values in [('target_speed', truth), ('origin_speed', inputs.v_lag0.to_numpy()[:, None])]:
        for label, low, high in [('lt400', -np.inf, 400), ('400_550', 400, 550),
                                 ('550_700', 550, 700), ('ge700', 700, np.inf)]:
            masks[f'{name}_{label}'] = np.isfinite(values) & (values >= low) & (values < high)
        masks[f'{name}_missing'] = ~np.isfinite(values)
    delta = (inputs.v_lag0 - inputs.v_lag24).to_numpy()[:, None]
    for label, mask in [('fall', delta <= -50), ('stable', np.abs(delta) < 50),
                         ('rise', delta >= 50), ('missing', ~np.isfinite(delta))]:
        masks[f'origin_delta24_{label}'] = mask
    series = observations.set_index(pd.DatetimeIndex(pd.to_datetime(observations.timestamp_utc, utc=True))).speed_kms.sort_index()
    valid = series.dropna()
    positions = valid.index.get_indexer(origins, method='ffill')
    if (positions < 0).any():
        raise ValueError('No causal official observation for an origin')
    age = (origins.asi8 - valid.index.asi8[positions]) / 3.6e12
    gap = np.column_stack([series.reindex(origins - pd.Timedelta(hours=h)).isna().to_numpy()
                           for h in range(24)]).sum(1)
    for name, values, threshold in [('official_age_gt6', age, 6), ('past24_has_gap', gap, 0),
                                    ('image_age_gt24', inputs.img_age_h.to_numpy(), 24),
                                    ('pending_enlil', inputs.enlil_n_pending.to_numpy(), 0)]:
        for state in (False, True):
            masks[f'{name}_{state}'] = (np.isfinite(values) & ((values > threshold) == state))[:, None]
        masks[f'{name}_missing'] = (~np.isfinite(values))[:, None]
    for period in PERIODS:
        masks[period] = (context['period'] == period)[:, None]
    return masks


def timing(prediction, truth):
    shifts = {str(shift): error_summary(prediction[:, 12 + shift:60 + shift], truth[:, 12:60], True)
              for shift in (-12, -6, 0, 6, 12)}
    adequate = np.isfinite(truth).sum(1) >= 48
    y, pred = truth[adequate], prediction[adequate]
    result = {'fixed_leads': [13, 60], 'shifted_prediction_minus_truth': shifts,
              'peak_range_origins': int(adequate.sum())}
    if not adequate.any():
        return result
    valid = np.isfinite(y)
    observed_range = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    predicted_range = np.max(np.where(valid, pred, -np.inf), axis=1) - np.min(np.where(valid, pred, np.inf), axis=1)
    peak_shift = np.argmax(np.where(valid, pred, -np.inf), axis=1) - np.nanargmax(y, axis=1)
    nonconstant = observed_range > 0
    result.update(median_observed_range=float(np.median(observed_range)),
                  median_predicted_range=float(np.median(predicted_range)),
                  median_range_ratio=float(np.median(predicted_range[nonconstant] / observed_range[nonconstant])) if nonconstant.any() else None,
                  nonconstant_range_origins=int(nonconstant.sum()),
                  median_peak_shift_hours=float(np.median(peak_shift)),
                  median_absolute_peak_shift_hours=float(np.median(abs(peak_shift))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline', type=Path, default=ROOT.parent / 'drop035_v650_confirm_20261108')
    parser.add_argument('--observations', type=Path, default=ROOT.parent / 'offset_roll_complete_ablation_20261106/competition-observations.parquet')
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = module(ROOT / 'analyze_results.py', 'roundselect_diagnosis_gate')
    runs = analysis.completed_runs(args.root)  # Must precede observation access and evaluation.
    scorer = module(REPO / 'leaderboard/scorer/competition.py', 'roundselect_diagnosis_scorer')
    kernel = module(ROOT.parent / 'post_ablation_five_axes_20261108/analysis_kernel.py', 'roundselect_diagnosis_kernel')
    observations = scorer.load_observations(args.observations)
    context = scorer.prepare_context(observations)
    truth = context['truth']
    weights = kernel.metric_weights(truth, np.arange(len(truth)))
    masks = strata(context, observations, args.data)
    output = {'classification': 'postfit descriptive matched-seed diagnostic', 'seeds': {}, 'limits': LIMITS,
              'observations': analysis.identity(args.observations), 'data': analysis.identity(args.data),
              'scorer_version': scorer.VERSION, 'energy_units': 'km^2/s^2; global per-lead weights / 72',
              'past24_definition': 'origin-23h through origin inclusive; absent or NaN hourly observations',
              'difference_direction': 'treatment minus control; negative error/energy difference is improvement',
              'offset_change_direction': 'within each arm: final minus no_offset; control/treatment labels in offset_energy_change mean no_offset/final'}
    for seed, treatment in zip(SEEDS, runs, strict=True):
        arms = {}
        for arm, run in [('control', args.baseline / 'runs' / f'drop035_v650_s{seed}'), ('treatment', treatment)]:
            frames, replay = reconstruct(run)
            arms[arm] = {'replay': replay, 'arrays': {}, 'scores': {}}
            for variant, table in frames.items():
                array = scorer.align_predictions(table, context)
                official = scorer.score_competition(table, context)
                assert_official_parity(kernel, array, context, official)
                arms[arm]['arrays'][variant] = array
                arms[arm]['scores'][variant] = official
            stored = json.loads((run / 'local_score.json').read_text())['competition']
            for scope in ('total', 'f1', 'f2', 'f3'):
                for metric in ('s', 'rmse', 'trmse', 'timse'):
                    np.testing.assert_allclose(arms[arm]['scores']['final'][scope][metric], stored[scope][metric], rtol=0, atol=1e-8)
        a, b = arms['control'], arms['treatment']
        result = {'arms': {name: {'replay': arm['replay'], 'official_scores': arm['scores'],
                                 'timing': timing(arm['arrays']['final'], truth)} for name, arm in arms.items()},
                  'strata': paired_strata(a['arrays']['final'], b['arrays']['final'], truth, weights, masks),
                  'no_offset_strata': paired_strata(a['arrays']['no_offset'], b['arrays']['no_offset'], truth, weights, masks),
                  'offset_energy_change': {name: paired_strata(arm['arrays']['no_offset'], arm['arrays']['final'],
                                                               truth, weights, {'all': True})['all']
                                           for name, arm in arms.items()},
                  'offset_score_change': {name: {scope: {metric: arm['scores']['final'][scope][metric] - arm['scores']['no_offset'][scope][metric]
                                                        for metric in ('s', 'rmse', 'trmse', 'timse')}
                                                for scope in ('total', 'f1', 'f2', 'f3')}
                                          for name, arm in arms.items()},
                  'score_differences': {variant: {scope: {metric: b['scores'][variant][scope][metric] - a['scores'][variant][scope][metric]
                                                        for metric in ('s', 'rmse', 'trmse', 'timse')}
                                                 for scope in ('total', 'f1', 'f2', 'f3')}
                                        for variant in ('final', 'no_offset')}}
        output['seeds'][str(seed)] = result
    analysis.write(args.root / 'diagnosis.json', output)
    lines = ['# Post-fit descriptive diagnosis', '', *[f'- {limit}' for limit in LIMITS], '',
             '| Seed | Final ΔS | No-offset ΔS | ΔMSE energy | ΔTIMSE energy |',
             '| --- | ---: | ---: | ---: | ---: |']
    for seed, result in output['seeds'].items():
        delta = result['strata']['all']['energy_difference_treatment_minus_control']
        lines.append(f"| {seed} | {result['score_differences']['final']['total']['s']:.8f} | "
                     f"{result['score_differences']['no_offset']['total']['s']:.8f} | {delta['mse']:.8f} | {delta['timse']:.8f} |")
    lines += ['', 'All differences are treatment minus control. See diagnosis.json for every fixed stratum,',
              'all five timing shifts, peak/range summaries, offset replay and official score parity.',
              'No candidate selection, causal mechanism conclusion, or forecast correction is made.']
    (args.root / 'diagnosis.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
