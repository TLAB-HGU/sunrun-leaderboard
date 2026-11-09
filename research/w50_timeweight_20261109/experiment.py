"""Fresh retained-parent training with one fixed calendar-row objective multiplier."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import xgboost as xgb

CAMPAIGN = Path(__file__).resolve().parent
REPO = CAMPAIGN.parents[1]
PARENT = CAMPAIGN.parent / 'drop035_v650_confirm_20261108'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_modules():
    """Private globals prevent changing the parent or another imported runner."""
    baseline = load_module(PARENT / 'baseline.py', 'timeweight_baseline')
    previous = sys.modules.get('baseline')
    sys.modules['baseline'] = baseline
    try:
        core = load_module(PARENT / 'core.py', 'timeweight_core')
    finally:
        if previous is None:
            del sys.modules['baseline']
        else:
            sys.modules['baseline'] = previous
    return baseline, core


def fixed_config(core, seed, experiment):
    return core.normalize_config({'experiment': experiment, 'seed': seed,
                                     'exclude_features': ['ch_ch_t035_equatorial_area', 'strip_v650_dark']})


def assert_parent_contract(core, config, seed):
    names = core.feature_names(config)
    if sum(map(len, names.values())) != 28:
        raise ValueError('Frozen parent requires exactly 28 source features')
    for period in ('F1', 'F2', 'F3'):
        path = PARENT / 'runs' / f'drop035_v650_s{seed}' / f'{period}_metadata.json'
        parent = json.loads(path.read_text())
        expected = core.normalize_config(parent['config'])
        if {k: v for k, v in expected.items() if k != 'experiment'} != {
                k: v for k, v in config.items() if k != 'experiment'}:
            raise ValueError(f'{period}: settings differ from matched parent')
        columns = {str(start): core.expanded_names(names, start, 24) for start in (0, 24, 48)}
        if names != parent['features'] or columns != parent['expanded_features']:
            raise ValueError(f'{period}: actual feature order differs from matched parent')
        if parent['rounds'] != 900 or parent['training_cutoff'] != core.baseline.CUTOFFS[period]:
            raise ValueError(f'{period}: parent rounds or cutoff differs')
    return names


def score_outputs(core, out, observations):
    # Every period has frozen complete forecasts before official truth is opened.
    frame = core.prediction_frame(out)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    scorer = load_module(REPO / 'leaderboard/scorer/competition.py', 'timeweight_official_scorer')
    context = scorer.prepare_context(scorer.load_observations(observations))
    score = scorer.score_competition(frame, context)
    aligned = scorer.align_predictions(frame, context)
    for scope in ('total', 'f1', 'f2', 'f3'):
        if score[scope]['n_replaced']:
            raise ValueError('Official scorer replaced forecasts')
        mask = np.ones(len(aligned), bool) if scope == 'total' else context['period'] == scope.upper()
        score[scope]['mae'] = float(np.nanmean(np.abs(aligned[mask] - context['truth'][mask])))
    result = {'competition': score, 'observations': core.file_identity(observations),
                  'scorer': core.file_identity(scorer.__file__), 'scorer_version': scorer.VERSION}
    core.write_json(out / 'local_score.json', result)
    core.write_json(out / 'submission.json', dict(result, local_only=True, published=False))
    return score


POLICY = {'name': 'calendar_row_multiplier', 'half_life_days': 180, 'floor': .1,
          'age_reference': 'max_stage_origin', 'normalization': 'mean_one_per_stage',
          'gradient_and_hessian': True, 'dmatrix_weight': False,
          'application': 'multiply both parent gradient and Hessian once; DMatrix weights unset',
          'tail_renormalization_after_multiplier': False}


def row_weights(origins, cutoff):
    origins = pd.DatetimeIndex(origins)
    if (not len(origins) or origins.hasnans or origins.tz is None or origins.has_duplicates
            or not origins.is_monotonic_increasing):
        raise ValueError('Stage origins must be nonempty, unique, sorted and timezone aware')
    if not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff, tz='UTC')).all():
        raise ValueError('Stage origins violate 144h training cutoff')
    age = (origins.max() - origins).total_seconds().to_numpy() / 86400.
    raw = .1 + .9 * np.exp(-np.log(2.) * age / 180.)
    return {'origins': origins.asi8.copy(), 'age_days': age, 'raw': raw, 'normalized': raw / raw.mean()}


def validate_weights(receipt, origins):
    if not np.array_equal(receipt['origins'], pd.DatetimeIndex(origins).asi8):
        raise ValueError('Row weights misaligned with stage origins')
    raw, normalized = receipt['raw'], receipt['normalized']
    if (raw.shape != (len(origins),) or normalized.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(normalized).all()
            or (raw <= 0).any() or (normalized <= 0).any()
            or not np.isclose(normalized.mean(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Invalid positive finite mean1 row weights')
    np.testing.assert_array_equal(normalized, raw / raw.mean())


def weighted_objective(core, target, config, receipt, origins):
    validate_weights(receipt, origins)
    if len(target) != len(origins):
        raise ValueError('Training targets and weight rows differ')
    parent = core.make_objective(target, config)
    multiplier = receipt['normalized'][:, None].copy()

    def objective(prediction, matrix):
        gradient, hessian = parent(prediction, matrix)
        return ((gradient * multiplier).astype(np.float32), (hessian * multiplier).astype(np.float32))
    return objective


def original_observed(data, period, origins, target):
    rows = pd.read_parquet(data, columns=['origin', 'h', 'y'],
                           filters=[(f'fit_{period}', '==', True)], use_threads=False)
    rows['origin'] = pd.to_datetime(rows.origin, utc=True)
    rows = rows.sort_values(['origin', 'h'], kind='stable')
    if (len(rows) != target.size or rows.duplicated(['origin', 'h']).any()
            or not np.array_equal(pd.DatetimeIndex(rows.origin).asi8, np.repeat(origins.asi8, 72))
            or not np.array_equal(rows.h.to_numpy(), np.tile(np.arange(1, 73), len(origins)))):
        raise ValueError('Original-label keys differ from prepared training grid')
    labels = rows.y.to_numpy(float).reshape(target.shape)
    if np.isinf(labels).any():
        raise ValueError('Infinite original labels')
    observed = np.isfinite(labels)
    if not np.array_equal(labels[observed], target[observed]):
        raise ValueError('Originally observed labels changed')
    return observed


def weight_statistics(receipt, target, observed):
    r = receipt['normalized']
    tail = np.where(target >= 550., 4., 1.)
    point = tail / tail.mean(axis=0)
    hessian = point + .1
    weighted_point, weighted_hessian = r[:, None] * point, r[:, None] * hessian
    groups = {'all': np.ones_like(observed), 'ge550': target >= 550., 'ge700': target >= 700.,
              'observed': observed, 'imputed': ~observed,
              'old_ge700': (receipt['age_days'][:, None] >= 180) & (target >= 700.)}
    result = {'rows': len(r), 'origin_start': str(pd.Timestamp(receipt['origins'][0], tz='UTC')),
              'origin_end': str(pd.Timestamp(receipt['origins'][-1], tz='UTC')), 'row_ess': float(r.sum() ** 2 / np.sum(r ** 2)),
              'raw_minmax': [float(receipt['raw'].min()), float(receipt['raw'].max())],
              'normalized_minmax': [float(r.min()), float(r.max())], 'normalized_mean': float(r.mean()),
              'raw_min': float(receipt['raw'].min()), 'raw_max': float(receipt['raw'].max()),
              'normalized_min': float(r.min()), 'normalized_max': float(r.max()),
              'ess': float(r.sum() ** 2 / np.sum(r ** 2)),
              'old_definition': 'age_days >= 180', 'row_ess_is_not_independent_sample_count': True,
              'hessian_is_diagonal_surrogate_not_spectral_cell_loss': True, 'groups': {}}
    for name, mask in groups.items():
        rows = mask.any(axis=1)
        selected = r[rows]
        result['groups'][name] = {'pairs': int(mask.sum()), 'origins': int(rows.sum()),
            'row_ess_any_qualifying_lead': float(selected.sum() ** 2 / np.sum(selected ** 2)) if len(selected) else None,
            'parent_point_mass_per_lead': (point * mask).sum(axis=0).tolist(),
            'actual_point_mass_per_lead': (weighted_point * mask).sum(axis=0).tolist(),
            'parent_hessian_mass_per_lead': (hessian * mask).sum(axis=0).tolist(),
            'actual_hessian_mass_per_lead': (weighted_hessian * mask).sum(axis=0).tolist(),
            'parent_point_mass_fraction': float(point[mask].sum() / point.sum()),
            'actual_point_mass_fraction': float(weighted_point[mask].sum() / weighted_point.sum())}
    return result


def fit_head(core, x, y, config, receipt, origins, threads, rounds=900):
    matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
    if matrix.get_weight().size:
        raise ValueError('DMatrix weights must remain unset')
    params = dict(core.baseline.PARAMS, seed=config['seed'], device='cpu', nthread=threads,
                  num_target=24, base_score=float(y.mean()))
    model = xgb.train(params, matrix, num_boost_round=rounds,
                      obj=weighted_objective(core, y, config, receipt, origins))
    if model.num_boosted_rounds() != rounds:
        raise ValueError('Incomplete fresh fixed-budget model')
    return model, params


def fit_stage(core, train, query, target, observed, config, out, period, stage, threads, progress):
    out.mkdir(parents=True)
    receipt = row_weights(train['origins'], core.baseline.CUTOFFS[period])
    validate_weights(receipt, train['origins'])
    np.savez_compressed(out / 'row_weights.npz', **receipt)
    stats = weight_statistics(receipt, target, observed)
    core.write_json(out / 'weight_statistics.json', stats)
    pred = np.empty((len(query['origins']), 72), np.float64)
    parameters = []
    for block, start in enumerate((0, 24, 48)):
        model, params = fit_head(core, core.design(train, start, 24), target[:, start:start + 24],
                                config, receipt, train['origins'], threads)
        model.save_model(out / f'b{block}.ubj')
        pred[:, start:start + 24] = model.inplace_predict(core.design(query, start, 24))
        parameters.append(params)
        progress['unit'] += 1
        print(json.dumps({'period': period, 'stage': stage, 'block': block, 'horizon': start + 24,
                          **progress}), flush=True)
        del model
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite fresh component prediction')
    return pred, {'parameters': parameters, 'weight_statistics': stats}


def run_period(core, config, data, original_labels, out, period, threads, identity):
    baseline = core.baseline
    names = core.feature_names(config)
    train, target = baseline.load_grid(data, period, names, True)
    query, query_truth = baseline.load_grid(data, period, names, False)
    if query_truth is not None:
        raise ValueError('Query loader must not return evaluation targets')
    if len(query['origins']) != baseline.QUERY_COUNTS[period] or not np.isfinite(target).all():
        raise ValueError('Invalid official query count or prepared targets')
    parent = json.loads((PARENT / 'runs' / f"drop035_v650_s{config['seed']}" / f'{period}_metadata.json').read_text())
    origins = train['origins']
    if (len(origins) != parent['train_origins'] or str(origins.min()) != parent['train_start']
            or str(origins.max()) != parent['train_end']):
        raise ValueError('Original full training rows or ranges changed')
    observed = original_observed(original_labels, period, origins, target)
    reference, calibration, boundary = baseline.prior_split(origins)
    for name, indices in [('reference', reference), ('calibration', calibration)]:
        state = parent['frozen_state']
        if (len(indices) != state[f'{name}_rows'] or str(origins[indices].min()) != state[f'{name}_start']
                or str(origins[indices].max()) != state[f'{name}_end']):
            raise ValueError('Original prior split changed')
    model_dir = out / f'{period}_models'
    model_dir.mkdir()
    progress = {'unit': 0, 'units': 6}
    prior, prior_report = fit_stage(core, baseline.subset(train, reference), baseline.subset(train, calibration),
                                    target[reference], observed[reference], config, model_dir / 'prior',
                                    period, 'prior', threads, progress)
    raw, full_report = fit_stage(core, train, query, target, observed, config, model_dir / 'full',
                                 period, 'full', threads, progress)
    truth, cal_origins = target[calibration], origins[calibration].asi8
    offset, rows = core.estimate_offset(prior, truth, config)
    state = {'policy': core.POLICY, 'reference_rows': len(reference), 'calibration_rows': len(calibration),
             'reference_start': str(origins[reference].min()), 'reference_end': str(origins[reference].max()),
             'calibration_start': str(origins[calibration].min()), 'calibration_end': str(origins[calibration].max()),
             'split_boundary': str(boundary), 'purge_hours': 144, 'window': 336, 'alpha': .25,
             'scope': 'horizon', 'applied': True, 'offset_rows': rows, 'offset': offset.tolist(),
             'offset_start': str(origins[calibration[-rows]]), 'offset_end': str(origins[calibration[-1]])}
    core.write_json(model_dir / 'frozen_calibration.json', state)
    np.savez_compressed(model_dir / 'fitting_subsets.npz', reference=reference, calibration=calibration,
                        offset=calibration[-336:], full=np.arange(len(origins)), train_origins=origins.asi8)
    np.savez_compressed(model_dir / 'prior_predictions.npz', raw=prior, spectral=core.component_values(prior, config),
                        training_truth=truth, origins=cal_origins)
    final, no_offset = core.apply_offset(raw, offset, config)
    component = core.component_values(raw, config)
    query_origins = query['origins'].asi8
    np.savez_compressed(out / f'{period}_predictions.npz', pred=final, origins=query_origins)
    np.savez_compressed(out / f'{period}_no_offset_predictions.npz', pred=no_offset, origins=query_origins)
    np.savez_compressed(out / f'{period}_raw_predictions.npz', raw=raw, component=component,
                        preclip=component + offset[None, :], offset=offset, origins=query_origins)
    report = {'experiment': config['experiment'], 'config': config, 'period': period, 'seed': config['seed'],
              'parameters': dict(baseline.PARAMS, seed=config['seed'], nthread=threads),
              'clipping': {'component_cells': int(np.count_nonzero(raw != component)),
                           'final_cells': int(np.count_nonzero(component + offset[None, :] != final))},
              'rounds': 900, 'smoke': False, 'fresh_training': True, 'reused_model_or_predictions': False,
              'row_weight_policy': POLICY, 'stages': {'prior': prior_report, 'full': full_report},
              'row_weight_statistics': {'prior': prior_report['weight_statistics'], 'full': full_report['weight_statistics']},
              'features': names, 'feature_count': 28, 'expanded_feature_count': 212,
              'expanded_features': {str(s): core.expanded_names(names, s, 24) for s in (0, 24, 48)},
              'training_cutoff': baseline.CUTOFFS[period], 'maturity_hours': 144,
              'train_origins': len(origins), 'train_start': str(origins.min()), 'train_end': str(origins.max()),
              'query_origins': len(query_origins), 'query_start': str(query['origins'].min()),
              'query_end': str(query['origins'].max()), 'frozen_state': state, 'identity': identity}
    core.write_json(out / f'{period}_metadata.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--original-labels', type=Path)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), required=True)
    parser.add_argument('--threads', type=int, choices=(4,), required=True)
    args = parser.parse_args(argv)
    if args.experiment != f'timeweight180_s{args.seed}':
        parser.error('experiment must be timeweight180_s followed by the selected seed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fresh CPU GBM requires CUDA_VISIBLE_DEVICES=""')
    if args.original_labels is None:
        args.original_labels = args.data.with_name('df_ffill.parquet')
    baseline, core = parent_modules()
    config = fixed_config(core, args.seed, args.experiment)
    names = assert_parent_contract(core, config, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('F*_models')) or (args.out / 'complete.json').exists():
        raise ValueError('Output contains models; use a fresh run directory')
    git = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True, check=False)
    campaign_provenance = json.loads((CAMPAIGN / 'provenance.json').read_text())
    identity = {'data': core.file_identity(args.data), 'original_labels': core.file_identity(args.original_labels),
                'experiment': core.file_identity(__file__), 'baseline': core.file_identity(baseline.__file__),
                'shared_core': core.file_identity(core.__file__), 'git_commit': git.stdout.strip() or None,
                'actual_git_commit': git.stdout.strip() or None, 'source_commit': campaign_provenance['commit'],
                'campaign_provenance': campaign_provenance, 'git_status': dirty.stdout,
                'fresh_training': True, 'reused_model_or_predictions': False, 'seed': args.seed,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'pandas': pd.__version__, 'xgboost': xgb.__version__},
                'started_utc': datetime.now(UTC).isoformat()}
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), Path(baseline.__file__), Path(core.__file__)]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    core.write_json(args.out / 'resolved_config.json', config)
    core.write_json(args.out / 'provenance.json', identity)
    core.write_json(args.out / 'feature_contract.json', {'features': names, 'unchanged_feature_names': True,
                    'changed_values': [], 'parent': f'drop035_v650_s{args.seed}', 'row_weight_policy': POLICY})
    for period in baseline.CUTOFFS:
        print(json.dumps({'period': period, 'stage': 'fresh_training', 'unit': 0, 'units': 6}), flush=True)
        run_period(core, config, args.data, args.original_labels, args.out, period, args.threads, identity)
    core.write_json(args.out / 'predictions_frozen.json', {
        period: core.file_identity(args.out / f'{period}_predictions.npz') for period in baseline.CUTOFFS})
    score = score_outputs(core, args.out, args.observations)
    core.write_json(args.out / 'complete.json', {'status': 'complete', 'exit_code': 0, 'seed': args.seed,
                    'fresh_training': True, 'reused_model_or_predictions': False, 'publish': 'not_required',
                    'publish_reason': 'pending matched multi-seed gate; local research only',
                    'total_s': score['total']['s'], 'completed_utc': datetime.now(UTC).isoformat()})
    print(json.dumps({'stage': 'complete', 'seed': args.seed, 'total_s': score['total']['s']}), flush=True)


if __name__ == '__main__':
    main()
