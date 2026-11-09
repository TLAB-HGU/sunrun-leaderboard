"""Fresh retained-parent full24 spectral training with one_output_per_tree only."""
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
    baseline = load_module(PARENT / 'baseline.py', 'output_strategy_baseline')
    previous = sys.modules.get('baseline')
    sys.modules['baseline'] = baseline
    try:
        core = load_module(PARENT / 'core.py', 'output_strategy_core')
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


STRATEGY = 'one_output_per_tree'
PARAMETER_DIFF = {'multi_strategy': {'parent': 'multi_output_tree', 'treatment': STRATEGY}}


def configure_strategy(baseline, core, config, threads):
    if 'reuse_run' in config:
        raise ValueError('Fresh output strategy never reuses fitted state')
    lock = json.loads((CAMPAIGN / 'feature-lock.json').read_text())
    names = assert_parent_contract(core, config, config['seed'])
    if names != lock['source_features'] or any(core.expanded_names(names, start, 24) != lock['expanded_features'][str(start)] for start in (0, 24, 48)):
        raise ValueError('Retained feature lock differs')
    parent = dict(baseline.PARAMS, seed=config['seed'], nthread=threads)
    if parent != dict(lock['parent_parameters'], seed=config['seed'], nthread=threads):
        raise ValueError('Parent parameter lock differs')
    treatment = dict(parent, multi_strategy=STRATEGY)
    if treatment != dict(lock['treatment_parameters'], seed=config['seed'], nthread=threads):
        raise ValueError('Treatment parameter lock differs')
    # Only the freshly imported private baseline module changes; no disk/global source edit.
    baseline.PARAMS = dict(baseline.PARAMS, multi_strategy=STRATEGY)
    return {'parent_parameters': parent, 'treatment_parameters': treatment,
            'strategy_parameter_diff': PARAMETER_DIFF}


def training_subsets(baseline, grid, target, parent, period, smoke):
    origins = grid['origins']
    if (origins.hasnans or origins.has_duplicates or not origins.is_monotonic_increasing
            or origins.tz is None or not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(baseline.CUTOFFS[period], tz='UTC')).all()):
        raise ValueError('Original training origin order/maturity violated')
    if (len(origins) != parent['train_origins'] or str(origins.min()) != parent['train_start']
            or str(origins.max()) != parent['train_end'] or parent['training_cutoff'] != baseline.CUTOFFS[period]
            or parent['maturity_hours'] != 144 or target.shape != (len(origins), 72) or not np.isfinite(target).all()):
        raise ValueError('Original training rows, targets or cutoff changed')
    reference, calibration, _ = baseline.prior_split(origins)
    for name, indices in [('reference', reference), ('calibration', calibration)]:
        frozen = parent['frozen_state']
        if (len(indices) != frozen[f'{name}_rows'] or str(origins[indices].min()) != frozen[f'{name}_start']
                or str(origins[indices].max()) != frozen[f'{name}_end']):
            raise ValueError('Original prior/calibration split changed')
    full = np.arange(len(origins))
    if smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 128)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 64)).astype(int)]
        full = np.linspace(0, len(origins) - 1, min(len(origins), 192)).astype(int)
    return {'train_origins': origins.asi8, 'reference': reference, 'calibration': calibration,
            'full': full, 'offset': calibration[-min(336, len(calibration)):]}


def traced_objective(original, target, config, expected, trace, proof_path):
    if target.ndim != 2 or target.shape[1] != 24 or not np.array_equal(target, expected):
        raise ValueError('Objective must retain the exact full stage Nx24 training targets')
    parent_objective = original(target, config)
    trace.update(expected_shape=list(target.shape), calls=0, matrix_columns=212,
                 base_score=float(target.mean()), all_callbacks_full_matrix=True,
                 gradient_dtype='float32', hessian_dtype='float32', dmatrix_weight_count=0)

    def objective(prediction, matrix):
        if (np.asarray(prediction).shape != target.shape or not np.isfinite(prediction).all()
                or matrix.num_row() != len(target) or matrix.num_col() != 212 or matrix.get_weight().size):
            raise ValueError('Full24 callback shape/input or unweighted DMatrix contract violated')
        gradient, hessian = parent_objective(prediction, matrix)
        if (gradient.shape != target.shape or hessian.shape != target.shape
                or gradient.dtype != np.float32 or hessian.dtype != np.float32
                or not np.isfinite(gradient).all() or not np.isfinite(hessian).all() or (hessian < 0).any()):
            raise ValueError('Retained full24 objective derivative contract violated')
        if trace['calls'] == 0:
            np.testing.assert_array_equal(matrix.get_label().reshape(target.shape), target.astype(np.float32))
            np.testing.assert_array_equal(prediction, np.full(target.shape, np.float32(target.mean())))
            rows = np.unique(np.linspace(0, len(target) - 1, min(8, len(target))).astype(int))
            np.savez_compressed(proof_path, sample_indices=rows, prediction=np.asarray(prediction)[rows],
                target=target[rows], gradient=gradient[rows], hessian=hessian[rows],
                tail_weight_normalizer=np.where(target >= 550., 4., 1.).mean(axis=0))
        trace['calls'] += 1
        return gradient, hessian  # Parent arrays returned unchanged, no reweighting.
    return objective


def model_structure(path, rounds, threads):
    model = xgb.Booster(params={'device': 'cpu', 'nthread': threads}, model_file=path)
    runtime = json.loads(model.save_config())['learner']
    model_params = runtime['learner_model_param']
    tree_params = runtime['gradient_booster']['gbtree_model_param']
    structure = {'rounds': model.num_boosted_rounds(), 'scalar_trees': int(tree_params['num_trees']),
                 'num_target': int(model_params['num_target']), 'num_feature': model.num_features(),
                 'num_parallel_tree': int(tree_params['num_parallel_tree']),
                 'multi_strategy': runtime['learner_train_param']['multi_strategy']}
    if structure != {'rounds': rounds, 'scalar_trees': rounds * 24, 'num_target': 24,
                     'num_feature': 212, 'num_parallel_tree': 1, 'multi_strategy': STRATEGY}:
        raise ValueError('Saved scalar strategy tree/round/target/input structure differs')
    return structure


def run_period(core, config, data, out, period, threads, smoke, identity):
    if 'reuse_run' in config or core.baseline.PARAMS['multi_strategy'] != STRATEGY:
        raise ValueError('Only fresh scalar output-strategy fitting is permitted')
    parent = json.loads((PARENT / 'runs' / f"drop035_v650_s{config['seed']}" / f'{period}_metadata.json').read_text())
    original_load, original_objective = core.baseline.load_grid, core.make_objective
    original_train = core.xgb.train
    captured, traces = {}, []
    rounds = 2 if smoke else 900
    model_dir = out / f'{period}_models'

    def load_grid(source, requested_period, names, training):
        grid, target = original_load(source, requested_period, names, training)
        if training:
            captured['subsets'] = training_subsets(core.baseline, grid, target, parent, period, smoke)
            captured['target'] = target
        elif target is not None:
            raise ValueError('Query loader returned evaluation targets')
        return grid, target

    def make_objective(target, received_config):
        position = len(traces)
        if position >= 6 or received_config != config:
            raise ValueError('Unexpected number/configuration of full24 objective constructions')
        stage, block = ('prior' if position < 3 else 'full'), position % 3
        indices = captured['subsets']['reference' if stage == 'prior' else 'full']
        expected = captured['target'][indices, block * 24:(block + 1) * 24]
        trace = {'period': period, 'stage': stage, 'block': block, 'expected_calls': rounds,
                 'first_input_proof': str(model_dir / stage / f'b{block}_objective_first.npz')}
        traces.append(trace)
        return traced_objective(original_objective, target, config, expected, trace, Path(trace['first_input_proof']))

    def train(*args, **kwargs):
        model = original_train(*args, **kwargs)
        live_config = json.loads(model.save_config())
        learner = live_config['learner']
        model_params = learner['learner_model_param']
        if (learner['learner_train_param']['multi_strategy'] != STRATEGY
                or int(model_params['num_target']) != 24
                or int(model_params['num_feature']) != 212):
            raise ValueError('Live trained strategy/target/input configuration differs')
        trace = traces[-1]
        live_path = model_dir / trace['stage'] / f"b{trace['block']}_live_config.json"
        core.write_json(live_path, live_config)
        trace['live_config'] = str(live_path)
        return model

    core.baseline.load_grid, core.make_objective = load_grid, make_objective
    core.xgb.train = train
    try:
        report = core.run_period(config, data, out, period, threads, smoke, identity)
    finally:
        core.baseline.load_grid, core.make_objective = original_load, original_objective
        core.xgb.train = original_train
    if len(traces) != 6 or any(trace['calls'] != rounds for trace in traces):
        raise ValueError('Every booster requires one full Nx24 callback per boosting iteration')
    np.savez_compressed(model_dir / 'fitting_subsets.npz', **captured['subsets'])
    structures = []
    for trace in traces:
        core.write_json(model_dir / trace['stage'] / f"b{trace['block']}_objective_trace.json", trace)
        structures.append({'stage': trace['stage'], 'block': trace['block'],
                           **model_structure(model_dir / trace['stage'] / f"b{trace['block']}.ubj", rounds, threads)})
    report.update(fresh_training=True, reused_model_or_predictions=False, multi_strategy=STRATEGY,
                  strategy_parameter_diff=PARAMETER_DIFF, objective_traces=traces, model_structure=structures,
                  parent_parameters=parent['parameters'], treatment_parameters=report['parameters'])
    core.write_json(out / f'{period}_metadata.json', report)
    return report


def score_outputs(core, out, observations):
    # Check all frozen period receipts BEFORE loading the scorer/official truth.
    frozen = json.loads((out / 'predictions_frozen.json').read_text())
    if set(frozen) != {'F1', 'F2', 'F3'}:
        raise ValueError('All three forecast periods must freeze before scoring')
    for period in ('F1', 'F2', 'F3'):
        meta = json.loads((out / f'{period}_metadata.json').read_text())
        if meta['smoke'] or not meta.get('fresh_training') or meta.get('reused_model_or_predictions') is not False:
            raise ValueError('Only completed fresh nonsmoke forecasts may be scored')
        if core.file_identity(out / f'{period}_predictions.npz') != frozen[period]:
            raise ValueError('Frozen forecast receipt changed before scoring')
    frame = core.prediction_frame(out)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    scorer = load_module(REPO / 'leaderboard/scorer/competition.py', 'output_strategy_official_scorer')
    context = scorer.prepare_context(scorer.load_observations(observations))
    score = scorer.score_competition(frame, context)
    aligned = scorer.align_predictions(frame, context)
    for scope in ('total', 'f1', 'f2', 'f3'):
        if score[scope]['n_replaced']:
            raise ValueError('Official scorer replaced forecasts')
        mask = np.ones(len(aligned), bool) if scope == 'total' else context['period'] == scope.upper()
        score[scope]['mae'] = float(np.nanmean(abs(aligned[mask] - context['truth'][mask])))
    result = {'competition': score, 'observations': core.file_identity(observations),
              'scorer': core.file_identity(scorer.__file__), 'scorer_version': scorer.VERSION}
    core.write_json(out / 'local_score.json', result)
    core.write_json(out / 'submission.json', dict(result, local_only=True, published=False))
    return score


def source_provenance(paths, smoke):
    git = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, check=False)
    actual = git.stdout.decode().strip() if git.returncode == 0 else None
    receipts = []
    for path in paths:
        relative = path.resolve().relative_to(REPO)
        committed = subprocess.run(['git', '-C', str(REPO), 'show', f'{actual}:{relative.as_posix()}'], capture_output=True, check=False)
        equal = committed.returncode == 0 and committed.stdout == path.read_bytes()
        receipts.append({'path': str(path.resolve()), 'repository_path': relative.as_posix(),
                         'commit': actual, 'head_bytes_equal': equal, 'bytes': path.stat().st_size})
    if not smoke and (actual is None or not all(receipt['head_bytes_equal'] for receipt in receipts)):
        raise ValueError('Actual fitting sources/contracts must exactly match executed Git HEAD')
    status = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True, check=False)
    return {'actual_git_commit': actual, 'git_commit': actual, 'git_status': status.stdout,
            'executed_source_head_proof': receipts, 'source_proof_required_for_actual_fit': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--period', choices=('F1', 'F2', 'F3'))
    args = parser.parse_args(argv)
    if args.experiment != f'output_strategy_s{args.seed}':
        parser.error('experiment must be output_strategy_s followed by the selected seed')
    if args.period and not args.smoke:
        parser.error('A single period may only be requested for a nonscoring smoke')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.threads != (1 if args.smoke else 4) or xgb.__version__ != '2.1.4':
        raise ValueError('Pinned CPU fitting requires CUDA invisible, threads4 actual/1 smoke, XGBoost2.1.4')
    baseline, core = parent_modules()
    config = fixed_config(core, args.seed, args.experiment)
    parameter_proof = configure_strategy(baseline, core, config, args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob('F*_models')) or (args.out / 'complete.json').exists():
        raise ValueError('Output contains fitted artifacts; use a fresh run directory')
    sources = [Path(__file__), Path(baseline.__file__), Path(core.__file__), CAMPAIGN / 'strategy.md',
               CAMPAIGN / 'evaluation-contract.json', CAMPAIGN / 'feature-lock.json']
    identity = source_provenance(sources, args.smoke)
    campaign = json.loads((CAMPAIGN / 'provenance.json').read_text())
    identity.update(source_commit=campaign['commit'], campaign_provenance=campaign,
                    data=core.file_identity(args.data), experiment=core.file_identity(__file__),
                    baseline=core.file_identity(baseline.__file__), shared_core=core.file_identity(core.__file__),
                    seed=args.seed, fresh_training=True, reused_model_or_predictions=False,
                    versions={'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgb.__version__},
                    started_utc=datetime.now(UTC).isoformat())
    snapshots = args.out / 'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for path in sources:
        (snapshots / path.name).write_bytes(path.read_bytes())
    core.write_json(args.out / 'provenance.json', identity)
    core.write_json(args.out / 'resolved_config.json', config)
    core.write_json(args.out / 'parameter_diff.json', parameter_proof)
    core.write_json(args.out / 'feature_contract.json', json.loads((CAMPAIGN / 'feature-lock.json').read_text()))
    periods = (args.period,) if args.period else ('F1', 'F2', 'F3')
    for period in periods:
        print(json.dumps({'period': period, 'stage': 'fresh_training', 'unit': 0, 'units': 6}), flush=True)
        run_period(core, config, args.data, args.out, period, args.threads, args.smoke, identity)
    core.write_json(args.out / 'predictions_frozen.json', {
        period: core.file_identity(args.out / f'{period}_predictions.npz') for period in periods})
    completion = {'status': 'smoke_complete' if args.smoke else 'complete', 'exit_code': 0, 'seed': args.seed,
                  'smoke': args.smoke, 'fresh_training': True, 'reused_model_or_predictions': False,
                  'multi_strategy': STRATEGY, 'publish': 'not_required',
                  'publish_reason': 'nonscoring smoke' if args.smoke else 'pending independent matched-seed gate; no external publication',
                  'completed_utc': datetime.now(UTC).isoformat()}
    if not args.smoke:
        score = score_outputs(core, args.out, args.observations)
        completion['total_s'] = score['total']['s']
    core.write_json(args.out / 'complete.json', completion)
    print(json.dumps({'stage': completion['status'], 'seed': args.seed}), flush=True)


if __name__ == '__main__':
    main()
