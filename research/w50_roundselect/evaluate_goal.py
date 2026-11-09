"""Read-only, fail-closed official-S goal gate; never fits or modifies goal state.

Exit 0 PASS requires all three seeds <= the fixed 50% threshold plus verified
scientific integrity. Exit 1 FAIL means complete forecasts freshly scored above
that threshold. Exit 2 PARTIAL means incomplete forecasts or missing evidence.

The independent audit schema is explicit: status='pass', integrity_pass=true,
reviewer, unresolved_required_gaps=[], checks keyed by REQUIRED_CHECKS with
passed=true and nonempty evidence paths, and inputs containing current path /
bytes / mtime_ns receipts for all files returned by integrity_files(). Missing
fields never default to success. Receipts use no hashes. Audit checks include
saved-booster replay; this lightweight evaluator does not import XGBoost.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PARENT = ROOT.parent / 'drop035_v650_confirm_20261108'
OBSERVATIONS = ROOT.parent / 'offset_roll_complete_ablation_20261106/competition-observations.parquet'
SCORER = REPO / 'leaderboard/scorer/competition.py'
SEEDS = (42, 43, 44)
PERIODS = ('F1', 'F2', 'F3')
SCOPES = ('total', 'f1', 'f2', 'f3')
QUERY_COUNTS = {'F1': 2089, 'F2': 2113, 'F3': 2113}
BEST_SINGLE_S = 97.49244208059969
HALF_S_TARGET = 48.746221040299844
REQUIRED_CHECKS = (
    'causal_fitting', 'fresh_training', 'single_factor_diff', 'feature_identity',
    'official_score_replay', 'model_replay', 'provenance_complete', 'device_policy',
    'preregistration',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def identity(path):
    path = path.resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def current_receipts(receipts, required):
    require(isinstance(receipts, list), 'identity receipt list missing')
    indexed = {}
    for receipt in receipts:
        path = Path(receipt['path']).resolve()
        require(path not in indexed, f'duplicate identity receipt: {path}')
        require(identity(path) == receipt, f'stale identity receipt: {path}')
        indexed[path] = receipt
    require({path.resolve() for path in required} <= set(indexed),
            'identity receipts do not cover all required artifacts')


def completed_runs(root):
    runs = []
    for seed in SEEDS:
        run = root / 'runs' / f'roundselect_s{seed}'
        require((run / 'train.exit').is_file(), f'{run.name}: waiting for training completion')
        require((run / 'train.exit').read_text().strip() == '0',
                f'{run.name}: training has not exited successfully')
        complete = read_json(run / 'complete.json')
        require(complete.get('status') == 'complete' and complete.get('exit_code') == 0
                and complete.get('seed') == seed, f'{run.name}: incomplete completion receipt')
        for period in PERIODS:
            meta = read_json(run / f'{period}_metadata.json')
            require(meta.get('seed') == seed and meta.get('period') == period
                    and meta.get('smoke') is False, f'{run.name}/{period}: incomplete metadata')
            require((run / f'{period}_predictions.npz').is_file(),
                    f'{run.name}/{period}: missing predictions')
        require((run / 'predictions.parquet').is_file(), f'{run.name}: missing combined forecasts')
        runs.append(run)
    return runs


def prediction_frame(run):
    """Validate all period arrays and their combined copy before truth access."""
    frames = []
    for period in PERIODS:
        with np.load(run / f'{period}_predictions.npz', allow_pickle=False) as saved:
            origins = pd.DatetimeIndex(saved['origins'], tz='UTC')
            values = saved['pred']
        require(len(origins) == QUERY_COUNTS[period]
                and values.shape == (len(origins), 72), f'{run.name}/{period}: incomplete grid')
        require(not origins.hasnans and not origins.has_duplicates
                and origins.is_monotonic_increasing, f'{run.name}/{period}: invalid origins')
        require(np.isfinite(values).all() and (values > 0).all(),
                f'{run.name}/{period}: invalid predictions')
        frames.append(pd.DataFrame({
            'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
            'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
            'pred_kms': values.ravel(),
        }))
    frame = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(run / 'predictions.parquet'),
                                  check_dtype=False, check_exact=True)
    return frame


def score_frame(scorer, context, frame):
    score = scorer.score_competition(frame, context)
    aligned = scorer.align_predictions(frame, context)
    for scope in SCOPES:
        require(score[scope]['n_replaced'] == 0, f'{scope}: official scorer replaced forecasts')
        for metric in ('s', 'rmse', 'trmse', 'timse'):
            value = score[scope][metric]
            require(math.isfinite(value) and value >= 0, f'{scope}/{metric}: invalid score')
        mask = np.ones(len(aligned), bool) if scope == 'total' else context['period'] == scope.upper()
        score[scope]['pooled_mae'] = float(np.nanmean(abs(aligned[mask] - context['truth'][mask])))
    return score


def fresh_scores(frames):
    spec = importlib.util.spec_from_file_location('roundselect_goal_official_scorer', SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    require(scorer.VERSION == 'lead-timse-6h-v1', 'official scorer version changed')
    context = scorer.prepare_context(scorer.load_observations(OBSERVATIONS))
    return [score_frame(scorer, context, frame) for frame in frames], scorer, context


def control_and_metadata(run, seed, period):
    meta = read_json(run / f'{period}_metadata.json')
    require(meta.get('fresh_training') is True and meta.get('reused_model_or_predictions') is False,
            f'{run.name}/{period}: fresh training not established')
    require(meta.get('prior_rounds') == 900, f'{run.name}/{period}: prior budget changed')
    counts = meta['selected_rounds']
    require(isinstance(counts, list) and len(counts) == 3
            and all(type(n) is int and 1 <= n <= 900 for n in counts), 'invalid selected counts')
    model_dir = run / f'{period}_models'
    heads = read_json(model_dir / 'tree_selection.json')
    require(heads == meta['tree_selection'] and len(heads) == 3, 'tree selection receipt differs')
    for block, (head, count) in enumerate(zip(heads, counts)):
        curve = np.asarray(head['native_rmse'], dtype=float)
        require(curve.shape == (900,) and np.isfinite(curve).all(), 'invalid selection curve')
        require(count == int(np.argmin(curve)) + 1 and head['block'] == block
                and head['selected_rounds'] == count and head['full_rounds'] == count
                and head['prior_rounds'] == 900 and head['inference_iteration_range'] == [0, count],
                'selected prefix/full-model metadata mismatch')
        for stage in ('prior', 'full'):
            model = model_dir / stage / f'b{block}.ubj'
            require(model.is_file() and model.stat().st_size > 0, f'missing model: {model}')
    parent = PARENT / 'runs' / f'drop035_v650_s{seed}'
    original = read_json(parent / f'{period}_metadata.json')
    for key in ('features', 'expanded_features', 'feature_count', 'expanded_feature_count',
                'training_cutoff', 'maturity_hours', 'train_origins', 'train_start', 'train_end'):
        require(meta[key] == original[key], f'{run.name}/{period}: parent {key} differs')
    require({k: v for k, v in meta['config'].items() if k != 'experiment'} ==
            {k: v for k, v in original['config'].items() if k != 'experiment'}, 'parent config differs')
    frozen_path = (run.parents[1] / 'frozen_controls' / f'drop035_v650_s{seed}' /
                   f'{period}_models/prior_predictions.npz')
    with np.load(model_dir / 'prior900_predictions.npz', allow_pickle=False) as fresh, \
            np.load(frozen_path, allow_pickle=False) as frozen:
        for key in ('origins', 'training_truth'):
            require(np.array_equal(fresh[key], frozen[key]), f'prior900 control {key} mismatch')
        require(fresh['raw'].shape == frozen['raw'].shape and np.isfinite(fresh['raw']).all(),
                'prior900 control shape or finiteness mismatch')
        difference = float(np.max(abs(fresh['raw'] - frozen['raw'])))
        require(math.isfinite(difference) and difference <= 1e-6, 'prior900 endpoint control failed')
    control = read_json(model_dir / 'control.json')
    require(control == meta['control'] and control.get('passed') is True
            and control.get('tolerance') == 1e-6
            and control.get('max_absolute_difference') == difference, 'control receipt mismatch')


def integrity_files(root, runs):
    files = [root / 'results.json', root / 'bootstrap.npz', root / 'strategy.md',
             root / 'experiment.py', SCORER, OBSERVATIONS]
    for seed, run in zip(SEEDS, runs):
        files.extend(run / name for name in ('train.exit', 'complete.json', 'predictions.parquet',
                                             'provenance.json', 'resolved_config.json'))
        parent = PARENT / 'runs' / f'drop035_v650_s{seed}'
        files.append(parent / 'predictions.parquet')
        for period in PERIODS:
            files.extend([run / f'{period}_predictions.npz', run / f'{period}_metadata.json',
                          parent / f'{period}_metadata.json',
                          root / 'frozen_controls' / f'drop035_v650_s{seed}' /
                          f'{period}_models/prior_predictions.npz'])
            model_dir = run / f'{period}_models'
            files.extend(model_dir / name for name in ('control.json', 'tree_selection.json',
                         'prior900_predictions.npz', 'prior_predictions.npz',
                         'fitting_subsets.npz', 'frozen_calibration.json'))
            files.extend(model_dir / stage / f'b{block}.ubj'
                         for stage in ('prior', 'full') for block in range(3))
    return files


def verify_evidence(root, runs, treatment, scorer, context):
    result = read_json(root / 'results.json')
    require(result['status'] == 'complete' and result['seeds'] == list(SEEDS), 'analysis incomplete')
    require(result['best_reference_s'] == BEST_SINGLE_S and result['half_s_target'] == HALF_S_TARGET,
            'frozen reference or 50% target changed')
    bootstrap = result['bootstrap']
    for key, value in {'replicates': 2000, 'block_hours': 168, 'seed': 20261104,
                       'period_stratified': True, 'empirical_timse_quantiles_recomputed': True}.items():
        require(bootstrap.get(key) == value, f'bootstrap {key} mismatch')
    with np.load(root / 'bootstrap.npz', allow_pickle=False) as saved:
        draws = saved['reduction']
    require(draws.shape == (2000,) and np.isfinite(draws).all(), 'invalid bootstrap population')
    ci = np.asarray(result['paired_reduction_ci95'], dtype=float)
    require(ci.shape == (2,) and np.isfinite(ci).all() and ci[0] > 0 and ci[0] <= ci[1]
            and np.allclose(ci, np.quantile(draws, [.025, .975]), rtol=0, atol=1e-12),
            'positive paired 95% interval not verified')
    control = [score_frame(scorer, context, pd.read_parquet(
        PARENT / 'runs' / f'drop035_v650_s{seed}' / 'predictions.parquet')) for seed in SEEDS]
    require(abs(control[0]['total']['s'] - BEST_SINGLE_S) <= 1e-8, 'frozen best no longer reproduces')
    for arm, scores in [('control', control), ('treatment', treatment)]:
        for scope in SCOPES:
            for metric in ('s', 'rmse', 'trmse', 'timse', 'pooled_mae'):
                values = np.asarray([score[scope][metric] for score in scores])
                summary = result['summaries'][arm][scope][metric]
                recorded = np.asarray(summary['per_seed'], dtype=float)
                require(recorded.shape == (3,) and np.allclose(values, recorded, rtol=0, atol=1e-8)
                        and abs(float(values.mean()) - summary['mean']) <= 1e-8,
                        f'analysis disagrees with fresh scores: {arm}/{scope}/{metric}')
    gains = 1 - np.array([s['total']['s'] for s in treatment]) / np.array([s['total']['s'] for s in control])
    require((gains > 0).all() and gains.mean() >= .01, 'matched candidate gate failed')
    for metric in ('rmse', 'pooled_mae'):
        require(np.mean([s['total'][metric] for s in treatment]) <=
                1.01 * np.mean([s['total'][metric] for s in control]), f'auxiliary {metric} gate failed')
    for period in ('f1', 'f2', 'f3'):
        require(np.mean([s[period]['s'] for s in treatment]) <=
                1.02 * np.mean([s[period]['s'] for s in control]), f'{period} auxiliary gate failed')
    require(result.get('candidate_gate_passed') is True and result.get('auxiliary_gate_passed') is True,
            'analysis gates did not pass')
    prediction_files = [run / 'predictions.parquet' for run in runs] + [
        PARENT / 'runs' / f'drop035_v650_s{seed}' / 'predictions.parquet' for seed in SEEDS]
    current_receipts(result['inputs'] + [result['observations'], result['scorer']],
                     prediction_files + [OBSERVATIONS, SCORER])
    for seed, run in zip(SEEDS, runs):
        for period in PERIODS:
            control_and_metadata(run, seed, period)
    audit = read_json(root / 'verification-independent.json')
    require(audit.get('status') == 'pass' and audit.get('integrity_pass') is True,
            'independent integrity audit did not pass')
    require(bool(audit.get('reviewer')) and audit.get('unresolved_required_gaps') == [],
            'independent reviewer or explicit gap clearance missing')
    require(not audit.get('gaps'), 'independent audit still lists gaps')
    for name in REQUIRED_CHECKS:
        check = audit['checks'][name]
        require(check.get('passed') is True and isinstance(check.get('evidence'), list)
                and bool(check['evidence']), f'{name}: independent evidence missing')
        for entry in check['evidence']:
            path = Path(entry)
            require((path if path.is_absolute() else root / path).is_file(),
                    f'{name}: evidence artifact absent: {path}')
    current_receipts(audit['inputs'], integrity_files(root, runs))


def evaluate(root):
    report = {'verdict': 'PARTIAL', 'exit_code': 2, 'goal_achieved': False,
              'fixed_best_single_s': BEST_SINGLE_S, 'required_max_s': HALF_S_TARGET,
              'seeds': list(SEEDS), 'all_seed_scores': [], 'failures': [], 'gaps': [],
              'fresh_official_scoring': False, 'workflow_state_modified': False}
    try:
        runs = completed_runs(root)
        paths = [run / name for run in runs for name in
                 ['train.exit', 'complete.json', 'predictions.parquet', *[
                     f'{p}_{suffix}' for p in PERIODS for suffix in ('metadata.json', 'predictions.npz')]]]
        before = [identity(path) for path in paths + [OBSERVATIONS, SCORER]]
        frames = [prediction_frame(run) for run in runs]
        treatment, scorer, context = fresh_scores(frames)
        current_receipts(before, paths + [OBSERVATIONS, SCORER])
        values = [score['total']['s'] for score in treatment]
        report.update(all_seed_scores=values, fresh_official_scoring=True)
        failures = [f'seed {seed}: S={value:.15g} > {HALF_S_TARGET:.15g}'
                    for seed, value in zip(SEEDS, values) if value > HALF_S_TARGET]
        if failures:
            report.update(verdict='FAIL', exit_code=1, failures=failures)
            return report
        verify_evidence(root, runs, treatment, scorer, context)
        current_receipts(before, paths + [OBSERVATIONS, SCORER])
        report.update(verdict='PASS', exit_code=0, goal_achieved=True)
    except Exception as error:  # noqa: BLE001 - every evidence error must fail closed as PARTIAL
        # Malformed/corrupt evidence is PARTIAL, never an uncaught exit-1 score FAIL.
        report['gaps'].append(f'{type(error).__name__}: {error}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = evaluate(args.root.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
