"""Official matched-seed analysis, gated on all fresh fits having completed.

This reporting process reads evaluation truth only after every treatment forecast
has frozen. It never writes fitting inputs or fitted model state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASELINE = ROOT.parent / 'drop035_v650_confirm_20261108'
OBS = ROOT.parent / 'offset_roll_complete_ablation_20261106/competition-observations.parquet'
SEEDS = (42, 43, 44)
SCOPES = ('total', 'f1', 'f2', 'f3')
METRICS = ('s', 'rmse', 'trmse', 'timse', 'pooled_mae')
BEST_SINGLE_S = 97.49244208059969
HALF_S_TARGET = BEST_SINGLE_S / 2


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def describe(values, lower_is_worse=False):
    values = np.asarray(values, dtype=float)
    return {'per_seed': values.tolist(), 'mean': float(values.mean()),
            'sample_variance': float(values.var(ddof=1)),
            'sample_sd': float(values.std(ddof=1)),
            'worst': float(values.min() if lower_is_worse else values.max())}


def completed_runs(root):
    """Check all treatment artifacts before any official observations are opened."""
    runs = []
    for seed in SEEDS:
        run = root / 'runs' / f'age_retrain_s{seed}'
        if (run / 'train.exit').read_text().strip() != '0':
            raise ValueError(f'{run.name}: unsuccessful training exit')
        complete = json.loads((run / 'complete.json').read_text())
        if complete.get('status') != 'complete' or complete.get('fresh_training') is not True:
            raise ValueError(f'{run.name}: successful completion receipt missing')
        for name in ('predictions.parquet', 'provenance.json', 'local_score.json'):
            if not (run / name).is_file():
                raise ValueError(f'{run.name}: missing {name}')
        for period in ('F1', 'F2', 'F3'):
            meta = json.loads((run / f'{period}_metadata.json').read_text())
            if (meta.get('smoke') or meta.get('seed') != seed or meta.get('rounds') != 900
                    or meta.get('fresh_training') is not True
                    or meta.get('reused_model_or_predictions') is not False):
                raise ValueError(f'{run.name}/{period}: incomplete or mismatched fit')
            if not (run / f'{period}_predictions.npz').is_file():
                raise ValueError(f'{run.name}/{period}: missing period forecasts')
        runs.append(run)
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--observations', type=Path, default=OBS)
    args = parser.parse_args(argv)
    runs = completed_runs(args.root)
    for seed, run in zip(SEEDS, runs):
        for period in ('F1', 'F2', 'F3'):
            treatment_meta = json.loads((run / f'{period}_metadata.json').read_text())
            control_meta = json.loads((args.baseline / 'runs' / f'drop035_v650_s{seed}'
                                       / f'{period}_metadata.json').read_text())
            for key in ('features', 'expanded_features', 'training_cutoff', 'maturity_hours', 'rounds'):
                if treatment_meta[key] != control_meta[key]:
                    raise ValueError(f'{run.name}/{period}: baseline {key} changed')
    scorer = module(REPO / 'leaderboard/scorer/competition.py', 'age_training_scorer')
    stats = module(ROOT.parent / 'offset_roll_complete_ablation_20261106/analyze_results.py',
                   'age_training_bootstrap')
    # Every treatment's three-period predictions are complete before this read.
    context = scorer.prepare_context(scorer.load_observations(args.observations))
    scores = {'control': [], 'treatment': []}
    arrays, inputs = [], []
    for arm, arm_scores in scores.items():
        for seed, treatment in zip(SEEDS, runs):
            run = args.baseline / 'runs' / f'drop035_v650_s{seed}' if arm == 'control' else treatment
            prediction = run / 'predictions.parquet'
            table = pd.read_parquet(prediction)
            score = scorer.score_competition(table, context)
            array = scorer.align_predictions(table, context)
            if not np.isfinite(array).all() or (array <= 0).any():
                raise ValueError(f'{run.name}: invalid official-grid forecasts')
            stored = json.loads((run / 'local_score.json').read_text())['competition']
            for scope in SCOPES:
                if score[scope]['n_replaced'] != 0:
                    raise ValueError(f'{run.name}/{scope}: scorer fallback used')
                for metric in ('s', 'rmse', 'trmse', 'timse'):
                    if not np.isclose(score[scope][metric], stored[scope][metric], rtol=0, atol=1e-8):
                        raise ValueError(f'{run.name}/{scope}/{metric}: stored score mismatch')
                mask = np.ones(len(array), bool) if scope == 'total' else context['period'] == scope.upper()
                score[scope]['pooled_mae'] = float(np.nanmean(abs(array[mask] - context['truth'][mask])))
            arm_scores.append(score)
            arrays.append(array)
            inputs.append(identity(prediction))
    if not np.isclose(scores['control'][0]['total']['s'], BEST_SINGLE_S, rtol=0, atol=1e-8):
        raise ValueError('Frozen best-single reference no longer reproduces')
    exact, draws, rejected = stats.bootstrap_scores(context, np.stack(arrays), 2000, 168, seed=20261104)
    for i, score in enumerate(scores['control'] + scores['treatment']):
        if not np.isclose(exact['s'][i], score['total']['s'], rtol=0, atol=1e-8):
            raise ValueError('Bootstrap point score differs from official scorer')
    reduction = 1 - exact['s'][3:] / exact['s'][:3]
    boot = (1 - draws['s'][:, 3:] / draws['s'][:, :3]).mean(1)
    delta_boot = (draws['s'][:, 3:] - draws['s'][:, :3]).mean(1)
    summaries = {arm: {scope: {metric: describe([score[scope][metric] for score in arm_scores])
                              for metric in METRICS} for scope in SCOPES}
                 for arm, arm_scores in scores.items()}
    ci = np.quantile(boot, [.025, .975]).tolist()
    companion_gate = all(summaries['treatment']['total'][metric]['mean'] <=
                         1.01 * summaries['control']['total'][metric]['mean']
                         for metric in ('rmse', 'pooled_mae'))
    period_gate = all(summaries['treatment'][p]['s']['mean'] <=
                      1.02 * summaries['control'][p]['s']['mean'] for p in ('f1', 'f2', 'f3'))
    candidate = bool(reduction.mean() >= .01 and reduction.min() > 0 and ci[0] > 0
                     and companion_gate and period_gate)
    reference_reductions = 1 - exact['s'][3:] / BEST_SINGLE_S
    result = {
        'status': 'complete', 'metric': 'official S, user-confirmed W', 'seeds': list(SEEDS),
        'classification': 'fresh-training single-input replacement bugfix ablation',
        'summaries': summaries, 'scores': scores,
        'paired_reduction': describe(reduction, lower_is_worse=True),
        'paired_reduction_ci95': ci,
        'paired_delta_s_ci95': np.quantile(delta_boot, [.025, .975]).tolist(),
        'best_reference_s': BEST_SINGLE_S, 'half_s_target': HALF_S_TARGET,
        'versus_fixed_best_single_reduction': describe(reference_reductions, lower_is_worse=True),
        'candidate_gate_passed': candidate,
        'auxiliary_gate_passed': bool(companion_gate and period_gate),
        'score_50pct_gate_passed': bool((exact['s'][3:] <= HALF_S_TARGET).all()),
        'goal_50pct_achieved': False,
        'goal_status_reason': 'Score report alone cannot establish independent completion audit.',
        'adopted': False,
        'sota_comparison': {'reference_id': 'e2000fs_calib_offset_roll_drop_t035_v650_s42',
                            'reference_s': BEST_SINGLE_S,
                            'reference_revision': 'f756e1623af7f0cbd9fd9c0fc181902f542580bf',
                            'snapshot_only': True,
                            'reduction': describe(reference_reductions, lower_is_worse=True),
                            'all_seeds_at_least_2pct_better': bool((reference_reductions >= .02).all())},
        'bootstrap': {'replicates': 2000, 'block_hours': 168, 'seed': 20261104,
                      'rejected': rejected, 'period_stratified': True,
                      'empirical_timse_quantiles_recomputed': True},
        'inputs': inputs, 'observations': identity(args.observations),
        'scorer': identity(REPO / 'leaderboard/scorer/competition.py'),
        'scorer_version': scorer.VERSION,
        'limits': ['Intervals condition on three fitted seeds and exposed official periods.',
                   'Repeated leaderboard selection bias is not removed by bootstrap.',
                   'Current input issue-time availability remains incompletely audited.',
                   'SOTA comparison is against the pinned snapshot, not a fresh live-board query.']}
    write(args.root / 'results.json', result)
    np.savez_compressed(args.root / 'bootstrap.npz', reduction=boot, delta_s=delta_boot)
    print(json.dumps({'mean_reduction': float(reduction.mean()), 'ci95': ci,
                      'candidate_gate_passed': candidate,
                      'score_50pct_gate_passed': result['score_50pct_gate_passed']}))


if __name__ == '__main__':
    main()
