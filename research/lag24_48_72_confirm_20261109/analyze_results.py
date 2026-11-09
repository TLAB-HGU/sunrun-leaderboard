"""Verify and report the single preregistered three-lag addition after all fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
ADDITIONS = ['v_lag24', 'v_lag48', 'v_lag72']
BOOTSTRAP_SEED = 20261108
BLOCK_HOURS = 168
HELPER_PATH = ROOT.parent / 'feature_addition_all_20261108/analyze_results.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = load_module('lag_joint_statistics', HELPER_PATH)


def ordered_recipes(assignment):
    recipes = assignment['experiments']
    if len(recipes) != 3 or sorted(row['seed'] for row in recipes) != list(SEEDS):
        raise ValueError('Joint experiment requires exactly one run for each seed42/43/44')
    names = [row['experiment'] for row in recipes]
    if len(set(names)) != 3 or any(Path(name).name != name for name in names):
        raise ValueError('Experiment names must be unique safe path components')
    for row in recipes:
        if row['config'].get('seed') != row['seed'] or row['config'].get('add_features') != ADDITIONS:
            raise ValueError('Seed or joint-addition recipe differs from preregistration')
    return sorted(recipes, key=lambda row: row['seed'])


def require_completed(run, seed, treatment):
    required = ['train.exit', 'training-validation.json', 'complete.json',
                'predictions.parquet', 'local_score.json']
    if any(not (run / name).is_file() for name in required):
        raise ValueError(f'{run.name}: complete validated artifacts required before analysis')
    if (run / 'train.exit').read_text().strip() != '0':
        raise ValueError(f'{run.name}: training failed')
    for period in ('F1', 'F2', 'F3'):
        report = stats.read_json(run / f'{period}_metadata.json')
        if report.get('smoke') or report['rounds'] != 900 or report['seed'] != seed:
            raise ValueError(f'{run.name}: full matching-seed training required')
        if treatment:
            if report['config'].get('add_features') != ADDITIONS:
                raise ValueError(f'{run.name}: treatment metadata disagrees with recipe')
            if report['feature_count'] != 31 or report['expanded_feature_count'] != 215:
                raise ValueError(f'{run.name}: wrong feature union size')
        elif report['feature_count'] != 28 or report['expanded_feature_count'] != 212:
            raise ValueError(f'{run.name}: wrong matched parent configuration')


def improvement_gate(baseline, treatment, deltas, ci95, replicates):
    checks = {
        'bootstrap_2000': replicates >= 2000,
        'mean_s_gain_at_least_1pct': treatment['total']['s']['mean'] <= .99 * baseline['total']['s']['mean'],
        'all_three_seeds_improve': len(deltas) == 3 and max(deltas) < 0,
        'ci_upper_below_zero': len(ci95) == 2 and ci95[1] < 0,
        'mae_rmse_deterioration_at_most_1pct': all(
            treatment['total'][metric]['mean'] <= 1.01 * baseline['total'][metric]['mean']
            for metric in ('mae', 'rmse')),
        'each_period_s_deterioration_at_most_2pct': all(
            treatment[period]['s']['mean'] <= 1.02 * baseline[period]['s']['mean']
            for period in ('f1', 'f2', 'f3')),
    }
    return {'passed': all(checks.values()), 'checks': checks}


def matched_deltas(parent_scores, treatment_scores):
    if set(parent_scores) != set(SEEDS) or set(treatment_scores) != set(SEEDS):
        raise ValueError('Paired differences require all three matching seeds')
    return [float(treatment_scores[seed] - parent_scores[seed]) for seed in SEEDS]


def analyze(root=ROOT, replicates=2000):
    root = Path(root).resolve()
    if replicates < 1:
        raise ValueError('Positive bootstrap replicate count required')
    assignment = stats.read_json(root / 'assignment.global.json')
    recipes = ordered_recipes(assignment)
    controls = {seed: Path(assignment['baseline_runs'][str(seed)]) for seed in SEEDS}
    treatments = {row['seed']: root / 'runs' / row['experiment'] for row in recipes}
    for seed in SEEDS:
        require_completed(controls[seed], seed, False)
        require_completed(treatments[seed], seed, True)
    parent_lock = stats.read_json(root / 'parent-lock.json')
    if parent_lock.get('state') != 'verified':
        raise ValueError('Verified frozen external parent reference required')
    scorer = load_module('lag_joint_official_scorer', root / 'competition.py')
    observations = Path(assignment['observations'])
    context = scorer.prepare_context(scorer.load_observations(observations))
    # These helpers independently rescore all four scopes and reject any replaced forecast.
    control_values = [stats.score_run(run.parent.parent, {'experiment': run.name}, scorer, context)
                      for run in controls.values()]
    treatment_values = [stats.score_run(run.parent.parent, {'experiment': run.name}, scorer, context)
                        for run in treatments.values()]
    baseline = stats.summarize([value[1] for value in control_values])
    treatment = stats.summarize([value[1] for value in treatment_values])
    if not np.isclose(control_values[0][1]['total']['s'], parent_lock['total_s'], atol=1e-8, rtol=0):
        raise ValueError('Frozen leaderboard winner does not match seed42 parent score')
    deltas = matched_deltas(dict(zip(SEEDS, baseline['total']['s']['per_seed'])),
                            dict(zip(SEEDS, treatment['total']['s']['per_seed'])))
    sources = [root / 'assignment.global.json', root / 'parent-lock.json', root / 'competition.py',
               root / 'analyze_results.py', HELPER_PATH, observations]
    sources += [run / name for run in [*controls.values(), *treatments.values()]
                for name in ('predictions.parquet', 'local_score.json', 'training-validation.json', 'complete.json')]
    signature = {'replicates': replicates, 'block_hours': BLOCK_HOURS, 'seed': BOOTSTRAP_SEED,
                 'inputs': [stats.identity(path) for path in sources]}
    cache = root / 'analysis' / 'bootstrap-cache.json'
    cached = stats.read_json(cache) if cache.exists() else {}
    if cached.get('signature') == signature:
        uncertainty = cached['uncertainty']
    else:
        arrays = np.stack([value[0] for value in control_values + treatment_values])
        exact, draws, rejected = stats.bootstrap_scores(context, arrays, replicates, BLOCK_HOURS, BOOTSTRAP_SEED)
        official = [value[1]['total']['s'] for value in control_values + treatment_values]
        if not np.allclose(exact['s'], official, atol=1e-8, rtol=1e-10):
            raise ValueError('Bootstrap identity scores disagree with official scorer')
        delta_draws = (draws['s'][:, 3:] - draws['s'][:, :3]).mean(axis=1)
        uncertainty = dict(stats.distribution(delta_draws, float(np.mean(deltas))),
                           rejected_draws=rejected, pairing='same period-stratified origin draws across all six models')
        stats.atomic(cache, {'signature': signature, 'uncertainty': uncertainty})
    mean_delta = float(np.mean(deltas))
    external_s = float(parent_lock['total_s'])
    result = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'Exploratory joint addition selected from seed42 official-period results; no independent generalization claim',
        'seeds': list(SEEDS), 'add_features': ADDITIONS, 'baseline': baseline, 'treatment': treatment,
        'matched_seed_delta_s': deltas, 'delta_s': stats.describe(deltas),
        'mean_delta_s': mean_delta,
        'improvement_fraction': -mean_delta / baseline['total']['s']['mean'],
        'paired_seed_improvement_fractions': [(b - t) / b for b, t in zip(
            baseline['total']['s']['per_seed'], treatment['total']['s']['per_seed'])],
        'period_mean_delta_s': {scope: treatment[scope]['s']['mean'] - baseline[scope]['s']['mean']
                                for scope in ('f1', 'f2', 'f3')},
        'bootstrap': dict(replicates=replicates, block_hours=BLOCK_HOURS, seed=BOOTSTRAP_SEED, **uncertainty),
        'multiplicity': 'One preregistered joint condition; no within-campaign BH correction. Broader feature-selection bias remains.',
        'gate': improvement_gate(baseline, treatment, deltas, uncertainty['ci95'], replicates),
        'external_SOTA': {'experiment': parent_lock['experiment'], 'revision': parent_lock['revision'],
                          'total_s': external_s, 'target_2pct_s': .98 * external_s,
                          'per_seed_gain_fraction': [(external_s - value) / external_s
                                                     for value in treatment['total']['s']['per_seed']],
                          'mean_gain_fraction': 1 - treatment['total']['s']['mean'] / external_s,
                          'mean_2pct_target_met': treatment['total']['s']['mean'] <= .98 * external_s},
        'publication_policy': 'User explicitly requested all three seed results regardless of performance gate; no seed selection or ensemble.',
        'failure_analysis': ('Observed paired gain passes all preregistered performance checks; independent confirmation remains unproven.'
                             if improvement_gate(baseline, treatment, deltas, uncertainty['ci95'], replicates)['passed'] else
                             'Performance gate not met; see checks. Validated training artifacts and independent official-score parity support a valid exploratory result, not an implementation/protocol failure diagnosis.'),
        'provenance': signature,
    }
    stats.atomic(root / 'analysis' / 'results.json', result)
    (root / 'analysis' / 'report.md').write_text(render_report(result))
    return result


def render_report(result):
    baseline, treatment = result['baseline'], result['treatment']
    lines = ['# Joint lag24/48/72 addition', '', result['classification'], '',
             '|Seed|Parent S|Joint addition S|ΔS|Improvement|', '|---|---:|---:|---:|---:|']
    for i, seed in enumerate(SEEDS):
        lines.append(f"|{seed}|{baseline['total']['s']['per_seed'][i]:.6f}|{treatment['total']['s']['per_seed'][i]:.6f}|"
                     f"{result['matched_seed_delta_s'][i]:+.6f}|{100 * result['paired_seed_improvement_fractions'][i]:+.3f}%|")
    lines += ['', f"Mean S: {baseline['total']['s']['mean']:.6f} → {treatment['total']['s']['mean']:.6f}; "
              f"improvement {100 * result['improvement_fraction']:+.3f}%.",
              f"Treatment sample SD {treatment['total']['s']['std']:.6f}; worst seed S {treatment['total']['s']['worst']:.6f}.",
              f"Paired95% ΔS interval: {result['bootstrap']['ci95']}; {result['bootstrap']['replicates']} draws of168h blocks.",
              f"Performance gate passed: **{result['gate']['passed']}**.", '',
              '|Metric|Parent mean|Joint mean|', '|---|---:|---:|']
    for metric in stats.REPORT_METRICS:
        lines.append(f"|{metric.upper()}|{baseline['total'][metric]['mean']:.6f}|{treatment['total'][metric]['mean']:.6f}|")
    lines += ['', '|Period|Parent mean S|Joint mean S|ΔS|', '|---|---:|---:|---:|']
    for period in ('f1', 'f2', 'f3'):
        lines.append(f"|{period.upper()}|{baseline[period]['s']['mean']:.6f}|{treatment[period]['s']['mean']:.6f}|{result['period_mean_delta_s'][period]:+.6f}|")
    lines += ['', 'Gate checks: ' + json.dumps(result['gate']['checks']), '', result['failure_analysis'], '',
              f"Frozen external best single S={result['external_SOTA']['total_s']:.6f}; separate2% target={result['external_SOTA']['target_2pct_s']:.6f}.",
              result['multiplicity'], '', result['publication_policy'], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.root, args.bootstrap)
    print(json.dumps({'status': result['status'], 'mean_delta_s': result['mean_delta_s'],
                      'gate_passed': result['gate']['passed']}))
