"""Report eight frozen additive hypotheses; never publish or select during fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
AXES = tuple(range(1, 9))
BOOTSTRAP_SEED = 20261109
BLOCK_HOURS = 168
HELPER_PATH = ROOT.parent / 'feature_addition_all_20261108/analyze_results.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = load_module('axes8_statistics', HELPER_PATH)


def conditions_from(assignment):
    rows = assignment['experiments']
    if len(rows) != 24 or len({r['experiment'] for r in rows}) != 24:
        raise ValueError('Exactly24 unique runs required for frozen eight-axis family')
    result = {}
    for axis in AXES:
        recipes = sorted([r for r in rows if r['axis'] == axis], key=lambda r: r['seed'])
        if [r['seed'] for r in recipes] != list(SEEDS):
            raise ValueError('Each axis requires seeds42/43/44 exactly once')
        for recipe in recipes:
            if Path(recipe['experiment']).name != recipe['experiment']:
                raise ValueError('Unsafe experiment name')
            if recipe['config']['seed'] != recipe['seed']:
                raise ValueError('Recipe/config seed mismatch')
            features = recipe['config']['add_features']
            if not features or len(features) != len(set(features)) or features != recipes[0]['config']['add_features']:
                raise ValueError('Each axis requires the same explicit additive union across seeds')
        result[axis] = recipes
    return result


def require_completed(run, seed, additions=None):
    for filename in ('train.exit', 'training-validation.json', 'complete.json',
                     'predictions.parquet', 'local_score.json'):
        if not (run / filename).is_file():
            raise ValueError(f'{run}: missing {filename}')
    if (run / 'train.exit').read_text().strip() != '0':
        raise ValueError(f'{run}: nonzero training exit')
    if not stats.read_json(run / 'training-validation.json'):
        raise ValueError(f'{run}: official validation receipt required')
    if additions is not None and stats.read_json(run / 'complete.json').get('publish') != 'not_required':
        raise ValueError(f'{run}: report-only completion receipt required')
    for period in ('F1', 'F2', 'F3'):
        meta = stats.read_json(run / f'{period}_metadata.json')
        if meta.get('smoke') or meta['seed'] != seed or meta['rounds'] != 900:
            raise ValueError(f'{run}: full matching-seed metadata required')
        if additions is not None and meta['config']['add_features'] != additions:
            raise ValueError(f'{run}: additive union differs from frozen recipe')


def run_status(root, recipe, state):
    row = state.get(recipe['experiment'], {})
    status = row.get('state', recipe.get('state', 'waiting'))
    if status in ('blocked_source', 'failed', 'cancelled_by_user'):
        return {'state': status, 'reason': row.get('reason', row.get('error', recipe.get('reason')))}
    out = root / 'runs' / recipe['experiment']
    if (out / 'train.exit').exists() and (out / 'train.exit').read_text().strip() != '0':
        return {'state': 'failed', 'reason': 'nonzero train.exit'}
    if status == 'trained_validated':
        try:
            require_completed(out, recipe['seed'], recipe['config']['add_features'])
        except (OSError, ValueError, KeyError) as error:
            return {'state': 'failed', 'reason': str(error)}
        return {'state': 'complete'}
    return {'state': status, 'reason': row.get('reason')}


def adjust_family(rows):
    if sorted(r['axis'] for r in rows) != list(AXES):
        raise ValueError('BH family must contain all eight frozen hypotheses')
    pvalues = [r.get('p_centered_bootstrap', 1.) if r['state'] == 'complete' else 1. for r in rows]
    if not all(p is not None and np.isfinite(p) and 0 <= p <= 1 for p in pvalues):
        raise ValueError('Invalid bootstrap p-value')
    for row, p, q in zip(rows, pvalues, stats.bh_adjust(pvalues)):
        row.update(family_p=p, q_bh=q, bh_family_size=8)


def analyze(root=ROOT, replicates=2000):
    root = Path(root).resolve()
    if replicates < 1:
        raise ValueError('Positive bootstrap count required')
    assignment = stats.read_json(root / 'assignment.global.json')
    conditions = conditions_from(assignment)
    state = {}
    for filename in ('scheduler-state.json', 'runtime.json'):
        if (root / filename).exists():
            state.update(stats.read_json(root / filename))
    rows = []
    for axis, recipes in conditions.items():
        runs = {r['experiment']: dict(seed=r['seed'], **run_status(root, r, state)) for r in recipes}
        states = [r['state'] for r in runs.values()]
        status = ('complete' if all(s == 'complete' for s in states) else
                  'blocked_source' if 'blocked_source' in states else
                  'failed' if any(s in ('failed', 'cancelled_by_user') for s in states) else
                  'running' if any(s != 'waiting' for s in states) else 'waiting')
        rows.append(dict(axis=axis, state=status, runs=runs, add_features=recipes[0]['config']['add_features']))
    baseline = None
    controls = []
    scorer = context = weights = None
    provenance = stats.read_json(root / 'provenance.json') if (root / 'provenance.json').exists() else {}
    for row in rows:
        if row['state'] != 'complete':
            continue
        try:
            parent_paths = [Path(assignment['baseline_runs'][str(seed)]) for seed in SEEDS]
            for path, seed in zip(parent_paths, SEEDS):
                require_completed(path, seed)
            # Official targets are loaded only after all six fits for this comparison.
            if context is None:
                scorer = load_module('axes8_official_scorer', root / 'competition.py')
                observations = Path(assignment['observations'])
                if not observations.is_absolute():
                    observations = root / observations
                context = scorer.prepare_context(scorer.load_observations(observations))
                controls = [stats.score_run(p.parent.parent, {'experiment': p.name}, scorer, context) for p in parent_paths]
                baseline = stats.summarize([v[1] for v in controls])
            recipes = conditions[row['axis']]
            paths = parent_paths + [root / 'runs' / r['experiment'] for r in recipes]
            sources = [root / 'assignment.global.json', root / 'competition.py', Path(__file__), HELPER_PATH, observations]
            if (root / 'provenance.json').exists():
                sources.append(root / 'provenance.json')
            sources += [p / name for p in paths for name in ('predictions.parquet', 'local_score.json',
                        'training-validation.json', 'complete.json', 'train.exit',
                        'F1_metadata.json', 'F2_metadata.json', 'F3_metadata.json')]
            signature = dict(replicates=replicates, block_hours=BLOCK_HOURS, seed=BOOTSTRAP_SEED,
                             inputs=[stats.identity(p) for p in sources])
            cache_path = root / 'analysis' / 'cache' / f"axis{row['axis']}.json"
            cached = stats.read_json(cache_path) if cache_path.exists() else {}
            if cached.get('signature') == signature:
                values = cached['values']
            else:
                treatments = [stats.score_run(root, r, scorer, context) for r in recipes]
                if weights is None:
                    weights, rejected = stats.shared_bootstrap_weights(
                        context, root / 'analysis/shared-bootstrap', replicates, BLOCK_HOURS, BOOTSTRAP_SEED,
                        {'observations': stats.identity(observations), 'scorer': stats.identity(root / 'competition.py'),
                         'helper': stats.identity(HELPER_PATH)})
                exact, draws, rejected = stats.bootstrap_scores_shared(
                    context, np.stack([v[0] for v in controls + treatments]), weights, rejected)
                if not np.allclose(exact['s'], [v[1]['total']['s'] for v in controls + treatments], atol=1e-8, rtol=1e-10):
                    raise ValueError('Bootstrap identity disagrees with official scorer')
                delta = exact['s'][3:] - exact['s'][:3]
                summary = stats.summarize([v[1] for v in treatments])
                values = dict(summary=summary, per_seed_delta_s=delta.tolist(), delta_s=stats.describe(delta),
                              mean_delta_s=float(delta.mean()), improved_seeds=int((delta < 0).sum()),
                              improvement_fraction=1-summary['total']['s']['mean']/baseline['total']['s']['mean'],
                              per_seed_improvement_fraction=(1-exact['s'][3:]/exact['s'][:3]).tolist(),
                              period_mean_delta_s={p: summary[p]['s']['mean']-baseline[p]['s']['mean'] for p in ('f1', 'f2', 'f3')},
                              bootstrap_rejected_draws=rejected,
                              **stats.distribution((draws['s'][:, 3:]-draws['s'][:, :3]).mean(1), float(delta.mean())))
                stats.atomic(cache_path, dict(signature=signature, values=values))
            row.update(values)
        except (OSError, KeyError, ValueError, TypeError) as error:
            row.update(state='analysis_blocked', analysis_error=str(error))
    complete = all(r['state'] == 'complete' and 'summary' in r for r in rows)
    adjust_family(rows)
    for row in rows:
        row['gate'] = stats.gate(row, baseline, complete, replicates)
    result = dict(status='complete' if complete else 'provisional',
                  created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
                  rows=rows, counts=dict(Counter(r['state'] for r in rows)), baseline=baseline,
                  parent=provenance.get('parent'), basis_runs=assignment.get('baseline_runs'),
                  bootstrap=dict(replicates=replicates, block_hours=BLOCK_HOURS, seed=BOOTSTRAP_SEED),
                  family_size=8, final_adoption_claims_allowed=complete,
                  policy='Report only; no publishing. Partial results cannot adopt a new baseline. Official-period exploratory comparisons do not establish independent generalization.')
    stats.atomic(root / 'analysis/results.json', result)
    (root / 'analysis/report.md').write_text(render_report(result))
    return result


def render_report(result):
    lines = ['# Eight-axis additive ablation', '', f"Status: {result['status']}; {result['counts']}", '',
             result['policy'], '', '|Axis|State|Improved seeds|Mean S|SD|Worst S|Mean ΔS|95% ΔS CI|BH q|Gate|',
             '|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    for row in result['rows']:
        if 'summary' not in row:
            lines.append(f"|{row['axis']}|{row['state']}|—|—|—|—|—|—|{row['q_bh']:.4f}|False|")
            continue
        s = row['summary']['total']['s']
        lines.append(f"|{row['axis']}|{row['state']}|{row['improved_seeds']}/3|{s['mean']:.6f}|{s['std']:.6f}|{s['worst']:.6f}|{row['mean_delta_s']:+.6f}|{row['ci95']}|{row['q_bh']:.4f}|{row['gate']['passed']}|")
    lines += ['', 'All F1/F2/F3 and total S, MAE, RMSE, TRMSE and TIMSE per-seed summaries are in results.json.',
              'Frozen BH family:8; blocked/incomplete axes use p=1. Gates require all8 complete, all3 paired seeds improve, mean gain≥1%, CI upper<0, q<0.05, MAE/RMSE deterioration≤1%, each-period S deterioration≤2%.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.root, args.bootstrap)
    print(json.dumps({'status': result['status'], 'counts': result['counts']}))
