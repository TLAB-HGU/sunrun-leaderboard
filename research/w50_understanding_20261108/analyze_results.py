"""Paired official-score uncertainty for the preregistered input repair."""
import json

import numpy as np
import pandas as pd
from age_repair import OBS, REPO, ROOT, module, write


def describe(values):
    values = np.asarray(values, dtype=float)
    return {'per_seed': values.tolist(), 'mean': float(values.mean()),
            'sample_variance': float(values.var(ddof=1)),
            'sample_sd': float(values.std(ddof=1)), 'worst': float(values.max())}


def main():
    scorer = module(REPO / 'leaderboard/scorer/competition.py', 'repair_analysis_scorer')
    stats = module(ROOT.parent / 'offset_roll_complete_ablation_20261106/analyze_results.py', 'repair_stats')
    context = scorer.prepare_context(scorer.load_observations(OBS))
    arrays, scores = [], {'control': [], 'treatment': []}
    identity = {}
    for arm, arm_scores in scores.items():
        for seed in (42, 43, 44):
            run = ROOT / 'runs' / f'age_repair_s{seed}'
            assert (run / 'train.exit').read_text().strip() == '0'
            assert json.loads((run / 'complete.json').read_text())['status'] == 'complete'
            table = pd.read_parquet(run / ('control.parquet' if arm == 'control' else 'predictions.parquet'))
            score = scorer.score_competition(table, context)
            array = scorer.align_predictions(table, context)
            for scope in ('total', 'f1', 'f2', 'f3'):
                assert score[scope]['n_replaced'] == 0
                mask = np.ones(len(array), bool) if scope == 'total' else context['period'] == scope.upper()
                score[scope]['pooled_mae'] = float(np.nanmean(np.abs(array[mask] - context['truth'][mask])))
            arrays.append(array)
            arm_scores.append(score)
    exact, draws, rejected = stats.bootstrap_scores(context, np.stack(arrays), 2000, 168)
    for arm, offset in [('control', 0), ('treatment', 3)]:
        for i, score in enumerate(scores[arm]):
            assert abs(exact['s'][offset + i] - score['total']['s']) < 1e-8
    reduction = 1 - exact['s'][3:] / exact['s'][:3]
    boot = (1 - draws['s'][:, 3:] / draws['s'][:, :3]).mean(1)
    delta_boot = (draws['s'][:, 3:] - draws['s'][:, :3]).mean(1)
    summaries = {arm: {scope: {metric: describe([s[scope][metric] for s in values])
                              for metric in ('s', 'rmse', 'trmse', 'timse', 'pooled_mae')}
                       for scope in ('total', 'f1', 'f2', 'f3')}
                 for arm, values in scores.items()}
    ci = np.quantile(boot, [.025, .975]).tolist()
    accepted = bool(reduction.mean() >= .01 and reduction.min() > 0 and ci[0] > 0
                    and all(summaries['treatment']['total'][metric]['mean'] <=
                            1.01 * summaries['control']['total'][metric]['mean']
                            for metric in ('rmse', 'pooled_mae'))
                    and all(summaries['treatment'][p]['s']['mean'] <=
                            1.02 * summaries['control'][p]['s']['mean']
                            for p in ('f1', 'f2', 'f3')))
    for path in [OBS, ROOT / 'age_repair.py', ROOT / 'strategy.md']:
        identity[str(path)] = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    result = {'status': 'complete', 'metric': 'official S, user-confirmed W', 'summaries': summaries,
              'paired_reduction': dict(describe(reduction), worst=float(reduction.min())), 'paired_reduction_ci95': ci,
              'paired_delta_s_ci95': np.quantile(delta_boot, [.025, .975]).tolist(),
              'best_reference_s': float(exact['s'][0]), 'half_s_target': float(exact['s'][0] / 2),
              'best_reference_seed42_reduction': float(reduction[0]),
              'best_reference_seed42_reduction_ci95': np.quantile(1 - draws['s'][:, 3] / draws['s'][:, 0], [.025, .975]).tolist(),
              'goal_50pct_achieved': bool((exact['s'][3:] <= exact['s'][0] / 2).all()),
              'candidate_gate_passed': accepted, 'adopted': False,
              'bootstrap': {'replicates': 2000, 'block_hours': 168, 'seed': 20261104, 'rejected': rejected},
              'identity': identity,
              'limits': ['Inference-only replacement: corrected training untested.',
                         'Conditional intervals on three fitted seeds and exposed periods.',
                         'No control for historical leaderboard selection.',
                         'Fresh file hashes rejected by automatic approval hook.']}
    write(ROOT / 'results.json', result)
    np.savez_compressed(ROOT / 'bootstrap.npz', reduction=boot, delta_s=delta_boot)
    print(json.dumps({'mean_reduction': float(reduction.mean()), 'ci95': ci, 'candidate_gate_passed': accepted}))


if __name__ == '__main__':
    main()
