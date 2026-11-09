"""Saved-tree prefixes on original fitting rows; no new fits or query labels."""
import argparse
import json
import os
import resource
from pathlib import Path

import experiment
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
PREFIXES = (1, 100, 300, 600, 900)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--period', choices=('F1', 'F2', 'F3'), required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only model diagnostics require invisible CUDA')
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    out = ROOT / 'runs' / f'age_retrain_s{args.seed}'
    metadata = json.loads((out / f'{args.period}_metadata.json').read_text())
    assert metadata['fresh_training'] and metadata['rounds'] == 900 and not metadata['smoke']
    baseline, _ = experiment.parent_modules()
    grid, truth = baseline.load_grid(assignment['data'], args.period, metadata['features'], True)
    grid, _ = experiment.repair_grid(grid, metadata['features'],
        experiment.repaired_age(ROOT / 'raw_age.parquet'), training=True)
    moments = {prefix: {'n': 0, 'square': 0., 'absolute': 0., 'weighted_square': 0.,
                       'tail_n': 0, 'tail_error': 0., 'tail_square': 0.}
               for prefix in PREFIXES}
    for block, start in enumerate((0, 24, 48)):
        target = truth[:, start:start + 24]
        design = np.hstack([grid['C'], grid['E'][:, start:start + 24].reshape(len(target), -1)])
        weights = np.where(target >= 550., 4., 1.)
        weights /= weights.mean(axis=0)
        extreme = target >= 700.
        model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
        model.load_model(out / f'{args.period}_models/full/b{block}.ubj')
        model.set_param({'device': 'cpu', 'nthread': 1})
        assert model.num_boosted_rounds() == 900
        for prefix in PREFIXES:
            error = model.inplace_predict(design, iteration_range=(0, prefix)).astype(np.float64) - target
            assert np.isfinite(error).all()
            row = moments[prefix]
            row['n'] += error.size
            row['square'] += float((error**2).sum())
            row['absolute'] += float(np.abs(error).sum())
            row['weighted_square'] += float((error**2 * weights).sum())
            row['tail_n'] += int(extreme.sum())
            row['tail_error'] += float(error[extreme].sum())
            row['tail_square'] += float((error[extreme]**2).sum())
    result = {'seed': args.seed, 'period': args.period, 'training_origins': len(truth),
              'query_truth_read': False, 'new_fit': False, 'curves': [],
              'limits': ['Original fitting rows: not generalization evidence or an internal selection fold.',
                         'RMSE/weighted MSE are diagnostics, not the full spectral objective or official S.',
                         'Prefixes are fixed descriptive checkpoints; no round/parameter selection.',
                         'A descending training curve does not prove optimization optimality or a skill gain.']}
    for prefix, row in moments.items():
        result['curves'].append({'rounds': prefix, 'rmse': float(np.sqrt(row['square'] / row['n'])),
              'mae': row['absolute'] / row['n'], 'weighted_mse': row['weighted_square'] / row['n'],
              'tail_ge700_n': row['tail_n'], 'tail_ge700_bias': row['tail_error'] / row['tail_n'],
              'tail_ge700_rmse': float(np.sqrt(row['tail_square'] / row['tail_n']))})
    (out / f'{args.period}_training-fit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
