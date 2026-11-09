"""Existing prior-fit/calibration procedure diagnostics, entirely before cutoff."""
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
        raise ValueError('CPU XGBoost requires invisible CUDA')
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    out = ROOT / 'runs' / f'age_retrain_s{args.seed}'
    metadata = json.loads((out / f'{args.period}_metadata.json').read_text())
    baseline, _ = experiment.parent_modules()
    grid, truth = baseline.load_grid(assignment['data'], args.period, metadata['features'], True)
    grid, _ = experiment.repair_grid(grid, metadata['features'],
        experiment.repaired_age(ROOT / 'raw_age.parquet'), training=True)
    reference, calibration, boundary = baseline.prior_split(grid['origins'])
    result = {'seed': args.seed, 'period': args.period, 'query_truth_read': False,
              'new_fit': False, 'parameters_selected': False,
              'scope': 'Existing training-only prior/reference and offset-calibration procedure',
              'boundary': str(boundary), 'groups': {},
              'limits': ['No new evaluation fold, model score or adoption criterion.',
                         'Fixed prefix checkpoints only; ongoing900-round fits unchanged.',
                         'Both groups are strictly before the official training cutoff.',
                         'Differences can reflect temporal regime change as well as fitting capacity.']}
    for name, indices in [('prior_fitting_rows', reference), ('existing_offset_calibration_rows', calibration)]:
        totals = {prefix: {'square': 0., 'absolute': 0., 'tail_error': 0., 'n': 0, 'tail_n': 0}
                  for prefix in PREFIXES}
        for block, start in enumerate((0, 24, 48)):
            target = truth[indices, start:start + 24]
            design = np.hstack([grid['C'][indices],
                grid['E'][indices, start:start + 24].reshape(len(indices), -1)])
            model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
            model.load_model(out / f'{args.period}_models/prior/b{block}.ubj')
            model.set_param({'device': 'cpu', 'nthread': 1})
            assert model.num_boosted_rounds() == 900
            extreme = target >= 700.
            for prefix in PREFIXES:
                error = model.inplace_predict(design, iteration_range=(0, prefix)).astype(np.float64) - target
                row = totals[prefix]
                row['square'] += float((error**2).sum())
                row['absolute'] += float(np.abs(error).sum())
                row['tail_error'] += float(error[extreme].sum())
                row['n'] += error.size
                row['tail_n'] += int(extreme.sum())
        result['groups'][name] = {'origins': len(indices),
            'start': str(grid['origins'][indices].min()), 'end': str(grid['origins'][indices].max()),
            'curves': [{'rounds': prefix, 'rmse': float(np.sqrt(row['square'] / row['n'])),
                        'mae': row['absolute'] / row['n'],
                        'tail_ge700_n': row['tail_n'],
                        'tail_ge700_bias': row['tail_error'] / row['tail_n'] if row['tail_n'] else None}
                       for prefix, row in totals.items()]}
    (out / f'{args.period}_prior-fit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
