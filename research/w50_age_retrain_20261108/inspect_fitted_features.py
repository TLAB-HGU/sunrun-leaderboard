"""Descriptive split counts from saved vector-leaf trees; never reads targets."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import xgboost as xgb

ROOT = Path(__file__).resolve().parent


def describe(run, period):
    metadata = json.loads((run / f'{period}_metadata.json').read_text())
    totals, age_by_block = Counter(), {}
    for block, start in enumerate((0, 24, 48)):
        names = metadata['expanded_features'][str(start)]
        booster = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
        path = run / f'{period}_models/full/b{block}.ubj'
        booster.load_model(path)
        assert booster.num_features() == len(names) == 212
        model = json.loads(booster.save_raw(raw_format='json'))
        trees = model['learner']['gradient_booster']['model']['trees']
        counts = Counter()
        for tree in trees:
            for feature, left in zip(tree['split_indices'], tree['left_children'], strict=True):
                if left >= 0:  # leaf entries carry a default feature index; exclude them.
                    counts[names[feature].split('@h', 1)[0]] += 1
        totals.update(counts)
        age_by_block[str(start)] = {'img_age_h_splits': counts['img_age_h'],
                                    'all_splits': counts.total(), 'trees': len(trees)}
    return {'source_split_counts': dict(totals.most_common()), 'blocks': age_by_block,
            'img_age_h_split_fraction': totals['img_age_h'] / totals.total()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', choices=(42, 43, 44), type=int, required=True)
    parser.add_argument('--period', choices=('F1', 'F2', 'F3'), required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only XGBoost inspection requires invisible CUDA')
    out = ROOT / 'runs' / f'age_retrain_s{args.seed}'
    result = {'seed': args.seed, 'period': args.period,
              'parent': describe(args.parent_run, args.period),
              'treatment': describe(out, args.period),
              'query_or_training_targets_read': False,
              'limits': ['Unweighted split occurrence is not causal attribution, gain or forecast skill.',
                         'Counts across repeated source columns depend on expansion and subsampling.',
                         'This inspection does not alter the ongoing experiment or select features.']}
    (out / f'{args.period}_split-inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
