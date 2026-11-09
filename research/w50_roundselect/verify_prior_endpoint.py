"""Independently verify a completed prior head against the frozen parent."""
import argparse
import json
import os
from pathlib import Path

import experiment
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--period', choices=('F1', 'F2', 'F3'), required=True)
    parser.add_argument('--block', type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU verification requires invisible CUDA')
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, args.seed, f'roundselect_s{args.seed}')
    names = core.feature_names(config)
    train, truth = baseline.load_grid(assignment['data'], args.period, names, True)
    _, calibration, _ = baseline.prior_split(train['origins'])
    start = args.block * 24
    design = core.design(baseline.subset(train, calibration), start, 24)
    out = ROOT / 'runs' / f'roundselect_s{args.seed}'
    model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    model.load_model(out / f'{args.period}_models/prior/b{args.block}.ubj')
    model.set_param({'device': 'cpu', 'nthread': 1})
    assert model.num_boosted_rounds() == 900 and model.num_features() == design.shape[1] == 212
    prediction = model.inplace_predict(design)
    anchor = ROOT / 'frozen_controls' / f'drop035_v650_s{args.seed}' / f'{args.period}_models/prior_predictions.npz'
    with np.load(anchor, allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], train['origins'][calibration].asi8)
        assert np.array_equal(saved['training_truth'], truth[calibration])
        difference = float(np.max(np.abs(prediction - saved['raw'][:, start:start + 24])))
    assert difference <= 1e-6
    result = {'seed': args.seed, 'period': args.period, 'block': args.block, 'status': 'pass',
              'prior_rounds': 900, 'expanded_columns': 212, 'query_truth_read': False,
              'max_parent_prediction_difference': difference,
              'scope': 'Completed prior head only; remaining control heads and full selected refit still pending'}
    (out / f'{args.period}_prior900_b{args.block}_control_replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
