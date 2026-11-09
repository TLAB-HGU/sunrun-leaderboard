"""Independent period model/calibration replay without that period's query truth."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--period', choices=('F1', 'F2', 'F3'), required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Model replay requires invisible CUDA for CPU XGBoost')
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    out = ROOT / 'runs' / f'age_retrain_s{args.seed}'
    metadata = json.loads((out / f'{args.period}_metadata.json').read_text())
    parent = ROOT.parent / 'drop035_v650_confirm_20261108'
    reference = json.loads((parent / 'runs' / f'drop035_v650_s{args.seed}' /
                            f'{args.period}_metadata.json').read_text())
    assert metadata['fresh_training'] and not metadata['reused_model_or_predictions']
    assert metadata['features'] == reference['features']
    assert metadata['expanded_features'] == reference['expanded_features']
    assert metadata['rounds'] == 900 and not metadata['smoke']
    spec = importlib.util.spec_from_file_location('independent_replay_baseline', parent / 'baseline.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    grid, labels = baseline.load_grid(assignment['data'], args.period, metadata['features'], False)
    assert labels is None
    raw = pd.read_parquet(ROOT / 'raw_age.parquet', columns=['origin', 'img_age_h'],
                          filters=[('h', '==', 1)], use_threads=False)
    raw = raw.drop_duplicates('origin').sort_values('origin')
    origins = pd.DatetimeIndex(pd.to_datetime(raw.origin, utc=True))
    ages = raw.img_age_h.to_numpy()
    observed = np.isfinite(ages)
    # Independently locate the latest real frame record at or before each origin.
    source = origins.asi8[observed] - pd.to_timedelta(ages[observed], unit='h').asi8
    def repair(inputs):
        locations = np.searchsorted(origins.asi8[observed], inputs['origins'].asi8, side='right') - 1
        assert (locations >= 0).all()
        values = ((inputs['origins'].asi8 - source[locations]) / 3600e9).astype(np.float32)
        assert np.isfinite(values).all() and (values >= 0).all()
        inputs['C'][:, metadata['features']['constant'].index('img_age_h')] = values

    repair(grid)
    train, training_labels = baseline.load_grid(assignment['data'], args.period, metadata['features'], True)
    reference_rows, calibration_rows, boundary = baseline.prior_split(train['origins'])
    calibration = baseline.subset(train, calibration_rows)
    repair(calibration)
    assert (train['origins'] + pd.Timedelta(hours=144)
            <= pd.Timestamp(baseline.CUTOFFS[args.period], tz='UTC')).all()
    assert (train['origins'][reference_rows] + pd.Timedelta(hours=144) <= boundary).all()
    assert metadata['frozen_state']['calibration_rows'] == len(calibration_rows)
    predictions, prior_predictions, rounds = [], [], {}
    for stage in ('prior', 'full'):
        for block in range(3):
            model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
            model.load_model(out / f'{args.period}_models' / stage / f'b{block}.ubj')
            model.set_param({'device': 'cpu', 'nthread': 1})
            rounds[f'{stage}/{block}'] = model.num_boosted_rounds()
            assert rounds[f'{stage}/{block}'] == 900
            inputs = grid if stage == 'full' else calibration
            start = block * 24
            design = np.hstack([inputs['C'], inputs['E'][:, start:start + 24].reshape(len(inputs['origins']), -1)])
            assert design.shape[1] == len(metadata['expanded_features'][str(start)]) == 212
            if stage == 'full':
                predictions.append(model.inplace_predict(design).astype(np.float64))
            else:
                prior_predictions.append(model.inplace_predict(design).astype(np.float64))
    prior = np.hstack(prior_predictions)
    truth = training_labels[calibration_rows]
    with np.load(out / f'{args.period}_models/prior_predictions.npz') as saved:
        assert np.array_equal(saved['origins'], calibration['origins'].asi8)
        assert np.array_equal(saved['training_truth'], truth)
        prior_difference = float(np.max(np.abs(saved['raw'] - prior)))
    assert prior_difference <= 1e-6
    assert len(truth) >= 336 and metadata['frozen_state']['offset_rows'] == 336
    offset = .25 * (truth[-336:] - np.clip(prior[-336:], 1, 3000)).mean(0)
    with np.load(out / f'{args.period}_raw_predictions.npz') as state:
        offset_difference = float(np.max(np.abs(state['offset'] - offset)))
    assert offset_difference <= 1e-6
    prediction = np.clip(np.clip(np.hstack(predictions), 1, 3000) + offset, 1, 3000)
    with np.load(out / f'{args.period}_predictions.npz') as saved:
        assert np.array_equal(saved['origins'], grid['origins'].asi8)
        difference = float(np.max(np.abs(saved['pred'] - prediction)))
    assert difference <= 1e-6
    receipt = {'period': args.period, 'seed': args.seed, 'status': 'pass',
               'max_prediction_difference': difference, 'model_rounds': rounds,
               'max_prior_prediction_difference': prior_difference,
               'max_training_only_offset_difference': offset_difference,
               'training_maturity_and_calibration_purge_verified': True,
               'calibration_rows': len(calibration_rows),
               'query_truth_read': False, 'feature_names_order_match_parent': True,
               'independent_age_method': 'searchsorted latest real raw-frame origin before query',
               'query_origins': len(grid['origins']), 'source_features': 28, 'expanded_columns': 212}
    (out / f'{args.period}_model-replay.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
