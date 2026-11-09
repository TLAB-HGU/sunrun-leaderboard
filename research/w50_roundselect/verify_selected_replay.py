"""Independent completed-period replay of selected priors/full fits and offsets."""
import argparse
import json
import os
from pathlib import Path

import experiment
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
        raise ValueError('CPU replay requires invisible CUDA')
    out = ROOT / 'runs' / f'roundselect_s{args.seed}'
    meta = json.loads((out / f'{args.period}_metadata.json').read_text())
    assignment = json.loads((ROOT / 'assignment.json').read_text())
    data_stat = Path(assignment['data']).stat()
    assert data_stat.st_size == meta['identity']['data']['bytes']
    assert data_stat.st_mtime_ns == meta['identity']['data']['mtime_ns']
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, args.seed, out.name)
    names = experiment.assert_parent_contract(core, config, args.seed)
    assert meta['features'] == names and meta['config'] == config
    train, truth = baseline.load_grid(assignment['data'], args.period, names, True)
    query, query_truth = baseline.load_grid(assignment['data'], args.period, names, False)
    assert query_truth is None
    reference, calibration, boundary = baseline.prior_split(train['origins'])
    offset_rows = calibration[-336:]
    stopping = calibration[train['origins'][calibration] + pd.Timedelta(hours=144)
                           <= train['origins'][offset_rows[0]]]
    assert len(stopping) == {'F1': 1706, 'F2': 1682, 'F3': 1730}[args.period]
    assert (train['origins'][reference] + pd.Timedelta(hours=144) <= boundary).all()
    model_dir = out / f'{args.period}_models'
    with np.load(model_dir / 'fitting_subsets.npz', allow_pickle=False) as saved:
        for key, indices in [('reference', reference), ('calibration', calibration),
                             ('stopping', stopping), ('offset', offset_rows)]:
            assert np.array_equal(saved[key], indices)
        assert np.array_equal(saved['train_origins'], train['origins'].asi8)
    heads = json.loads((model_dir / 'tree_selection.json').read_text())
    assert heads == meta['tree_selection'] and len(heads) == 3
    prior, endpoint, full = [], [], []
    curve_differences, rounds = [], {}
    for block, head in enumerate(heads):
        start = block * 24
        curve = np.asarray(head['native_rmse'])
        assert curve.shape == (900,) and np.isfinite(curve).all()
        count = int(np.argmin(curve)) + 1
        assert count == meta['selected_rounds'][block] == head['selected_rounds'] == head['full_rounds']
        assert head['inference_iteration_range'] == [0, count]
        design = core.design(train, start, 24)
        model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
        model.load_model(model_dir / 'prior' / f'b{block}.ubj')
        model.set_param({'device': 'cpu', 'nthread': 1})
        assert model.num_boosted_rounds() == 900 and model.num_features() == design.shape[1] == 212
        rounds[f'prior/{block}'] = model.num_boosted_rounds()
        endpoint.append(model.inplace_predict(design[calibration]).astype(np.float64))
        prior.append(model.inplace_predict(design[calibration], iteration_range=(0, count)).astype(np.float64))
        error = (model.inplace_predict(design[stopping], iteration_range=(0, count)).astype(np.float64)
                 - truth[stopping, start:start + 24].astype(np.float32))
        difference = abs(float(np.sqrt(np.mean(error**2))) - curve[count - 1])
        assert difference <= 1e-5
        curve_differences.append(difference)
        model.load_model(model_dir / 'full' / f'b{block}.ubj')
        model.set_param({'device': 'cpu', 'nthread': 1})
        assert model.num_boosted_rounds() == count and model.num_features() == 212
        rounds[f'full/{block}'] = model.num_boosted_rounds()
        full.append(model.inplace_predict(core.design(query, start, 24)).astype(np.float64))
    prior, endpoint, full = np.hstack(prior), np.hstack(endpoint), np.hstack(full)
    anchor = ROOT / 'frozen_controls' / f'drop035_v650_s{args.seed}' / f'{args.period}_models/prior_predictions.npz'
    with np.load(anchor, allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], train['origins'][calibration].asi8)
        assert np.array_equal(saved['training_truth'], truth[calibration])
        control_difference = float(np.max(np.abs(endpoint - saved['raw'])))
    assert control_difference <= 1e-6
    with np.load(model_dir / 'prior_predictions.npz', allow_pickle=False) as saved:
        prior_difference = float(np.max(np.abs(prior - saved['raw'])))
        assert np.array_equal(saved['origins'], train['origins'][calibration].asi8)
        assert np.array_equal(saved['training_truth'], truth[calibration])
    offset = .25 * (truth[offset_rows] - np.clip(prior[-336:], 1, 3000)).mean(0)
    with np.load(out / f'{args.period}_raw_predictions.npz', allow_pickle=False) as saved:
        offset_difference = float(np.max(np.abs(offset - saved['offset'])))
        full_difference = float(np.max(np.abs(full - saved['raw'])))
    prediction = np.clip(np.clip(full, 1, 3000) + offset, 1, 3000)
    with np.load(out / f'{args.period}_predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], query['origins'].asi8)
        prediction_difference = float(np.max(np.abs(prediction - saved['pred'])))
    assert max(prior_difference, offset_difference, full_difference, prediction_difference) <= 1e-6
    # Existing model-fitting subsets only: diagnose transfer from selection
    # data to later offset rows without creating an experiment-selection fold.
    fit_diagnostics = {}
    for name, indices in [('stopping', np.searchsorted(calibration, stopping)),
                          ('offset_before_fitting_offset', np.arange(len(calibration) - 336, len(calibration)))]:
        observed = truth[calibration][indices]
        values = {}
        for label, fitted in [('prior900', endpoint), ('selected_prior', prior)]:
            error = fitted[indices] - observed
            tail = observed >= 700.
            values[label] = {'rmse': float(np.sqrt(np.mean(error**2))),
                             'mae': float(np.mean(np.abs(error))), 'bias': float(error.mean()),
                             'tail_ge700_n': int(tail.sum()),
                             'tail_ge700_bias': float(error[tail].mean()) if tail.any() else None}
        fit_diagnostics[name] = {'origins': len(indices), 'raw_prior_metrics': values}
    result = {'status': 'pass', 'seed': args.seed, 'period': args.period,
              'model_rounds': rounds, 'selected_rounds': meta['selected_rounds'],
              'max_parent900_difference': control_difference, 'max_prior_difference': prior_difference,
              'max_offset_difference': offset_difference, 'max_full_raw_difference': full_difference,
              'max_final_difference': prediction_difference,
              'selected_native_rmse_replay_differences': curve_differences,
              'stopping_rows': len(stopping), 'offset_rows': 336,
              'training_only_transfer_diagnostics': fit_diagnostics,
              'fitting_subsets_and_purge_verified': True, 'query_truth_read': False}
    (out / f'{args.period}_model-replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
