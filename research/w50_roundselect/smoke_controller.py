"""Bounded synthetic CPU preflight on the actual training host."""
import json
import os
from pathlib import Path

import experiment
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preflight requires invisible CUDA')
    rng = np.random.default_rng(20261108)
    x = rng.normal(size=(128, 212)).astype(np.float32)
    xs = rng.normal(size=(32, 212)).astype(np.float32)
    coefficient = rng.normal(size=(212, 24))
    y = 450 + 8 * x @ coefficient
    ys = 450 + 8 * xs @ coefficient
    _, core = experiment.parent_modules()
    config = experiment.fixed_config(core, 42, 'roundselect_s42')
    with_eval, curve, _ = experiment.train_head(core, x, y, config, 1, 12, stopping=(xs, ys))
    without_eval, _, _ = experiment.train_head(core, x, y, config, 1, 12)
    assert with_eval.num_boosted_rounds() == 12 and len(curve) == 12
    endpoint_difference = float(np.max(np.abs(with_eval.inplace_predict(xs) - without_eval.inplace_predict(xs))))
    assert endpoint_difference == 0
    selected = experiment.first_minimum(curve, 12)
    prediction = with_eval.inplace_predict(xs, iteration_range=(0, selected))
    out = ROOT / 'smoke'
    out.mkdir(exist_ok=True)
    with_eval.save_model(out / 'prior.ubj')
    replay = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    replay.load_model(out / 'prior.ubj')
    replay.set_param({'device': 'cpu', 'nthread': 1})
    prefix_difference = float(np.max(np.abs(prediction - replay.inplace_predict(xs, iteration_range=(0, selected)))))
    assert prefix_difference == 0 and prediction.shape == (32, 24)
    result = {'status': 'pass', 'kind': 'synthetic compatibility only, not a scientific experiment',
              'device': 'cpu', 'threads': 1, 'xgboost': xgb.__version__,
              'inputs': 212, 'outputs': 24, 'smoke_rounds': 12,
              'eval_matrix_endpoint_difference': endpoint_difference,
              'prefix_serialization_difference': prefix_difference, 'selected_smoke_prefix': selected,
              'official_data_or_truth_read': False}
    (ROOT / 'remote-smoke-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
