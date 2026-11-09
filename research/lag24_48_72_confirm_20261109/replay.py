"""Reproduce full saved-model forecasts using the additive input contract."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
import xgboost as xgb
import baseline
import core


def replay(run, data, parent=False):
    results = {}
    for period in baseline.CUTOFFS:
        meta = json.loads((run / f'{period}_metadata.json').read_text())
        config = core.normalize_config({'seed': meta['seed'], 'add_features': []}) if parent else meta['config']
        names = core.feature_names(config)
        grid, truth = baseline.load_grid(data, period, names, False)
        if truth is not None:
            raise ValueError('Query truth leaked into inference')
        raw = np.empty((len(grid['origins']), 72))
        for block, start in enumerate((0, 24, 48)):
            model = xgb.Booster(params={'nthread': 1, 'device': 'cpu'})
            model.load_model(run / f'{period}_models/full/b{block}.ubj')
            if model.num_boosted_rounds() != 900:
                raise ValueError('Not a full 900-round booster')
            raw[:, start:start + 24] = model.inplace_predict(core.design(grid, start, 24))
        offset = np.asarray(meta['frozen_state']['offset'])
        prediction, _ = core.apply_offset(raw, offset, config)
        with np.load(run / f'{period}_predictions.npz') as saved:
            if not np.array_equal(saved['origins'], grid['origins'].asi8):
                raise ValueError('Replay origin mismatch')
            error = float(np.max(np.abs(saved['pred'] - prediction)))
        if error > 1e-6:
            raise ValueError(f'{period} saved model replay error {error}')
        results[period] = {'max_abs_error': error, 'boosters': 3, 'rounds': 900}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent', action='store_true')
    args = parser.parse_args()
    print(json.dumps(replay(args.run, args.data, args.parent)))
