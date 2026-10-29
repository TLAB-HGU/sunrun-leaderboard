"""Shared helpers for the XGBoost block experiments (DirMO-24 on the Top-30 groups, DirRecMO-24 on all features)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parent / 'feat30_pointweight4'))
from common import paths  # noqa: E402
import evaluation as ev  # noqa: E402

CONFIG = json.loads((HERE / 'config/xgb.json').read_text())
FROZEN = json.loads((HERE.parent / 'feat30_pointweight4/config/frozen.json').read_text())
H = 72
BLOCKS = ((1, 24), (25, 48), (49, 72))


def xgb_params(smoke=False, n_jobs=None):
    params = dict(CONFIG['recipe'], n_jobs=n_jobs or CONFIG['n_jobs'], multi_strategy=CONFIG['multi_strategy'])
    if smoke:
        params['n_estimators'] = 2
    return params


def load_cache(cache):
    cache = Path(cache)
    names = json.loads((cache / 'names.json').read_text())
    bundles = {n: dict(np.load(cache / f'{n}.npz')) for n in ('train', 'f1', 'official')}
    series = pd.read_parquet(cache / 'speed.parquet')['speed']
    return names, bundles, series


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)


def stages(bundles, series, smoke=False):
    """Yield (stage, fit_idx, val_idx, target_bundle, target_origins, truth) for the F1 and F2 models."""
    train = bundles['train']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(series, origins)
    v = CONFIG['select_validation']
    val = np.flatnonzero((origins >= pd.Timestamp(v['start'])) & (origins <= pd.Timestamp(v['end'])))
    if smoke:
        val = val[::100]
    for stage, grid in (('f1', 'f1'), ('final', 'official')):
        fit = np.flatnonzero(origins + pd.Timedelta(hours=H) < pd.Timestamp(CONFIG['stage_cutoffs'][stage], tz='UTC'))
        assert len(fit) and len(val) and np.isfinite(y[fit]).all() and np.isfinite(y[val]).all()
        if smoke:
            fit = fit[::100]
        target = bundles[grid]
        yield stage, fit, val, target, pd.DatetimeIndex(target['origins'], tz='UTC'), origins, y


def loss_frame(model, rounds):
    result = model.evals_result()
    return pd.DataFrame({'round': np.arange(1, rounds + 1), 'train_mse': np.square(result['validation_0']['rmse']),
                         'validation_mse': np.square(result['validation_1']['rmse'])})


def finish(out, oo_list, preds, smoke, reference, name, expected_mse=None):
    """Assemble the F1+F2 leaderboard frame, score it with the frozen scorer, diff against the published forecast."""
    oo = oo_list[0].append(oo_list[1])
    pred = np.vstack(preds)
    frame = ev.frame(oo, pred)
    result = {'rows': len(frame)}
    if not smoke:
        assert len(frame) == 357840 and not frame.duplicated(ev.KEY).any()
        result['official'] = ev.official_score(frame)
        result['mse'] = result['official']['mse']
        if expected_mse is not None:
            np.testing.assert_allclose(result['mse'], expected_mse, rtol=1e-9)
    frame.to_parquet(Path(out) / 'predictions.parquet', index=False)
    np.savez(Path(out) / 'forecast.npz', origins_ns=oo.asi8, predictions=pred)
    if reference and not smoke:
        theirs = np.load(reference)
        np.testing.assert_array_equal(oo.asi8, theirs['origins_ns'])
        result['reference_max_abs_diff'] = float(np.abs(pred - theirs['predictions']).max())
    (Path(out) / 'result.json').write_text(json.dumps(ev.json_safe(result), indent=2) + '\n')
    return result
