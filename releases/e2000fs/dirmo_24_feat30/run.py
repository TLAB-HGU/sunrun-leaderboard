"""Frozen post-hoc Top30 DirMO-24 ablation. Run with --smoke before full execution."""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'analysis/forecast_strategies_e2000'
CACHE = PARENT / 'outputs/cache'
IMPORTANCE = ROOT / 'analysis/e2000fs-visualizations/worktree/docs/visualizations/e2000fs_dirmo_24/feature_importance.csv'
RECEIPT = IMPORTANCE.with_name('feature_importance_receipt.json')
SCORER = PARENT / 'publication/worktree/scorer/scoring.py'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    tmp.replace(path)


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


def design(bundle, names, selected, start):
    ci = [i for i, n in enumerate(names['constant']) if n in selected]
    ei = [i for i, n in enumerate(names['varying']) if n in selected]
    return np.hstack([bundle['C'][:, ci], bundle['E'][:, start:start + 24, :][:, :, ei].reshape(len(bundle['C']), -1)]).astype(np.float32)


def main(args, logs):
    t0 = time.monotonic()
    ranking = pd.read_csv(IMPORTANCE).pivot(index='feature', columns='stage', values='perm_delta_mse_mean')
    ranking['weighted_delta_mse'] = (ranking.f1 * 2137 + ranking.final * 2833) / 4970
    ranking = ranking.reset_index().sort_values(['weighted_delta_mse', 'feature'], ascending=[False, True])
    selected = ranking.head(30).feature.tolist()
    ranking.to_csv(logs / 'feature_ranking.csv', index=False)
    names = json.loads((CACHE / 'names.json').read_text())
    assert len(set(selected)) == 30
    assert sum(n in selected for n in names['constant']) == 21
    assert sum(n in selected for n in names['varying']) == 9
    assert set(selected) <= set(names['constant'] + names['varying'])
    params = json.loads(RECEIPT.read_text())['recipe']
    params.update(n_jobs=4, n_estimators=2 if args.smoke else 900, multi_strategy='multi_output_tree')
    manifest = {'experiment': HERE.name, 'posthoc_official_exposed': True, 'selected': selected,
                'ranking': '(2137*F1+2833*F2)/4970', 'columns_per_block': 237, 'params': params,
                'python': sys.version, 'platform': platform.platform(), 'xgboost': xgb.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__, 'argv': sys.argv,
                'hashes': {str(p): digest(p) for p in [Path(__file__), IMPORTANCE, RECEIPT, SCORER, *sorted(CACHE.iterdir())] if p.is_file()}}
    write(logs / 'manifest.json', manifest)
    bundles = {n: dict(np.load(CACHE / f'{n}.npz')) for n in ('train', 'f1', 'official')}
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed']
    train = bundles['train']
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y = targets(series, origins)
    val = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    if args.smoke:
        val = val[::100]
    output = logs / 'artifacts'
    output.mkdir()
    predictions = []
    for stage, grid, cutoff, expected in [('f1', 'f1', '2026-02-26', 2137), ('final', 'official', '2026-05-29', 2833)]:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
        assert len(fit) and len(val) and np.isfinite(y[fit]).all() and np.isfinite(y[val]).all()
        if args.smoke:
            fit = fit[::100]
        target = bundles[grid]
        oo = pd.DatetimeIndex(target['origins'], tz='UTC')
        assert len(oo) == expected and oo.is_unique and oo.is_monotonic_increasing
        pred = np.empty((len(oo), 72))
        for b, start in enumerate((0, 24, 48)):
            tag = f'{stage}_b{b:02}'
            print(f'{tag}: fitting {len(fit)} origins, rounds={params["n_estimators"]}', flush=True)
            X = design(train, names, selected, start)
            Xt = design(target, names, selected, start)
            assert X.shape[1] == Xt.shape[1] == 237
            model = xgb.XGBRegressor(**params)
            model.fit(X[fit], y[fit, start:start + 24], eval_set=[(X[fit], y[fit, start:start + 24]), (X[val], y[val, start:start + 24])], verbose=True)
            ev = model.evals_result()
            pd.DataFrame({'round': np.arange(1, params['n_estimators'] + 1), 'train_mse': np.square(ev['validation_0']['rmse']), 'validation_mse': np.square(ev['validation_1']['rmse'])}).to_csv(logs / f'{tag}_loss.csv', index=False)
            path = output / f'{tag}.ubj'
            model.get_booster().save_model(path)
            raw = model.predict(Xt)
            reloaded = xgb.Booster(model_file=path)
            np.testing.assert_array_equal(raw, reloaded.inplace_predict(Xt))
            pred[:, start:start + 24] = np.clip(raw, 1., 3000.)
            print(f'{tag}: saved; reload parity passed', flush=True)
        assert np.isfinite(pred).all()
        np.save(output / f'{stage}_predictions.npy', pred)
        predictions.append(pd.DataFrame({'origin_last_input_utc': np.repeat(oo.strftime('%Y-%m-%dT%H:%M:%SZ'), 72), 'horizon_hours': np.tile(np.arange(1, 73), len(oo)), 'pred_kms': pred.ravel()}))
    frame = pd.concat(predictions, ignore_index=True)
    assert len(frame) == 357840 and not frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    frame.to_parquet(output / 'predictions.parquet', index=False)
    if args.smoke:
        print('SMOKE PASSED: six fits, feature mapping, cutoffs, grid and reload parity', flush=True)
    else:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info('tlabtlab/sunrun-lb-store').sha
        tables = {}
        for name in ('truth', 'folds', 'naive'):
            path = Path(hf_hub_download('tlabtlab/sunrun-lb-store', f'{name}.parquet', repo_type='dataset', revision=revision))
            tables[name] = pd.read_parquet(path)
            manifest['hashes'][str(path)] = digest(path)
        spec = importlib.util.spec_from_file_location('frozen_scoring', SCORER)
        scoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoring)
        errors = scoring.validate(frame, tables['folds'])
        if errors:
            raise ValueError(errors)
        result = scoring.score(frame, tables['truth'], tables['folds'], tables['naive'])
        joined = frame.merge(tables['truth'], on=scoring.KEY, validate='one_to_one')
        assert len(joined) == len(frame)
        np.testing.assert_allclose(np.mean((joined.pred_kms - joined.target_kms) ** 2), result['mse'], rtol=1e-9)
        parent = {'mse': 3480.2701508981454, 'mse_f1': 3878.8761443485832, 'mse_f2': 3179.592068298927}
        result['parent_comparison'] = {k: {'parent': v, 'candidate': result[k], 'delta': result[k] - v, 'percent': 100 * (result[k] / v - 1)} for k, v in parent.items()}
        result['posthoc_official_exposed'] = True
        write(output / 'scores.json', result)
        manifest['evaluation_revision'] = revision
        write(logs / 'manifest.json', manifest)
        print(json.dumps(result['parent_comparison'], indent=2), flush=True)
    write(logs / 'status.json', {'status': 'completed', 'exit_code': 0, 'elapsed_seconds': time.monotonic() - t0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a directory name')
    logs = HERE / 'logs' / args.run_id
    logs.mkdir(exist_ok=False)
    # Redirect native descriptors too: warnings, child/native failures and tracebacks share the run log.
    fd = os.open(logs / 'console.log', os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    faulthandler.enable()
    write(logs / 'status.json', {'status': 'running', 'pid': os.getpid(), 'smoke': args.smoke})
    try:
        main(args, logs)
    except BaseException:
        traceback.print_exc()
        write(logs / 'status.json', {'status': 'failed', 'exit_code': 1})
        raise
