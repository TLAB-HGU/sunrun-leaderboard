"""e2000fs_feat30_pointweight4: per-horizon LightGBM with pointwise extreme weighting, train -> predict -> score.

One independent model per horizon (72) on the frozen Top-30 feature groups, three stages (internal
validation, F1, final/F2). Training weight is 4 if this horizon's target is >= 550 km/s else 1 (mean
normalized). HPO parameters are frozen in ``config/frozen.json``; nothing is tuned here.

    python reproduce/feat30_pointweight4/train_predict.py --cache reproduce/outputs/pointweight4_cache
    python reproduce/feat30_pointweight4/train_predict.py --smoke --cache <cache>   # 3 trees, tiny subsets
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
from common import paths  # noqa: E402
import evaluation as ev  # noqa: E402

FROZEN = json.loads((HERE / 'config/frozen.json').read_text())
H = 72


def point_weights(target):
    """Only this horizon's observed high target receives extra training weight."""
    target = np.asarray(target)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError('Expected finite one-dimensional training targets')
    weights = np.where(target >= FROZEN['point_weight']['threshold_kms'], FROZEN['point_weight']['weight'], 1.)
    return weights / weights.mean()


def targets(series, origins):
    times = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, H)


def design(bundle, names, selected, start):
    ci = [i for i, n in enumerate(names['constant']) if n in selected]
    ei = [i for i, n in enumerate(names['varying']) if n in selected]
    return np.hstack([bundle['C'][:, ci], bundle['E'][:, start:start + 24, :][:, :, ei].reshape(len(bundle['C']), -1)]).astype(np.float32)


def temporal_split(origins):
    start = pd.Timestamp(FROZEN['internal_validation']['start'])
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < start)
    val = np.flatnonzero((origins >= start) & (origins <= pd.Timestamp(FROZEN['internal_validation']['end'])))
    assert len(fit) and len(val) and not np.intersect1d(fit, val).size
    assert (origins[fit] + pd.Timedelta(hours=72)).max() < origins[val].min()
    return fit, val


def load_data(cache):
    names = json.loads((cache / 'names.json').read_text())
    selected = FROZEN['selected_groups']
    assert len(set(selected)) == 30
    train = dict(np.load(cache / 'train.npz'))
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    series = pd.read_parquet(cache / 'speed.parquet')['speed']
    y = targets(series, origins)
    assert np.isfinite(y).all()
    blocks = [design(train, names, selected, h) for h in (0, 24, 48)]
    assert all(x.shape == (len(origins), 237) for x in blocks)
    return names, selected, origins, series, y, blocks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fit_horizon(x, y, xv, yv, params, weights, folder):
    history = {}
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y, sample_weight=weights, eval_set=[(x, y), (xv, yv)], eval_names=['train', 'validation'],
              eval_metric='l2', callbacks=[lgb.record_evaluation(history)])
    tmp = folder / 'model.txt.tmp'
    model.booster_.save_model(str(tmp))
    restored = lgb.Booster(model_file=str(tmp))
    np.testing.assert_allclose(model.predict(xv), restored.predict(xv, num_threads=2), rtol=0, atol=0)
    tmp.replace(folder / 'model.txt')
    pd.DataFrame(dict(round=np.arange(1, len(history['train']['l2']) + 1), train_mse=history['train']['l2'],
                      validation_mse=history['validation']['l2'])).to_csv(folder / 'loss.csv', index=False)
    return restored


def point_metrics(y, prediction, origins):
    result = ev.metrics(y, prediction, origins)
    low = y < 550
    result['mse_lt550'] = float(np.mean((prediction[low] - y[low]) ** 2)) if low.any() else None
    result['n_pairs_lt550'] = int(low.sum())
    result['bias_lt550'] = float(np.mean(prediction[low] - y[low])) if low.any() else None
    return result


def execute(cache, out, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    names, selected, origins, series, y, blocks = load_data(cache)
    fit, val = temporal_split(origins)
    params = dict(FROZEN['params'], n_jobs=FROZEN['n_jobs'])
    if smoke:
        params['n_estimators'] = 3
        fit, val = fit[::max(1, len(fit) // 128)], val[::max(1, len(val) // 24)]
    stages = [('internal', fit, [x[val] for x in blocks], origins[val], y[val])]
    for stage, grid, cutoff in [('f1', 'f1', FROZEN['stage_cutoffs']['f1_train_cutoff_utc']),
                                ('final', 'official', FROZEN['stage_cutoffs']['final_train_cutoff_utc'])]:
        train = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
        bundle = dict(np.load(cache / f'{grid}.npz'))
        oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
        query = [design(bundle, names, selected, b) for b in (0, 24, 48)]
        yy = targets(series, oo)
        if smoke:
            train = train[::max(1, len(train) // 128)]
            oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
        stages.append((stage, train, query, oo, yy))
    results, outputs = {}, {}
    for stage, train, query, oo, yy in stages:
        assert (origins[train] + pd.Timedelta(hours=72)).max() < oo.min()
        pred = np.empty_like(yy)
        for h in range(H):
            folder = out / stage / f'h{h + 1:02}'
            folder.mkdir(parents=True, exist_ok=True)
            model_path, receipt = folder / 'model.txt', folder / 'receipt.json'
            if receipt.exists():  # resume an interrupted run
                assert json.loads(receipt.read_text())['sha256'] == digest(model_path)
                model = lgb.Booster(model_file=str(model_path))
            else:
                print(f'[{time.time() - started:7.0f}s] {stage} h{h + 1:02}/{H}', flush=True)
                # Refits use training-only diagnostics; internal validation is purged.
                diag = val if stage == 'internal' else train[:min(648, len(train))]
                model = fit_horizon(blocks[h // 24][train], y[train, h], blocks[h // 24][diag], y[diag, h],
                                    params, point_weights(y[train, h]), folder)
                write_json(receipt, dict(sha256=digest(model_path), train_count=len(train),
                                         validation_held_out=stage == 'internal'))
            pred[:, h] = np.clip(model.predict(query[h // 24], num_threads=2), 1., 3000.)
        assert np.isfinite(pred).all()
        np.savez_compressed(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
        results[stage] = point_metrics(yy, pred, oo)
        outputs[stage] = (oo, yy, pred)
        write_json(out / 'results.json', results)
    oo = outputs['f1'][0].append(outputs['final'][0])
    yy = np.vstack([outputs[s][1] for s in ('f1', 'final')])
    pred = np.vstack([outputs[s][2] for s in ('f1', 'final')])
    results['combined'] = point_metrics(yy, pred, oo)
    frame = ev.frame(oo, pred)
    if not smoke:
        assert len(frame) == FROZEN['expected']['rows'] and not frame.duplicated(ev.KEY).any()
        results['combined']['official'] = ev.official_score(frame)
        np.testing.assert_allclose(results['combined']['mse'], results['combined']['official']['mse'], rtol=1e-12)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    np.savez(out / 'forecast.npz', origins_ns=oo.asi8, predictions=pred)
    results['elapsed_seconds'] = round(time.time() - started, 1)
    write_json(out / 'results.json', results)
    return results


def compare_reference(out, reference):
    """Largest absolute difference to a published ``forecast.npz`` (same origins, same 72 horizons)."""
    ours, theirs = np.load(out / 'forecast.npz'), np.load(reference)
    np.testing.assert_array_equal(ours['origins_ns'], theirs['origins_ns'])
    return float(np.abs(ours['predictions'] - theirs['predictions']).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=str(paths.OUTPUT_ROOT / 'pointweight4_cache'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reference', default=None, help='published forecast.npz to diff against')
    args = parser.parse_args()
    out = Path(args.out) if args.out else paths.OUTPUT_ROOT / ('pointweight4_smoke' if args.smoke else 'pointweight4')
    results = execute(Path(args.cache), out, args.smoke)
    print(json.dumps({k: results[k].get('mse') if isinstance(results[k], dict) else results[k] for k in results}, indent=2))
    if args.reference:
        print('max |prediction - reference| =', compare_reference(out, args.reference))


if __name__ == '__main__':
    main()
