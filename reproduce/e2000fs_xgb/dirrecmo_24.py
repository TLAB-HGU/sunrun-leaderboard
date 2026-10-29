"""e2000fs_dirrecmo_24: DirRecMO-24 XGBoost on all 111 features (90 origin features + 20 horizon features).

Three 24-output joint-tree blocks. Block b sees the origin features, the last 24 observed speeds, the trajectory of
all earlier blocks (observed values in training = teacher forcing, the model's own predictions at inference) and the
20 horizon features of its own 24 hours. F1 models use targets < 2026-02-26, F2 models targets < 2026-05-29.

    python reproduce/e2000fs_xgb/dirrecmo_24.py --cache reproduce/outputs/pointweight4_cache [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgb_common as C  # noqa: E402
from common import paths  # noqa: E402

LAG = 24


def features(bundle, traj, s, e):
    """DirRecMO block input: ctx | past | trajectory of horizons 1..s-1 | horizon features of s..e."""
    earlier = traj[:, LAG:LAG + s - 1]
    return np.hstack([bundle['C'], bundle['past'], earlier,
                      bundle['E'][:, s - 1:e].reshape(len(bundle['C']), -1)]).astype(np.float32)


def fit_block(tag, X, Y, Xv, Yv, params, rounds):
    model = xgb.XGBRegressor(**params)
    model.fit(X, Y, eval_set=[(X, Y), (Xv, Yv)], verbose=False)
    return model, C.loss_frame(model, rounds)


def take(bundle, index):
    return {k: bundle[k][index] for k in ('C', 'E', 'past')}


def run(cache, out, smoke=False, reference=None, workers=3):
    out = Path(out)
    (out / 'loss').mkdir(parents=True, exist_ok=True)
    started = time.time()
    names, bundles, series = C.load_cache(cache)
    params = C.xgb_params(smoke)
    train = bundles['train']
    oo_list, preds = [], []
    for stage, fit, val, target, oo, origins, y in C.stages(bundles, series, smoke):
        inp, inp_v = take(train, fit), take(train, val)
        traj = np.hstack([inp['past'].astype(np.float64), y[fit]])
        traj_v = np.hstack([inp_v['past'].astype(np.float64), y[val]])
        print(f'[{time.time() - started:6.0f}s] {stage}: fitting 3 blocks on {len(fit)} origins, rounds={params["n_estimators"]}', flush=True)
        jobs = [(f'{stage}_b{b:02}', features(inp, traj, s, e), y[fit, s - 1:e], features(inp_v, traj_v, s, e), y[val, s - 1:e])
                for b, (s, e) in enumerate(C.BLOCKS)]
        fitted = Parallel(n_jobs=min(workers, len(jobs)), backend='loky', max_nbytes='16M')(
            delayed(fit_block)(*job, params, params['n_estimators']) for job in jobs)
        for (tag, *_), (_, loss) in zip(jobs, fitted):
            loss.to_csv(out / 'loss' / f'{tag}_loss.csv', index=False)
        # inference: each block feeds on the trajectory predicted so far
        T = np.hstack([target['past'].astype(np.float64), np.full((len(oo), C.H), np.nan)])
        for b, (s, e) in enumerate(C.BLOCKS):
            T[:, LAG + s - 1:LAG + e] = np.asarray(fitted[b][0].predict(features(target, T, s, e)), dtype=np.float64).reshape(len(oo), -1)
        pred = np.clip(T[:, LAG:], 1., 3000.)
        assert np.isfinite(pred).all()
        oo_list.append(oo)
        preds.append(pred)
    result = C.finish(out, oo_list, preds, smoke, reference, 'dirrecmo_24')
    result['elapsed_seconds'] = round(time.time() - started, 1)
    (out / 'result.json').write_text(json.dumps(C.ev.json_safe(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=str(paths.OUTPUT_ROOT / 'pointweight4_cache'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reference', default=None, help='published forecast.npz to diff against')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    out = args.out or paths.OUTPUT_ROOT / ('dirrecmo_24_smoke' if args.smoke else 'dirrecmo_24')
    result = run(Path(args.cache), out, args.smoke, args.reference, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != 'official'}, indent=2))


if __name__ == '__main__':
    main()
