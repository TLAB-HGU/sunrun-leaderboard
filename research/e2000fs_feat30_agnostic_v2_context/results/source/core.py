"""Shared frozen data, mature splits and checkpointed CPU training."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = HERE.parent / 'e2000fs_feat30_followups'
PARENT = HERE.parent / 'e2000fs_dirmo_24_feat30/logs/full_01'
CACHE = HERE.parent / 'forecast_strategies_e2000/outputs/cache'
FOLDS = [('fold0', '2025-06-01', '2025-06-04', '2025-08-22'),
         ('fold1', '2025-09-01', '2025-09-04', '2025-11-22'),
         ('fold2', '2025-12-01', '2025-12-04', '2026-02-22')]
CUTOFFS = {name: cutoff for name, cutoff, _, _ in FOLDS} | {'f1': '2026-02-26', 'final': '2026-05-29'}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (Path, pd.Timestamp)):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')
    temp.replace(path)


def save(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def status(directory, phase, state='running', **extra):
    record = dict(status=state, phase=phase, pid=os.getpid(), utc=pd.Timestamp.now(tz='UTC').isoformat(), **extra)
    write(Path(directory) / 'status.json', record)
    print(json.dumps(record), flush=True)


def manifest(smoke):
    parent = json.loads((PARENT / 'manifest.json').read_text())
    for path, sha in parent['hashes'].items():
        if not Path(path).is_file() or digest(path) != sha:
            raise RuntimeError(f'Frozen parent source/input changed: {path}')
    paths = [*sorted(HERE.glob('*.py')), *sorted(CACHE.glob('*.npz')), CACHE / 'speed.parquet',
             CACHE / 'names.json', CACHE / 'receipt.json', PARENT / 'manifest.json',
             V1 / 'run.py', V1 / 'evaluation.py',
             *sorted((PARENT / 'artifacts').glob('*.ubj')),
             *sorted((PARENT / 'artifacts').glob('*predictions.npy'))]
    hashes = {str(p): digest(p) for p in paths}
    result = dict(protocol='feat30-agnostic-v2', smoke=smoke, hashes=hashes, threads=3,
                  folds=[list(f) for f in FOLDS], cutoffs=CUTOFFS, mse_budget=1.03, selected_features=parent['selected'],
                  seed=42, rounds=3 if smoke else 900, python=sys.version, xgboost=xgb.__version__,
                  posthoc_official_exposed=True, no_upload=True, no_neural=True,
                  study_id_not_expos_eid=True, parent_mse=3249.842329310296)
    result['identity'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def freeze(directory, record):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError(f'Refusing changed code/data/config/mode: {directory}')
    write(path, record)


def load_data():
    return import_file('feat30_v1_data', V1 / 'run.py').load_data()[2]


def take(bundle, indices):
    return {k: v[indices] for k, v in bundle.items()}


def design(bundle, block=None):
    e = bundle['E'] if block is None else bundle['E'][:, block * 24:(block + 1) * 24]
    return np.hstack([bundle['C'], e.reshape(len(e), -1)]).astype(np.float32)


def split(data, stage, smoke=False):
    oo = data['train']['origins']
    fit = np.flatnonzero(oo + pd.Timedelta(hours=72) < pd.Timestamp(CUTOFFS[stage], tz='UTC'))
    if stage.startswith('fold'):
        _, _, lo, hi = next(f for f in FOLDS if f[0] == stage)
        valid = np.flatnonzero((oo >= pd.Timestamp(lo, tz='UTC')) &
                               (oo < pd.Timestamp(hi, tz='UTC') + pd.Timedelta(days=1)))
    else:
        valid = split(data, 'fold2', False)[1]
    if smoke:
        # Sparse training sample spans full history, preserving enough purge gaps.
        fit = fit[np.linspace(0, len(fit)-1, min(64, len(fit))).astype(int)]
        valid = valid[np.linspace(0, len(valid)-1, min(16, len(valid))).astype(int)]
    if not len(fit) or not len(valid):
        raise RuntimeError(f'Empty split {stage}')
    assert (oo[fit] + pd.Timedelta(hours=72) < pd.Timestamp(CUTOFFS[stage], tz='UTC')).all()
    if stage.startswith('fold'):
        assert oo[fit].max() + pd.Timedelta(hours=72) < oo[valid].min()
    return fit, valid


def params(smoke):
    result = dict(json.loads((PARENT / 'manifest.json').read_text())['params'])
    result.update(n_jobs=3, n_estimators=3 if smoke else 900)
    return result


def fit_model(directory, X, y, Xv, yv, smoke, overrides=None, classifier=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    modelpath = directory / 'model.ubj'; receiptpath = directory / 'receipt.json'
    if receiptpath.exists():
        receipt = json.loads(receiptpath.read_text())
        if not modelpath.exists() or digest(modelpath) != receipt['sha256']:
            raise RuntimeError(f'Corrupt checkpoint {directory}')
        booster = xgb.Booster(model_file=str(modelpath)); booster.set_param({'nthread': 3})
        return booster
    if not np.isfinite(y).all() or not np.isfinite(yv).all() or np.isinf(X).any() or np.isinf(Xv).any():
        raise ValueError('Invalid targets or infinite inputs; native NaN inputs are preserved')
    p = params(smoke)
    if classifier:
        p.pop('multi_strategy', None)
        p.update(objective='binary:logistic', eval_metric='logloss', base_score=float(np.clip(np.mean(y), 1e-4, 1-1e-4)))
    else:
        p.update(base_score=float(np.asarray(y, dtype=np.float32).mean()))
    p.update(overrides or {})
    print(f'FIT {directory}: X={X.shape}, y={y.shape}', flush=True)
    model = (xgb.XGBClassifier if classifier else xgb.XGBRegressor)(**p)
    model.fit(X, np.asarray(y, dtype=np.float32), eval_set=[(X, np.asarray(y, dtype=np.float32)),
                                                         (Xv, np.asarray(yv, dtype=np.float32))], verbose=100)
    histories = model.evals_result()
    columns = {}
    for label, data in histories.items():
        for metric, values in data.items():
            columns[('train' if label == 'validation_0' else 'validation') + '_' + metric] = values
    history = pd.DataFrame(columns); history.insert(0, 'round', np.arange(1, len(history)+1))
    history.to_csv(directory / 'loss.csv', index=False)
    temp = directory / 'model.tmp.ubj'; model.get_booster().save_model(temp)
    reload = xgb.Booster(model_file=str(temp)); reload.set_param({'nthread': 3})
    d = xgb.DMatrix(Xv, nthread=3)
    np.testing.assert_array_equal(model.get_booster().predict(d), reload.predict(d))
    temp.replace(modelpath)
    write(receiptpath, dict(sha256=digest(modelpath), reload_parity=True, rows=len(X), targets=1 if y.ndim==1 else y.shape[1],
                            columns=X.shape[1], native_loss_history=True, validation_may_overlap_final_fit=True))
    return reload


def prediction(model, X):
    return model.predict(xgb.DMatrix(X, nthread=3))


def parent_predict(bundle, stage=None, directory=None, train=None, valid=None, smoke=False):
    output = []
    for b in range(3):
        if stage in ('f1', 'final'):
            model = xgb.Booster(model_file=str(PARENT / 'artifacts' / f'{stage}_b{b:02}.ubj'))
            model.set_param({'nthread': 3})
        else:
            model = fit_model(Path(directory) / stage / f'b{b:02}', design(train, b), train['Y'][:, b*24:(b+1)*24],
                              design(valid, b), valid['Y'][:, b*24:(b+1)*24], smoke)
        output.append(prediction(model, design(bundle, b)))
    return np.clip(np.hstack(output), 1, 3000)


def wait_common(common, identity):
    while True:
        path = Path(common) / 'status.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['status'] == 'failed':
                raise RuntimeError('Shared preparation failed: ' + str(record))
            if record['status'] == 'completed':
                if json.loads((Path(common)/'manifest.json').read_text())['identity'] != identity:
                    raise RuntimeError('Common baseline identity mismatch')
                return
        print('Waiting for frozen three-fold common baseline', flush=True)
        time.sleep(10)
