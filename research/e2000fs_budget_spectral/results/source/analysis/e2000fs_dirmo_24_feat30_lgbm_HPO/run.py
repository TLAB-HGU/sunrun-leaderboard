"""Three independent resumable TPE studies; temporal validation then frozen refit."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm'
CACHE = ROOT / 'analysis/forecast_strategies_e2000/outputs/cache'
RUN = HERE / 'runs/main'
EXPERIMENT = 'e2000fs_dirmo_24_feat30_lgbm_HPO'
SEEDS = (42, 43, 44)
SEARCH_KEYS = ('n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'subsample', 'colsample_bytree', 'min_child_samples', 'min_child_weight', 'reg_alpha', 'reg_lambda')
spec = importlib.util.spec_from_file_location('frozen_lgbm', BASE / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write(path, value)


def temporal_split(origins):
    start = pd.Timestamp('2025-12-04', tz='UTC')
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < start)
    val = np.flatnonzero((origins >= start) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    assert len(fit) and len(val) and not np.intersect1d(fit, val).size
    assert (origins[fit] + pd.Timedelta(hours=72)).max() < origins[val].min()
    return fit, val


def load_data():
    manifest = read(BASE / 'logs/full_01/manifest.json')
    for p in [BASE / 'run.py', *sorted(CACHE.iterdir())]:
        assert base.digest(p) == manifest['hashes'][str(p)], f'Frozen source/input changed: {p}'
    names = read(CACHE / 'names.json')
    selected = manifest['selected']
    assert len(set(selected)) == 30
    train = dict(np.load(CACHE / 'train.npz'))
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed']
    y = base.targets(series, origins)
    assert np.isfinite(y).all()
    blocks = [base.design(train, names, selected, h) for h in (0, 24, 48)]
    assert all(x.shape == (len(origins), 237) for x in blocks)
    return manifest, names, origins, y, blocks


def parameters(trial):
    depth = trial.suggest_int('max_depth', 2, 8)
    return dict(n_estimators=trial.suggest_int('n_estimators', 300, 1800, step=100),
                learning_rate=trial.suggest_float('learning_rate', .01, .15, log=True),
                max_depth=depth, num_leaves=trial.suggest_int('num_leaves', 4, min(64, 2**depth)),
                subsample=trial.suggest_float('subsample', .6, 1.),
                colsample_bytree=trial.suggest_float('colsample_bytree', .5, 1.),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 200),
                min_child_weight=trial.suggest_float('min_child_weight', .001, 100., log=True),
                reg_alpha=trial.suggest_float('reg_alpha', 0., 20.),
                reg_lambda=trial.suggest_float('reg_lambda', 0., 50.))


def full_params(params):
    return dict(objective='regression', subsample_freq=1, random_state=42,
                device_type='cpu', verbosity=-1, n_jobs=4, **params)


def open_study(index, directory):
    # Each worker has a separate DB; no cross-study SQLite writer contention.
    return optuna.create_study(study_name=f'{EXPERIMENT}_study{index + 1}',
        storage=f'sqlite:///{directory / "optuna.sqlite3"}', direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEEDS[index]),
        pruner=optuna.pruners.NopPruner(), load_if_exists=True)


def run_study(index, smoke=False):
    directory = (HERE / 'runs/smoke' if smoke else RUN) / f'study{index + 1}'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, names, origins, y, blocks = load_data()
    fit, val = temporal_split(origins)
    if smoke:
        fit, val = fit[::100], val[::100]
    identity = dict(source_sha256=base.digest(__file__), parent_sha256=base.digest(BASE / 'logs/full_01/manifest.json'),
                    input_sha256={str(p): base.digest(p) for p in sorted(CACHE.iterdir())},
                    seed=SEEDS[index], model_seed=42, threads=4, goal=1 if smoke else 50,
                    fit_count=len(fit), validation_count=len(val),
                    fit_target_end=str((origins[fit] + pd.Timedelta(hours=72)).max()),
                    validation_start=str(origins[val].min()), validation_end=str(origins[val].max()),
                    smoke=smoke, optuna_version=optuna.__version__, lightgbm_version=lgb.__version__,
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    sampler_seed_policy='study_seed + 1009 * attempted_trial_count')
    if (directory / 'identity.json').exists():
        assert read(directory / 'identity.json') == identity, 'Refusing changed run identity'
    else:
        write(directory / 'identity.json', identity)
    study = open_study(index, directory)
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    baseline = {k: manifest['params'].get(k, 0.) for k in SEARCH_KEYS}
    study.enqueue_trial(baseline, skip_if_exists=True)
    fit_x, val_x = [x[fit] for x in blocks], [x[val] for x in blocks]
    def objective(trial):
        sampled = parameters(trial)
        params = full_params(sampled)
        if smoke:
            params['n_estimators'] = 2
        errors = []
        history_dir = directory / 'trials' / f'{trial.number:04}'
        history_dir.mkdir(parents=True, exist_ok=True)
        for h in range(72):
            history = {}
            model = lgb.LGBMRegressor(**params)
            model.fit(fit_x[h // 24], y[fit, h],
                      eval_set=[(fit_x[h // 24], y[fit, h]), (val_x[h // 24], y[val, h])],
                      eval_names=['train', 'validation'], eval_metric='l2',
                      callbacks=[lgb.record_evaluation(history)])
            pred = np.clip(model.predict(val_x[h // 24]), 1., 3000.)
            assert np.isfinite(pred).all()
            errors.append(float(np.mean((pred - y[val, h])**2)))
            pd.DataFrame({'round': np.arange(1, params['n_estimators'] + 1),
                'train_mse': history['train']['l2'], 'validation_mse': history['validation']['l2']}).to_csv(history_dir / f'h{h + 1:02}_loss.csv', index=False)
        value = float(np.mean(errors))
        write(history_dir / 'result.json', dict(params=sampled, value=value, mse_by_horizon=errors))
        trial.set_user_attr('mse_by_horizon', errors)
        return value
    goal = identity['goal']
    while len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < goal:
        # Recreate TPE with a deterministic per-trial seed, independent of process lifetime.
        attempt = len(study.trials)
        study.sampler = optuna.samplers.TPESampler(seed=SEEDS[index] + 1009 * attempt)
        before = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        study.optimize(objective, n_trials=1, catch=(RuntimeError, lgb.basic.LightGBMError), gc_after_trial=True)
        completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        write(directory / 'status.json', dict(status='completed' if completed == goal else 'running', completed=completed,
            attempted=len(study.trials), goal=goal, best_value=study.best_value if completed else None))
        if completed == before:
            recent = study.trials[-3:]
            if len(recent) == 3 and all(t.state == optuna.trial.TrialState.FAIL for t in recent):
                raise RuntimeError('Three consecutive failed trials; preserving study for diagnosis')
    trial = study.best_trial
    write(directory / 'best.json', dict(study=index + 1, trial=trial.number, value=trial.value, params=trial.params, completed=goal))
    study.trials_dataframe().to_csv(directory / 'trials.csv', index=False)


def select_best(winners):
    return min(winners, key=lambda w: (w['value'], w['study'], w['trial']))


def refit(best):
    directory = RUN / 'final'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'artifacts'
    output.mkdir(exist_ok=True)
    parent, names, origins, y, blocks = load_data()
    manifest = dict(parent)
    manifest.update(experiment=EXPERIMENT, params=full_params(best['params']), hpo_selection=best,
                    final_validation_note='Training-overlapping diagnostic only; no early stopping or parameter selection. HPO uses strict disjoint temporal validation.')
    manifest['hashes'] = dict(parent['hashes'])
    manifest['hashes'][str(Path(__file__))] = base.digest(__file__)
    if (directory / 'manifest.json').exists():
        assert read(directory / 'manifest.json') == manifest, 'Refusing changed final identity'
    write(directory / 'manifest.json', manifest)
    write(directory / 'status.json', dict(status='running'))
    _, diagnostic_val = temporal_split(origins)
    frames = []
    for stage, grid, cutoff, expected in [('f1', 'f1', '2026-02-26', 2137), ('final', 'official', '2026-05-29', 2833)]:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
        target = dict(np.load(CACHE / f'{grid}.npz'))
        oo = pd.DatetimeIndex(target['origins'], tz='UTC')
        assert len(oo) == expected and oo.is_unique and oo.is_monotonic_increasing
        query = [base.design(target, names, parent['selected'], h) for h in (0, 24, 48)]
        pred = np.empty((len(oo), 72))
        for h in range(72):
            path = output / f'{stage}_b{h // 24:02}_h{h + 1:02}.txt'
            if not path.exists():
                history = {}
                model = lgb.LGBMRegressor(**manifest['params'])
                model.fit(blocks[h // 24][fit], y[fit, h],
                    eval_set=[(blocks[h // 24][fit], y[fit, h]), (blocks[h // 24][diagnostic_val], y[diagnostic_val, h])], eval_names=['train', 'validation'],
                    callbacks=[lgb.record_evaluation(history)])
                pd.DataFrame({'round': np.arange(1, len(history['train']['l2']) + 1),
                              'train_mse': history['train']['l2'], 'validation_mse': history['validation']['l2']}).to_csv(directory / f'{stage}_h{h + 1:02}_loss.csv', index=False)
                temp = path.with_suffix('.txt.tmp')
                model.booster_.save_model(str(temp))
                reloaded = lgb.Booster(model_file=str(temp))
                np.testing.assert_array_equal(model.predict(query[h // 24]), reloaded.predict(query[h // 24], num_threads=4))
                temp.replace(path)
            pred[:, h] = np.clip(lgb.Booster(model_file=str(path)).predict(query[h // 24], num_threads=4), 1., 3000.)
        assert np.isfinite(pred).all()
        np.save(output / f'{stage}_predictions.npy', pred)
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(oo.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
            'horizon_hours': np.tile(np.arange(1, 73), len(oo)), 'pred_kms': pred.ravel()}))
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 357840 and not frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    frame.to_parquet(output / 'predictions.parquet', index=False)
    from huggingface_hub import hf_hub_download
    tables = {n: pd.read_parquet(hf_hub_download('tlabtlab/sunrun-lb-store', f'{n}.parquet', repo_type='dataset', revision=parent['evaluation_revision'])) for n in ('truth', 'folds', 'naive')}
    spec = importlib.util.spec_from_file_location('frozen_scoring', base.SCORER)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    assert base.digest(base.SCORER) == parent['hashes'][str(base.SCORER)]
    assert not scoring.validate(frame, tables['folds'])
    scores = scoring.score(frame, tables['truth'], tables['folds'], tables['naive'])
    joined = frame.merge(tables['truth'], on=scoring.KEY, validate='one_to_one')
    np.testing.assert_allclose(np.mean((joined.pred_kms - joined.target_kms)**2), scores['mse'], rtol=1e-9)
    old = read(BASE / 'logs/full_01/artifacts/scores.json')
    scores['parent_comparison'] = {k: dict(parent=old[k], candidate=scores[k], percent=100*(scores[k]/old[k]-1)) for k in ('mse', 'mse_f1', 'mse_f2')}
    scores['posthoc_official_exposed'] = True
    write(output / 'scores.json', scores)
    write(directory / 'status.json', dict(status='completed'))


def campaign(publish=False):
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes, handles = [], []
    for index in range(3):
        log = (RUN / f'study{index+1}.log').open('a')
        handles.append(log)
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__)), 'study', '--index', str(index)], stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'Study exit codes: {codes}')
    winners = [read(RUN / f'study{i+1}/best.json') for i in range(3)]
    best = dict(select_best(winners), study_winners=winners)
    write(RUN / 'best.json', best)
    refit(best)
    if publish:
        for step in ('prepare', 'timing', 'build', 'dryrun', 'push', 'github', 'submit', 'verify'):
            subprocess.run([sys.executable, str(HERE / 'publish.py'), step], check=True)
    write(RUN / 'status.json', dict(status='completed', published=publish, best=best))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['study', 'campaign'])
    parser.add_argument('--index', type=int, choices=range(3), default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    try:
        if args.step == 'study':
            run_study(args.index, args.smoke)
        else:
            campaign(args.publish)
    except BaseException:
        traceback.print_exc()
        raise
