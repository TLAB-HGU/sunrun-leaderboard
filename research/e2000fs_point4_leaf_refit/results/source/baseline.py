"""Frozen seasonal pointweight4 comparator; no official fitting or scoring."""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import shutil
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analysis/e2000fs_feat30_pointweight/run.py'
spec = importlib.util.spec_from_file_location('advance_pointweight', SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
d, np, pd, lgb = p.d, p.np, p.pd, p.lgb
REFERENCE = SOURCE.parent / 'runs/full_20261028_v1'
FOLDS = [('dev0', '2024-12-04', '2025-02-22'), ('dev1', '2025-03-04', '2025-05-22'),
         ('dev2', '2025-09-04', '2025-11-22'), ('confirmation', '2025-12-04', '2026-02-22')]
point_weights = p.point_weights


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if d.digest(path) != expected:
            raise ValueError(f'Frozen file changed: {path}')


def mature_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=72) < cutoff)


def checkpoint(folder, context):
    receipt = d.read(folder / 'receipt.json')
    if receipt['context'] != context:
        raise ValueError('Checkpoint identity changed')
    verify_hashes({str(folder / name): digest for name, digest in receipt['sha256'].items()})
    return lgb.Booster(model_file=str(folder / 'model.txt'))


def execute(smoke=False):
    out = HERE / 'runs' / ('smoke_20261029_v1' if smoke else 'full_20261029_v1') / 'baseline'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='verify_inputs')
            parent = d.read(REFERENCE / 'manifest.json')
            verify_hashes(parent['source_sha256'])
            verify_hashes(parent['input_sha256'])
            hpo = d.imported('advance_baseline_hpo', d.HPO / 'run.py')
            ev = d.imported('advance_baseline_evaluation', d.FOLLOW / 'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            distribution = d.read(p.REFERENCE / 'manifest.json')
            assert data_parent == distribution['parent']
            params = dict(parent['params'], n_jobs=2)
            if smoke:
                params['n_estimators'] = 3
            paths = [Path(__file__), SOURCE, Path(d.__file__), d.HPO / 'run.py',
                     d.FOLLOW / 'evaluation.py', REFERENCE / 'manifest.json',
                     REFERENCE / 'results.json', REFERENCE / 'internal_predictions.npz']
            paths += sorted((REFERENCE / 'internal').glob('h*/model.txt'))
            paths += sorted((REFERENCE / 'internal').glob('h*/receipt.json'))
            paths += sorted((REFERENCE / 'internal').glob('h*/loss.csv'))
            hashes = {str(path): d.digest(path) for path in paths}
            manifest = dict(experiment='e2000fs_feat30_pointweight4_seasonal_baseline', smoke=smoke,
                            parent_experiment=parent['experiment'], params=params,
                            source_sha256=hashes, input_sha256=parent['input_sha256'],
                            folds=[list(row) for row in FOLDS], weighting=parent['weighting'],
                            confirmation='original internal predictions; previously exposed',
                            versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__))
            if (out / 'manifest.json').exists():
                if d.read(out / 'manifest.json') != manifest:
                    raise ValueError('Run identity changed; preserve existing run')
            else:
                d.write(out / 'manifest.json', manifest)
            identity = d.digest(out / 'manifest.json')
            results, outputs = {}, {}
            for stage, start, end in FOLDS:
                cutoff = pd.Timestamp(start, tz='UTC')
                train = mature_indices(origins, cutoff)
                ix = np.flatnonzero((origins >= cutoff) &
                                    (origins < pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)))
                if smoke:
                    train = train[np.linspace(0, len(train)-1, min(512, len(train)), dtype=int)]
                    ix = ix[:16]
                oo, yy = origins[ix], y[ix]
                assert len(train) and len(ix) and (origins[train] + pd.Timedelta(hours=72)).max() < oo.min()
                reuse = stage == 'confirmation' and not smoke
                if reuse:
                    old_train, old_val = hpo.temporal_split(origins)
                    np.testing.assert_array_equal(old_train, train)
                    np.testing.assert_array_equal(old_val, ix)
                    saved = np.load(REFERENCE / 'internal_predictions.npz')
                    np.testing.assert_array_equal(saved['origins'], oo.asi8)
                pred = np.empty_like(yy)
                for h in range(72):
                    folder = out / stage / f'h{h+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    context = dict(manifest_sha256=identity, stage=stage, horizon=h+1,
                                   train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest())
                    if (folder / 'receipt.json').exists():
                        model = checkpoint(folder, context)
                    else:
                        d.status(out, 'running', stage=stage, horizon=h+1)
                        if reuse:
                            source = REFERENCE / 'internal' / f'h{h+1:02}'
                            receipt = d.read(source / 'receipt.json')
                            assert receipt['sha256'] == d.digest(source / 'model.txt')
                            assert receipt['train_count'] == len(train) and receipt['validation_held_out']
                            for name in ('model.txt', 'loss.csv'):
                                shutil.copyfile(source / name, folder / name)
                            model = lgb.Booster(model_file=str(folder / 'model.txt'))
                        else:
                            model = d.controls_fit(blocks[h//24][train], y[train, h],
                                                   blocks[h//24][ix], yy[:, h], params,
                                                   point_weights(y[train, h]), folder)
                        d.write(folder / 'receipt.json', dict(context=context, reused=reuse,
                                sha256={name:d.digest(folder/name) for name in ('model.txt', 'loss.csv')}))
                    pred[:, h] = np.clip(model.predict(blocks[h//24][ix], num_threads=2), 1., 3000.)
                assert np.isfinite(pred).all()
                results[stage] = d.point_metrics(ev, yy, pred, oo)
                if reuse:
                    np.testing.assert_array_equal(pred, saved['pred'])
                    old_metrics = d.read(REFERENCE / 'results.json')['internal']
                    for key, value in results[stage].items():
                        assert value == old_metrics[key], key
                results[stage]['training'] = dict(rows=len(train), last_origin=str(origins[train].max()),
                    last_target=str((origins[train]+pd.Timedelta(hours=72)).max()), reused=reuse)
                d.array_save(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
                outputs[stage] = (oo, yy, pred)
                d.write(out / 'results.json', results)
            stages = ['dev0', 'dev1', 'dev2']
            oo = outputs['dev0'][0].append([outputs[s][0] for s in stages[1:]])
            yy, pred = [np.vstack([outputs[s][i] for s in stages]) for i in (1, 2)]
            results['pooled'] = d.point_metrics(ev, yy, pred, oo)
            d.array_save(out / 'pooled_predictions.npz', pred=pred, origins=oo.asi8)
            d.write(out / 'results.json', results)
            verify_hashes(hashes)
            (out / 'REPORT.md').write_text('# Seasonal pointweight4 baseline\n\n' +
                ('SMOKE ONLY.\n\n' if smoke else 'Confirmation reuses previously exposed original internal models.\n\n') +
                '\n'.join(f'- {stage}: MSE={row["mse"]:.3f}, low={row["mse_lt550"]}, high={row["mse_ge550"]}'
                          for stage, row in results.items()) + '\n')
            completed = ['manifest.json', 'results.json'] + [f'{s}_predictions.npz' for s in results]
            d.write(out / 'completion.json', dict(sha256={str(out/f):d.digest(out/f) for f in completed}))
            d.status(out, 'completed', smoke=smoke)
        except Exception as error:
            d.status(out, 'failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    execute(parser.parse_args().smoke)
