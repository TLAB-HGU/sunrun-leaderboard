"""One frozen smooth error-tail LightGBM experiment, with purged validation."""
import argparse
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('smooth_tail_helpers', ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/run.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
methods = d.imported('smooth_tail_methods', HERE / 'methods.py')
np, pd, lgb = d.np, d.pd, d.lgb
REFERENCE = ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1'
POINT = ROOT / 'analysis/e2000fs_feat30_pointweight/runs/full_20261028_v1'
SETTINGS = dict(alpha=3., cutoff=100., scale=50.)


def comparison(candidate, baseline, block, point):
    low = candidate['mse_lt550'] - baseline['mse_lt550']
    high_point = candidate['mse_ge550'] - point['mse_ge550']
    high_block = candidate['mse_ge550'] - block['mse_ge550']
    return dict(low_delta_vs_l2=low, high_delta_vs_point4=high_point,
                high_delta_vs_block4=high_block,
                satisfies_point4_and_low=bool(low <= 0 and high_point <= 0),
                satisfies_block4_and_low=bool(low <= 0 and high_block <= 0))


def train_indices(origins, first_query):
    indices = np.flatnonzero(origins + pd.Timedelta(hours=72) < first_query)
    if not len(indices):
        raise ValueError('No mature training targets')
    return indices


def save_model(x, y, xv, yv, params, folder):
    """Centered targets supply a train-only mean intercept for custom objectives."""
    offset = float(np.mean(y))
    history = {}
    custom_params = dict(params, objective=methods.objective(**SETTINGS))
    model = lgb.LGBMRegressor(**custom_params)
    model.fit(x, y - offset, eval_set=[(x, y - offset), (xv, yv - offset)],
              eval_names=['train', 'validation'], eval_metric='l2',
              callbacks=[lgb.record_evaluation(history)])
    temporary = folder / 'model.txt.tmp'
    model.booster_.save_model(str(temporary))
    restored = lgb.Booster(model_file=str(temporary))
    np.testing.assert_allclose(model.predict(xv) + offset,
                               restored.predict(xv, num_threads=2) + offset, rtol=0, atol=0)
    temporary.replace(folder / 'model.txt')
    d.write(folder / 'offset.json', dict(offset=offset, policy='mean of training targets only'))
    pd.DataFrame(dict(round=np.arange(1, len(history['train']['l2'])+1),
                      train_mse=history['train']['l2'], validation_mse=history['validation']['l2'])).to_csv(folder / 'loss.csv', index=False)
    return restored, offset


def validate_receipt(folder, context):
    receipt = d.read(folder / 'receipt.json')
    if any(receipt.get(k) != value for k, value in context.items()):
        raise ValueError('Checkpoint context changed')
    for filename, expected in receipt['sha256'].items():
        if d.digest(folder / filename) != expected:
            raise ValueError(f'Checkpoint file changed: {filename}')
    return lgb.Booster(model_file=str(folder / 'model.txt')), d.read(folder / 'offset.json')['offset']


def execute(smoke=False):
    out = HERE / 'runs' / ('smoke_20261028_v1' if smoke else 'full_20261028_v1')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='load_frozen_inputs')
            parent = d.read(REFERENCE / 'manifest.json')
            point_manifest = d.read(POINT / 'manifest.json')
            hpo = d.imported('smooth_tail_hpo', d.HPO / 'run.py')
            ev = d.imported('smooth_tail_evaluation', d.FOLLOW / 'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            assert data_parent == parent['parent']
            params = dict(parent['params'], n_jobs=2)
            assert params == point_manifest['params']
            fit, val = hpo.temporal_split(origins)
            if smoke:
                params['n_estimators'] = 3
                fit = fit[::max(1, len(fit)//512)]
                val = val[::max(1, len(val)//24)]
            paths = [Path(__file__), HERE / 'methods.py', Path(d.__file__), d.HPO / 'run.py',
                     d.FOLLOW / 'evaluation.py', REFERENCE / 'manifest.json',
                     REFERENCE / 'controls/results.json', POINT / 'manifest.json', POINT / 'results.json',
                     HERE / 'research/RESEARCH.md']
            paths += list((REFERENCE / 'controls').glob('*_l2.npz'))
            paths += list((REFERENCE / 'controls').glob('*_extreme4.npz'))
            paths += list(POINT.glob('*_predictions.npz'))
            hashes = {str(p): d.digest(p) for p in paths}
            for path, expected in parent['inherited_hashes'].items():
                if d.digest(path) != expected:
                    raise ValueError(f'Frozen input changed: {path}')
            manifest = dict(experiment='e2000fs_feat30_smooth_error_tail', smoke=smoke,
                            params=params, settings=SETTINGS, source_sha256=hashes,
                            input_sha256=parent['inherited_hashes'],
                            loss='0.5*e^2 + 1.5*50^2*softplus((e^2-100^2)/50^2); e=prediction-truth',
                            initialization='train-target mean centering; add stored mean at inference; custom objective initial raw score zero',
                            paper='https://arxiv.org/abs/2505.17777',
                            adaptation='smooth symmetric error-tail surrogate inspired by shortfall risk; not paper exact shortfall/bisection; cutoff and scale are fixed engineering choices',
                            selection='one fixed arm; no hyperparameter selection; internal then fixed F1/F2 diagnostics',
                            acceptance='low MSE <= L2 and high MSE <= point4; also report block4 threshold',
                            limitation='official-exposed follow-up; point estimates, no statistical noninferiority claim; no promotion/upload',
                            fit_last_target=str((origins[fit]+pd.Timedelta(hours=72)).max()),
                            validation_first_origin=str(origins[val].min()),
                            versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__))
            if (out / 'manifest.json').exists():
                if d.read(out / 'manifest.json') != manifest:
                    raise ValueError('Run identity changed')
            else:
                d.write(out / 'manifest.json', manifest)
            manifest_hash = d.digest(out / 'manifest.json')
            reference = d.read(REFERENCE / 'controls/results.json')
            point = d.read(POINT / 'results.json')
            series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
            results, outputs = {}, {}
            for stage in ('internal', 'f1', 'final'):
                if stage == 'internal':
                    train, query, oo, yy = fit, [x[val] for x in blocks], origins[val], y[val]
                else:
                    grid, cutoff = ('f1', '2026-02-26') if stage == 'f1' else ('official', '2026-05-29')
                    train = train_indices(origins, pd.Timestamp(cutoff, tz='UTC'))
                    bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
                    oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
                    query = [hpo.base.design(bundle, names, parent['selected_groups'], b) for b in (0, 24, 48)]
                    yy = hpo.base.targets(series, oo)
                    if smoke:
                        train = train[::max(1, len(train)//512)]
                        oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
                assert (origins[train]+pd.Timedelta(hours=72)).max() < oo.min()
                pred = np.empty_like(yy)
                for horizon in range(72):
                    folder = out / stage / f'h{horizon+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    context = dict(manifest_sha256=manifest_hash, stage=stage, horizon=horizon+1,
                                   train_count=len(train), train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                                   validation_held_out=stage == 'internal')
                    if (folder / 'receipt.json').exists():
                        model, offset = validate_receipt(folder, context)
                    else:
                        d.status(out, 'running', stage=stage, horizon=horizon+1, total_horizons=72)
                        diag = val if stage == 'internal' else train[:min(648, len(train))]
                        model, offset = save_model(blocks[horizon//24][train], y[train, horizon],
                                                   blocks[horizon//24][diag], y[diag, horizon], params, folder)
                        d.write(folder / 'receipt.json', dict(**context, sha256={f:d.digest(folder / f) for f in ('model.txt','offset.json','loss.csv')}))
                    pred[:, horizon] = np.clip(model.predict(query[horizon//24], num_threads=2)+offset, 1., 3000.)
                assert np.isfinite(pred).all()
                d.array_save(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
                results[stage] = d.point_metrics(ev, yy, pred, oo)
                if not smoke:
                    for arm in ('l2', 'extreme4'):
                        with np.load(REFERENCE / 'controls' / f'{stage}_{arm}.npz') as saved:
                            np.testing.assert_array_equal(saved['origins'], oo.asi8)
                    with np.load(POINT / f'{stage}_predictions.npz') as saved:
                        np.testing.assert_array_equal(saved['origins'], oo.asi8)
                    results[stage]['comparison'] = comparison(results[stage], reference['l2'][stage], reference['extreme4'][stage], point[stage])
                outputs[stage] = (oo, yy, pred)
                d.write(out / 'results.json', results)
            oo = outputs['f1'][0].append(outputs['final'][0])
            yy = np.vstack([outputs[s][1] for s in ('f1', 'final')])
            pred = np.vstack([outputs[s][2] for s in ('f1', 'final')])
            results['combined'] = d.point_metrics(ev, yy, pred, oo)
            frame = ev.frame(oo, pred)
            if not smoke:
                assert len(frame) == 357840 and not frame.duplicated(ev.KEY).any()
                results['combined']['official'] = ev.official_score(frame, parent['parent'])
                np.testing.assert_allclose(results['combined']['mse'], results['combined']['official']['mse'], rtol=1e-12)
                results['combined']['comparison'] = comparison(results['combined'], reference['l2']['combined'], reference['extreme4']['combined'], point['combined'])
            frame.to_parquet(out / 'predictions.parquet', index=False)
            d.write(out / 'results.json', results)
            lines = ['# Smooth error-tail follow-up', '', '| Stage | MSE | <550 MSE | >=550 MSE | Low + point4 | Low + block4 |', '|---|---:|---:|---:|---|---|']
            for stage, row in results.items():
                values = ['N/A' if row[k] is None else f'{row[k]:.3f}' for k in ('mse','mse_lt550','mse_ge550')]
                cmp = row.get('comparison', {})
                lines.append(f"| {stage} | {' | '.join(values)} | {cmp.get('satisfies_point4_and_low','smoke')} | {cmp.get('satisfies_block4_and_low','smoke')} |")
            (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
            d.status(out, 'completed', smoke=smoke, results=str(out / 'results.json'))
        except Exception as error:
            d.status(out, 'failed', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    execute(parser.parse_args().smoke)
