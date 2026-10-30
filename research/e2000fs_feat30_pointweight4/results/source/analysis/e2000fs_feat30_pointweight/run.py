"""Pointwise extreme weighting: one fixed arm against frozen block-weight controls."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('distribution_helpers', ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/run.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
np, pd, lgb = d.np, d.pd, d.lgb
REFERENCE = ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1'


def point_weights(target):
    """Only this horizon's observed high target receives extra training weight."""
    target = np.asarray(target)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError('Expected finite one-dimensional training targets')
    weights = np.where(target >= 550., 4., 1.)
    return weights / weights.mean()


def comparison(candidate, baseline, extreme):
    low = candidate['mse_lt550'] - baseline['mse_lt550']
    high = candidate['mse_ge550'] - extreme['mse_ge550']
    return dict(low_delta_vs_l2=low, high_delta_vs_extreme4=high,
                satisfies_both=bool(low <= 0 and high <= 0))


def execute(smoke=False):
    out = HERE / 'runs' / ('smoke_20261028_v1' if smoke else 'full_20261028_v1')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            d.status(out, 'running', phase='load_frozen_inputs')
            parent = d.read(REFERENCE / 'manifest.json')
            hpo = d.imported('pointweight_hpo', d.HPO / 'run.py')
            ev = d.imported('pointweight_evaluation', d.FOLLOW / 'evaluation.py')
            data_parent, names, origins, y, blocks = hpo.load_data()
            assert data_parent == parent['parent']
            fit, val = hpo.temporal_split(origins)
            params = dict(parent['params'])
            params['n_jobs'] = 2
            if smoke:
                params['n_estimators'] = 3
                fit, val = fit[::max(1, len(fit)//128)], val[::max(1, len(val)//24)]
            paths = [Path(__file__), Path(d.__file__), d.HPO / 'run.py', d.FOLLOW / 'evaluation.py',
                     REFERENCE / 'manifest.json', REFERENCE / 'controls/results.json']
            paths += list((REFERENCE / 'controls').glob('*_l2.npz'))
            paths += list((REFERENCE / 'controls').glob('*_extreme4.npz'))
            hashes = {str(p): d.digest(p) for p in paths}
            for p, expected in parent['inherited_hashes'].items():
                assert d.digest(p) == expected, p
            identity = dict(experiment='e2000fs_feat30_pointweight4', smoke=smoke, params=params,
                            source_sha256=hashes, input_sha256=parent['inherited_hashes'],
                            weighting='per-horizon y>=550:4 else:1, mean normalized; removes block/ramp trigger',
                            acceptance='low MSE <= same-HPO L2 AND high MSE <= same-HPO block extreme4',
                            selection='one fixed arm; no official-based tuning; no automatic promotion',
                            limitation='official-exposed follow-up; point estimates, no statistical noninferiority claim',
                            fit_last_target=str((origins[fit]+pd.Timedelta(hours=72)).max()),
                            validation_first_origin=str(origins[val].min()))
            if (out / 'manifest.json').exists():
                assert d.read(out / 'manifest.json') == identity, 'Run identity changed'
            else:
                d.write(out / 'manifest.json', identity)
            stages = [('internal', fit, [x[val] for x in blocks], origins[val], y[val])]
            series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
            for stage, grid, cutoff in [('f1', 'f1', '2026-02-26'), ('final', 'official', '2026-05-29')]:
                train = np.flatnonzero(origins+pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
                bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
                oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
                query = [hpo.base.design(bundle, names, parent['selected_groups'], b) for b in (0, 24, 48)]
                yy = hpo.base.targets(series, oo)
                if smoke:
                    train = train[::max(1, len(train)//128)]
                    oo, yy, query = oo[:16], yy[:16], [x[:16] for x in query]
                stages.append((stage, train, query, oo, yy))
            results, outputs = {}, {}
            reference = d.read(REFERENCE / 'controls/results.json')
            for stage, train, query, oo, yy in stages:
                assert (origins[train]+pd.Timedelta(hours=72)).max() < oo.min()
                pred = np.empty_like(yy)
                for h in range(72):
                    folder = out / stage / f'h{h+1:02}'
                    folder.mkdir(parents=True, exist_ok=True)
                    model_path, receipt = folder / 'model.txt', folder / 'receipt.json'
                    if receipt.exists():
                        assert d.read(receipt)['sha256'] == d.digest(model_path)
                        model = lgb.Booster(model_file=str(model_path))
                    else:
                        d.status(out, 'running', stage=stage, horizon=h+1, total_horizons=72)
                        # Refits use training-only diagnostics; internal validation is purged.
                        diag = val if stage == 'internal' else train[:min(648, len(train))]
                        model = d.controls_fit(blocks[h//24][train], y[train, h],
                                               blocks[h//24][diag], y[diag, h], params,
                                               point_weights(y[train, h]), folder)
                        d.write(receipt, dict(sha256=d.digest(model_path), train_count=len(train),
                                             validation_held_out=stage == 'internal'))
                    pred[:, h] = np.clip(model.predict(query[h//24], num_threads=2), 1., 3000.)
                assert np.isfinite(pred).all()
                d.array_save(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
                results[stage] = d.point_metrics(ev, yy, pred, oo)
                if not smoke:
                    for arm in ('l2', 'extreme4'):
                        saved = np.load(REFERENCE / 'controls' / f'{stage}_{arm}.npz')
                        np.testing.assert_array_equal(saved['origins'], oo.asi8)
                    results[stage]['comparison'] = comparison(results[stage], reference['l2'][stage], reference['extreme4'][stage])
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
                results['combined']['comparison'] = comparison(results['combined'], reference['l2']['combined'], reference['extreme4']['combined'])
            frame.to_parquet(out / 'predictions.parquet', index=False)
            d.write(out / 'results.json', results)
            lines = ['# Pointwise extreme4 follow-up', '', '| Stage | MSE | <550 MSE | >=550 MSE | Both conditions |', '|---|---:|---:|---:|---|']
            for stage, row in results.items():
                values = ['N/A' if row[k] is None else f'{row[k]:.3f}' for k in ('mse', 'mse_lt550', 'mse_ge550')]
                lines.append(f"| {stage} | {' | '.join(values)} | {row.get('comparison', {}).get('satisfies_both', 'smoke')} |")
            (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
            d.status(out, 'completed', smoke=smoke, results=str(out / 'results.json'))
        except Exception as exc:
            d.status(out, 'failed', error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    execute(parser.parse_args().smoke)
