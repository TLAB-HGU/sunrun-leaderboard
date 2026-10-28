"""Publish the frozen pointwise-weight experiment and reproducible diagnostics."""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PUB = Path(__file__).resolve().parent
ROOT = PUB.parents[2]
RUN = PUB.parent / 'runs/full_20261028_v1'
WORK = PUB / 'worktree'
EXPERIMENT = 'e2000fs_feat30_pointweight4'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

P = load('pointweight_publisher_base', ROOT / 'analysis/forecast_strategies_kumo/publish.py')
read, write, sha = P.read_json, P.write_json, P.sha

class Publisher(P.Publisher):
    def git(self, *args, capture=False):
        if args and args[0] == 'commit':
            args = ('commit', '-m', 'E2000FS: publish pointwise extreme weighting and diagnostics')
        return super().git(*args, capture=capture)

def build():
    assert read(RUN / 'status.json')['status'] == 'completed'
    manifest = read(RUN / 'manifest.json')
    for field in ('source_sha256', 'input_sha256'):
        for path, digest in manifest[field].items():
            assert sha(path) == digest, path
    result = read(RUN / 'results.json')
    frame = pd.read_parquet(RUN / 'predictions.parquet')
    origins, values = P.validate_frame(frame)
    hpo = load('pointweight_publish_hpo', ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm_HPO/run.py')
    parent = read(ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1/manifest.json')
    names = read(hpo.CACHE / 'names.json')
    selected = parent['selected_groups']
    feature_names = [n for n in names['constant'] if n in selected] + [n for _ in range(24) for n in names['varying'] if n in selected]
    release = WORK / 'releases' / EXPERIMENT
    viz = WORK / 'docs/visualizations' / EXPERIMENT
    release.mkdir(parents=True, exist_ok=True)
    viz.mkdir(parents=True, exist_ok=True)
    archived = {}
    for filename, digest in manifest['source_sha256'].items():
        source = Path(filename)
        if source.suffix != '.py': continue
        relative = source.relative_to(ROOT)
        target = release / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        archived[str(relative)] = digest
    for loss in RUN.glob('*/h*/loss.csv'):
        target = release / 'loss_history' / loss.relative_to(RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(loss, target)
    timing, importance, model_hashes = {}, [], {}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for ax, stage in zip(axes, ('internal', 'f1', 'final')):
        losses = [pd.read_csv(RUN / stage / f'h{h+1:02}/loss.csv') for h in range(72)]
        ax.plot(np.mean([v.train_mse for v in losses], axis=0), label='mean training MSE (72 horizons)')
        ax.plot(np.mean([v.validation_mse for v in losses], axis=0), label='held-out validation' if stage == 'internal' else 'training-subset diagnostic', linestyle='--')
        ax.set(title=stage, xlabel='Boosting round (zero-based)', ylabel='MSE'); ax.legend()
    fig.savefig(viz / 'loss_history.png', dpi=140); plt.close(fig)
    for stage, grid in [('f1', 'f1'), ('final', 'official')]:
        bundle = dict(np.load(hpo.CACHE / f'{grid}.npz'))
        query = [hpo.base.design(bundle, names, selected, b) for b in (0, 24, 48)]
        models = []
        for h in range(72):
            path = RUN / stage / f'h{h+1:02}/model.txt'
            assert sha(path) == read(path.with_name('receipt.json'))['sha256']
            model_hashes[str(path.relative_to(ROOT))] = sha(path)
            models.append(lgb.Booster(model_file=str(path)))
        expected = np.load(RUN / f'{stage}_predictions.npz')['pred']
        prediction = np.column_stack([np.clip(m.predict(query[h//24], num_threads=2), 1, 3000) for h, m in enumerate(models)])
        np.testing.assert_array_equal(prediction, expected)
        idx = np.linspace(0, len(expected)-1, 32).astype(int)
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            sample = np.column_stack([np.clip(m.predict(query[h//24][idx], num_threads=2), 1, 3000) for h, m in enumerate(models)])
            elapsed.append(time.perf_counter()-start)
            np.testing.assert_array_equal(sample, expected[idx])
        timing[stage] = dict(seconds=elapsed, seconds_per_origin=float(np.median(elapsed)/len(idx)), origins=len(expected), full_prediction_parity='exact')
        gains = np.sum([m.feature_importance(importance_type='gain') for m in models], axis=0)
        splits = np.sum([m.feature_importance(importance_type='split') for m in models], axis=0)
        grouped = pd.DataFrame(dict(feature=feature_names, gain=gains, split=splits)).groupby('feature').sum().reset_index()
        grouped['stage'] = stage; importance.append(grouped)
    importance = pd.concat(importance, ignore_index=True)
    importance.to_csv(viz / 'feature_importance.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    for ax, stage in zip(axes, ('f1', 'final')):
        top = importance[importance.stage == stage].nlargest(20, 'gain').sort_values('gain')
        ax.barh(top.feature, top.gain); ax.set(title=stage+' summed split gain', xlabel='Gain (training diagnostic)')
    fig.savefig(viz / 'feature_importance.png', dpi=140); plt.close(fig)
    write(viz / 'feature_importance_receipt.json', dict(method='native gain/split summed over 72 models; 24 copies grouped; no permutation claim', models=model_hashes, timing=timing))
    np.savez(release / 'forecast.npz', origins_ns=origins.asi8, predictions=values)
    config = dict(data=dict(feature_names=selected, ace_lag_hours=0, prefill='causal forward fill; leading missing retained',
        periods={p:dict(start=a,end=b,count=n) for p,(a,b,n) in P.GRID.items()},
        asset=dict(filename='forecast.npz',sha256=sha(release / 'forecast.npz')), archived_source_sha256=archived,
        input_and_source_sha256={**manifest['input_sha256'], **manifest['source_sha256']}, model_sha256=model_hashes),
        hyperparameters=dict(parameters=manifest['params'], method=manifest['weighting'], selection=manifest['selection']))
    csha = P.config_sha(config)
    description = ('Pointwise extreme weighting: independent LightGBM horizons; training weight4 if this target >=550, else1, mean normalized. '
        'Same frozen HPO as distribution controls. Removes the previous block-wide ramp trigger. Both-region acceptance failed. '
        'Inherited Top30 features were selected using previously scored official periods; this is official-exposed post-hoc research, not a fresh holdout. '
        'Targets mature72h before cutoff; F1 cutoff2026-02-26, F2 cutoff2026-05-29. Published script replays hash-pinned forecasts, training source archived. '
        'Inference timing: saved CPU models, two threads, median3 batches of32 origins; excludes feature assembly, loading and I/O.')
    measured = sum(v['seconds_per_origin']*v['origins'] for v in timing.values())/4970
    meta = dict(team_member='seongeun', experiment=EXPERIMENT, description=description, no_future_leakage=True,
                f1_training_data_end_utc='2026-02-26T00:00:00Z', config=config, inference_seconds_per_fold=measured)
    timing['end_to_end'] = dict(seconds=910.332, hours=910.332/3600,
        scope='Original full run 2026-10-28T12:29:23.549566Z to12:44:33.881205Z, internal+F1+F2 training/prediction/evaluation; excludes smoke, publication and prior HPO.')
    for filename, value in [('config.json',config), ('meta.yaml',meta), ('run-manifest.json',manifest), ('result.json',result), ('timing.json',timing)]:
        write(release / filename, value)
    (release / 'README.md').write_text(description+'\n')
    script_relative = Path('scripts/seongeun') / EXPERIMENT / (csha+'.py')
    script = WORK / script_relative; script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(P.REPLAY.replace('Kumo',EXPERIMENT).replace('__SHA__',repr(csha)).replace('__RELEASE__',repr(str(release.relative_to(WORK)))))
    replay = PUB / 'replay.parquet'
    subprocess.run([str(P.PYTHON),str(script),'--output',str(replay)],check=True)
    assert pd.read_parquet(replay).equals(frame)
    # Render one origin per day; preserve all forecasts in the release archive.
    series = pd.read_parquet(hpo.CACHE / 'speed.parquet')['speed']
    gif_receipt = {}
    months = (origins + pd.Timedelta(hours=1)).strftime('%Y-%m')
    for month in sorted(set(months)):
        ii = np.flatnonzero(months == month)[::24]
        images = []
        fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
        for i in ii:
            ax.clear(); origin = origins[i]
            hx = pd.date_range(origin-pd.Timedelta(hours=168), origin, freq='h')
            fx = pd.date_range(origin+pd.Timedelta(hours=1), periods=72, freq='h')
            ax.plot(np.arange(-168,1), series.reindex(hx), color='black', label='History')
            ax.plot(np.arange(1,73), series.reindex(fx), color='black', linestyle='--', label='Truth')
            ax.plot(np.arange(1,73), values[i], color='tab:red', label='Pointweight4')
            ax.axhline(550,color='gray',alpha=.5); ax.set(xlim=(-168,72),ylim=(200,900),title=f'{EXPERIMENT}: {origin}',xlabel='Hours from origin',ylabel='km/s'); ax.legend(loc='upper left')
            fig.canvas.draw(); images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P', palette=Image.Palette.ADAPTIVE))
        images[0].save(viz / f'{month}.gif',save_all=True,append_images=images[1:],duration=160,loop=0)
        plt.close(fig)
        gif_receipt[month] = dict(frames=len(ii), origins=[str(origins[i]) for i in ii])
    write(viz / 'gif_manifest.json', dict(sampling='every24th available origin within each month, fixed in advance', months=gif_receipt, prediction_sha256=sha(replay)))
    controls = read(ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1/controls/results.json')
    lines = ['# E2000FS Feat30 Pointweight4', '', description, '', '| Method | Overall MSE | <550 MSE | >=550 MSE |','|---|---:|---:|---:|']
    for name,row in [('L2',controls['l2']['combined']),('Block extreme4',controls['extreme4']['combined']),('Pointwise extreme4',result['combined'])]:
        lines.append(f"| {name} | {row['mse']:.6f} | {row['mse_lt550']:.6f} | {row['mse_ge550']:.6f} |")
    lines += ['', 'High-speed MSE improved, but ordinary MSE worsened. Both-region acceptance failed. These are point estimates.', '',
        '## Loss history', '', '![loss](loss_history.png)', '', 'Internal validation is chronologically held out with a72h purge. F1/F2 dashed lines are training-subset diagnostics, not held-out validation. Curves average72 horizon models.', '',
        '## Feature importance', '', '![importance](feature_importance.png)', '', 'Native LightGBM gain and split counts summed over72 models, grouped by feature. These are training diagnostics, not permutation or causal importance. Saved models reproduced all submitted predictions exactly.', '',
        '## Monthly forecast GIFs', '', 'One frame per24 available origins within each month (daily sampling); full4970-origin forecasts are in the release. Y-axis200–900km/s.', '']
    for month in gif_receipt: lines += [f'### {month}', '', f'![{month}]({month}.gif)', '']
    lines += [f'[Release](../../../releases/{EXPERIMENT}/)', f'[Replay script](../../../{script_relative})', '', 'Reproduce publication diagnostics with code/publish.py in the original workspace; private input data and model weights are not published.']
    (viz / 'README.md').write_text('\n'.join(lines)+'\n')
    (viz / 'code').mkdir(exist_ok=True); shutil.copyfile(__file__, viz / 'code/publish.py')
    index = WORK / 'docs/visualizations/README.md'
    entry = f'\n[{EXPERIMENT}]({EXPERIMENT}/): MSE3732.269, >=550 MSE6071.056; pointwise weighting follow-up, loss/importance and daily-sampled monthly GIFs. High-speed improvement accompanied by ordinary-region degradation.\n'
    if f']({EXPERIMENT}/)' not in index.read_text():
        index.write_text(index.read_text()+entry)
    paths = [script,index]+list(release.rglob('*'))+list(viz.rglob('*'))
    item = dict(experiment=EXPERIMENT, config_sha256=csha, script=str(script_relative), release=str(release.relative_to(WORK)),
        predictions=str(replay), prediction_sha256=sha(replay), meta=str(release/'meta.yaml'),
        scores={k:result['combined']['official'][v] for k,v in P.METRICS.items()},
        release_sha256={str(p.relative_to(WORK)):sha(p) for p in paths if p.is_file()})
    write(PUB/'build.json', {'pointweight':item}); write(PUB/'batch.json',['pointweight'])
    print('Exact full-model and archive replay parity verified; release and diagnostics built.',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('step',choices=['build','dryrun','push','github','submit','verify'])
    args=parser.parse_args(); publisher=Publisher(RUN,PUB,WORK)
    if args.step=='build': build()
    else: getattr(publisher,args.step)()
