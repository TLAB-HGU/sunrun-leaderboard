"""Frozen Kumo/XGBoost model ablation of the E2000 forecasting-strategy study."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / 'forecast_strategies_e2000'
sys.path.insert(0, str(BASE))
import data  # noqa: E402
import strategies as original  # noqa: E402

STRATEGIES = ('Recursive', 'DirRec', 'Direct', 'pooled')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def input_manifest(cache):
    """Bind existing admitted inputs and cache bytes; do not fetch observations."""
    cache = Path(cache)
    receipt = json.loads((cache / 'receipt.json').read_text())
    if receipt['lag_hours'] != 0 or receipt['prefill'] != 'ffill along origin':
        raise ValueError('the approved lag-0/prefill input contract differs')
    files = {}
    for item in receipt['inputs'].values():
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError(f'upstream input changed: {path}')
        files[str(path)] = item['sha256']
    for path in sorted(cache.iterdir()):
        if path.is_file():
            files[str(path)] = sha(path)
    for name in ('train', 'f1', 'official'):
        if receipt[f'{name}_lag2_parity']['mismatch'] != 0:
            raise ValueError('inherited feature parity failed')
    audit_root = ROOT / 'research/e0472-multioutput-s42-20261026/relocated-v1'
    for name in ('final-audit.json', 'raw-probes-causal.json', 'poison-bound.json',
                 'suvi-exact-byte-proof.json'):
        path = audit_root / name
        files[str(path)] = sha(path)
    if not json.loads((audit_root / 'final-audit.json').read_text())['passed']:
        raise ValueError('inherited source input audit did not pass')
    return files


def freeze_manifest(out, cache, config):
    from dataclasses import asdict
    sources = {str(p): sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
    sources.update({str(BASE / n): sha(BASE / n) for n in ('data.py', 'strategies.py', 'evaluate.py')})
    weights = {}
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download('nvidia/Kumo-Tabular', revision='v1.0.1',
                                     allow_patterns=['large/regressor.pt', 'config.json', '*.md']))
    # The actual SDM checkpoint filenames are pinned below by their Hub revision.
    # Only regression large files are fetched; other model sizes are out of scope.
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info('nvidia/Kumo-Tabular', revision='v1.0.1')
    matches = [s.rfilename for s in info.siblings
               if s.rfilename == 'large/regressor.pt']
    if not matches:
        raise ValueError('official large regression checkpoint could not be identified')
    for name in matches:
        path = Path(hf_hub_download('nvidia/Kumo-Tabular', name, revision='v1.0.1'))
        weights[name] = sha(path)
    manifest = {'config': asdict(config), 'input_sha256': input_manifest(cache),
                'source_sha256': sources, 'model_sha256': weights,
                'model_revision': info.sha, 'model_tag': 'v1.0.1',
                'package_versions': {n: importlib.metadata.version(n) for n in
                                     ('structured-data-models', 'torch', 'xgboost', 'numpy', 'pandas')},
                'cache': str(Path(cache).resolve()), 'snapshot': str(snapshot),
                'scope': 'analysis model ablation; no cumulative parent promotion',
                'epoch_history': 'not applicable: Kumo uses in-context inference, no gradient training',
                'prior_exposure': 'E2000 inputs inherited from research using later data; official periods exposed'}
    path = Path(out) / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('frozen experiment identity changed; use a new output directory')
    write_json(path, manifest)
    return manifest


@contextlib.contextmanager
def gpu1_lease():
    """Use the established lease for user-selected physical GPU1."""
    path = ROOT / 'leaderboard/store/ch-breakthrough-v2/gpu-leases/gpu1.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        print('waiting for physical GPU1 lease', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def metrics(pred, targets, observed):
    import numpy as np
    error = np.asarray(pred) - targets
    high = targets >= 550
    return {'val_mse_all': float(np.mean(error ** 2)),
            'val_mse_obs': float(np.mean(error[observed] ** 2)),
            'val_mse550': float(np.mean(error[high] ** 2)) if high.any() else None}


def execute(args):
    import evaluate
    import numpy as np
    from models import Config, fit_predict
    config = Config()
    if args.smoke:
        config = Config(size='small', n_estimators=1, context_rows=32, query_batch=8, xgb_rounds=3)
    out = Path(args.out)
    if not args.smoke:
        frozen = json.loads((out / 'manifest.json').read_text())
        from dataclasses import asdict
        if str(Path(args.cache).resolve()) != frozen['cache'] or asdict(config) != frozen['config']:
            raise ValueError('execution cache/config differs from frozen experiment')
        for package, expected in frozen['package_versions'].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(f'frozen package version changed: {package}')
        from huggingface_hub import hf_hub_download
        for name, expected in frozen['model_sha256'].items():
            weight = hf_hub_download('nvidia/Kumo-Tabular', name, revision=frozen['model_revision'],
                                     local_files_only=True)
            if sha(weight) != expected:
                raise ValueError(f'frozen model bytes changed: {name}')
        for group in ('source_sha256', 'input_sha256'):
            for path, expected in frozen[group].items():
                if sha(path) != expected:
                    raise ValueError(f'frozen file changed: {path}')
    train, f1, official, series, _ = data.load(args.cache)
    fit_select, valid = data.select_split(train)
    if not len(fit_select) or not len(valid):
        raise ValueError('empty chronological split')
    if args.smoke:
        valid = valid[:2]
        fit_select = fit_select[-40:]
    v = train.take(valid)
    vi, yv = original.Inputs.of(v), data.future_targets(series, v.origins)
    observed = evaluate.observed_mask(v.origins)
    specs = {'select': (fit_select, v, 'validation'),
             'f1': (data.f1_split(train), f1, 'f1'),
             'final': (data.final_split(train), official, 'official')}
    stages = ('select',) if args.smoke else ('select', 'f1', 'final')
    backends = ('kumo', 'xgb') if args.backend == 'both' else (args.backend,)
    for backend in backends:
        validation_metrics = None
        metas = {}
        combined = []
        for stage in stages:
            indices, target, tag = specs[stage]
            t = train.take(indices)
            inputs, Y = original.Inputs.of(t), data.future_targets(series, t.origins)
            if not np.isfinite(Y).all() or not np.isfinite(yv).all():
                raise ValueError('nonfinite filled targets')
            query = original.Inputs.of(target)
            progress = {'strategy': args.strategy, 'backend': backend, 'stage': stage,
                        'status': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            progress_file = out / 'progress' / f'{backend}_{args.strategy}.json'
            write_json(progress_file, progress)
            print(json.dumps(progress), flush=True)
            preds, meta = fit_predict(args.strategy, 'xgboost' if backend == 'xgb' else backend, inputs, Y, vi, yv,
                                      {tag: query}, origins=t.origins,
                                      output_dir=out / 'heads' / backend / args.strategy,
                                      stage=stage, config=config)
            meta['inference_seconds_per_fold'] = meta['inference_seconds_per_fold'][tag]
            meta['validation_loss_role'] = ('held-out selection diagnostic' if stage == 'select'
                                            else 'overlapping historical block diagnostic; never selection')
            pred = preds[tag]
            metas[stage] = meta
            if stage == 'select':
                validation_metrics = metrics(pred, yv, observed)
            else:
                frame = evaluate.prediction_frame(target.origins, pred)
                destination = out / 'predictions' / stage / f'{backend}_{args.strategy}.parquet'
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(destination, index=False)
                combined.append(frame)
            write_json(out / 'metrics' / f'{backend}_{args.strategy}.json',
                       {'validation': validation_metrics, 'stages': metas})
            write_json(progress_file, dict(progress, status='completed', metadata=meta))
        if not args.smoke:
            import pandas as pd
            frame = pd.concat(combined, ignore_index=True)
            if len(frame) != 357840 or frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
                raise ValueError('combined F1/F2 grid differs')
            filename = args.strategy if backend == 'kumo' else f'xgb_{args.strategy}'
            destination = out / 'predictions' / 'combined' / f'{filename}.parquet'
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp.parquet')
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
    receipt = {'strategy': args.strategy, 'backend': args.backend, 'status': 'completed', 'output': str(out)}
    if os.environ.get('EXPOS_OUT'):
        write_json(Path(os.environ['EXPOS_OUT']) / 'completion.json', receipt)
    return receipt


def report(out):
    import evaluate
    import pandas as pd
    out = Path(out)
    scorer = evaluate.Official()
    rows = {}
    for backend in ('kumo', 'xgb'):
        for strategy in STRATEGIES:
            name = strategy if backend == 'kumo' else f'xgb_{strategy}'
            path = out / 'predictions' / 'combined' / f'{name}.parquet'
            if not path.exists():
                raise ValueError(f'missing completed arm: {path}')
            row, by_h = scorer.row(path)
            meta = json.loads((out / 'metrics' / f'{backend}_{strategy}.json').read_text())
            row.update(meta['validation'])
            n = {'f1': 2137, 'final': 2833}
            # metadata fields are supplied by the model engine; do not include fit or IO time.
            row['inference_seconds_per_fold'] = sum(
                meta['stages'][s]['inference_seconds_per_fold'] * n[s] for s in n) / sum(n.values())
            row['fit_seconds'] = sum(meta['stages'][s]['fit_seconds'] for s in n)
            row['fit_timing_scope'] = 'original completed head fits; retry overhead excluded'
            row['mse_by_horizon'] = by_h.tolist()
            rows[f'{"Kumo" if backend == "kumo" else "XGB"}_{strategy}'] = row
    for strategy in STRATEGIES:
        a, b = rows[f'Kumo_{strategy}'], rows[f'XGB_{strategy}']
        a['matched_xgb_delta_percent'] = 100 * (a['mse'] / b['mse'] - 1)
    value = {'rows': rows, 'dataset_revision': scorer.revision,
             'provenance': json.loads((out / 'manifest.json').read_text())}
    write_json(out / 'report.json', value)
    table = pd.DataFrame(rows).T
    columns = ['mse', 'mse_f1', 'mse_f2', 'mse_obs', 'mse550', 'val_mse_obs',
               'inference_seconds_per_fold', 'matched_xgb_delta_percent']
    table[columns].to_csv(out / 'leaderboard.csv', index_label='strategy')
    display = table[columns].round(4).fillna('')
    header = '| strategy | ' + ' | '.join(columns) + ' |\n'
    header += '| --- | ' + ' | '.join(['---:'] * len(columns)) + ' |\n'
    lines = ['| ' + str(name) + ' | ' + ' | '.join(str(row[c]) for c in columns) + ' |'
             for name, row in display.iterrows()]
    (out / 'leaderboard.md').write_text(
        '# E2000FS Kumo scalar-model ablation\n\n'
        'Single seed 42; official periods previously exposed; no score-driven selection.\n\n'
        f'Evaluation dataset revision: `{scorer.revision}`\n\n' + header + '\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for strategy, ax in zip(STRATEGIES, axes.flat):
        for backend in ('Kumo', 'XGB'):
            ax.plot(range(1, 73), rows[f'{backend}_{strategy}']['mse_by_horizon'], label=backend)
        ax.set(title=strategy, xlabel='Horizon (hours)', ylabel='Official MSE ((km/s)^2)')
        ax.legend()
    fig.savefig(out / 'horizon_mse.png', dpi=150)
    fig.savefig(out / 'horizon_mse.svg')
    plt.close(fig)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare', 'run', 'report'))
    ap.add_argument('--out', type=Path, default=HERE / 'outputs')
    ap.add_argument('--cache', type=Path, default=BASE / 'outputs/cache')
    ap.add_argument('--strategy', choices=STRATEGIES, default='Recursive')
    ap.add_argument('--backend', choices=('kumo', 'xgb', 'both'), default='both')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.action == 'prepare':
        from models import Config
        freeze_manifest(args.out, args.cache, Config())
    elif args.action == 'report':
        report(args.out)
    elif args.backend == 'xgb':
        execute(args)
    else:
        with gpu1_lease():
            execute(args)


if __name__ == '__main__':
    main()
