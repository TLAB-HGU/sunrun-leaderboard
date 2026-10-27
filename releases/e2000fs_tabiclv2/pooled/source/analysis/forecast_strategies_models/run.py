"""Prepare, execute and score one frozen E2000FS scalar model ablation."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / 'forecast_strategies_e2000'
KUMO = HERE.parent / 'forecast_strategies_kumo'
sys.path.insert(0, str(BASE))
import data  # noqa: E402
import evaluate  # noqa: E402
import strategies  # noqa: E402

_spec = importlib.util.spec_from_file_location('kumo_contract_reader', KUMO / 'run.py')
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
MODELS = ('tabiclv2', 'tabldm', 'realmlp', 'lightgbm', 'mitra')
STRATEGIES = ('Recursive', 'DirRec', 'Direct', 'pooled')


def completed_prerequisite():
    """Require actual completion and all remote receipts, not merely a dead process."""
    state = json.loads((KUMO / 'outputs/supervision.json').read_text())
    verify = json.loads((KUMO / 'publication/verify.json').read_text())
    records = verify.get('records', {})
    return (state.get('status') == 'completed' and set(records) == set(STRATEGIES)
            and all(r.get('predictions_identical') and (r.get('scores_match') or r.get('mse_matches'))
                    for r in records.values()))


def config(model, smoke=False):
    value = {'model': model, 'seed': 42, 'physical_gpu': 1, 'context_rows': 5000,
             'estimators': 8, 'batch': 256, 'n_jobs': 4, 'epochs': 256,
             'smoke': smoke}
    if smoke:
        value.update(context_rows=32, estimators=1, batch=8, epochs=2)
    return value


def prepare(model, out):
    from adapters import prepare_identity
    if not completed_prerequisite():
        raise RuntimeError('Kumo remote completion has not been verified; environment stays unchanged')
    out = Path(out)
    cfg = config(model)
    model_identity = prepare_identity(model, cfg)
    sources = {str(p): C.sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
    sources.update({str(p): C.sha(p) for p in (KUMO / 'models.py', KUMO / 'run.py',
                                              BASE / 'data.py', BASE / 'evaluate.py', BASE / 'strategies.py',
                                              BASE / 'publication/worktree/scorer/scoring.py')})
    manifest = {'model': model, 'config': cfg,
                'cache': str(BASE / 'outputs/cache'), 'input_sha256': C.input_manifest(BASE / 'outputs/cache'),
                'source_sha256': sources, **model_identity,
                'scope': 'frozen model ablation; no champion promotion; all valid arms published',
                'prior_exposure': 'Inherited E2000 feature selection used later data; official periods already exposed'}
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('manifest identity differs; preserve existing run and choose new output')
    C.write_json(path, manifest)
    return manifest


def validate_manifest(out, model):
    manifest = json.loads((Path(out) / 'manifest.json').read_text())
    if manifest['model'] != model or manifest['config'] != config(model):
        raise ValueError('model/config mismatch')
    for key in ('input_sha256', 'source_sha256'):
        for path, expected in manifest[key].items():
            if C.sha(path) != expected:
                raise ValueError(f'frozen artifact differs: {path}')
    for package, expected in manifest.get('package_versions', {}).items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'frozen package differs: {package}')
    from adapters import prepare_identity
    current = prepare_identity(model, manifest['config'])
    if current != {key: manifest[key] for key in current}:
        raise ValueError('model identity changed')
    return manifest


def execute(args):
    from engine import fit_predict
    out = Path(args.out)
    cfg = config(args.model, args.smoke)
    if args.smoke:
        identity = 'synthetic-smoke-not-publishable'
    else:
        validate_manifest(out, args.model)
        identity = C.sha(out / 'manifest.json')
    train, f1, final, series, _ = data.load(BASE / 'outputs/cache')
    fit, vi = data.select_split(train)
    if args.smoke:
        fit, vi = fit[-40:], vi[:2]
    validation = train.take(vi)
    Xv = strategies.Inputs.of(validation)
    Yv = data.future_targets(series, validation.origins)
    observed = evaluate.observed_mask(validation.origins)
    specs = [('select', fit, validation, 'validation')]
    if not args.smoke:
        specs += [('f1', data.f1_split(train), f1, 'f1'),
                  ('final', data.final_split(train), final, 'official')]
    meta, combined = {}, []
    for stage, idx, target, tag in specs:
        C.write_json(out / 'progress' / f'{args.strategy}.json',
                     {'model': args.model, 'strategy': args.strategy, 'stage': stage, 'status': 'running'})
        b = train.take(idx)
        pred, timing = fit_predict(args.strategy, args.model, strategies.Inputs.of(b),
                                   data.future_targets(series, b.origins), Xv, Yv,
                                   {tag: strategies.Inputs.of(target)}, b.origins,
                                   out / 'heads', stage, cfg, identity)
        timing['inference_seconds_per_fold'] = timing['inference_seconds_per_fold'][tag]
        meta[stage] = timing
        if stage == 'select':
            internal = C.metrics(pred[tag], Yv, observed)
        else:
            frame = evaluate.prediction_frame(target.origins, pred[tag])
            path = out / 'predictions' / stage / f'{args.strategy}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            combined.append(frame)
        C.write_json(out / 'metrics' / f'{args.strategy}.json', {'validation': internal, 'stages': meta})
    if not args.smoke:
        frame = pd.concat(combined, ignore_index=True)
        if len(frame) != 357840 or frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
            raise ValueError('combined grid mismatch')
        path = out / 'predictions/combined' / f'{args.strategy}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.parquet')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    receipt = {'model': args.model, 'strategy': args.strategy, 'status': 'completed', 'smoke': args.smoke}
    C.write_json(out / 'progress' / f'{args.strategy}.json', receipt)
    if os.environ.get('EXPOS_OUT'):
        C.write_json(Path(os.environ['EXPOS_OUT']) / 'completion.json', receipt)


def report(model, out):
    out = Path(out)
    rows = {}
    reference = json.loads((KUMO / 'outputs/report.json').read_text())
    scorer = evaluate.Official(revision=reference['dataset_revision'])
    for strategy in STRATEGIES:
        # Comparator reuse is conditional on the actual ordered scalar contexts.
        for stage in ('select', 'f1', 'final'):
            for head in range(1, 73):
                name = f'h{head:02}'
                current = json.loads((out / 'heads' / strategy / stage / name / 'receipt.json').read_text())
                for backend, code in (('kumo', 'kumo'), ('xgb', 'xgboost')):
                    path = KUMO / 'outputs/heads' / backend / strategy / stage / code / strategy / name / 'receipt.json'
                    reference_head = json.loads(path.read_text())
                    if current['context_hashes'] != reference_head['context_hashes']:
                        raise ValueError(f'comparator scalar context differs: {strategy}/{stage}/{name}/{backend}')
        row, by_h = scorer.row(out / 'predictions/combined' / f'{strategy}.parquet')
        meta = json.loads((out / 'metrics' / f'{strategy}.json').read_text())
        row.update(meta['validation'])
        row['inference_seconds_per_fold'] = sum(meta['stages'][s]['inference_seconds_per_fold'] * n
                                               for s, n in [('f1', 2137), ('final', 2833)]) / 4970
        row['fit_seconds'] = sum(meta['stages'][s]['fit_seconds'] for s in ('f1', 'final'))
        row['mse_by_horizon'] = by_h.tolist()
        for comparator in ('Kumo', 'XGB'):
            filename = strategy if comparator == 'Kumo' else f'xgb_{strategy}'
            baseline, _ = scorer.row(KUMO / 'outputs/predictions/combined' / f'{filename}.parquet')
            if abs(baseline['mse'] - reference['rows'][f'{comparator}_{strategy}']['mse']) > 1e-6:
                raise ValueError('comparator score changed under the pinned evaluation contract')
            row[f'delta_vs_{comparator}_percent'] = 100 * (
                row['mse'] / baseline['mse'] - 1)
        rows[f'{model}_{strategy}'] = row
    C.write_json(out / 'report.json', {'rows': rows, 'dataset_revision': scorer.revision,
                                     'provenance': json.loads((out / 'manifest.json').read_text())})
    table = pd.DataFrame(rows).T
    table.drop(columns='mse_by_horizon').to_csv(out / 'leaderboard.csv', index_label='candidate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--strategy', choices=STRATEGIES, default='Recursive')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out = args.out or HERE / 'outputs' / args.model
    if args.action == 'prepare':
        prepare(args.model, args.out)
    elif args.action == 'report':
        report(args.model, args.out)
    elif args.model == 'lightgbm':
        execute(args)
    else:
        with C.gpu1_lease():
            execute(args)


if __name__ == '__main__':
    main()
