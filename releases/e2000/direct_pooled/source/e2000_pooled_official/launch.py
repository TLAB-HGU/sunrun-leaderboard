"""Final-only E1983 confirmation; no official-score recipe selection or upload."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd

from experiments.cumulative_20261024.gate import locked
from experiments.e2000_njobs_all import runner
from experiments.e2000_njobs_all.inputs import (
    CUTOFF, atomic_json, check_ref, digest, prediction_frame, reference, sha, split_masks,
)
from experiments.e2000_njobs_all.model import ModelConfig, PooledRegressor
from experiments.e2000_parallel6 import launch as prior

HERE = Path(__file__).resolve().parent
LB = HERE.parents[1]
PYTHON = LB / '.venv-fusion/bin/python'
MODULE = 'experiments.e2000_pooled_official.launch'
SUPERVISOR = LB.parent / 'research/e2000/pooled-official/supervise.py'


def code_hashes():
    files = set(runner.code_hashes()) | set(prior.code_hashes())
    files.update([str(HERE / 'launch.py'), str(HERE / '__init__.py'), str(SUPERVISOR)])
    return {p: sha(p) for p in sorted(files)}


def command(bundle):
    return shlex.join(['env', 'OMP_NUM_THREADS=8', 'OMP_THREAD_LIMIT=8', 'OMP_DYNAMIC=FALSE',
                      'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1',
                      str(PYTHON), '-m', MODULE, 'execute', '--bundle', str(Path(bundle).resolve())])


def files(spec, bundle):
    return sorted(set(prior.registered_files(spec)) | set(code_hashes()) | {
        str(bundle / name) for name in ('inputs.json', 'intent.json')})


def register(inputs, bundle):
    spec = json.loads(Path(inputs).read_text())
    prior.preflight(spec)
    baseline = prior.verify_baseline(spec)
    bundle = Path(bundle).resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    atomic_json(bundle / 'inputs.json', spec)
    atomic_json(bundle / 'intent.json', {
        'baseline': baseline, 'input_sha256': digest(spec), 'code_hashes': code_hashes(),
        'recipe': asdict(ModelConfig()), 'seed': 42, 'omp_thread_limit': 8,
        'purpose': 'User-requested official confirmation of E1983; same scientific recipe',
        'training_target_cutoff': CUTOFF.isoformat(), 'upload_requested': False})
    core = prior.protected()
    event = core.register('D_E2000_MULTIOUTPUT_S42', 'codex-e2000-pooled-confirmation',
        'User-requested E1983 pooled direct XGBoost official confirmation, seed42, unchanged recipe/input. '
        'Final train targets before 2026-05-29; one official evaluation. Existing internal E1983 reused; '
        'runtime OpenMP thread limit 8, XGBoost n_jobs=-1. No official tuning or upload.',
        command(bundle), kind='reference', code=files(spec, bundle), reference='E1983')
    ids = {'pooled_internal': 'E1983', 'pooled_official': event['id']}
    atomic_json(bundle / 'registrations.json', ids)
    return ids


def verify(bundle):
    bundle = Path(bundle).resolve()
    spec = json.loads((bundle / 'inputs.json').read_text())
    intent = json.loads((bundle / 'intent.json').read_text())
    ids = json.loads((bundle / 'registrations.json').read_text())
    core = prior.protected()
    eid = ids['pooled_official']
    entry = core.state()['experiments'][eid]
    if entry['cmd'] != command(bundle) or entry.get('flags') or ids['pooled_internal'] != 'E1983':
        raise ValueError('registration identity or integrity mismatch')
    if (intent['code_hashes'] != code_hashes() or intent['input_sha256'] != digest(spec)
            or intent['recipe'] != asdict(ModelConfig()) or intent['seed'] != 42
            or intent['training_target_cutoff'] != CUTOFF.isoformat() or intent['omp_thread_limit'] != 8):
        raise ValueError('frozen recipe/input mismatch')
    for path in files(spec, bundle):
        if entry['code'].get(path) != sha(path):
            raise ValueError('registered artifact changed: ' + path)
    if intent['baseline'] != prior.verify_baseline(spec):
        raise ValueError('immutable baseline changed')
    prior.preflight(spec)
    return core, spec, eid, entry


def final_fit(spec, out, config=None):
    config = ModelConfig() if config is None else config
    origins, x, y, mask, official_origins, official_x = runner.load_data(spec)
    _, valid_mask = split_masks(origins, mask)
    va = valid_mask.any(axis=1)
    context = {'code_hashes': code_hashes(), 'data_hashes': {'input_envelope': digest(spec)},
               'feature_names': list(x.columns), 'cutoff': CUTOFF.isoformat(),
               'grid': {'origins': digest(official_origins.astype(str).tolist()), 'horizons': 72},
               'source_availability': spec['source_receipts'], 'stage': 'official'}
    print(f'pooled: final fit {len(origins)} origins, {int(mask.sum())} observed targets, seed42', flush=True)
    kwargs = dict(observed_mask=mask, validation=(x.to_numpy()[va], y[va], valid_mask[va]),
                  output_dir=out / 'official', context=context, validation_role='overlapping_diagnostic')
    model = PooledRegressor(config).fit(x.to_numpy(), y, **kwargs)
    values = model.predict(official_x.to_numpy())
    restored = PooledRegressor(config).fit(x.to_numpy(), y, **kwargs)
    if not np.array_equal(values, restored.predict(official_x.to_numpy())):
        raise ValueError('checkpoint replay mismatch')
    frame = prediction_frame(official_origins, values)
    if len(frame) != 203976:
        raise ValueError('incomplete prediction grid')
    from scorer import scoring
    folds = pd.DataFrame({'origin_last_input_utc': official_origins.strftime('%Y-%m-%dT%H:%M:%SZ')})
    errors = scoring.validate(frame, folds)
    if errors:
        raise ValueError('official grid validator: ' + str(errors))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'predictions_official_s42.parquet.tmp'
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, out / 'predictions_official_s42.parquet')
    runner.curves(out / 'official')
    result = {'status': 'completed', 'seed': 42, 'model_count': 1, 'architecture': 'pooled_direct',
              'checkpoint_replay_exact': True, 'validation_role': 'overlapping_diagnostic_not_selection',
              'prediction': reference(out / 'predictions_official_s42.parquet'),
              'training_target_cutoff': CUTOFF.isoformat(), 'recipe': asdict(config),
              'input_sha256': digest(spec), 'code_hashes': code_hashes(), 'internal_reference': 'E1983'}
    atomic_json(out / 'confirmation.json', result)
    return result


def identity(spec):
    return {'input_sha256': digest(spec), 'code_hashes': code_hashes(), 'arm': 'pooled_official'}


def execute(bundle):
    core, spec, eid, entry = verify(bundle)
    out = core.RUNS / eid
    if Path(os.environ.get('EXPOS_OUT', '')).resolve() != out.resolve():
        raise ValueError('execute only via registered ExpOS run')
    with locked(out / 'execution'):
        path = out / 'execution.json'
        expected = identity(spec)
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['identity'] != expected:
                raise ValueError('execution identity mismatch')
            if receipt['status'] == 'completed':
                for ref in receipt['outputs']:
                    check_ref(ref)
                return receipt
        if entry.get('lbeval'):
            raise ValueError('official-exposed run cannot refit')
        atomic_json(path, {'status': 'running', 'identity': expected})
        try:
            result = final_fit(spec, out)
            outputs = [reference(p) for p in sorted(out.rglob('*')) if p.is_file() and p != path
                       and p.suffix in ('.csv', '.png', '.svg', '.parquet', '.checkpoint', '.json')]
            receipt = {'status': 'completed', 'identity': expected, 'confirmation': result, 'outputs': outputs}
            atomic_json(path, receipt)
            return receipt
        except BaseException as error:
            atomic_json(path, {'status': 'failed', 'identity': expected, 'error': str(error)})
            raise


def evaluate_once(bundle):
    core, spec, eid, entry = verify(bundle)
    if not entry['runs'] or entry['runs'][-1]['rc'] != 0:
        raise ValueError('successful terminal ExpOS run required')
    out = core.RUNS / eid
    for name, expected in entry['runs'][-1]['outputs'].items():
        if sha(out / name) != expected:
            raise ValueError('run output changed: ' + name)
    receipt = json.loads((out / 'execution.json').read_text())
    if receipt['status'] != 'completed' or receipt['identity'] != identity(spec):
        raise ValueError('completed execution identity required')
    for ref in receipt['outputs']:
        check_ref(ref)
    with locked(out / 'official-evaluation'):
        records = core.state()['experiments'][eid].get('lbeval', [])
        if len(records) > 1:
            raise ValueError('multiple official evaluations')
        if records:
            score = records[0]
        else:
            from expos.lbeval import lb_eval
            score = lb_eval(eid)
        internal = json.loads((core.RUNS / 'E1983/internal.json').read_text())
        report = {'experiment_name': 'e2000_pooled_official', 'id': eid, 'internal_reference': 'E1983',
                  'internal_mse': internal['mse'], 'official': score, 'seed': 42,
                  'single_seed_limit': 'Legacy three-seed gate remains unsatisfied', 'upload_requested': False}
        atomic_json(out / 'report.json', report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'execute', 'evaluate'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    if args.command == 'register':
        if args.inputs is None:
            parser.error('--inputs required')
        result = register(args.inputs, args.bundle)
    else:
        result = execute(args.bundle) if args.command == 'execute' else evaluate_once(args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
