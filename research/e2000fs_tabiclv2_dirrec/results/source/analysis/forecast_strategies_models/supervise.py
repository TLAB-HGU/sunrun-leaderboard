"""Durable followup queue; never mutate the environment before Kumo completes."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LB = ROOT / 'leaderboard'
PYTHON = ROOT / '.venv/bin/python'
KUMO = HERE.parent / 'forecast_strategies_kumo'
ORDER = ('tabiclv2', 'tabldm', 'realmlp', 'lightgbm', 'mitra')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def prerequisite_status():
    state_path = KUMO / 'outputs/supervision.json'
    verify_path = KUMO / 'publication/verify.json'
    if not state_path.exists():
        return 'waiting', 'Kumo supervisor has not recorded status'
    state = json.loads(state_path.read_text())
    if state.get('status') == 'failed':
        return 'prerequisite_failed', state.get('error', 'Kumo failed')
    if state.get('status') != 'completed' or not verify_path.exists():
        return 'waiting', 'Kumo inference/publication is still in progress'
    records = json.loads(verify_path.read_text()).get('records', {})
    if set(records) != {'Recursive', 'DirRec', 'Direct', 'pooled'} or not all(r.get('predictions_identical') and
                                  (r.get('scores_match') or r.get('mse_matches')) for r in records.values()):
        return 'prerequisite_failed', 'Kumo remote verification is incomplete'
    return 'ready', None


def command(args, cwd=HERE):
    print('+', shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def validate_queue(path):
    if path is None:
        return
    frozen = json.loads(Path(path).read_text())
    for source, expected in frozen['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen queue code changed: {source}')


def dependencies(model):
    """Explicit package bounds preserve shared torch until a supported Mitra migration exists."""
    common = ['torch==2.7.1+cu126', 'huggingface_hub==2.3.0', 'numpy==2.4.6', 'pandas==2.3.3']
    packages = {
        'tabiclv2': ['structured-data-models==0.2.1'],
        'tabldm': ['Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@03cfbb9e7069b821dcef32676dc19b40d294e348'],
        'realmlp': ['pytabkit==1.7.3'],
        'lightgbm': ['lightgbm==4.7.0'],
    }
    if model not in packages:
        raise RuntimeError('Mitra environment migration requires reusable official adapter proof first')
    return ['uv', 'pip', 'install', '--python', str(PYTHON), '--extra-index-url',
            'https://download.pytorch.org/whl/cu126', '--index-strategy', 'unsafe-best-match',
            *common, *packages[model]]


def register(model, out):
    sys.path.insert(0, str(LB))
    from expos import core
    path = out / 'registration.json'
    if path.exists():
        return json.loads(path.read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    experiments = {}
    code = [str(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    for strategy in ('Recursive', 'DirRec', 'Direct', 'pooled'):
        cmd = shlex.join([str(PYTHON), str(HERE / 'run.py'), 'run', '--model', model,
                         '--strategy', strategy, '--out', str(out)])
        event = core.register('D154', 'fm-model-ablation',
                              f'Frozen {model} replacement for e2000fs {strategy}; seed42 latest5000 scalar context; '
                              'same lag0/prefill/splits as Kumo. User-authorized GPU1/official weights. '
                              'All valid arms publish without official-score selection. No parent promotion.',
                              cmd, kind='full', code=code, reference='E0607')
        experiments[strategy] = event['id']
    value = {'model': model, 'experiments': experiments, 'config': manifest['config']}
    save(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'outputs')
    parser.add_argument('--once', action='store_true', help='Check prerequisites without waiting')
    parser.add_argument('--queue-manifest', type=Path)
    args = parser.parse_args()
    state_path = args.out / 'supervision.json'
    validate_queue(args.queue_manifest)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'models': {}, 'status': 'waiting_for_kumo'}
    state['lightpfn'] = {'status': 'unsupported_regression',
                         'reason': 'Official LightPFN 1.1.0 supports classification only; no discretization adaptation'}
    from mitra_adapter import recipe_status
    state['mitra_recipe_preflight'] = recipe_status()
    os.environ.update(CUDA_VISIBLE_DEVICES='1', CC='/usr/bin/gcc-12', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    while True:
        status, reason = prerequisite_status()
        if status == 'ready':
            break
        state.update(status=status, prerequisite_reason=reason)
        save(state_path, state)
        print(f'{status}: {reason}', flush=True)
        if status == 'prerequisite_failed' or args.once:
            return
        time.sleep(30)
    for model in ORDER:
        validate_queue(args.queue_manifest)
        if state['models'].get(model, {}).get('status') in ('completed', 'blocked'):
            continue
        state.update(status='running', current_model=model)
        state['models'][model] = {'status': 'running'}
        save(state_path, state)
        out = args.out / model
        try:
            if model == 'mitra':
                from mitra_adapter import MitraAdapter
                # Preserve the source-backed blocker receipt without installing an incompatible stack.
                MitraAdapter().fit(None, None, None, None, out)
            command(dependencies(model))
            # Capture the resolved environment before any candidate fit.
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'environment.freeze.txt').open('w') as stream:
                subprocess.run(['uv', 'pip', 'freeze', '--python', str(PYTHON)], stdout=stream, check=True)
            command([PYTHON, HERE / 'run.py', 'prepare', '--model', model, '--out', out])
            command([PYTHON, HERE / 'run.py', 'run', '--model', model, '--smoke', '--out', out / 'smoke'])
            registration = register(model, out)
            sys.path.insert(0, str(LB))
            from expos import core
            for strategy, eid in registration['experiments'].items():
                state['models'][model].update(strategy=strategy, eid=eid)
                save(state_path, state)
                command([PYTHON, '-m', 'expos', 'run', eid, '--timeout', '172800'], cwd=LB)
                latest = core.state()['experiments'][eid]['runs'][-1]
                if latest['rc'] != 0:
                    raise RuntimeError(f'{eid} returned {latest["rc"]}; log {latest["log"]}')
            command([PYTHON, HERE / 'run.py', 'report', '--model', model, '--out', out])
            for step in ('build', 'dryrun', 'push', 'github', 'submit', 'verify'):
                command([PYTHON, HERE / 'publish.py', '--model', model, '--out', out, step])
            state['models'][model]['status'] = 'completed'
        except Exception as error:
            state['models'][model].update(status='blocked' if model == 'mitra' else 'failed', error=str(error))
            print(f'{model}: {type(error).__name__}: {error}', flush=True)
        save(state_path, state)
    complete = all(item['status'] == 'completed' for item in state['models'].values()) and len(state['models']) == len(ORDER)
    state['status'] = 'completed' if complete else 'completed_with_blockers'
    save(state_path, state)


if __name__ == '__main__':
    main()
