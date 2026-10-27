"""Owned cumulative launch version for the explicit seed42/72-output request.

Retains protected hashes, immutable registration, input/source/poison gates,
locked completion, checkpoint identities and a single official confirmation.
Does not alter canonical runners or protected evaluation/promotion gates.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
from dataclasses import asdict

from experiments.cumulative_20261024.gate import locked
from .inputs import atomic_json, check_ref, digest, input_audit, reference, sha
from .runner import code_hashes, run
from .model import ModelConfig
from experiments.e2000_njobs_all import launch as baseline_launch

HERE = Path(__file__).resolve().parent
LB = HERE.parents[1]
PYTHON = LB / '.venv-fusion/bin/python'
MODULE = 'experiments.e2000_parallel6.launch'
BASELINE_BUNDLE = LB.parent / 'research/e2000/registered-njobs-all-v1'
BASELINE_ID = 'E1983'


def execution_command(bundle, arm):
    return shlex.join(['env', 'OMP_NUM_THREADS=1', 'OMP_DYNAMIC=FALSE', 'OPENBLAS_NUM_THREADS=1',
                      'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1',
                      str(PYTHON), '-m', MODULE, 'execute', '--bundle', str(Path(bundle).resolve()), '--arm', arm])


def protected():
    from expos import core
    changed = core.check_protected()
    if changed:
        raise ValueError('protected evaluator/data changed: ' + str(changed))
    return core


def preflight(spec):
    protected()
    audit = input_audit(spec)
    if not audit['passed']:
        raise ValueError(json.dumps(audit))
    return audit


def registered_files(spec):
    files = set(code_hashes())
    for key in ('artifacts',):
        files.update(ref['path'] for ref in spec[key].values())
    for key in ('dependencies', 'source_receipts'):
        files.update(ref['path'] for ref in spec.get(key, []))
    files.add(spec['poison']['path'])
    poison = json.loads(check_ref(spec['poison']).read_text())
    files.update(poison['builder_code_sha256'])
    for ref in spec['source_receipts']:
        r = json.loads(check_ref(ref).read_text())
        files.add(r['path'])
        files.update(e['path'] for e in r.get('evidence', {}).values())
        if r.get('exact_byte_identity'):
            files.add(r['exact_byte_identity']['path'])
    files.update(baseline_launch.code_hashes())
    files.update(str(BASELINE_BUNDLE / p) for p in ('inputs.json', 'intent.json', 'registrations.json'))
    files.update(str(LB.parent / 'research/expos/runs' / BASELINE_ID / p)
                 for p in ('execution.json', 'internal.json'))
    return sorted(files)


def validate_baseline_receipt(spec, execution, internal, expected_code):
    expected = {'input_sha256': digest(spec), 'code_hashes': expected_code, 'arm': 'pooled'}
    if execution.get('status') != 'completed' or execution.get('identity') != expected:
        raise ValueError('baseline execution identity mismatch or incomplete run')
    recipe = dict(internal['recipe'])
    current = asdict(ModelConfig())
    recipe.pop('n_jobs')
    current.pop('n_jobs')
    if (recipe != current or internal.get('seed') != 42 or internal.get('arm') != 'pooled'
            or internal.get('input_sha256') != digest(spec)):
        raise ValueError('baseline scientific recipe/input mismatch')


def verify_baseline(spec):
    """Read-only reuse of the completed paired control; never relabel its code."""
    core = protected()
    entry = core.state()['experiments'][BASELINE_ID]
    if entry.get('flags') or not entry['runs'] or entry['runs'][-1]['rc'] != 0:
        raise ValueError('baseline has integrity flags or no successful terminal run')
    old_spec = json.loads((BASELINE_BUNDLE / 'inputs.json').read_text())
    old_intent = json.loads((BASELINE_BUNDLE / 'intent.json').read_text())
    old_ids = json.loads((BASELINE_BUNDLE / 'registrations.json').read_text())
    if (old_ids['pooled'] != BASELINE_ID or digest(old_spec) != digest(spec)
            or old_intent['code_hashes'] != baseline_launch.code_hashes()
            or entry['cmd'] != baseline_launch.execution_command(BASELINE_BUNDLE, 'pooled')):
        raise ValueError('baseline registration/input/command mismatch')
    for path, expected in entry['code'].items():
        file = Path(path) if Path(path).is_absolute() else LB / path
        if sha(file) != expected:
            raise ValueError('baseline registered file changed: ' + path)
    out = core.RUNS / BASELINE_ID
    for name, expected in entry['runs'][-1]['outputs'].items():
        if sha(out / name) != expected:
            raise ValueError('baseline output differs from immutable ExpOS run record: ' + name)
    execution = json.loads((out / 'execution.json').read_text())
    internal = json.loads((out / 'internal.json').read_text())
    validate_baseline_receipt(spec, execution, internal, old_intent['code_hashes'])
    for ref in execution['outputs']:
        check_ref(ref)
    return {'experiment_id': BASELINE_ID, 'registration_hash': entry['hash'],
            'input_sha256': digest(spec), 'code_hashes': old_intent['code_hashes'],
            'execution': reference(out / 'execution.json'), 'internal': reference(out / 'internal.json'),
            'original_bundle': str(BASELINE_BUNDLE)}


def register(spec_path, bundle):
    spec = json.loads(Path(spec_path).read_text())
    preflight(spec)
    baseline = verify_baseline(spec)
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise ValueError('new owned registration bundle required')
    bundle.mkdir(parents=True)
    core = protected()
    direction = 'D_E2000_MULTIOUTPUT_S42'
    if direction not in core.state()['directions']:
        core.add_direction(direction, 'User-fixed seed42 direct 72-output XGBoost',
                           'New architecture; no E0003/D33/D45 mixture or output correction; official once regardless of internal score.')
    atomic_json(bundle / 'inputs.json', spec)
    intent = {'version': 1, 'experiment_name': 'e2000', 'seeds': [42], 'horizons': 72,
              'xgboost_n_jobs': 1, 'parallel_workers': 6, 'execution_version': 'parallel6',
              'baseline_reference': baseline,
              'code_hashes': code_hashes(), 'input_sha256': digest(spec),
              'official_once_even_if_internal_negative': True, 'upload_only_if_internal_improves': True}
    atomic_json(bundle / 'intent.json', intent)
    atomic_json(bundle / 'baseline-reference.json', baseline)
    ids = {'pooled': BASELINE_ID}
    for arm in ('multioutput',):
        command = execution_command(bundle, arm)
        event = core.register(direction, 'codex-multioutput-s42',
            f'e2000 user-authorized six concurrent horizon processes, XGBoost n_jobs=1 each: same frozen inputs, seed42; '
            '72 independent outputs vs pooled direct XGBoost control. No output correction or ensemble. '
            'Multioutput gets one official confirmation even for an internal negative; no official tuning.',
            command, kind='reference' if arm == 'pooled' else 'full',
            code=registered_files(spec) + [str(bundle / 'inputs.json'), str(bundle / 'intent.json'),
                                           str(bundle / 'baseline-reference.json')],
            reference=ids.get('pooled'))
        ids[arm] = event['id']
        # Publish each registration immediately, so interruption never loses an assigned EID.
        atomic_json(bundle / 'registrations.json', ids)
    return ids


def verify(bundle, arm):
    if arm != 'multioutput':
        raise ValueError('completed E1983 control is reused read-only, never executed in this version')
    core = protected()
    bundle = Path(bundle)
    spec = json.loads((bundle / 'inputs.json').read_text())
    intent = json.loads((bundle / 'intent.json').read_text())
    if (intent['code_hashes'] != code_hashes() or intent['input_sha256'] != digest(spec)
            or intent['seeds'] != [42] or intent['horizons'] != 72
            or intent.get('xgboost_n_jobs') != 1 or intent.get('parallel_workers') != 6):
        raise ValueError('frozen execution intent drift')
    ids = json.loads((bundle / 'registrations.json').read_text())
    eid = ids[arm]
    entry = core.state()['experiments'][eid]
    if entry['cmd'] != execution_command(bundle, arm):
        raise ValueError('registered command/arm mismatch')
    if entry.get('flags'):
        raise ValueError('registered experiment has integrity flags')
    for path in registered_files(spec) + [str(bundle / 'inputs.json'), str(bundle / 'intent.json'),
                                         str(bundle / 'baseline-reference.json')]:
        if entry['code'].get(path) != sha(path):
            raise ValueError('registered artifact changed: ' + path)
    baseline = verify_baseline(spec)
    if (ids.get('pooled') != BASELINE_ID or intent.get('baseline_reference') != baseline
            or json.loads((bundle / 'baseline-reference.json').read_text()) != baseline):
        raise ValueError('frozen baseline reference changed')
    preflight(spec)
    return core, spec, ids, entry


def execute(bundle, arm):
    core, spec, ids, entry = verify(bundle, arm)
    out = core.RUNS / ids[arm]
    if Path(os.environ.get('EXPOS_OUT', '')).resolve() != out.resolve():
        raise ValueError('execute only through registered expos run')
    with locked(out / 'execution'):
        receipt = out / 'execution.json'
        identity = {'input_sha256': digest(spec), 'code_hashes': code_hashes(), 'arm': arm}
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior['identity'] != identity:
                raise ValueError('execution identity drift')
            if prior['status'] == 'completed':
                for ref in prior['outputs']:
                    check_ref(ref)
                return prior
        if entry.get('lbeval'):
            raise ValueError('official-exposed recipe cannot resume training')
        atomic_json(receipt, {'status': 'running', 'identity': identity})
        try:
            result = run(spec, out, arm=arm)
            outputs = [reference(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.suffix in ('.csv', '.png', '.svg', '.parquet', '.checkpoint', '.json')
                       and p != receipt]
            value = {'status': 'completed', 'identity': identity, 'internal': result, 'outputs': outputs}
            atomic_json(receipt, value)
            return value
        except BaseException as e:
            atomic_json(receipt, {'status': 'failed', 'identity': identity, 'error': str(e)})
            raise


def evaluate_once(bundle):
    core, spec, ids, entry = verify(bundle, 'multioutput')
    for arm, eid in [('multioutput', ids['multioutput'])]:
        receipt = json.loads((core.RUNS / eid / 'execution.json').read_text())
        expected = {'input_sha256': digest(spec), 'code_hashes': code_hashes(), 'arm': arm}
        if receipt['identity'] != expected:
            raise ValueError('baseline/candidate execution identity mismatch')
        if receipt['status'] != 'completed':
            raise ValueError('both internal runs must complete before evaluation')
        for ref in receipt['outputs']:
            check_ref(ref)
    out = core.RUNS / ids['multioutput']
    control = json.loads((core.RUNS / ids['pooled'] / 'internal.json').read_text())
    candidate = json.loads((out / 'internal.json').read_text())
    if (control['observed_cells'] != candidate['observed_cells']
            or control['feature_names'] != candidate['feature_names']
            or control['train_target_cutoff'] != candidate['train_target_cutoff']):
        raise ValueError('paired internal rows/features/cutoff mismatch')
    with locked(out / 'official-evaluation'):
        # Re-read inside lock; never score the same EID twice.
        records = core.state()['experiments'][ids['multioutput']].get('lbeval', [])
        if records:
            if len(records) != 1:
                raise ValueError('more than one official evaluation already exists')
            score = records[0]
        else:
            from expos.lbeval import lb_eval
            score = lb_eval(ids['multioutput'])
        control = json.loads((core.RUNS / ids['pooled'] / 'internal.json').read_text())
        candidate = json.loads((out / 'internal.json').read_text())
        delta = candidate['mse'] / control['mse'] - 1
        report = {'experiment_name': 'e2000', 'ids': ids, 'xgboost_n_jobs': 1, 'parallel_workers': 6,
                  'reused_completed_baseline': BASELINE_ID,
                  'internal_control_mse': control['mse'],
                  'internal_candidate_mse': candidate['mse'], 'internal_relative_delta': delta,
                  'official': score, 'upload_eligible': delta < 0,
                  'single_seed_limit': 'seed42 only; legacy three-seed gate is preserved, not satisfied',
                  'historical_comparators': {eid: score['summary']['mse_seedmean'] / value - 1
                                            for eid, value in {'E1813': 3277.1166, 'E1789': 3180.2038}.items()}}
        atomic_json(out / 'report.json', report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'register', 'execute', 'evaluate'])
    p.add_argument('--inputs', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--arm', choices=['pooled', 'multioutput'])
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.command in ('audit', 'register') and a.inputs is None:
        p.error('--inputs required')
    if a.command != 'audit' and a.bundle is None:
        p.error('--bundle required')
    if a.command == 'execute' and a.arm is None:
        p.error('--arm required')
    if a.command == 'audit':
        result = input_audit(json.loads(a.inputs.read_text()))
        protected()
    elif a.command == 'register':
        result = register(a.inputs, a.bundle)
    elif a.command == 'execute':
        result = execute(a.bundle, a.arm)
    else:
        result = evaluate_once(a.bundle)
    if a.report:
        atomic_json(a.report, result)
    print(json.dumps(result, indent=2))
    return 2 if result.get('passed') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
