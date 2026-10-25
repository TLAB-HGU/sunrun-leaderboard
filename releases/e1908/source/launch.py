"""Guarded ExpOS launch path for cumulative direct forecasting.

No command scores the final window automatically. Confirmation reservation is
bookkeeping under the existing selection/arrival contract, never new permission.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import gate
import runner
from poison import validate_poison

HERE = Path(__file__).resolve().parent
LB = Path('/home/t-lab01/sunrun/leaderboard')
sys.path.insert(0, str(LB))
PYTHON = LB / '.venv-fusion/bin/python'


def protected():
    from expos import core
    gate.require(not core.check_protected(), 'protected evaluator/data changed')
    return core


def canonical_files():
    hashes = runner.code_hashes()
    gate.require(all(value != 'missing' for value in hashes.values()), 'incomplete canonical runner bundle')
    return [HERE / name for name in hashes]


def no_exposure(manifest):
    entries = gate.graph_entries()
    gate.require(not any(e.get('id') == manifest['experiment_id'] and
                         e.get('event') in ('official', 'lbeval') for e in entries),
                 'official-exposed candidate cannot enter selection')


def check_current(manifest, inputs, state, parent=None, factorial=None):
    gate.preflight(manifest, parent, factorial)
    runner.validate_recipe(manifest['recipe'])
    gate.require(manifest['code_sha256'] == gate.digest(runner.code_hashes()), 'code bundle drift')
    canonical_files()
    gate.require(manifest['contract']['evaluation_sha256'] == __import__('build_inputs').evaluator_hash(),
                 'evaluator identity drift')
    gate.require(inputs['source_availability'] == manifest['contract']['sources'], 'source identity drift')
    gate.require(gate.digest(inputs) == manifest['contract']['inputs_sha256'],
                 'frozen input artifacts/features/block mapping changed')
    validate_registered_poison(inputs, manifest['recipe'])
    if Path(state).exists():
        current = gate.read(state)
        expected = parent if parent is not None else manifest
        gate.require(current['accepted_sha256'] == gate.digest(expected) and
                     current['accepted_manifest'] == expected, 'stale parent or campaign')
        gate.require(current['confirmation'] is None, 'campaign frozen after final confirmation')
    else:
        gate.require(parent is None and manifest['change_kind'] == 'baseline',
                     'initialize a reproduced baseline before branching')
    protected()


def verify_registration(manifest, intent, inputs_path):
    """Bind graph EID to immutable code, inputs and EID-free registration intent."""
    core = protected()
    registered = core.state()['experiments'].get(manifest['experiment_id'])
    gate.require(registered is not None and not registered.get('flags'), 'missing/flagged ExpOS registration')
    body = gate.read(intent)
    expected = {k: v for k, v in manifest.items() if k != 'experiment_id'}
    gate.require(body == expected, 'manifest differs from registered intent')
    code = registered['code']
    for path in [*canonical_files(), Path(intent).resolve(), Path(inputs_path).resolve()]:
        gate.require(code.get(str(path)) == runner.file_hash(path), 'registered code/input changed: ' + str(path))
    gate.require(manifest['code_sha256'] == gate.digest(runner.code_hashes()), 'registered bundle drift')
    validate_registered_poison(gate.read(inputs_path), manifest['recipe'], code)
    return registered


def validate_registered_poison(inputs, recipe, registered_code=None):
    receipt = validate_poison(inputs, recipe)
    if registered_code is not None:
        for name, sha in receipt['builder_code_sha256'].items():
            gate.require(registered_code.get(name) == sha,
                         'poison feature builder was not registered: ' + name)
        ref = inputs['poison']
        gate.require(registered_code.get(str(Path(ref['path']).resolve())) == ref['sha256'],
                     'poison receipt was not registered')
    return receipt


def register(manifest_path, inputs_path, state, bundle, parent_path=None, factorial_path=None):
    m = gate.read(manifest_path)
    inputs = gate.read(inputs_path)
    parent = gate.read(parent_path) if parent_path else None
    factorial = gate.read(factorial_path) if factorial_path else None
    check_current(m, inputs, state, parent, factorial)
    bundle = Path(bundle).resolve()
    gate.require(not bundle.exists(), 'new experiment bundle directory required')
    bundle.mkdir(parents=True)
    intent = bundle / 'intent.json'
    frozen_inputs = bundle / 'inputs.json'
    gate.atomic(intent, {k: v for k, v in m.items() if k != 'experiment_id'})
    gate.atomic(frozen_inputs, inputs)
    command = [str(PYTHON), str(HERE / 'launch.py'), 'execute', '--intent', str(intent),
               '--inputs', str(frozen_inputs), '--state', str(Path(state).resolve())]
    poison = validate_registered_poison(inputs, m['recipe'])
    registered_files = [*canonical_files(), intent, frozen_inputs,
                        Path(inputs['poison']['path']).resolve(),
                        *[Path(p) for p in poison['builder_code_sha256']]]
    for name, original in [('parent', parent_path), ('factorial', factorial_path)]:
        if original:
            target = bundle / (name + '.json')
            gate.atomic(target, gate.read(original))
            registered_files.append(target)
            command += ['--' + name, str(target)]
    core = protected()
    direction = 'D_CUMULATIVE_20261024'
    if direction not in core.state()['directions']:
        core.add_direction(direction, 'Cumulative direct forecast experiments',
                           'Frozen direct parent, paired single-axis and factorial evidence; final whole-period confirmation once.')
    event = core.register(direction, 'gjc-cumulative', m['rationale'], shlex.join(command),
                          kind='reference' if parent is None else 'full',
                          code=[str(p) for p in registered_files],
                          reference=parent['experiment_id'] if parent else None)
    m['experiment_id'] = event['id']
    gate.atomic(bundle / 'manifest.json', m)
    return {'experiment_id': event['id'], 'manifest': str(bundle / 'manifest.json'),
            'next_command': shlex.join([str(PYTHON), '-m', 'expos', 'run', event['id'], '--timeout', '86400'])}


def execute(args):
    out = Path(os.environ.get('EXPOS_OUT', '')).resolve()
    gate.require(out.parent == LB.parent / 'research/expos/runs', 'execute only within a registered ExpOS run')
    m = gate.read(args.intent)
    m['experiment_id'] = out.name
    inputs = gate.read(args.inputs)
    parent = gate.read(args.parent) if args.parent else None
    factorial = gate.read(args.factorial) if args.factorial else None
    verify_registration(m, args.intent, args.inputs)
    check_current(m, inputs, args.state, parent, factorial)
    no_exposure(m)
    if not Path(args.state).exists():
        gate.initialize(args.state, m)
    identity = {'manifest_sha256': gate.digest(m), 'inputs_sha256': runner.file_hash(args.inputs),
                'stage': 'preofficial'}
    receipt = out / 'execution.json'
    with gate.locked(out / 'execution'):
        if receipt.exists():
            existing = gate.read(receipt)
            gate.require(existing['identity'] == identity, 'execution identity drift')
            if existing['status'] == 'completed':
                gate.require(all(runner.file_hash(p) == h for p, h in existing['outputs'].items()),
                             'completed output changed')
                return existing
        gate.atomic(receipt, dict(identity=identity, status='running'))
        try:
            outputs = runner.run_stage(m, inputs, out, n_jobs=args.n_jobs)
            result = dict(identity=identity, status='completed',
                          outputs={p: runner.file_hash(p) for p in outputs})
            gate.atomic(receipt, result)
            return result
        except BaseException as error:
            gate.atomic(receipt, dict(identity=identity, status='failed', error=str(error)))
            raise


def final_confirmation(args):
    m = gate.read(args.manifest)
    state = gate.read(args.state)
    expected = dict(manifest_sha256=gate.digest(m), contract_sha256=gate.digest(m['contract']))
    gate.require(state['confirmation'] == expected, 'accepted frozen confirmation reservation required')
    gate.require(state['accepted_sha256'] == gate.digest(m) and state['accepted_manifest'] == m
                 and state['campaign_id'] == m['campaign_id'], 'stale accepted confirmation parent')
    intent = Path(args.manifest).with_name('intent.json')
    verify_registration(m, intent, args.inputs)
    gate.require(not any(e.get('id') == m['experiment_id'] and e.get('event') == 'lbeval'
                         for e in gate.graph_entries()), 'confirmation already scored; never tune/rerun')
    out = LB.parent / 'research/expos/runs' / m['experiment_id']
    receipt = out / 'confirmation-execution.json'
    with gate.locked(out / 'confirmation-execution'):
        state = gate.read(args.state)
        gate.require(state['confirmation'] == expected and state['accepted_sha256'] == gate.digest(m)
                     and state['accepted_manifest'] == m and state['campaign_id'] == m['campaign_id'],
                     'confirmation state changed while acquiring lock')
        gate.require(not any(e.get('id') == m['experiment_id'] and e.get('event') == 'lbeval'
                             for e in gate.graph_entries()), 'confirmation already scored; never tune/rerun')
        if receipt.exists():
            prior = gate.read(receipt)
            gate.require(prior['manifest_sha256'] == gate.digest(m), 'confirmation identity drift')
            if prior['status'] == 'completed':
                gate.require(all(runner.file_hash(p) == h for p, h in prior['outputs'].items()),
                             'completed confirmation changed')
                return prior
        gate.atomic(receipt, dict(manifest_sha256=gate.digest(m), status='running'))
        try:
            paths = runner.run_stage(m, gate.read(args.inputs), out, stage='official', n_jobs=args.n_jobs)
            result = dict(manifest_sha256=gate.digest(m), status='completed',
                          outputs={p: runner.file_hash(p) for p in paths},
                          note='Predictions only. Use the existing protected evaluator exactly once; no upload.')
            gate.atomic(receipt, result)
            return result
        except BaseException as error:
            gate.atomic(receipt, dict(manifest_sha256=gate.digest(m), status='failed', error=str(error)))
            raise


def main():
    parser = gate.JsonParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'register', 'execute', 'confirm'])
    for name in ('manifest', 'inputs', 'state', 'bundle', 'parent', 'factorial', 'intent'):
        parser.add_argument('--' + name)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    try:
        for name in ('inputs', 'state'):
            gate.require(getattr(args, name), '--' + name + ' required')
        if args.command == 'execute':
            gate.require(args.intent, '--intent required')
            result = execute(args)
        else:
            gate.require(args.manifest, '--manifest required')
            if args.command == 'confirm':
                result = final_confirmation(args)
            elif args.command == 'register':
                gate.require(args.bundle, '--bundle required')
                result = register(args.manifest, args.inputs, args.state, args.bundle, args.parent, args.factorial)
            else:
                check_current(gate.read(args.manifest), gate.read(args.inputs), args.state,
                              gate.read(args.parent) if args.parent else None,
                              gate.read(args.factorial) if args.factorial else None)
                result = {'preflight': 'passed', 'training_started': False}
        print(json.dumps(dict(ok=True, result=result), sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(ok=False, error=str(error)), sort_keys=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
