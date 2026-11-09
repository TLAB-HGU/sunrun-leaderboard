"""Freeze exactly three fresh seeds of the user-selected joint lag addition."""
import argparse
import ast
import difflib
import io
import tarfile
import json
from pathlib import Path
import subprocess

import baseline
import core
from scheduler_support import atomic, identity, now
from transport_support import remote, REMOTE, RPY

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT.parent / 'feature_addition_all_20261108'
PARENT = ROOT.parent / 'drop035_v650_confirm_20261108'
ADDITIONS = ['v_lag24', 'v_lag48', 'v_lag72']
PARENT_ID = 'e2000fs_calib_offset_roll_drop_t035_v650_s42'


def recipes():
    return [dict(experiment=f'lag24_48_72_s{seed}', seed=seed, family='pooling',
                 leaderboard_experiment=f'e2000fs_calib_offset_roll_drop_t035_v650_lag24_48_72_s{seed}',
                 stage='ablation', kind='joint_addition', execution_host='golab2', threads=4,
                 config={'seed': seed, 'add_features': list(ADDITIONS)}) for seed in (42, 43, 44)]


def scientific_functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name != 'main'}


def validate_source(root=ROOT):
    for name in ('baseline.py', 'competition.py', 'inventory.json', 'replay.py'):
        if (root / name).read_bytes() != (SOURCE / name).read_bytes():
            raise ValueError('Validated source differs: ' + name)
    if scientific_functions(root / 'core.py') != scientific_functions(SOURCE / 'core.py'):
        raise ValueError('Scientific training functions changed from validated additive runner')
    base = core.feature_names(core.normalize_config({'seed': 42, 'add_features': []}))
    names = core.feature_names(core.normalize_config({'seed': 42, 'add_features': ADDITIONS}))
    if sum(map(len, names.values())) != 31:
        raise ValueError('Expected28 parent plus3 source features')
    for start in (0, 24, 48):
        expanded = core.expanded_names(names, start, 24)
        if len(expanded) != 215 or expanded[:212] != core.expanded_names(base, start, 24):
            raise ValueError('Original212 expanded input prefix must be preserved')
        if expanded[212:] != ADDITIONS:
            raise ValueError('Joint additions/order differ')
    return names



def stage_remote(root, assignment, gate):
    remote_parent = '/home/yehwan/sunrun-drop035-v650-confirm-20261108'
    previous = json.loads(remote(f"from pathlib import Path;print((Path({remote_parent!r})/'assignment.json').read_text())", timeout=30).stdout)
    proofs = {}
    for field in ('data', 'observations'):
        payload = remote(f"import sys;from pathlib import Path;sys.stdout.buffer.write(Path({previous[field]!r}).read_bytes())", timeout=180).stdout
        expected = Path(assignment[field]).read_bytes()
        if payload != expected:
            raise ValueError('Local/remote frozen bytes disagree: ' + field)
        proofs[field] = dict(exact_byte_equality=True, bytes=len(payload), local=assignment[field], remote=previous[field])
    atomic(root / 'cross-host-input-equality.json', dict(passed=True, utc=now(), comparisons=proofs))
    distant = dict(assignment, host='golab2', python=RPY, data=previous['data'], observations=previous['observations'])
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path in sorted(root.iterdir()):
            if path.is_file() and (path.suffix == '.py' or path.name in ('strategy.md', 'inventory.json', 'parent-lock.json', 'provenance.json', 'source.diff', 'configuration-diff.json', 'manifest.json')):
                archive.add(path, arcname=path.name)
    code = f"""import io,json,sys,tarfile
from pathlib import Path
root=Path({REMOTE!r})
if root.exists():raise RuntimeError('Refuse existing remote campaign')
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:archive.extractall(root,filter='data')
(root/'assignment.json').write_text(json.dumps({distant!r},indent=2))
(root/'scheduler-state.json').write_text(json.dumps({ {r['experiment']: {'state': 'waiting'} for r in assignment['experiments']}!r},indent=2))
(root/'runs').mkdir()
def identity(p):
 s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
proof={gate!r}
proof['golab2']=dict(input_consistency=dict(input_identity=identity(Path({previous['data']!r}))),source_identity={{f:identity(root/f) for f in ('baseline.py','core.py','competition.py','inventory.json','replay.py','parent-lock.json')}})
(root/'preflight.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof['golab2']))
"""
    result = remote(code, input=stream.getvalue(), timeout=60)
    gate['golab2'] = json.loads(result.stdout)
    gate['cross_host_input_equality'] = proofs
    atomic(root / 'preflight.json', gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / 'prepared.json').exists():
        raise RuntimeError('Campaign already frozen; resume existing services instead')
    parent = json.loads((root / 'parent-lock.json').read_text())
    if parent.get('state') != 'verified' or parent.get('experiment') != PARENT_ID:
        raise ValueError('Current leaderboard parent must be explicitly verified as the supported D model')
    if not (root / 'strategy.md').is_file():
        raise ValueError('Preregistered strategy.md required before freezing')
    features = validate_source(root)
    old = json.loads((SOURCE / 'assignment.global.json').read_text())
    prior = json.loads((SOURCE / 'preflight.json').read_text())
    if not prior.get('passed') or not prior['cross_host_input_equality'].get('passed'):
        raise ValueError('Inherited full-data and three-seed reproduction evidence missing')
    data = Path(old['data'])
    if identity(data) != prior['local']['input_consistency']['input_identity']:
        raise ValueError('Frozen prepared training data changed')
    entries = recipes()
    assignment = dict(campaign=root.name, host='local', python=str(REPO / '.venv/bin/python'),
                      data=str(data), observations=old['observations'], observation_revision=old['observation_revision'],
                      threads=4, max_jobs=3, max_cpu_threads_total=24, reserve_memory_gib=6,
                      experiments=entries, publication='explicit_user_request', parent=parent,
                      baseline_runs={str(seed): str(PARENT / 'runs' / f'drop035_v650_s{seed}') for seed in (42, 43, 44)})
    for name in ('assignment.json', 'assignment.global.json'):
        atomic(root / name, assignment)
    state = {r['experiment']: {'state': 'waiting'} for r in entries}
    for name in ('scheduler-state.json', 'runtime.json'):
        atomic(root / name, state)
    atomic(root / 'registry.json', {})
    atomic(root / 'manifest.json', dict(campaign=root.name, hosts=['golab2'], condition='joint_three_lag_addition',
          seeds=[42, 43, 44], new_training_runs=3, publication='explicit_user_request', parent=parent))
    atomic(root / 'provenance.json', dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
          git_dirty=subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True).splitlines(),
          parent=PARENT_ID, parent_lock=parent, source_campaign=str(SOURCE),
          scientific_parent_campaign=str(PARENT), observation_revision=old['observation_revision'], frozen_utc=now(),
          additions=ADDITIONS, retained_features=baseline.PARENT_NAMES,
          interpretation='Exploratory user-selected combination of seed42 findings; no independent confirmation or automatic promotion'))
    atomic(root / 'configuration-diff.json', dict(before={'add_features': [], 'source_features': baseline.PARENT_NAMES},
          after={'add_features': ADDITIONS, 'source_features': features}, source_count_before=28,
          source_count_after=31, expanded_count_before=212, expanded_count_after=215))
    diff = ''.join(''.join(difflib.unified_diff((SOURCE / f).read_text().splitlines(True), (root / f).read_text().splitlines(True),
                   fromfile=str(SOURCE / f), tofile=str(root / f))) for f in ('baseline.py', 'core.py'))
    (root / 'source.diff').write_text(diff)
    gate = dict(passed=True, utc=now(), source_equivalence='Exact baseline/scorer/inventory/replay bytes; equal scientific function ASTs',
                inherited_three_seed_reproduction=str(SOURCE / 'preflight.json'),
                local=dict(input_consistency=prior['local']['input_consistency'], source_identity={
                    f: identity(root / f) for f in ('baseline.py', 'core.py', 'competition.py', 'inventory.json', 'replay.py', 'parent-lock.json')}))
    atomic(root / 'preflight.json', gate)
    (root / 'runs').mkdir(exist_ok=True)
    stage_remote(root, assignment, gate)
    atomic(root / 'prepared.json', dict(utc=now(), training_host='golab2', remote=REMOTE, training_runs=3, source_features=31, expanded_features=215,
                                     publication='explicit_user_request', parent=PARENT_ID))
    print((root / 'prepared.json').read_text())


if __name__ == '__main__':
    main()
