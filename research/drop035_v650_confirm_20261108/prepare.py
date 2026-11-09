"""Prepare only the user-frozen three-seed combination, preserving the old campaign."""
import difflib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scheduler_support import atomic, now
from transport_support import remote, REMOTE, RPY

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent / 'offset_roll_complete_ablation_20261106'
EXCLUDE = ['ch_ch_t035_equatorial_area', 'strip_v650_dark']


def main():
    if (ROOT / 'prepared.json').exists():
        raise RuntimeError('Already prepared')
    original = json.loads((OLD / 'assignment.global.json').read_text())
    recipes = []
    for seed in (42, 43, 44):
        recipes.append(dict(experiment=f'drop035_v650_s{seed}', family='pooling', stage='ablation',
                            variant='two_feature_removal', category='combined', seed=seed, threads=4,
                            execution_host='local' if seed == 42 else 'golab2',
                            config={'seed': seed, 'exclude_features': EXCLUDE}))
    assignment = dict(campaign=ROOT.name, host='local', python=str(REPO / '.venv/bin/python'),
                      data=original['data'], observations=original['observations'],
                      observation_revision=original['observation_revision'], threads=4,
                      experiments=recipes, max_jobs=1, max_cpu_threads_total=24, reserve_memory_gib=6)
    atomic(ROOT / 'assignment.global.json', assignment)
    atomic(ROOT / 'assignment.json', dict(assignment, experiments=recipes[:1]))
    atomic(ROOT / 'registry.json', {r['experiment']: r for r in recipes[:1]})
    atomic(ROOT / 'runtime.json', {r['experiment']: {'state': 'waiting'} for r in recipes})
    atomic(ROOT / 'scheduler-state.json', {recipes[0]['experiment']: {'state': 'waiting'}})
    atomic(ROOT / 'manifest.json', dict(campaign=ROOT.name, hosts=['local', 'golab2'], global_inventory='assignment.global.json', no_automatic_expansion=True))
    atomic(ROOT / 'provenance.json', dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                        parent='original A', authorization='explicit user-selected exploratory combination 2026-11-08',
                                        observation_revision=original['observation_revision'], source_campaign=OLD.name,
                                        scientific_scope='Only two source-feature removals; three fresh seeds and fixed equal-weight mean; no tuning'))
    atomic(ROOT / 'configuration-diff.json', {'before': {'exclude_features': [], 'alpha': .25, 'spectral_hessian_only': False},
                                           'after': {'exclude_features': EXCLUDE, 'alpha': .25, 'spectral_hessian_only': False}})
    source_diff = []
    for name in ('core.py', 'baseline.py', 'competition.py'):
        before = (OLD / name).read_text(); after = (ROOT / name).read_text()
        assert before == after, 'Shared scientific implementation changed'
        source_diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(OLD / name), tofile=str(ROOT / name)))
    (ROOT / 'source.diff').write_text(''.join(source_diff) or '# Scientific modules are byte-identical to the validated campaign; configuration-only combination.\n')
    remote_old = '/home/yehwan/sunrun-offset-roll-complete-ablation-20261106'
    raw = json.loads(remote(f"from pathlib import Path;print((Path({remote_old!r})/'assignment.json').read_text())", timeout=30).stdout)
    remote_assignment = dict(assignment, host='golab2', python=RPY, max_jobs=2, data=raw['data'], observations=raw['observations'], experiments=recipes[1:])
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for name in ('core.py', 'baseline.py', 'competition.py', 'scheduler_support.py', 'host_scheduler.py', 'provenance.json', 'strategy.md', 'configuration-diff.json', 'source.diff'):
            archive.add(ROOT / name, arcname=name)
    remote_state = {r['experiment']: {'state': 'waiting'} for r in recipes[1:]}
    code = f'''import sys,io,tarfile,json
from pathlib import Path
root=Path({REMOTE!r})
if root.exists():raise RuntimeError('Refuse existing remote campaign')
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:archive.extractall(root,filter='data')
(root/'assignment.json').write_text(json.dumps({remote_assignment!r},indent=2))
(root/'scheduler-state.json').write_text(json.dumps({remote_state!r}))
(root/'runs').mkdir()
print('prepared')
'''
    remote(code, input=payload.getvalue(), timeout=60)
    (ROOT / 'runs').mkdir(exist_ok=True)
    atomic(ROOT / 'prepared.json', {'utc': now(), 'remote': REMOTE, 'experiments': 3})


if __name__ == '__main__':
    main()
