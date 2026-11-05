"""Remove four unlaunched queued strategies experiments named by the user (2026-11-04)."""
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'leaderboard'))
import lbstatus

TARGETS = {'e2000fs_mimo', 'e2000fs_recmo_24', 'e2000fs_dirmo_6', 'e2000fs_rectify'}
REASON = '2026-11-04 user removed from the queue before launch (queue items 3,4,5,8)'


def matches(e):
    return e.get('member') == 'seongeun' and e.get('status') == '대기' and e.get('experiment') in TARGETS


for name in TARGETS:
    assert not (ROOT / 'runs' / name / 'launch.json').exists(), f'Already launched: {name}'

registry_path = ROOT / 'registry.json'
registry = json.loads(registry_path.read_text())
for name in TARGETS:
    registry.pop(name, None)
tmp = registry_path.with_suffix('.new.json')
tmp.write_text(json.dumps(registry, indent=2) + '\n')
tmp.replace(registry_path)

# A controller tick may have admitted one of them between the check and the write.
for name in TARGETS:
    assert not (ROOT / 'runs' / name / 'launch.json').exists(), f'Launched during removal: {name}'

manifest_path = ROOT / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['user_excluded_experiments'] = sorted(set(manifest.get('user_excluded_experiments', [])) | TARGETS)
for entry in manifest['experiments']:
    if entry['experiment'] in TARGETS:
        entry['state'] = 'excluded_by_user'
        entry['exclusion_reason'] = REASON
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')

lbstatus.leaderboard_submissions = lambda entries, dataset: {}


def remove(entries):
    for e in entries:
        if e.get('member') == 'seongeun' and e.get('experiment') in TARGETS:
            assert matches(e), 'Status changed; refuse removal of active/completed run'
    return [e for e in entries if not matches(e)]


lbstatus.commit(remove, message='Remove user-cancelled queued mimo/recmo_24/dirmo_6/rectify')
api = HfApi()
head = api.repo_info(lbstatus.SPACE, repo_type='space').sha
actual = json.loads(Path(hf_hub_download(lbstatus.SPACE, 'status.json', repo_type='space', revision=head)).read_text())
assert not any(e.get('experiment') in TARGETS for e in actual['active'])
assert not any(n in json.loads(registry_path.read_text()) for n in TARGETS)
receipt = {'removed': sorted(TARGETS), 'count': len(TARGETS), 'reason': REASON,
           'remaining_registry': len(registry), 'revision': head}
(ROOT / 'removed-queued-named-20261104.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
