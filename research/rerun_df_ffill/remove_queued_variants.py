"""Remove only unlaunched queued pooled/recursive experiments requested by user."""
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'leaderboard'))
import lbstatus

def matches(e):
    return e.get('member')=='seongeun' and e.get('status')=='대기' and any(s in e['experiment'] for s in ('_pooled','_recursive'))

api=HfApi();head=api.repo_info(lbstatus.SPACE,repo_type='space').sha
status=json.loads(Path(hf_hub_download(lbstatus.SPACE,'status.json',repo_type='space',revision=head)).read_text())
targets={e['experiment'] for e in status['active'] if matches(e)}
for name in targets:
    assert not (ROOT/'runs'/name/'launch.json').exists(),f'Already launched: {name}'
registry_path=ROOT/'registry.json';registry=json.loads(registry_path.read_text())
for name in targets:registry.pop(name,None)
tmp=registry_path.with_suffix('.new.json');tmp.write_text(json.dumps(registry,indent=2)+'\n');tmp.replace(registry_path)
manifest_path=ROOT/'manifest.json';manifest=json.loads(manifest_path.read_text())
manifest['user_excluded_experiments']=sorted(set(manifest.get('user_excluded_experiments',[]))|targets)
for entry in manifest['experiments']:
    if entry['experiment'] in targets:
        entry['state']='excluded_by_user'
        entry['exclusion_reason']='User removed queued experiments containing _pooled or _recursive'
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
lbstatus.leaderboard_submissions=lambda entries,dataset:{}
def remove(entries):
    for e in entries:
        if e.get('member')=='seongeun' and e.get('experiment') in targets:
            assert matches(e),'Status changed; refuse removal of active/completed run'
    return [e for e in entries if not(matches(e) and e['experiment'] in targets)]
lbstatus.commit(remove,message='Remove user-cancelled queued pooled and recursive experiments')
head=api.repo_info(lbstatus.SPACE,repo_type='space').sha
actual=json.loads(Path(hf_hub_download(lbstatus.SPACE,'status.json',repo_type='space',revision=head)).read_text())
assert not any(matches(e) for e in actual['active'])
assert not any(n in registry for n in targets)
receipt={'removed':sorted(targets),'count':len(targets),'remaining_matching_queued':0,'completed_and_running_preserved':True,'revision':head}
(ROOT/'removed-queued-variants.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
