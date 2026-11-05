"""Cancel every dirrec-family experiment without deleting historical artifacts."""
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'leaderboard'))
import lbstatus

mp=ROOT/'manifest.json';manifest=json.loads(mp.read_text())
names={e['experiment'] for e in manifest['experiments'] if 'dirrec' in e['experiment'].lower()}
for name in names:
    assert not (ROOT/'runs'/name/'launch.json').exists(),f'Unexpected launched run: {name}'
manifest['user_excluded_experiments']=sorted(set(manifest.get('user_excluded_experiments',[]))|names)
for entry in manifest['experiments']:
    if entry['experiment'] in names:
        entry['state']='excluded_by_user';entry['exclusion_reason']='User removed all dirrec-family experiments'
mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
p=ROOT/'registry.json';registry=json.loads(p.read_text());removed=set(registry)&names
for name in names:registry.pop(name,None)
tmp=p.with_suffix('.new.json');tmp.write_text(json.dumps(registry,indent=2)+'\n');tmp.replace(p)
lbstatus.leaderboard_submissions=lambda entries,dataset:{}
lbstatus.commit(lambda entries:[e for e in entries if not(e.get('member')=='seongeun' and 'dirrec' in e.get('experiment','').lower())],message='Remove all user-cancelled dirrec-family experiments')
api=HfApi();head=api.repo_info(lbstatus.SPACE,repo_type='space').sha
actual=json.loads(Path(hf_hub_download(lbstatus.SPACE,'status.json',repo_type='space',revision=head)).read_text())
assert not any(e.get('member')=='seongeun' and 'dirrec' in e.get('experiment','').lower() for e in actual['active'])
receipt={'excluded':sorted(names),'count':len(names),'removed_from_execution_queue':len(removed),'active_dirrec_rows':0,'revision':head}
(ROOT/'removed-dirrec.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
