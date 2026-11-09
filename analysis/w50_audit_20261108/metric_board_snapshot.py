"""Read-only board ranking snapshot at one immutable revision."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from huggingface_hub import HfApi, hf_hub_download
api=HfApi();repo='tlabtlab/sunrun-lb-store';revision=api.repo_info(repo,repo_type='dataset').sha
files=api.list_repo_files(repo,repo_type='dataset',revision=revision)
names=[f for f in files if f.startswith('results/') and f.endswith('.json')]
def read(name):
 p=hf_hub_download(repo,name,repo_type='dataset',revision=revision)
 row=json.loads(Path(p).read_text());c=row.get('competition',{})
 return {'path':name,'experiment':row.get('experiment'),'s':c.get('total',{}).get('s'),'version':c.get('version'),'submitted_utc':row.get('submitted_utc')}
with ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(read,names))
scored=sorted([r for r in rows if r['version']=='lead-timse-6h-v1' and r['s'] is not None],key=lambda r:r['s'])
out={'dataset':repo,'revision':revision,'records_read':len(rows),'scored_records':len(scored),'ranking':scored}
Path(__file__).with_name('metric_board_snapshot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'revision':revision,'records':len(rows),'top10':scored[:10]},indent=2))
