"""Hide dominated and duplicate finished reruns; retain useful comparisons."""
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd,HfApi
from huggingface_hub.utils import HfHubHTTPError
import hide_legacy as shared

ROOT=Path(__file__).resolve().parent
rows={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/submission.json')}
metrics={n:[r['competition']['total'][k] for k in ('s','timse','rmse')] for n,r in rows.items()}
reasons={}
for n,v in metrics.items():
    better=[m for m,w in metrics.items() if all(a<=b for a,b in zip(w,v)) and any(a<b for a,b in zip(w,v))]
    if better:reasons[n]={'reason':'dominated on total S, TIMSE and RMSE','comparators':better}
for n in ('e2000fs_dirmo_24_feat30_pointweight4','e2000fs_calib_point4_144h_rawdf'):
    if n in rows:
        reference='e2000fs_calib_point4_144h'
        assert rows[n]['competition']==rows[reference]['competition']
        reasons[n]={'reason':'identical complete score report','comparator':reference}
names=set(reasons)
api=HfApi()
for attempt in range(5):
    revision=api.repo_info(shared.DATASET,repo_type='dataset').sha
    files=[f for f in api.list_repo_files(shared.DATASET,repo_type='dataset',revision=revision) if f.startswith('results/') and f.endswith('.json')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(lambda f:json.loads(shared.read(shared.DATASET,f,revision,'dataset')),files))
    records=shared.submit.with_extras(api,shared.DATASET,records,revision=revision)
    parent=api.repo_info(shared.SPACE,repo_type='space').sha
    policy=json.loads(shared.read(shared.SPACE,'visibility.json',parent,'space'))
    for name in sorted(names):
        rule={'member':'seongeun','experiment':name}
        if rule not in policy.setdefault('hidden',[]):policy['hidden'].append(rule)
    assert shared.RULE in policy.get('hidden_before',[]),'Legacy withdrawal cutoff must remain'
    architecture=json.loads(shared.read(shared.SPACE,'architecture-code.json',parent,'space'))
    status=json.loads(shared.read(shared.SPACE,'status.json',parent,'space'))
    shared.board.visibility_policy=lambda:policy
    page=shared.board.render(records,architecture=architecture)
    page=shared.lbstatus.bold_experiment_names(shared.lbstatus.mark_leaders(shared.lbstatus.inject(page,status.get('active',[]))),policy)
    page=shared.lbstatus.apply_architecture_links(page,architecture)
    if api.repo_info(shared.DATASET,repo_type='dataset').sha!=revision:continue
    try:
        commit=api.create_commit(shared.SPACE,repo_type='space',parent_commit=parent,commit_message='Hide dominated and duplicate completed reruns; retain useful comparisons',operations=[CommitOperationAdd('visibility.json',(json.dumps(policy,indent=2,ensure_ascii=False)+'\n').encode()),CommitOperationAdd('index.html',page.encode())])
    except HfHubHTTPError as error:
        if getattr(error.response,'status_code',None) in (409,412) and attempt<4:continue
        raise
    local=ROOT.parents[1]/'leaderboard/visibility.json'
    local_policy=json.loads(local.read_text())
    for name in sorted(names):
        rule={'member':'seongeun','experiment':name}
        if rule not in local_policy.setdefault('hidden',[]):local_policy['hidden'].append(rule)
    local.write_text(json.dumps(local_policy,indent=2,ensure_ascii=False)+'\n')
    head=api.repo_info(shared.SPACE,repo_type='space').sha
    actual=shared.read(shared.SPACE,'index.html',head,'space')
    start,stop,cols=shared.board.ranking(actual);labels=[name for name,_ in cols]
    visible=[]
    for row in re.findall(r'<tr\b[^>]*>.*?</tr>',actual[start:stop],re.S):
        cells=list(shared.board.CELL.finditer(row))
        if len(cells)!=len(cols):continue
        member=html.unescape(re.sub(r'<[^>]+>','',cells[labels.index('member')].group(2)))
        dated=re.search(r'data-submitted-utc="([^"]+)"',row)
        if member=='seongeun' and dated:
            assert not shared.board.hidden_before(member,dated[1],policy),'Old result became visible'
            visible.append(html.unescape(re.sub(r'<[^>]+>','',cells[labels.index('experiment')].group(2))))
    completed=[p.parent.name for p in (ROOT/'runs').glob('*/submission.json')]
    assert not names.intersection(visible),names.intersection(visible)
    assert set(completed)-names<=set(visible),(set(completed)-names-set(visible))
    receipt={'commit_url':commit.commit_url,'hidden_count':len(names),'hidden':reasons,'completed_results_visible':len(set(completed)-names),'visible_old_score_rows':0}
    (ROOT/'curated-results-visibility.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    break
else:raise RuntimeError('Concurrent changes prevented publication')
