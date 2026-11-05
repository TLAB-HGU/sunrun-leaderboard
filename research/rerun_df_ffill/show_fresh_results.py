"""Show all fresh reruns while retaining the withdrawn old-input results."""
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd,HfApi
from huggingface_hub.utils import HfHubHTTPError
import hide_legacy as shared

ROOT=Path(__file__).resolve().parent
names=set(json.loads((ROOT/'registry.json').read_text()))
api=HfApi()
for attempt in range(5):
    revision=api.repo_info(shared.DATASET,repo_type='dataset').sha
    files=[f for f in api.list_repo_files(shared.DATASET,repo_type='dataset',revision=revision) if f.startswith('results/') and f.endswith('.json')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(lambda f:json.loads(shared.read(shared.DATASET,f,revision,'dataset')),files))
    records=shared.submit.with_extras(api,shared.DATASET,records,revision=revision)
    parent=api.repo_info(shared.SPACE,repo_type='space').sha
    policy=json.loads(shared.read(shared.SPACE,'visibility.json',parent,'space'))
    removed=[e for e in policy.get('hidden',[]) if e.get('member')=='seongeun' and e.get('experiment') in names]
    policy['hidden']=[e for e in policy.get('hidden',[]) if e not in removed]
    assert shared.RULE in policy.get('hidden_before',[]),'Legacy withdrawal cutoff must remain'
    architecture=json.loads(shared.read(shared.SPACE,'architecture-code.json',parent,'space'))
    status=json.loads(shared.read(shared.SPACE,'status.json',parent,'space'))
    shared.board.visibility_policy=lambda:policy
    page=shared.board.render(records,architecture=architecture)
    page=shared.lbstatus.bold_experiment_names(shared.lbstatus.mark_leaders(shared.lbstatus.inject(page,status.get('active',[]))),policy)
    page=shared.lbstatus.apply_architecture_links(page,architecture)
    if api.repo_info(shared.DATASET,repo_type='dataset').sha!=revision:continue
    try:
        commit=api.create_commit(shared.SPACE,repo_type='space',parent_commit=parent,commit_message='Show every fresh-df seongeun rerun; retain old-input withdrawal',operations=[CommitOperationAdd('visibility.json',(json.dumps(policy,indent=2,ensure_ascii=False)+'\n').encode()),CommitOperationAdd('index.html',page.encode())])
    except HfHubHTTPError as error:
        if getattr(error.response,'status_code',None) in (409,412) and attempt<4:continue
        raise
    local=ROOT.parents[1]/'leaderboard/visibility.json'
    local_policy=json.loads(local.read_text());local_policy['hidden']=[e for e in local_policy.get('hidden',[]) if not(e.get('member')=='seongeun' and e.get('experiment') in names)]
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
    assert set(completed)<=set(visible),(set(completed)-set(visible))
    receipt={'commit_url':commit.commit_url,'completed_results_visible':len(completed),'hidden_rules_removed':len(removed),'visible_old_score_rows':0,'all_future_campaign_results_visible':True}
    (ROOT/'fresh-results-visible.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    break
else:raise RuntimeError('Concurrent changes prevented publication')
