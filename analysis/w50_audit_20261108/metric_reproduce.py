"""Read-only official-score replay; writes only this audit's metric artifacts."""
from pathlib import Path
import importlib.util, json, subprocess, sys
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAM = ROOT/'research/post_ablation_five_axes_20261108'
spec=importlib.util.spec_from_file_location('official_competition', ROOT/'leaderboard/scorer/competition.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
def identity(path):
 p=Path(path); return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
a=json.loads((CAM/'assignment.json').read_text()); lock=json.loads((CAM/'baseline-lock.json').read_text()); sota=json.loads((CAM/'sota-reference.json').read_text())
obs=c.load_observations(a['observations']); ctx=c.prepare_context(obs)
result={'scorer':identity(c.__file__),'observations':identity(a['observations']), 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'metric':'S; W is absent from official implementation', 'version':c.VERSION,'seeds':{},'periods':{},'observation_hour_validity':{}}
for period,bounds in c.PERIODS.items():
 mask=ctx['period']==period;truth=ctx['truth'][mask]
 result['periods'][period]={'bounds_end_exclusive':bounds,'origins':int(mask.sum()),'first_origin':str(ctx['origins'][mask][0]),'last_origin':str(ctx['origins'][mask][-1]),'finite_truth_pairs':int(np.isfinite(truth).sum()),'total_pairs':int(truth.size),'missing_fraction':float(np.isnan(truth).mean())}
for hour in range(6):
 sel=obs[pd.to_datetime(obs.timestamp_utc).dt.hour.mod(6).eq(hour)]
 result['observation_hour_validity'][str(hour)]={'total':len(sel),'valid':int(sel.speed_kms.notna().sum())}
for seed in (42,43,44):
 frames=[];ids=[]
 for period in c.PERIODS:
  path=CAM/f'reference/s{seed}/{period}_predictions.npz';ids.append(identity(path))
  with np.load(path) as z: pred=z['pred'];origins=pd.to_datetime(z['origins'],utc=True)
  assert pred.shape==(len(origins),72)
  frames.append(pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),'pred_kms':pred.ravel()}))
 frame=pd.concat(frames,ignore_index=True);score=c.score_competition(frame,ctx);aligned=c.align_predictions(frame,ctx)
 assert np.isfinite(aligned).all() and (aligned>0).all()
 diffs={scope:score[scope]['s']-lock['incumbent_audit']['scores'][str(seed)][scope]['s'] for scope in ('total','f1','f2','f3')}
 assert max(abs(v) for v in diffs.values())<1e-10
 for scope in ('total','f1','f2','f3'):
  sel=np.ones(len(aligned),bool) if scope=='total' else ctx['period']==scope.upper()
  score[scope]['mae']=float(np.nanmean(abs(aligned[sel]-ctx['truth'][sel])))
 result['seeds'][str(seed)]={'scores':score,'score_diff_from_lock':diffs,'inputs':ids}
values=np.array([result['seeds'][str(s)]['scores']['total']['s'] for s in (42,43,44)])
result['seed_summary']={'mean_s':float(values.mean()),'sample_variance':float(values.var(ddof=1)),'sample_sd':float(values.std(ddof=1)),'worst_s':float(values.max()),'hypothetical_half_s':float(values.mean()/2)}
def cached(name):return hf_hub_download(sota['dataset'],name,repo_type='dataset',revision=sota['revision'],local_files_only=True)
store=c.load_observations(cached('competition-observations.parquet'));assert obs.equals(store)
frame=pd.read_parquet(cached(sota['forecast_path']));actual=c.score_competition(frame,ctx);assert abs(actual['total']['s']-sota['total_s'])<1e-10
result['sota']={**sota,'replayed_scores':actual,'forecast_identity':identity(cached(sota['forecast_path'])),'store_observations_identity':identity(cached('competition-observations.parquet')),'provenance':json.loads(Path(cached('competition-provenance.json')).read_text()),'incumbent_mean_relative_excess':float(values.mean()/actual['total']['s']-1)}
(OUT/'metric_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'seed_summary':result['seed_summary'],'periods':result['periods'],'sota_s':actual['total']['s'],'all_assertions_passed':True},indent=2))
# Audit newer completed campaign separately from the adopted incumbent.
new={}
for seed in (42,43,44):
 run=ROOT/f'research/drop035_v650_confirm_20261108/runs/drop035_v650_s{seed}'
 frame=pd.read_parquet(run/'predictions.parquet');score=c.score_competition(frame,ctx);arr=c.align_predictions(frame,ctx)
 for scope in ('total','f1','f2','f3'):
  sel=np.ones(len(arr),bool) if scope=='total' else ctx['period']==scope.upper()
  err=arr[sel]-ctx['truth'][sel]
  score[scope]['mae']=float(np.nanmean(abs(err)))
  score[scope]['mae_lead_mean']=float(np.nanmean(abs(err),axis=0).mean())
 expected=json.loads((run/'submission.json').read_text())['competition']
 assert abs(score['total']['s']-expected['total']['s'])<1e-10
 new[str(seed)]={'scores':score,'prediction':identity(run/'predictions.parquet'),'exit':(run/'train.exit').read_text().strip()}
result['newer_drop035_v650']=new
new_values=np.array([v['scores']['total']['s'] for v in new.values()])
result['newer_seed_summary']={'mean_s':float(new_values.mean()),'sample_variance':float(new_values.var(ddof=1)),'sample_sd':float(new_values.std(ddof=1)),'worst_s':float(new_values.max()),'best_selected_seed_s':float(new_values.min()),'hypothetical_half_best_s':float(new_values.min()/2)}
for seed in (42,43,44):
 frames=[]
 for period in c.PERIODS:
  with np.load(CAM/f'reference/s{seed}/{period}_predictions.npz') as z: pred=z['pred'];origins=pd.to_datetime(z['origins'],utc=True)
  frames.append(pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),'pred_kms':pred.ravel()}))
 arr=c.align_predictions(pd.concat(frames),ctx)
 for scope in ('total','f1','f2','f3'):
  sel=np.ones(len(arr),bool) if scope=='total' else ctx['period']==scope.upper()
  result['seeds'][str(seed)]['scores'][scope]['mae_lead_mean']=float(np.nanmean(abs(arr[sel]-ctx['truth'][sel]),axis=0).mean())
(OUT/'metric_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['newer_seed_summary'],indent=2))
