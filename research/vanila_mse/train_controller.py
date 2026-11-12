"""Launch and validate three predeclared CPU MSE replicates in independent tmux sessions."""
import json,os,shlex,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PY=REPO/'.venv/bin/python'
SEEDS=(42,43,44)
THREADS=6

def write(path,data):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

def now():return datetime.now(timezone.utc).isoformat()

def memory_gib():
 return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2

def launch(seed):
 run=ROOT/'runs'/f'mse_s{seed}';run.mkdir(parents=True,exist_ok=True)
 session=f'vanila-mse-s{seed}'
 if (run/'launch.json').exists():raise RuntimeError(f'Refusing duplicate run {run}')
 cmd=[str(PY),'-u',str(ROOT/'experiment.py'),'--experiment','vanila_mse','--seeds',str(seed),'--out',str(run),'--device','cpu','--threads',str(THREADS),'--observations',str(ROOT/'input/competition-observations.parquet')]
 write(run/'launch.json',{'experiment':f'mse_s{seed}','seed':seed,'family':'vanila_mse','host':'local','training_host':'local','tmux_session':session,'started_utc':now(),'campaign_root':str(ROOT),'command':cmd,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'device':'cpu','threads':THREADS})
 shell='#!/bin/bash\nset -o pipefail\ncd '+shlex.quote(str(REPO))+'\n'
 shell+='env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(cmd)+' >> '+shlex.quote(str(run/'train.log'))+' 2>&1\n'
 shell+='code=$?\nprintf "%s\\n" "$code" > '+shlex.quote(str(run/'train.exit.tmp'))+'\nmv '+shlex.quote(str(run/'train.exit.tmp'))+' '+shlex.quote(str(run/'train.exit'))+'\nexit "$code"\n'
 (run/'run.sh').write_text(shell)
 subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(run/'run.sh')],check=True)
 return {'state':'running','started_utc':now(),'seed':seed,'tmux_session':session}

def validate(seed):
 run=ROOT/'runs'/f'mse_s{seed}'
 if (run/'train.exit').read_text().strip()!='0':raise RuntimeError(f'Seed {seed} failed; inspect {run}/train.log')
 required=[run/'results'/f's{seed}'/'predictions.parquet',run/'results'/f's{seed}'/'score.json']
 for period in ['F1','F2','F3']:
  required.extend([run/f'{period}_metadata.json',run/f'{period}_predictions.npz',run/'weights'/period/f's{seed}.npz'])
 for p in required:
  if not p.exists():raise RuntimeError(f'Missing {p}')
 score=json.loads((run/'results'/f's{seed}'/'score.json').read_text())['competition']
 if score['total']['n_replaced']!=0:raise RuntimeError('Unexpected persistence replacements')
 for period in ['F1','F2','F3']:
  m=json.loads((run/f'{period}_metadata.json').read_text())
  if m['seed']!=seed or m['rounds']!=900 or m['feature_count']!=31 or m['smoke']:raise RuntimeError('Model metadata mismatch')
 write(run/'complete.json',{'status':'complete','seed':seed,'competition':score,'publish':'not_required','publish_reason':'Seed replicate; fixed three-seed mean will be published as vanila_mse.'})
 return {'state':'complete','seed':seed,'total_s':score['total']['s'],'completed_utc':now()}

def main():
 state={};pending=list(SEEDS)
 while pending or any(v['state']=='running' for v in state.values()):
  for seed in SEEDS:
   name=f'mse_s{seed}'
   if state.get(name,{}).get('state')=='running' and (ROOT/'runs'/name/'train.exit').exists():state[name]=validate(seed)
  # Conservative admission: raw table loading plus fitting needs headroom, max 3 x 6 threads.
  if pending and memory_gib()>9:
   seed=pending.pop(0);state[f'mse_s{seed}']=launch(seed)
  write(ROOT/'runtime.json',state)
  if pending or any(v['state']=='running' for v in state.values()):time.sleep(15)
 write(ROOT/'training-complete.json',{'status':'complete','seeds':list(SEEDS),'completed_utc':now(),'runs':state})
 print(json.dumps({'event':'training_complete','runs':state}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(ROOT/'controller-error.json',{'utc':now(),'error':str(exc)})
  raise
