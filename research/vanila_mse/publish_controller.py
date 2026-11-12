"""Wait for the exact registered training cohort, then validate and publish its authorized fixed mean."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PY=REPO/'.venv/bin/python'

def main():
 while not (ROOT/'training-complete.json').exists():
  if (ROOT/'controller-error.json').exists():raise RuntimeError((ROOT/'controller-error.json').read_text())
  for seed in [42,43,44]:
   p=ROOT/'runs'/f'mse_s{seed}'/'train.exit'
   if p.exists() and p.read_text().strip()!='0':raise RuntimeError(f'Seed {seed} failed; publication blocked')
  time.sleep(30)
 subprocess.run([str(PY),'-u',str(ROOT/'publish_results.py'),'--observations',str(ROOT/'input/competition-observations.parquet'),'--threads','2','--publish'],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 subprocess.run([str(PY),str(REPO/'leaderboard/lbstatus.py'),'sync'],cwd=REPO,check=True)
 with (ROOT/'sunrun-complete.txt').open('w') as out:subprocess.run(['sunrun','--once'],cwd=REPO,stdout=out,stderr=subprocess.STDOUT,check=True)
 (ROOT/'publication-controller.exit').write_text('0\n')

if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'publication-controller.exit').write_text('1\n');raise
