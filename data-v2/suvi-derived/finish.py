"""Finish dependent SUVI transforms when their independently built inputs exist."""
import os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
STORE=HERE.parents[0]/'model-inputs/leaderboard-store/v2'
steps={
 'fe171-verify': [STORE/'fe171/source_manifest.json',STORE/'fe171/MANIFEST.json'],
 'chmask': [STORE/'suvi-fusion/suvi-hourly-v2.parquet'],
 'chmorph': [STORE/'suvi-fusion/suvi-hourly-v2.parquet',STORE/'ch/ch-hourly.parquet',STORE/'ch/ch-feature-manifest.json'],
}
started=time.monotonic()
while steps:
 for name,paths in list(steps.items()):
  if all(p.is_file() for p in paths):
   print(f'START {name}',flush=True)
   subprocess.run([sys.executable,str(HERE/'build.py'),name],check=True)
   del steps[name]
   print(f'COMPLETE {name}',flush=True)
 if steps:
  if time.monotonic()-started>21600: raise TimeoutError(f'Input prerequisites not ready after 6h: {steps}')
  print('WAIT '+', '.join(steps),flush=True)
  time.sleep(45)
subprocess.run([sys.executable,str(HERE/'verify.py')],check=True)
