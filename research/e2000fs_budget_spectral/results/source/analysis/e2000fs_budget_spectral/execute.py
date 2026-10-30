"""Detached supervisor: train, publish once, sync, report; internal decisions never stop it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LANE=ROOT/'research/tail-budget-20261029/lanes/spectral'
STATUS=ROOT/'leaderboard/lbstatus.py'
EXPERIMENT=HERE.name


def report(out, published, error=None):
    path=out/'results.json'
    results=json.loads(path.read_text()) if path.exists() else {}
    lines=['# spectral report', '', f'Publication verified: {published}. Infrastructure error: {error}.',
           'Published parent 3556.1984/5923.5239 is historical; timing differs.',
           'Paired controls use identical origin+144h<=cutoff rows. Single fixed candidate.', '',
           '|stage|control MSE|spectral MSE|delta|control >=550|spectral >=550|delta|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('D1','D2','D3','C','pooled_development','f1','final','combined'):
        if stage not in results: continue
        pair=results[stage]
        a,b=pair['control'],pair['spectral']
        lines.append(f"|{stage}|{a['mse']:.6f}|{b['mse']:.6f}|{b['mse']-a['mse']:+.6f}|{a['mse_ge550']:.6f}|{b['mse_ge550']:.6f}|{b['mse_ge550']-a['mse_ge550']:+.6f}|")
    lines += ['', 'Decision: '+json.dumps(results.get('decision',{})),
              'FreDF-inspired complex residual radial smooth-L1 adaptation; not a neural reproduction or performance guarantee.',
              'Loss convergence: per-round CSVs in '+str(out/'losses')+'.',
              'Hashes: manifest.json, model_receipts.json. Hessian evidence: '+str(LANE/'hessian-verification.json')+'.',
              'Submission and remote verification receipts: '+str(out/'publication')+'.',
              '27-day paired descriptive deltas saved per stage; overlapping horizons are not independent samples.',
              'Remaining limitation: official-exposed feature/window selection; no future performance claim.']
    (LANE/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve()
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(240):
            path=LANE/'STATUS.md'
            with path.open('a') as f: f.write(f'\nSupervisor alive UTC {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}, PID {os.getpid()}; publication will follow training.\n')
    threading.Thread(target=heartbeat,daemon=True).start()
    error=None; code=0
    try:
        subprocess.run([sys.executable,str(STATUS),'run',EXPERIMENT,'--',sys.executable,'-u',str(HERE/'run.py'),'--out',str(out)],cwd=ROOT,check=True)
        state=json.loads((out/'status.json').read_text())
        if state!={'status':'completed','smoke':False}: raise RuntimeError('not a completed full run')
        subprocess.run([sys.executable,'-u',str(HERE/'publish.py'),'--out',str(out)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(STATUS),'sync'],cwd=ROOT,check=True)
    except BaseException as exc:
        error=repr(exc); code=1
        subprocess.run([sys.executable,str(STATUS),'set',EXPERIMENT,'--status','실패','--note',f'Infrastructure failure: {error}; artifacts preserved {out}'],cwd=ROOT,check=False)
    finally:
        stop.set(); report(out,error is None,error)
        with (LANE/'STATUS.md').open('a') as f: f.write(f'\nSupervisor finished: published={error is None}; error={error}.\n')
    return code


if __name__=='__main__': raise SystemExit(main())
