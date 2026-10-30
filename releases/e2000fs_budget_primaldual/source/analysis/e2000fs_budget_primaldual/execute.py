"""Durable CPU3 training/publication supervisor with five-minute owned status updates."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RECORD=ROOT/'research/tail-budget-20261029/lanes/primaldual'
STATUS=ROOT/'leaderboard/lbstatus.py'
EXPERIMENT=HERE.name


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def heartbeat(out,stop):
    while not stop.is_set():
        state=read(out/'status.json',{})
        pub=read(out/'publication/verify.json',{})
        (RECORD/'STATUS.md').write_text(
            f'# primaldual status\nUpdated: {stamp()}\nSession: tb29-primaldual; supervisor PID {os.getpid()}. '
            f'Owned code: {HERE}; records: {RECORD}.\nRun: {out}\n'
            f'State: {json.dumps(state,ensure_ascii=False)}\nPublication verified: {bool(pub)}\n'
            'Fixed one candidate; corrected point4 controls; CPU3/BLAS1. '
            'Internal decisions never stop F1/F2 or submission. No claim of success before official scores.\n')
        stop.wait(300)


def report(out,phase,error=None):
    paired=read(out/'paired_results.json',{})
    lines=['# Primal-dual risk-budget report',f'Updated {stamp()}; phase: {phase}.',
           'Direct parent: pointweight4; historical published3556.1984/5923.5239 uses different maturity. '
           'All paired new controls and candidates use origin+144h<=cutoff, ACE lag0.',
           'M²FGB adaptation: one training-only high-risk constraint; normalized positive custom curvature; '
           'dual projected after each round. Inner90-day point4 chronological risk sets budget; '
           'no fairness theorem or performance guarantee transferred.',
           '', '| stage | control MSE | candidate MSE | delta | control >=550 | candidate >=550 | delta |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('D1','D2','D3','C','pooled_development','f1','final','combined'):
        if key not in paired: continue
        c,n=paired[key]['control'],paired[key]['candidate']
        h=lambda m: m['mse_ge550']
        hd=h(n)-h(c) if h(n) is not None and h(c) is not None else float('nan')
        lines.append(f"| {key} | {c['mse']:.6f} | {n['mse']:.6f} | {n['mse']-c['mse']:+.6f} | {h(c)} | {h(n)} | {hd:+.6f} |")
    def positive(key):
        if key not in paired: return False
        c,n=paired[key]['control'],paired[key]['candidate']
        return n['mse']<c['mse'] and n['mse_ge550'] is not None and n['mse_ge550']<=c['mse_ge550']
    internal=positive('pooled_development') and positive('C') and sum(positive(k) for k in ('D1','D2','D3'))>=2
    if 'combined' in paired:
        n=paired['combined']['candidate']
        target=n['mse']<3500 and n['mse_ge550'] is not None and n['mse_ge550']<6000
        lines += ['',f'Internal positive signal: {internal}; official paired favorable: {positive("combined")}; '
                  f'transferred: {internal and positive("combined")}; target met: {target}.',
                  f'F1 favorable: {positive("f1")}; F2 favorable: {positive("final")}.',
                  'These exposed periods are not untouched holdouts. Internal/empirical feasibility does not imply official feasibility.']
    lines += ['',f'Submission receipt: {out}/publication/submission.json',
              f'Remote verification: {read(out/"publication/verify.json",{})}',
              f'Source/input hashes: {out}/manifest.json; parent freeze: {RECORD}/preregistration.json.',
              f'Model hashes: {out}/model_receipts.json. Per-round convergence/dual traces: {out}/losses/.',
              'Paired27-day blocks, counts, bias, horizon bands and train-prior reweighting are saved per stage.']
    for trace in sorted((out/'losses').glob('*candidate*.csv')) if (out/'losses').exists() else []:
        rows=trace.read_text().splitlines()
        lines += [f'Final trace {trace.name}: `{rows[-1]}` (columns `{rows[0]}`).']
    if error: lines += ['',f'Infrastructure failure (not scientific negative): {error}']
    (RECORD/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve()
    stop=threading.Event()
    worker=threading.Thread(target=heartbeat,args=(out,stop),daemon=True); worker.start()
    try:
        if read(out/'status.json',{}).get('status')!='completed':
            subprocess.run([sys.executable,str(STATUS),'run',EXPERIMENT,'--',sys.executable,'-u',str(HERE/'run.py'),'--out',str(out)],cwd=ROOT,check=True)
        if read(out/'status.json')['smoke']:
            raise ValueError('No smoke publication')
        report(out,'training complete; publication pending')
        subprocess.run([sys.executable,'-u',str(HERE/'publish.py'),'--out',str(out)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(STATUS),'sync'],cwd=ROOT,check=True)
        report(out,'completed and submitted, sync complete')
        print('SUPERVISOR COMPLETE: published and synced',flush=True)
    except BaseException as exc:
        report(out,'infrastructure failure',traceback.format_exc())
        subprocess.run([sys.executable,str(STATUS),'set',EXPERIMENT,'--status','실패','--note',
                        f'Infrastructure failure: {type(exc).__name__}; preserved {out}'],cwd=ROOT,check=False)
        raise
    finally:
        stop.set(); worker.join()
        state=read(out/'status.json',{})
        (RECORD/'STATUS.md').write_text(f'# primaldual status\nUpdated {stamp()}\nSupervisor PID {os.getpid()}; session tb29-primaldual preserved.\n'
                                      f'State {state}; see REPORT.md and {out}/training.log for publication evidence/errors.\n')


if __name__=='__main__': main()
