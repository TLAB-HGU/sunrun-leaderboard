"""Persistent supervisor: full training, heartbeat, unconditional publication/report."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LANE=ROOT/'research/tail-budget-20261029/lanes/covshift'
EXPERIMENT=HERE.name
STATUS=ROOT/'leaderboard/lbstatus.py'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def heartbeat(out, phase):
    state={}
    for name in ('status.json','progress.json'):
        if (out/name).exists():
            state[name]=json.loads((out/name).read_text())
    (LANE/'STATUS.md').write_text(f'# covshift — {phase}\nUpdated {now()}; session tb29-covshift; supervisor PID {os.getpid()}.\nOwned code: {HERE}\nRun: {out}\n\n```json\n{json.dumps(state,indent=2)}\n```\n\nNo target/transfer claim before completed official results. D1/D2/D3/C checks never gate official fits/submission.\n')


def report(out, published, error=None):
    rows=['# covshift report',f'Updated {now()}; published={published}.',
          'Historical published parent 3556.1984 / 5923.5239 has the older origin+72h boundary; all paired controls here use origin+144h<=cutoff.',
          'DA-CART adapted to bounded blocked-OOF p_recent/p_all weights, preserving year×horizon×speed-band parent masses. Assumptions: stable conditional distribution and overlap, neither guaranteed.',
          '|stage|parent MSE|candidate MSE|delta|parent ≥550|candidate ≥550|delta|',
          '|---|---:|---:|---:|---:|---:|---:|']
    if (out/'results.json').exists():
        results=json.loads((out/'results.json').read_text())
        for stage in ('D1','D2','D3','C','development','f1','final','combined'):
            item=results.get(stage,{})
            if not all(a in item for a in ('parent','candidate')):
                continue
            p,c=item['parent'],item['candidate']
            rows.append(f"|{stage}|{p['mse']:.4f}|{c['mse']:.4f}|{c['mse']-p['mse']:+.4f}|{p['mse_ge550']:.4f}|{c['mse_ge550']:.4f}|{c['mse_ge550']-p['mse_ge550']:+.4f}|")
        rows+=['',f"Decision: `{json.dumps(results.get('decision',{}))}`.",
               'Internal/official/F1/F2 signs are retained as observed; no official-score selection. Negative results are scientific outcomes, execution/publication errors are infrastructure outcomes.']
        for stage in ('f1','final'):
            if stage in results and 'candidate' in results[stage]:
                p,c=results[stage]['parent'],results[stage]['candidate']
                rows.append(f"{stage} total/tail direction: {c['mse']-p['mse']:+.4f} / {c['mse_ge550']-p['mse_ge550']:+.4f}.")
    rows+=['',f'Artifacts: {out}',
           'Actual per-round convergence: losses/*.csv (fixed900rounds; no early stopping/round selection). Hashes: manifest.json, model_receipts.json; ratio/AUC/ESS/mass and paired27day evidence: diagnostics/.',
           'Submission/remote verification receipts: publication/submission.json, publication/verify.json; status-sync receipt: sync.log. Regression/poison evidence: lane VERIFICATION.json.']
    if (out/'manifest.json').exists():
        import hashlib
        rows += ['', 'Manifest SHA256: '+hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()]
    if (out/'losses').exists():
        import csv
        rows += ['', '|loss file|final train MSE|final validation MSE|', '|---|---:|---:|']
        for loss in sorted((out/'losses').glob('*.csv')):
            records=list(csv.DictReader(loss.open()))
            if records:
                last=records[-1]
                rows.append(f"|{loss.name}|{float(last['train_mse']):.4f}|{float(last['validation_mse']):.4f}|")
    if (out/'publication/verify.json').exists():
        rows += ['', 'Remote receipt: '+(out/'publication/verify.json').read_text()]
    if error:
        rows+=['',f'Infrastructure blocker: {error}']
    (LANE/'REPORT.md').write_text('\n\n'.join(rows)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    # Training wrapper ends before publication, so final sync cannot be overwritten by its exit post.
    log=HERE/'training.log'
    if not (out/'status.json').exists():
        with log.open('a') as stream:
            job=subprocess.Popen([sys.executable,str(STATUS),'run',EXPERIMENT,'--',sys.executable,'-u',str(HERE/'run.py'),'--out',str(out)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            while True:
                heartbeat(out,'training')
                try:
                    code=job.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if code:
            heartbeat(out,'infrastructure training failure')
            report(out,False,f'training rc={code}; see {log}')
            return code
    state=json.loads((out/'status.json').read_text())
    if state['status']!='completed' or state['smoke']:
        report(out,False,f'Incomplete full run: {state}')
        return 1
    report(out,False)
    heartbeat(out,'publishing')
    for attempt in range(3):
        with (HERE/'publication.log').open('a') as stream:
            job=subprocess.Popen([sys.executable,'-u',str(HERE/'publish.py'),'--out',str(out)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            while True:
                heartbeat(out,'publishing')
                try:
                    code=job.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if code==0:
            break
        time.sleep(30)
    if code:
        report(out,False,f'publication rc={code}; predictions preserved; execute.py can resume publication without retraining')
        heartbeat(out,'infrastructure publication failure')
        return code
    # Training wrapper has already exited: sync now retires the submitted entry.
    with (out/'sync.log').open('a') as stream:
        code=subprocess.call([sys.executable,str(STATUS),'sync'],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    report(out,True,None if code==0 else f'sync rc={code}')
    heartbeat(out,'submitted and verified' if code==0 else 'submitted; sync failed')
    return code


if __name__=='__main__':
    raise SystemExit(main())
