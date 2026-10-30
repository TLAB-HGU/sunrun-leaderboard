"""Preserve immutable full_v1 sources; recover worktree-aware publication lock.

The copied publisher assumes REPOSITORY/.git is a directory. In this workspace
it is a gitfile. Wait for the original supervisor to terminate, then publish
unchanged forecasts using the same frozen prepare/client and real git-dir lock.
No retraining, replacement, candidate selection or shared source changes.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LANE=ROOT/'research/tail-budget-20261029/lanes/spectral'
STATUS=ROOT/'leaderboard/lbstatus.py'
EXPERIMENT=HERE.name


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def publish(out):
    p=imported('frozen_spectral_publisher',HERE/'publish.py')
    pub=out/'publication'; pub.mkdir(parents=True,exist_ok=True)
    with (pub/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        item=p.prepare(out)
        # Lock existing stable Git common-directory inode, without shared-file edits.
        directory=subprocess.check_output(['git','-C',str(p.REPOSITORY),'rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            publisher=p.install_worktree(out,item)
            # One final client validation; no selection or repeated dry-run loop.
            publisher.dryrun()
            publisher.push()
            publisher.github()
            p.recover_submission(publisher)
            publisher.submit()
            publisher.verify()
        finally:
            os.close(fd)
    p.P.write_json(pub/'worktree-lock-recovery.json',dict(
        reason='Original frozen publisher assumes .git directory; actual .git is gitfile',
        recovery_source_sha256=p.P.sha(__file__),same_frozen_candidate=True,
        original_training_sources_preserved=True,replace=False))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--publish',action='store_true'); args=parser.parse_args()
    out=args.out.resolve()
    if args.publish:
        publish(out); return
    last=0
    while True:
        state=subprocess.check_output(['systemctl','--user','show','e2000fs-budget-spectral-run','--property=ActiveState','--value'],text=True).strip()
        if state not in ('active','activating','deactivating'): break
        if time.monotonic()-last>240:
            with (LANE/'STATUS.md').open('a') as f:
                f.write(f'\nWorktree-aware publication recovery supervisor PID {os.getpid()} alive; waiting original training supervisor.\n')
            last=time.monotonic()
        time.sleep(15)
    statepath=out/'status.json'
    if not statepath.exists() or json.loads(statepath.read_text())!={'status':'completed','smoke':False}:
        raise RuntimeError('Training not complete: no scientific result or publication claimed')
    if (out/'publication/verify.json').exists(): return
    recovery=imported('spectral_report',HERE/'execute.py')
    try:
        subprocess.run([sys.executable,str(STATUS),'run',EXPERIMENT,'--',sys.executable,'-u',str(Path(__file__)), '--publish','--out',str(out)],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(STATUS),'sync'],check=True,cwd=ROOT)
        recovery.report(out,True,None)
        with (LANE/'REPORT.md').open('a') as f:
            f.write('\nInfrastructure recovery: frozen original publisher assumed .git directory, but repository is a worktree. publication_recovery.py resolves actual common git directory for locking. Same forecasts/sources/constants; no retraining or score selection.\n')
        with (LANE/'STATUS.md').open('a') as f:
            f.write('\nPublication recovery completed: submission and remote prediction parity verified; sync completed. See REPORT.md.\n')
    except BaseException:
        recovery.report(out,False,traceback.format_exc()); raise


if __name__=='__main__': main()
