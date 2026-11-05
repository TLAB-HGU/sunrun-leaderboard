"""Reload this campaign controller only at its idle boundary; leave training alone."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PID = 21440
EXPECTED = str(ROOT / 'controller.py')
proc = Path('/proc') / str(PID)
for _ in range(1800):
    if not proc.exists():
        raise RuntimeError('Expected controller disappeared; refusing ambiguous restart')
    args = (proc / 'cmdline').read_bytes().split(b'\0')
    if EXPECTED.encode() not in args:
        raise RuntimeError('Controller identity changed')
    children = (proc / 'task' / str(PID) / 'children').read_text().strip()
    if not children and (proc / 'wchan').read_text().strip() in ('hrtimer_nanosleep', 'do_nanosleep'):
        os.kill(PID, signal.SIGTERM)
        for _ in range(100):
            if not proc.exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Controller did not exit')
        session = 'sunrun-ffill-20261103-controller'
        for _ in range(100):
            if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old wrapper session remained')
        subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(ROOT/'controller-run.sh')],check=True)
        # The old process must be gone before exposing this dependency-gated
        # recipe. Its runner's independent final suite must also have passed.
        ready = ROOT / 'hsspeak-runner-ready.json'
        for _ in range(1800):
            if ready.exists():
                proof = json.loads(ready.read_text())
                if proof.get('tests_passed') is not True:
                    raise RuntimeError('HSSPEAK runner lacks passing evidence')
                registry = ROOT / 'registry.json'
                recipes = json.loads(registry.read_text())
                recipes['e1604_hsspeak25'] = {'family':'special_hsspeak','threads':2}
                temporary = registry.with_suffix('.new.json')
                temporary.write_text(json.dumps(recipes,indent=2)+'\n')
                temporary.replace(registry)
                break
            time.sleep(1)
        else:
            raise RuntimeError('HSSPEAK runner readiness not established')
        (ROOT/'controller-reload.json').write_text(json.dumps({'old_pid':PID,'reloaded_utc':time.time(),'idle_boundary_verified':True})+'\n')
        print('Controller reloaded safely', flush=True)
        break
    time.sleep(1)
else:
    raise RuntimeError('No idle boundary within 30 minutes')
