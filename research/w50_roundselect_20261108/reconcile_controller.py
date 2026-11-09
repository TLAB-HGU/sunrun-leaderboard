"""Revalidate terminal receipts after the scheduler-only count-shadowing fix.

Never interrupts a fit, starts a fit, or changes a forecast. The original
scheduler source and terminal classification are preserved before replacement.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'rerun_df_ffill_20261103'))
from golab2_transport import remote

REMOTE = '/home/yehwan/w50-roundselect-workspace-20261108/research/w50_roundselect_20261108'
PYTHON = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'


def main():
    source = (ROOT / 'host_scheduler.py').read_bytes()
    code = '''import os,sys,json,shutil,subprocess,datetime
from pathlib import Path
root=Path(REMOTE_ROOT)
assignment=json.loads((root/'assignment.json').read_text())
expected={f'roundselect_s{s}' for s in (42,43,44)}
assert {r['experiment'] for r in assignment['experiments']}==expected
assert (root/'host-complete.json').exists(), 'Original scheduler is not terminal'
for name in expected:
    run=root/'runs'/name
    assert (run/'train.exit').read_text().strip()=='0', 'Fit not successfully terminal'
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().decode().split(chr(0))
    except (OSError,UnicodeError):continue
    assert str(root/'experiment.py') not in args, 'Scientific fit remains live'
    assert str(root/'host_scheduler.py') not in args, 'Original scheduler remains live'
archive=root/'scheduler_initial_terminal'
archive.mkdir(exist_ok=True)
for name in ('host_scheduler.py','scheduler-state.json','host-complete.json'):
    target=archive/name
    if not target.exists():shutil.copy2(root/name,target)
source=sys.stdin.buffer.read()
assert b'for head, selected_count in' in source
(root/'host_scheduler.py').write_bytes(source)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=subprocess.run([REMOTE_PYTHON,'-u',str(root/'host_scheduler.py')],env=env,capture_output=True)
sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr)
assert p.returncode==0, 'Corrected receipt validation failed to run'
terminal=json.loads((root/'host-complete.json').read_text())
assert terminal['training_success_all'] is True, 'Artifact validation still fails'
record={'status':'reconciled','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'artifact validator only','original_receipts':'scheduler_initial_terminal',
        'fresh_fits_started':False,'forecasts_modified':False,'research_goal_complete':False}
(root/'scheduler-reconciliation.json').write_text(json.dumps(record,indent=2)+chr(10))
print(json.dumps(record))
'''.replace('REMOTE_ROOT', repr(REMOTE)).replace('REMOTE_PYTHON', repr(PYTHON))
    try:
        result = remote(code, input=source, timeout=120)
    except subprocess.CalledProcessError as error:
        sys.stdout.buffer.write(error.stdout)
        sys.stderr.buffer.write(error.stderr)
        raise
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)


if __name__ == '__main__':
    main()
