from pathlib import Path
import os, signal, subprocess, time, json
from datetime import datetime, UTC

root = Path('/home/t-lab01/sunrun/research/rerun_df_ffill')
pid = 465256
proc = Path('/proc') / str(pid)
args = (proc / 'cmdline').read_bytes().decode().split('\0')
assert str(root / 'collect_remote17.py') in args
os.kill(pid, signal.SIGSTOP)
print('Old collector paused; its current publication continues to completion', flush=True)
while True:
    children = []
    for child in Path('/proc').iterdir():
        if not child.name.isdigit():
            continue
        try:
            fields = (child / 'stat').read_text().split(') ', 1)[1].split()
            if int(fields[1]) == pid and fields[0] != 'Z':
                children.append(int(child.name))
        except (OSError, ValueError, IndexError):
            pass
    if not children:
        break
    time.sleep(2)
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
for _ in range(100):
    if subprocess.run(['tmux', 'has-session', '-t', 'sunrun-remote17-collector-20261104'], capture_output=True).returncode:
        break
    time.sleep(.1)
assert subprocess.run(['tmux', 'has-session', '-t', 'sunrun-remote17-collector-20261104'], capture_output=True).returncode
exitfile = root / 'logs/remote17-collector.exit'
if exitfile.exists():
    exitfile.rename(root / 'logs/remote17-collector.before-remove-heavy.exit')
subprocess.run(['tmux', 'new-session', '-d', '-s', 'sunrun-remote17-collector-20261104', 'bash', str(root / 'remote17-collector-run.sh')], check=True)
receipt = {'utc': datetime.now(UTC).isoformat(), 'old_collector_pid': pid,
           'publication_allowed_to_finish': True, 'restarted': True,
           'excluded': ['e2000fs_calib5_detpool_all', 'e1604_hsspeak25']}
(root / 'collector-reload-heavy-20261104.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
