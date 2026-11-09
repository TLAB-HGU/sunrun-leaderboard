"""Prepare an isolated two-host campaign, preserving validated historical runs."""
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT.parent / 'rerun_df_ffill_20261103'))
from golab2_transport import remote as legacy_remote
import base64

def remote(code, *, input=None, timeout=120):
    if input is None:
        return legacy_remote(code, timeout=timeout)
    envelope=json.dumps(dict(code=code, data=base64.b64encode(input).decode())).encode()
    loader="import sys,json,base64,io;p=json.load(sys.stdin);sys.stdin=io.TextIOWrapper(io.BytesIO(base64.b64decode(p['data'])));exec(p['code'])"
    return subprocess.run(["ssh","-o","BatchMode=yes","golab2", "wsl -d Ubuntu -- python3 -c " + '"' + loader.replace('"', '\"') + '"'],input=envelope,capture_output=True,timeout=timeout,check=True)
REMOTE = '/home/yehwan/sunrun-drop035-v650-confirm-20261108'
RPY = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'
