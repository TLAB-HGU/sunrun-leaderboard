"""Existing authenticated golab2 transport, isolated campaign destination."""
from pathlib import Path
import base64
import importlib.util
import json
import subprocess

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('feature_addition_legacy_transport',
         ROOT.parent / 'rerun_df_ffill_20261103/golab2_transport.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
legacy_remote = _legacy.remote


def remote(code, *, input=None, timeout=120):
    # Large frozen assignments travel on stdin, never in the SSH argument list.
    if input is None:
        return legacy_remote(code, timeout=timeout)
    envelope = json.dumps({'code': code, 'data': base64.b64encode(input).decode()}).encode()
    loader = "import sys,json,base64,io;p=json.load(sys.stdin);sys.stdin=io.TextIOWrapper(io.BytesIO(base64.b64decode(p['data'])));exec(p['code'])"
    command = 'wsl -d Ubuntu -- python3 -c "' + loader.replace('"', '\\"') + '"'
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'golab2', command],
                          input=envelope, capture_output=True, timeout=timeout, check=True)

REMOTE = '/home/yehwan/sunrun-lag24-48-72-confirm-20261109'
REMOTE_PARENT = '/home/yehwan/sunrun-drop035-v650-confirm-20261108'
RPY = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'
