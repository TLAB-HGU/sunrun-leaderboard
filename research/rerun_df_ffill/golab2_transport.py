"""Explicit Windows OpenSSH -> Ubuntu transport for the 23:00 reservation."""
import base64
import subprocess


def remote(code, *, input=None, timeout=120):
    if input is None:
        command = 'wsl -d Ubuntu -- python3 -'
        input = code.encode()
    else:
        encoded = base64.b64encode(code.encode()).decode()
        command = ('wsl -d Ubuntu -- python3 -c "import base64;exec(base64.b64decode(\''
                   + encoded + '\'))"')
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           'golab2', command], input=input, capture_output=True,
                          timeout=timeout, check=True)
