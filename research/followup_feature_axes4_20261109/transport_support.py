"""Use the existing authenticated transport with an isolated destination."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('axes4_existing_transport',
    ROOT.parent / 'lag24_48_72_confirm_20261109/transport_support.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
remote = _module.remote
REMOTE = '/home/yehwan/sunrun-followup-feature-axes4-20261109'
RPY = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'
