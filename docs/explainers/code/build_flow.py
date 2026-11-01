"""Inline data/flow.json into templates/flow.html -> docs/explainers/e2000fs_calib4_detpool_gpd.html.

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/build_flow.py
"""
from pathlib import Path

HERE = Path('/home/t-lab01/sunrun/docs/explainers')
tpl = (HERE / 'templates/flow.html').read_text()
data = (HERE / 'data/flow.json').read_text()
assert '</script' not in data
out = HERE / 'e2000fs_calib4_detpool_gpd.html'
out.write_text(tpl.replace('__DATA__', data))
print(out, out.stat().st_size // 1024, 'KB')
