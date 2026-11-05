"""Inline the shared CSS and data/calib5.json -> docs/explainers/e2000fs_calib5_detpool_all.html

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/build_calib5.py
"""
from pathlib import Path
import re

HERE = Path('/home/t-lab01/sunrun/docs/explainers')
tpl = (HERE / 'templates/calib5.html').read_text()
css = re.search(r'<style>(.*?)</style>', (HERE / 'templates/flow.html').read_text(), re.S).group(1)
data = (HERE / 'data/calib5.json').read_text()
assert '</script' not in data
out = HERE / 'e2000fs_calib5_detpool_all.html'
out.write_text(tpl.replace('/*__CSS__*/', css).replace('__DATA_JSON__', data))
print(out, out.stat().st_size // 1024, 'KB')
