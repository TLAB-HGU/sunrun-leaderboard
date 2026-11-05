"""Inline the shared CSS and data/point4_144h.json -> docs/explainers/e2000fs_calib_point4_144h.html

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/build_point4_144h.py
"""
from pathlib import Path
import re

HERE = Path('/home/t-lab01/sunrun/docs/explainers')
tpl = (HERE / 'templates/point4_144h.html').read_text()
css = re.search(r'<style>(.*?)</style>', (HERE / 'templates/flow.html').read_text(), re.S).group(1)
data = (HERE / 'data/point4_144h.json').read_text()
assert '</script' not in data
out = HERE / 'e2000fs_calib_point4_144h.html'
out.write_text(tpl.replace('/*__CSS__*/', css).replace('__DATA_JSON__', data))
print(out, out.stat().st_size // 1024, 'KB')
