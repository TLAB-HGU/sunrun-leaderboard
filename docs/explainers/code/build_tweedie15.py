"""Inline the shared CSS (from templates/flow.html) and data/tweedie15.json into the
self-contained page docs/explainers/e2000fs_calib_tweedie15.html.

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/build_tweedie15.py
"""
from pathlib import Path
import re

HERE = Path('/home/t-lab01/sunrun/docs/explainers')
tpl = (HERE / 'templates/tweedie15.html').read_text()
css = re.search(r'<style>(.*?)</style>', (HERE / 'templates/flow.html').read_text(), re.S).group(1)
data = (HERE / 'data/tweedie15.json').read_text()
assert '</script' not in data
html = tpl.replace('/*__CSS__*/', css).replace('__DATA_JSON__', data)
out = HERE / 'e2000fs_calib_tweedie15.html'
out.write_text(html)
print(out, out.stat().st_size // 1024, 'KB')
