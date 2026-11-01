"""Inline the shared CSS/JS and the extracted numbers into one self-contained HTML per model.

The templates hold prose and structure; every number comes from docs/explainers/data/*.json,
which code/extract_*.py regenerate from the archived artefacts. Nothing is fetched at runtime,
so each built page opens offline and can be attached to a report as a single file.

Usage:  /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/build.py [detpool|mixture|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/t-lab01/sunrun')
HERE = ROOT / 'docs/explainers'
CSS = (HERE / 'shared/ui.css').read_text()
JS = (HERE / 'shared/ui.js').read_text()
APP_CSS = (HERE / 'app/app.css').read_text()
APP_JS = (HERE / 'app/app.js').read_text()

WORK = ('worked_gate_rows', 'worked_gate_tail', 'worked_main', 'worked_counter')


def trim_detpool(d: dict) -> dict:
    """Keep the numbers the page actually binds; drop the 72-row vectors of unused cases."""
    for k in WORK:
        e = d.get(k)
        if not e:
            continue
        if k != 'worked_main':
            e.pop('per_horizon', None)
            e.pop('scan', None)
    return d


def trim_mixture(d: dict) -> dict:
    for k in ('gate', 'examples'):
        pass
    return d


SPECS = {
    'detpool': dict(tpl='app_detpool.html', data='detpool_gpd.json', kind='app',
                    out='e2000fs_calib4_detpool_gpd.3d.html', trim=trim_detpool),
    'mixture': dict(tpl='app_mixture.html', data='mixture2comp.json', kind='app',
                    out='e2000fs_calib2_mixture2comp.html', trim=trim_mixture),
    # article-style walkthroughs (kept for reference, not built by default)
    'detpool-doc': dict(tpl='detpool_gpd.html', data='detpool_gpd.json', kind='doc',
                        out='e2000fs_calib4_detpool_gpd.doc.html', trim=trim_detpool),
    'mixture-doc': dict(tpl='mixture2comp.html', data='mixture2comp.json', kind='doc',
                        out='e2000fs_calib2_mixture2comp.doc.html', trim=trim_mixture),
}


def build(key: str) -> Path:
    spec = SPECS[key]
    tpl = (HERE / 'templates' / spec['tpl']).read_text()
    payload = json.loads((HERE / 'data' / spec['data']).read_text())
    payload = spec['trim'](payload)
    # raw-data chain (sources → alignment → features → worked mapping) shared by both pages
    raw = json.loads((HERE / 'data' / 'raw_chain.json').read_text())
    payload['raw'] = raw
    blob = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    css, js = (APP_CSS, APP_JS) if spec.get('kind') == 'app' else (CSS, JS)
    html = (tpl.replace('/*__CSS__*/', css)
               .replace('/*__JS__*/', js)
               .replace('__DATA_JSON__', blob))
    out = HERE / spec['out']
    out.write_text(html)
    print(f'{key}: {len(blob) / 1024:.0f} KB data -> {out} ({len(html) / 1024:.0f} KB)')
    return out


APPS = ('detpool', 'mixture')  # detpool 3D walkthrough -> *.3d.html; the main detpool page is built by build_flow.py

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for k in (APPS if which == 'all' else [which]):
        build(k)
