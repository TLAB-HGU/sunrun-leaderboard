"""Leaderboard records that get a dirmo_24-style visualization folder (user request 2026-10-27)."""
import json
from pathlib import Path

ANALYSIS = Path('/home/t-lab01/sunrun/analysis')
E2000_PUB = ANALYSIS / 'forecast_strategies_e2000/publication/submission.json'
KUMO_PUB = ANALYSIS / 'forecast_strategies_kumo/publication/submission.json'
DONE = {'e2000fs_dirmo_24'}


def records():
    out = []
    for source, path in (('e2000', E2000_PUB), ('kumo', KUMO_PUB)):
        for name, r in json.loads(path.read_text()).items():
            if r['experiment'] in DONE:
                continue
            out.append({'source': source, 'strategy': name, 'experiment': r['experiment'],
                        'member': r['team_member'], 'config_sha256': r['config_sha256'], 'mse': r['mse'],
                        'mse_f1': r['mse_f1'], 'mse_f2': r['mse_f2'],
                        'key': f"{r['team_member']}/{r['experiment']}/{r['config_sha256']}",
                        'renderer_id': r['experiment'].upper()})
    return out


def title(rec):
    if rec['source'] == 'kumo':
        return f"E2000FS Kumo {rec['strategy']} (seed 42)"
    if rec['experiment'].startswith('e2000_'):
        return f"E2000 {rec['strategy']} (seed 42)"
    return f"E2000FS {rec['strategy']} (seed 42)"
