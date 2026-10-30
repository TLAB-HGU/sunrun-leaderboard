"""Canonical v2 locations. Historical experiment inputs remain at their old paths.

Every database and model-input store lives under this directory (user instruction
2026-10-30): the raw RawDB in `raw-store/`, the model-input snapshots in
`model-inputs/leaderboard-store/`. The former `store/` symlinks inside the
workspace root and inside `leaderboard/` are gone; use `STORE_ROOT`/`STORE` and
`RAW_STORE` instead of pathing through them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LB = ROOT.parent / 'leaderboard'
# Model-input store: dataset snapshots written by builders/ and derived/.
STORE_ROOT = ROOT / 'model-inputs' / 'leaderboard-store'
STORE = STORE_ROOT / 'v2'
# Central RawDB (solar_raw.duckdb) and pinned historical snapshots.
RAW_STORE = ROOT / 'raw-store'
STAGE = Path.home() / '.local/state/sundb/stage'
# 2026-10-15 user directive: only user-requested sources stay in SolarWind DB; worker-collected
# datasets (CACTus, CCMC, SWPC events, GOES SEP/XRS, HMI, ENLIL ambient, NMDB, PUNCH, ACE K1) were removed.
# 2026-10-28 user: SWPC forecast discussion (SW-042) removed; RawDB records and historical snapshots are kept.
DATASETS = {
    'ace-snapshot': ['SW-003'], 'ch': ['SW-004', 'SW-005'],
    'suvi-fusion': ['SW-006'], 'suvi-meta': ['SW-007'],
    'donki': ['SW-009', 'SW-010', 'SW-025'], 'gong-mrzqs': ['SW-012'],
    'lasco-c2-512': ['SW-028'],
}

def dataset(name):
    if name not in DATASETS:
        raise KeyError(name)
    return STORE / name
