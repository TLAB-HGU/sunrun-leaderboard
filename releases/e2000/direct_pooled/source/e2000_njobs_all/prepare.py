"""Prepare complete features from the existing owned source builder, never collect.

Audit can run against preserved artifacts even when the old raw paths are gone.
Preparation refuses to turn absent label rows into NaN feature rows.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import atomic_json, input_audit, reference, sha, widen

HERE = Path(__file__).resolve().parent
LB = HERE.parents[1]
ROOT = LB.parent
V3 = HERE.parent / 'cumulative_lane12_20261024/materialized_v3'
SHARED = ROOT / 'research/sun-gjc-restart-20261024T145738Z/shared/receipts'
ARCHIVE = ROOT / 'data-v2/raw-store/v2/solar-pivot-noanalog72-20261015/pinned-v1/repo/leaderboard/store'
RELATIVE_INPUTS = ('ch-v1/ch-hourly.parquet', 'suvi-fusion/suvi-hourly-v2.parquet',
                   'ch-breakthrough-v2/upload/raw-ace-snapshot')
EXPECTED = {'ch-v1/ch-hourly.parquet': '96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a',
            'suvi-fusion/suvi-hourly-v2.parquet': '33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746',
            'ch-v1/ace.parquet': '700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af'}
RAW_TREE = '695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736'


def source_root(path=None):
    candidates = [Path(path)] if path is not None else [LB / 'store', ARCHIVE]
    for root in candidates:
        if all((root / name).exists() for name in RELATIVE_INPUTS):
            return root.resolve()
    raise FileNotFoundError('no complete frozen source root: ' + ', '.join(map(str, candidates)))


def source_files(root):
    root = Path(root)
    files = [root / name for name in EXPECTED]
    files += sorted((root / RELATIVE_INPUTS[2]).glob('year=*/part.parquet'))
    files += [p for p in sorted((root / 'headroom-audit/donki').glob('WSAEnlilSimulations_*.json'))
              if p.name[-12:-5] < '2026-06']
    files += sorted((root / 'donki-live').glob('WSAEnlilSimulations_*.json'))
    files += sorted((ROOT / 'data-v2/stage/suvi_hourly').glob('year=*/*.parquet'))
    return files


def verify_sources(root):
    import hashlib
    root = Path(root)
    for name, expected in EXPECTED.items():
        if sha(root / name) != expected:
            raise ValueError('frozen source mismatch: ' + name)
    raw = sorted((root / RELATIVE_INPUTS[2]).glob('year=*/part.parquet'))
    tree = hashlib.sha256(''.join(sha(p) for p in raw).encode()).hexdigest()
    if len(raw) != 5 or tree != RAW_TREE:
        raise ValueError('frozen raw ACE partition identity mismatch')
    return {'source_root': str(root), 'raw_ace_tree_sha256': tree,
            'files': [reference(p) for p in source_files(root)]}


def relocated_nc(path):
    """Resolve both original absolute and relative image paths after relocation."""
    if not path:
        return ''
    path = Path(path)
    old_root = ROOT / 'sun-img'
    if path.is_absolute():
        if not path.is_relative_to(old_root):
            return str(path)
        path = path.relative_to(old_root)
    return str(ROOT / 'data-v2/suvi' / path)


@contextmanager
def relocated_builder(root):
    # Import original code once, then change only this process's source globals.
    # The old modules remain byte-for-byte unchanged on disk.
    root = Path(root)
    p = importlib.import_module('experiments.cumulative_lane12_20261024.prepare_base111_v3')
    updates = [(p.HR, 'LB', root.parent), (p.HR, 'OUT', root / 'headroom-audit'),
               (p.ia, 'LB', root.parent), (p.ia, 'OUT', root / 'headroom-audit'),
               (p.A, 'LB', root.parent), (p.ST, 'SUVI_CACHE', root / 'suvi-fusion/suvi-hourly-v2.parquet'),
               (p.ST, 'SUNIMG', ROOT / 'data-v2/suvi'),
               (p.ST, 'resolve_nc', relocated_nc),
               (p.ST, 'META_GLOB', str(ROOT / 'data-v2/stage/suvi_hourly/year=*/*.parquet')),
               (p.L9, 'META_GLOB', str(ROOT / 'data-v2/stage/suvi_hourly/year=*/*.parquet')),
               (p.P2, 'CH', root / 'ch-v1/ch-hourly.parquet'),
               (p.L9, 'CH', root / 'ch-v1/ch-hourly.parquet'),
               (p.LE, 'LIVE', root / 'donki-live')]
    saved = [(m, name, getattr(m, name)) for m, name, _ in updates]
    saved += [(p.LE, 'RUNS', p.LE.RUNS), (p.LE, '_TABLE', p.LE._TABLE)]
    try:
        for module, name, value in updates:
            setattr(module, name, value)
        p.LE.RUNS = p.LE.enlil_runs()
        p.LE._TABLE = {}
        yield p
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)



def preserved_spec(root=None):
    """Diagnostic envelope only: intentionally fails if features/evidence are incomplete."""
    root = source_root(root)
    sources = [reference(SHARED / f'{s}.json') for s in ('SW-003', 'SW-004', 'SW-005', 'SW-006', 'SW-007', 'SW-010')]
    return {
        'version': 1,
        'artifacts': {'labels': reference(V3 / 'train_v3.parquet'),
                      'train_features': reference(V3 / 'train_v3.parquet'),
                      'official_features': reference(V3 / 'official_v3.parquet'),
                      'feature_names': reference(V3 / 'features_v3.json')},
        'source_receipts': sources,
        'dependencies': [reference(HERE.parent / 'cumulative_lane12_20261024' / f)
                         for f in ('prepare_base111_v3.py', 'prepare_base111_v2.py')],
        'raw_rebuild_paths': [str(root / name) for name in RELATIVE_INPUTS],
        'relocated_source_root': str(root),
        'note': 'Preserved diagnostic envelope; full feature rebuild and bound source/poison receipts required before registration.',
    }


def compare_overlap(labels, full, features):
    compare = labels[['origin', 'h'] + [c for c in features if c != 'h']].merge(
        full, on=['origin', 'h'], suffixes=('_old', '_new'), validate='one_to_one')
    if len(compare) != len(labels):
        raise ValueError('reconstructed grid lost preserved rows')
    differences = {}
    for c in features:
        if c == 'h':
            continue
        a, b = compare[c + '_old'].to_numpy(), compare[c + '_new'].to_numpy()
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-5, equal_nan=True)
        if not ok.all():
            differences[c] = int((~ok).sum())
    if differences:
        raise ValueError('feature reconstruction differs: ' + json.dumps(differences))
    return {'passed': True, 'observed_rows': len(compare), 'feature_count': len(features),
            'mismatches': differences, 'rtol': 1e-6, 'atol': 1e-5}


def rebuild(destination, root=None):
    root = source_root(root)
    identity = verify_sources(root)
    destination = Path(destination)
    if destination.exists():
        raise ValueError('new feature output path required')
    features = json.loads((V3 / 'features_v3.json').read_text())['features']
    labels = pd.read_parquet(V3 / 'train_v3.parquet')
    origins = pd.DatetimeIndex(sorted(pd.to_datetime(labels.origin, utc=True).unique()))
    with relocated_builder(root) as p:
        ace = p.A.filled_hourly()
        meta, block = p.ST.load_inputs()
        ch_cols = [c for c in features if c.startswith('ch_')]
        ch, _, _, _ = p.P2.filtered_table(origins, ch_cols)
        frames = []
        for start in range(0, len(origins), 512):
            oo = origins[start:start + 512]
            ch_row = pd.DataFrame({c: np.repeat(ch.reindex(oo)[c].to_numpy(), 72) for c in ch_cols})
            g = p.v3_base_grid_for(oo, ace, p.HR, meta, block, features, ch_row, lag_hours=2.)
            for name in features:
                if name != 'h':
                    g[name] = g[name].astype('float32')
            frames.append(g)
            print(f'feature rebuild {min(start + 512, len(origins))}/{len(origins)} origins', flush=True)
        dependencies = {str(Path(m.__file__).resolve()): sha(m.__file__) for m in
                        (p, p.P2, p.L9, p.HR, p.ia, p.A, p.ST, p.LE, p.L, p.L.SP, p.L.RL)}
    full = pd.concat(frames, ignore_index=True)
    _, wide = widen(full, features, expected_origins=origins)
    parity = compare_overlap(labels, full, features)
    if verify_sources(root) != identity:
        raise ValueError('source files changed during feature reconstruction')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.parquet.tmp')
    full.to_parquet(temporary, index=False)
    temporary.replace(destination)
    identity.update(output=reference(destination), rows=len(full), origins=len(origins),
                    wide_features=wide.shape[1], parity=parity, code_hashes=dependencies,
                    preparation_code=reference(__file__), source_admission_certified=False)
    atomic_json(destination.parent / 'rebuild-receipt.json', identity)
    return reference(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    root = source_root(args.source_root)
    spec = preserved_spec(root)
    if args.rebuild:
        spec['artifacts']['train_features'] = rebuild(args.out / 'train_complete.parquet', root)
    atomic_json(args.out / 'inputs.diagnostic.json', spec)
    audit = input_audit(spec)
    audit['raw_rebuild_missing'] = [str(root / name) for name in RELATIVE_INPUTS if not (root / name).exists()]
    atomic_json(args.out / 'input-audit.json', audit)
    print(json.dumps(audit, indent=2))
    return 0 if audit['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
