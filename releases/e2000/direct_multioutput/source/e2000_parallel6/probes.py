"""Recompute full production inputs under real temporal source perturbations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import atomic_json, digest, reference, sha
from .prepare import V3, relocated_builder, source_root


def delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError('probe shape/NaN mask changed')
    return float(np.max(np.abs(a - b)[np.isfinite(a)])) if np.isfinite(a).any() else 0.


def run(destination, corrected_metadata=None):
    features = json.loads((V3 / 'features_v3.json').read_text())['features']
    origin = pd.Timestamp('2025-06-02T00:00:00Z')
    oo = pd.DatetimeIndex([origin])
    cutoff = origin - pd.Timedelta('2h')
    cases = {}
    if corrected_metadata is not None:
        from .availability_rebuild import causal_builder
        context = causal_builder(corrected_metadata)
    else:
        context = relocated_builder(source_root())
    with context as p:
        ace = p.A.filled_hourly()
        meta, block = p.ST.load_inputs()
        ch_names = [c for c in features if c.startswith('ch_')]
        ch_raw = p.P2.load_ch30(ch_names)
        ch_meta = p.L9.load_meta()
        # Full raw ACE frame, through the actual production function.
        raw = pd.concat([pd.read_parquet(f) for f in sorted((source_root() / 'ch-breakthrough-v2/upload/raw-ace-snapshot').glob('year=*/part.parquet'))], ignore_index=True)
        raw['timestamp_utc'] = pd.to_datetime(raw.timestamp_utc, utc=True)
        raw = raw.drop_duplicates('timestamp_utc', keep='last').set_index('timestamp_utc').sort_index()
        original_ace_features = p.HR.ace_features

        def build(*, speed=ace, plasma=None, images=block, cm=ch_meta, cv=ch_raw, runs=None):
            vals, _ = p.P2.select_filtered(cv, cm, oo)
            rows = pd.DataFrame({c: np.repeat(vals[c].to_numpy(), 72) for c in ch_names})
            saved_runs, saved_table = p.LE.RUNS, p.LE._TABLE
            try:
                if plasma is not None:
                    # ace_via_production calls HR.ace_features, so preserve its original
                    # implementation while swapping only glob via a local temporary file.
                    import tempfile
                    with tempfile.TemporaryDirectory() as td:
                        file = Path(td) / 'raw.parquet'
                        plasma.reset_index().to_parquet(file, index=False)
                        old_glob = p.HR.glob.glob
                        def load(at):
                            p.HR.glob.glob = lambda pattern: [str(file)] if 'raw-ace-snapshot' in str(pattern) else old_glob(pattern)
                            try:
                                return original_ace_features(at)
                            finally:
                                p.HR.glob.glob = old_glob
                        p.HR.ace_features = load
                        if runs is not None:
                            p.LE.RUNS, p.LE._TABLE = runs, {}
                        return p.v3_base_grid_for(oo, speed, p.HR, meta, images, features, rows, lag_hours=2.)[features]
                if runs is not None:
                    p.LE.RUNS, p.LE._TABLE = runs, {}
                return p.v3_base_grid_for(oo, speed, p.HR, meta, images, features, rows, lag_hours=2.)[features]
            finally:
                p.HR.ace_features = original_ace_features
                p.LE.RUNS, p.LE._TABLE = saved_runs, saved_table

        base = build()

        def case(name, future, past, fc, pc, columns, identities):
            f, q = delta(base[columns], future[columns]), delta(base[columns], past[columns])
            # Future invariance covers the entire pipeline, not just that family.
            whole = delta(base, future)
            if f != 0 or whole != 0 or q <= 0 or fc <= 0 or pc <= 0:
                raise ValueError(f'{name}: invalid future/positive probe {f}/{whole}/{q}/{fc}/{pc}')
            cases[name] = {'origin': origin.isoformat(), 'future_delta': whole, 'past_delta': q,
                           'future_rows_changed': int(fc), 'past_rows_changed': int(pc),
                           'columns': columns, 'raw_sha256': identities}
            print(name, 'future', whole, 'past', q, flush=True)

        def identity(values):
            return __import__('hashlib').sha256(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes()).hexdigest()

        for name, cols in [('speed', [c for c in features if c.startswith('v_')]),
                           ('recurrence', [c for c in features if c.startswith('rec')])]:
            future, past = ace.copy(), ace.copy()
            fm, pm = ace.index > cutoff, ace.index <= cutoff
            future.loc[fm] += 100
            past.loc[pm] += 100
            case(name, build(speed=future), build(speed=past), fm.sum(), pm.sum(), cols,
                 {'original': identity(ace), 'future': identity(future), 'past': identity(past)})

        future, past = raw.copy(), raw.copy()
        fm, pm = raw.index > cutoff, raw.index <= cutoff
        numeric = [c for c in raw if c.startswith('ace_') and pd.api.types.is_numeric_dtype(raw[c])]
        future.loc[fm, numeric] += 100
        past.loc[pm, numeric] += 100
        case('plasma', build(plasma=future), build(plasma=past), fm.sum(), pm.sum(),
             [c for c in features if c.startswith('ace_') or c in ('speed_d24', 'epam_p1_d6')],
             {'original': identity(raw), 'future': identity(future), 'past': identity(past)})

        # CH cache cells inherit the selected image's availability clock.
        available = ch_meta.set_index('slot').available_at.reindex(ch_raw.index)
        fm, pm = available > origin, available <= origin
        future, past = ch_raw.copy(), ch_raw.copy()
        future.loc[fm] += .123
        past.loc[pm] += .123
        case('ch', build(cv=future), build(cv=past), fm.sum(), pm.sum(), ch_names,
             {'original': identity(ch_raw), 'future': identity(future), 'past': identity(past)})

        av = meta.set_index('slot')
        av = pd.concat([av.s3_modified, av.obs_end + pd.Timedelta('1h'), pd.Series(av.index, index=av.index)], axis=1).max(axis=1)
        avail = av.reindex(block.index)
        fm, pm = avail > origin, avail <= origin
        future, past = block.copy(), block.copy()
        future.loc[fm] += .123
        past.loc[pm] += .123
        case('strips', build(images=future), build(images=past), fm.sum(), pm.sum(),
             [c for c in features if c.startswith('strip_')],
             {'original': identity(block), 'future': identity(future), 'past': identity(past)})

        runs = p.LE.RUNS.copy()
        future, past = runs.copy(), runs.copy()
        fm, pm = runs.done > origin, runs.done <= origin
        future.loc[fm, 'arrival'] += pd.Timedelta('12h')
        past.loc[pm, 'arrival'] += pd.Timedelta('12h')
        future.loc[fm, 'kp'] += 1
        past.loc[pm, 'kp'] += 1
        case('enlil', build(runs=future), build(runs=past), fm.sum(), pm.sum(),
             [c for c in features if c.startswith('enlil_')],
             {'original': identity(runs), 'future': identity(future), 'past': identity(past)})
        codes = {str(Path(m.__file__).resolve()): sha(m.__file__) for m in
                 (p, p.P2, p.L9, p.HR, p.ia, p.A, p.ST, p.LE)}
    codes[str(Path(__file__).resolve())] = sha(__file__)
    from . import prepare
    codes[str(Path(prepare.__file__).resolve())] = sha(prepare.__file__)
    if corrected_metadata is not None:
        from . import availability_rebuild
        codes[str(Path(availability_rebuild.__file__).resolve())] = sha(availability_rebuild.__file__)
    receipt = {'groups': cases, 'builder_code_sha256': codes,
               'note': 'Actual full 72-row production builder under raw future/past source perturbations; input envelope binding added after source evidence completes.'}
    if corrected_metadata is not None:
        receipt['corrected_metadata'] = reference(corrected_metadata)
    atomic_json(destination, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corrected-metadata', type=Path)
    args = parser.parse_args()
    run(args.out, args.corrected_metadata)
