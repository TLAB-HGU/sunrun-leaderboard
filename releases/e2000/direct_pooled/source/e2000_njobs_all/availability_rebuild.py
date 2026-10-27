"""Rebuild inputs using exact-byte S3 publication times, without score access."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import prepare
from .inputs import atomic_json, reference, sha, widen
from .source_proof import digests, validate


def corrected_metadata(meta, receipts):
    """Admit byte-matched objects; defer eligibility to their actual publication."""
    verified = {}
    for receipt in receipts:
        # Availability failures do not invalidate byte identity; the selector
        # must instead wait until the exact object was actually published.
        checked = validate(receipt, receipt['head_headers'], receipt['earliest_origin'], receipt['obs_end'])
        errors = set(checked['errors']) - {'publication_after_origin', 'observation_after_origin_minus_1h'}
        if errors:
            raise ValueError('unverified object bytes: ' + str(sorted(errors)))
        if digests(receipt['local_path']) != {k: receipt[k] for k in ('local_sha256', 'local_md5', 'local_size')}:
            raise ValueError('local object changed after HEAD: ' + receipt['local_path'])
        key = receipt['remote_key']
        if key in verified:
            raise ValueError('duplicate object receipt: ' + key)
        verified[key] = receipt | checked
    rows = []
    for original_idx, row in meta.iterrows():
        key = str(Path(str(row.s3_key)).with_suffix('.nc'))
        if key not in verified or row.status != 'ok':
            continue
        receipt = verified[key]
        if pd.Timestamp(row.obs_end) != pd.Timestamp(receipt['obs_end']):
            raise ValueError('observation identity mismatch: ' + key)
        if Path(prepare.relocated_nc(str(row.nc_path))) != Path(receipt['local_path']):
            raise ValueError('local path identity mismatch: ' + key)
        if pd.isna(row.obs_end) or pd.isna(row.s3_modified):
            raise ValueError('missing source observation/publication time')
        if pd.Timestamp(row.obs_end) > pd.Timestamp(row.slot):
            raise ValueError('observation ends after metadata slot: ' + key)
        record = row.to_dict()
        record.update(original_slot_idx=int(original_idx), stage_s3_modified=row.s3_modified,
                      nc_last_modified=pd.Timestamp(receipt['remote_last_modified']),
                      local_sha256=receipt['local_sha256'], remote_key=key)
        record['s3_modified'] = max(pd.Timestamp(row.s3_modified), pd.Timestamp(receipt['remote_last_modified']),
                                    pd.Timestamp(row.obs_end) + pd.Timedelta(hours=1))
        rows.append(record)
    if not rows:
        raise ValueError('no byte-verified metadata')
    result = pd.DataFrame(rows).sort_values('slot').reset_index(drop=True)
    if result.slot.duplicated().any():
        raise ValueError('ambiguous duplicate verified slots')
    return result


def provisional_metadata(receipt_directory):
    """Full candidate set for HEAD selection closure only, never training proof."""
    known = {}
    for path in Path(receipt_directory).glob('*.json'):
        receipt = json.loads(path.read_text())
        checked = validate(receipt, receipt['head_headers'], receipt['earliest_origin'], receipt['obs_end'])
        errors = set(checked['errors']) - {'publication_after_origin', 'observation_after_origin_minus_1h'}
        if errors:
            raise ValueError('unverified provisional object: ' + str(sorted(errors)))
        known[receipt['remote_key']] = receipt | checked
    with prepare.relocated_builder(prepare.source_root()) as p:
        meta, _ = p.ST.load_inputs()
    meta = meta.copy()
    meta['original_slot_idx'] = np.arange(len(meta))
    meta['stage_s3_modified'] = meta.s3_modified
    meta['head_verified'] = False
    for index, row in meta.iterrows():
        key = str(Path(str(row.s3_key)).with_suffix('.nc'))
        if pd.notna(row.obs_end):
            meta.at[index, 's3_modified'] = max(row.s3_modified, row.obs_end + pd.Timedelta(hours=1))
        if key in known:
            receipt = known[key]
            if pd.Timestamp(row.obs_end) != pd.Timestamp(receipt['obs_end']):
                raise ValueError('provisional observation identity mismatch')
            meta.at[index, 's3_modified'] = max(meta.at[index, 's3_modified'], pd.Timestamp(receipt['remote_last_modified']))
            meta.at[index, 'head_verified'] = True
    return meta


def ch_metadata(meta):
    out = meta[meta.status.eq('ok')].drop_duplicates('slot').sort_values('slot').copy()
    out = out[out.obs_end.notna() & out.s3_modified.notna()].copy()
    def local_present(value):
        if pd.isna(value) or not str(value):
            return False
        path = Path(prepare.relocated_nc(str(value)))
        return path.is_file() and path.stat().st_size > 0
    # An absent body cannot provide a byte-verified source or suppress a
    # valid earlier source through a cache-only NaN row.
    out = out[out.nc_path.map(local_present)].copy()
    out['available_at'] = pd.concat([out.slot, out.s3_modified,
                                     out.obs_end + pd.Timedelta(hours=1)], axis=1).max(axis=1)
    return out.reset_index(drop=True)


@contextmanager
def causal_builder(corrected_meta_path):
    """Use one immutable corrected metadata/cache mapping for all feature paths."""
    meta = pd.read_parquet(corrected_meta_path)
    with prepare.relocated_builder(prepare.source_root()) as p:
        original_load, original_ch_load = p.ST.load_inputs, p.L9.load_meta
        _, block = original_load()
        # strip_features indexes block positionally by metadata slot_idx.
        # Reindex both on identical slots after removing unverified candidates.
        block = block.reindex(pd.DatetimeIndex(meta.slot))
        p.ST.load_inputs = lambda: (meta.copy(), block.copy())
        p.L9.load_meta = lambda: ch_metadata(meta)
        try:
            yield p
        finally:
            p.ST.load_inputs, p.L9.load_meta = original_load, original_ch_load


def changed_features(before, after, features):
    left = before[['origin'] + features].sort_values(['origin', 'h']).reset_index(drop=True)
    right = after[['origin'] + features].sort_values(['origin', 'h']).reset_index(drop=True)
    if (len(left) != len(right) or not np.array_equal(left.origin.to_numpy(), right.origin.to_numpy())
            or not np.array_equal(left.h.to_numpy(), right.h.to_numpy())):
        raise ValueError('changed output grid')
    any_changed = np.zeros(len(left), dtype=bool)
    counts = {}
    for feature in features:
        changed = ~np.isclose(left[feature], right[feature], equal_nan=True, rtol=1e-6, atol=1e-5)
        if changed.any():
            counts[feature] = int(changed.sum())
            any_changed |= changed
    return {'rows': len(left), 'changed_rows': int(any_changed.sum()),
            'changed_origins': int(left.loc[any_changed, 'origin'].nunique()), 'by_feature': counts}


def rebuild(head_directory, out_dir):
    head_directory, out_dir = Path(head_directory), Path(out_dir)
    closure_path = head_directory / 'closure.json'
    closure = json.loads(closure_path.read_text())
    if closure.get('passed') is not True or closure.get('new_objects') != 0:
        raise ValueError('full metadata selection closure required')
    closed_path = Path(closure['metadata_path'])
    if sha(closed_path) != closure['metadata_sha256']:
        raise ValueError('closure metadata changed')
    expected = json.loads((head_directory / 'selected_objects.json').read_text())
    receipts = []
    for obj in expected:
        path = head_directory / 'objects' / (hashlib.sha256(obj['url'].encode()).hexdigest() + '.json')
        if not path.exists():
            raise ValueError('HEAD collection incomplete: ' + str(path))
        receipt = json.loads(path.read_text())
        if any(receipt.get(k) != v for k, v in obj.items()):
            raise ValueError('HEAD receipt identity mismatch')
        receipts.append(receipt)
    outputs = [out_dir / name for name in ('corrected_metadata.parquet', 'train_complete_causal.parquet', 'official_causal.parquet')]
    if any(p.exists() for p in outputs):
        raise ValueError('new causal output paths required')
    source_identity = prepare.verify_sources(prepare.source_root())
    with prepare.relocated_builder(prepare.source_root()) as p:
        meta, _ = p.ST.load_inputs()
    receipt_paths = sorted((head_directory / 'objects').glob('*.json'))
    all_receipts = [json.loads(path.read_text()) for path in receipt_paths]
    # Verify every known object used to establish full-candidate availability,
    # including objects no longer selected after the publication correction.
    corrected_metadata(meta, all_receipts)
    corrected = provisional_metadata(head_directory / 'objects')
    pd.testing.assert_frame_equal(corrected, pd.read_parquet(closed_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_parquet(outputs[0], index=False)
    features = json.loads((prepare.V3 / 'features_v3.json').read_text())['features']
    report = {'reason': 'Exact .nc S3 LastModified replaces sibling .fits publication assumptions; no score-based selection.',
              'candidate_scope': 'Full original metadata with exact-byte publication corrections; latest eligible selections verified to closure. Unused candidates retained to preserve missing-value selection semantics.',
              'selection_closure': reference(closure_path),
              'corrected_metadata': reference(outputs[0]), 'verified_objects': len(all_receipts),
              'metadata_rows': len(corrected), 'source_identity': source_identity,
              'head_manifest': reference(head_directory / 'selected_objects.json'),
              'head_receipts': [reference(path) for path in receipt_paths],
              'preparation_code': reference(__file__), 'outputs': {}}
    report['owned_dependencies'] = [reference(prepare.__file__), reference(Path(__file__).with_name('source_proof.py'))]
    with causal_builder(outputs[0]) as p:
        ace = p.A.filled_hourly()
        meta, block = p.ST.load_inputs()
        for split, before_path, output in [('train', out_dir / 'train_complete.parquet', outputs[1]),
                                          ('official', prepare.V3 / 'official_v3.parquet', outputs[2])]:
            before = pd.read_parquet(before_path, columns=['origin'] + features)
            origins = pd.DatetimeIndex(sorted(pd.to_datetime(before.origin, utc=True).unique()))
            ch_cols = [c for c in features if c.startswith('ch_')]
            ch, _, _, _ = p.P2.filtered_table(origins, ch_cols)
            frames = []
            for start in range(0, len(origins), 512):
                oo = origins[start:start + 512]
                ch_row = pd.DataFrame({c: np.repeat(ch.reindex(oo)[c].to_numpy(), 72) for c in ch_cols})
                grid = p.v3_base_grid_for(oo, ace, p.HR, meta, block, features, ch_row, lag_hours=2.)
                for name in features:
                    if name != 'h':
                        grid[name] = grid[name].astype('float32')
                frames.append(grid)
                print(f'causal {split} {min(start+512, len(origins))}/{len(origins)}', flush=True)
            full = pd.concat(frames, ignore_index=True)
            _, wide = widen(full, features, expected_origins=origins)
            changes = changed_features(before, full, features)
            if any(not (c.startswith('ch_') or c.startswith('strip_') or c in ('img_age_h', 'img_stale')) for c in changes['by_feature']):
                raise ValueError('unexpected change outside image availability features')
            temp = output.with_suffix('.parquet.tmp')
            full.to_parquet(temp, index=False)
            temp.replace(output)
            report['outputs'][split] = {'artifact': reference(output), 'wide_features': wide.shape[1],
                                         'baseline': reference(before_path), 'changes': changes}
        report['code_hashes'] = {str(Path(m.__file__).resolve()): sha(m.__file__) for m in
                                 (p, p.P2, p.L9, p.HR, p.ia, p.A, p.ST, p.LE, p.L, p.L.SP, p.L.RL)}
        for path in (Path(prepare.__file__), Path(__file__), Path(__file__).with_name('source_proof.py')):
            report['code_hashes'][str(path.resolve())] = sha(path)
    if prepare.verify_sources(prepare.source_root()) != source_identity:
        raise ValueError('sources changed during rebuild')
    atomic_json(out_dir / 'availability-rebuild-receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-directory', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.head_directory, args.out_dir)


if __name__ == '__main__':
    main()
