"""Audit existing selected SUVI bytes against public S3 HEAD metadata only.

S3 documents single-part plaintext/SSE-S3 ETags as object MD5 digests:
https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html
No observation bodies are fetched. Local SHA256 is recorded independently.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import pandas as pd

from . import prepare
from .inputs import atomic_json


def digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
    return {'local_sha256': sha.hexdigest(), 'local_md5': md5.hexdigest(),
            'local_size': Path(path).stat().st_size}


def validate(local, headers, earliest_origin, obs_end):
    headers = {k.lower(): v for k, v in headers.items()}
    etag = headers.get('etag', '').strip('"')
    errors = []
    if not re.fullmatch('[0-9a-fA-F]{32}', etag):
        errors.append('etag_not_single_part_md5')
    elif etag.lower() != local['local_md5']:
        errors.append('md5_mismatch')
    if (headers.get('x-amz-server-side-encryption') not in (None, 'AES256')
            or any(k.startswith('x-amz-server-side-encryption-customer-') for k in headers)):
        errors.append('unsupported_sse')
    if int(headers.get('content-length', -1)) != local['local_size']:
        errors.append('size_mismatch')
    modified = pd.Timestamp(parsedate_to_datetime(headers['last-modified']))
    origin = pd.Timestamp(earliest_origin)
    if modified > origin:
        errors.append('publication_after_origin')
    if pd.isna(pd.Timestamp(obs_end)) or pd.Timestamp(obs_end) > origin - pd.Timedelta(hours=1):
        errors.append('observation_after_origin_minus_1h')
    return {'passed': not errors, 'errors': errors, 'remote_etag': etag,
            'remote_size': int(headers.get('content-length', -1)),
            'remote_sse': headers.get('x-amz-server-side-encryption'),
            'remote_last_modified': modified.isoformat()}


def cache_matches(receipt, identity, local):
    return receipt.get('passed') is True and all(receipt.get(k) == v for k, v in {**identity, **local}.items())


def verify_object(obj, destination, opener=urlopen, cache_directory=None):
    identity = dict(obj)
    target = destination / (hashlib.sha256(obj['url'].encode()).hexdigest() + '.json')
    try:
        local = digests(obj['local_path'])
        cache_path = target
        if not cache_path.exists() and cache_directory is not None:
            cache_path = Path(cache_directory) / target.name
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cache_matches(cached, identity, local):
                if cache_path != target:
                    atomic_json(target, cached)
                return cached
            # Re-evaluate previously strict AES256-only failures from retained HEAD.
            # Identity and local bytes must still match; preserve original request time.
            if (cached.get('head_headers') and
                    all(cached.get(k) == v for k, v in {**local, **{k: identity[k] for k in ('url', 'remote_key', 'local_path')}}.items())):
                checked = validate(local, cached['head_headers'], obj['earliest_origin'], obj['obs_end'])
                cached.update(identity)
                cached.update(checked)
                cached['proof_method'] = 's3_single_part_plaintext_or_sse_s3_etag_md5'
                atomic_json(target, cached)
                return cached
        for attempt in range(3):
            try:
                with opener(Request(obj['url'], method='HEAD'), timeout=20) as response:
                    headers = dict(response.headers.items())
                result = {**identity, **local, **validate(local, headers, obj['earliest_origin'], obj['obs_end']),
                          'head_headers': headers, 'checked_at': datetime.now(timezone.utc).isoformat(),
                          'proof_method': 's3_single_part_plaintext_or_sse_s3_etag_md5'}
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    except Exception as exc:
        result = {**identity, 'passed': False, 'errors': [f'{type(exc).__name__}: {exc}']}
    atomic_json(target, result)
    return result


def selections(metadata=None):
    features = json.loads((prepare.V3 / 'features_v3.json').read_text())['features']
    frames = []
    for split, filename in [('train', 'train_v3.parquet'), ('official', 'official_v3.parquet')]:
        origins = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(prepare.V3 / filename, columns=['origin']).origin, utc=True).unique()))
        frames.append(pd.DataFrame({'origin': origins, 'split': split}))
    origin_frame = pd.concat(frames, ignore_index=True)
    origins = pd.DatetimeIndex(origin_frame.origin)
    if metadata is None:
        context = prepare.relocated_builder(prepare.source_root())
    else:
        from .availability_rebuild import causal_builder
        context = causal_builder(metadata)
    with context as p:
        meta, block = p.ST.load_inputs()
        strip = p.ST.select_slots(origins, meta, block)
        active = strip.loc[~strip.missing & strip.img_stale.eq(0), 'slot_idx'].astype(int)
        if any(block.index[k] != meta.iloc[k].slot for k in active.unique()):
            raise ValueError('strip cache positional source mismatch')
        ch, slots, _, _ = p.P2.filtered_table(origins, [c for c in features if c.startswith('ch_')])
        by_slot = meta[meta.status.eq('ok')].drop_duplicates('slot', keep='last').set_index('slot', drop=False)
        records = []
        for i, origin in enumerate(origins):
            for family in ['strip', 'ch']:
                if family == 'strip':
                    idx = int(strip.iloc[i].slot_idx)
                    row = meta.iloc[idx] if idx >= 0 else None
                    used = idx >= 0  # age/stale features also use the selected object
                    stale = bool(strip.iloc[i].img_stale)
                else:
                    slot = pd.Timestamp(slots[i], tz='UTC')
                    row = by_slot.loc[slot] if pd.notna(slot) and slot in by_slot.index else None
                    used = row is not None  # eligibility also determines all-NaN missingness
                    stale = False
                record = {'origin': origin.isoformat(), 'split': origin_frame.iloc[i].split,
                          'family': family, 'used': used, 'stale': stale, 'missing': row is None,
                          'finite_values_used': bool(ch.iloc[i].notna().any()) if family == 'ch' else bool(used and not stale)}
                if row is not None:
                    key = str(row.s3_key)
                    key = str(Path(key).with_suffix('.nc'))
                    local_path = p.ST.resolve_nc(str(row.nc_path))
                    satellite = re.search(r'_G(\d{2})_', Path(key).name)
                    if not satellite:
                        raise ValueError('satellite missing from key: ' + key)
                    record.update(slot=row.slot.isoformat(), obs_end=row.obs_end.isoformat(),
                                  stage_s3_modified=row.s3_modified.isoformat(), local_path=local_path,
                                  remote_key=key, url=f'https://noaa-goes{satellite.group(1)}.s3.amazonaws.com/{key}')
                records.append(record)
    mapping = pd.DataFrame(records)
    if (mapping.used & mapping.missing).any():
        raise ValueError('used feature has no selected source metadata')
    objects = []
    for url, group in mapping[mapping.used].groupby('url'):
        for name in ['local_path', 'remote_key', 'obs_end']:
            if group[name].nunique() != 1:
                raise ValueError(f'inconsistent {name}: {url}')
        first = group.iloc[0]
        objects.append({name: first[name] for name in ['local_path', 'remote_key', 'url', 'obs_end']} |
                       {'earliest_origin': group.origin.min(), 'origin_family_uses': len(group)})
    return mapping, objects



def close_selection(destination, receipt_cache, workers):
    """Verify the latest eligible choice over full metadata to a fixed point."""
    import shutil
    from .availability_rebuild import provisional_metadata
    destination.mkdir(parents=True, exist_ok=True)
    receipts = destination / 'objects'
    receipts.mkdir(exist_ok=True)
    for path in Path(receipt_cache).glob('*.json'):
        target = receipts / path.name
        if not target.exists():
            shutil.copy2(path, target)
    for iteration in range(1, 21):
        metadata = provisional_metadata(receipts)
        metadata_path = destination / f'closure_metadata_{iteration:02d}.parquet'
        metadata.to_parquet(metadata_path, index=False)
        mapping, objects = selections(metadata_path)
        mapping.to_parquet(destination / f'closure_mapping_{iteration:02d}.parquet', index=False)
        unknown = [obj for obj in objects if not (receipts / (hashlib.sha256(obj['url'].encode()).hexdigest() + '.json')).exists()]
        print(f'closure iteration {iteration}: selected={len(objects)} new_objects={len(unknown)}', flush=True)
        if unknown:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda obj: verify_object(obj, receipts), unknown))
            byte_errors = [r for r in results if set(r.get('errors', [])) - {'publication_after_origin', 'observation_after_origin_minus_1h'}]
            if byte_errors:
                atomic_json(destination / 'closure_byte_failures.json', byte_errors)
                raise ValueError('closure encountered unverified bytes')
            continue
        atomic_json(destination / 'closure.json', {'passed': True, 'iterations': iteration,
                    'candidate_scope': 'full original stage metadata; latest eligible selection fixed point',
                    'metadata_path': str(metadata_path.resolve()), 'metadata_sha256': digests(metadata_path)['local_sha256'],
                    'selected_objects': len(objects), 'new_objects': 0})
        return metadata_path
    raise ValueError('source selection closure exceeded 20 rounds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--metadata', '--corrected-metadata', type=Path)
    parser.add_argument('--receipt-cache', '--cache-directory', type=Path)
    parser.add_argument('--closure', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be 1..16')
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = args.output / 'objects'
    receipts.mkdir(exist_ok=True)
    if args.closure:
        if args.receipt_cache is None:
            parser.error('--closure requires --receipt-cache')
        args.metadata = close_selection(args.output, args.receipt_cache, args.workers)
    mapping, objects = selections(args.metadata)
    mapping.to_parquet(args.output / 'origin_mapping.parquet', index=False)
    atomic_json(args.output / 'selected_objects.json', objects)
    print(f'selected {len(objects)} objects; {len(mapping)} origin/family mappings', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify_object, obj, receipts, cache_directory=args.receipt_cache) for obj in objects]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f'HEAD {len(results)}/{len(objects)} passed={sum(r["passed"] for r in results)}', flush=True)
    failures = [r for r in results if not r['passed']]
    atomic_json(args.output / 'failures.json', failures)
    atomic_json(args.output / 'summary.json', {'passed': not failures, 'object_count': len(objects),
                'verified_count': len(results) - len(failures), 'failed_count': len(failures),
                'mapping_count': len(mapping), 'missing_mapping_count': int(mapping.missing.sum()),
                'unused_mapping_count': int((~mapping.used).sum()), 'method': 'HEAD_only_no_body_download',
                'origin_mapping_sha256': digests(args.output / 'origin_mapping.parquet')['local_sha256']})
    print(f'complete: {len(results)-len(failures)}/{len(objects)} verified', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
