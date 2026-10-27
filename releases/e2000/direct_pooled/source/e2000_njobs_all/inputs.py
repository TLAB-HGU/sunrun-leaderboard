"""Origin-wide features, independent target masks, and immutable input checks."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 73))
CUTOFF = pd.Timestamp('2026-05-29T00:00:00Z')
TRAIN_END = pd.Timestamp('2026-03-01T00:00:00Z')
VALID_START = pd.Timestamp('2026-03-04T00:00:00Z')
VALID_END = pd.Timestamp('2026-05-25T23:00:00Z')
VARYING = ('rec26', 'rec27', 'rec28', 'rec_mean') + tuple(
    f'strip_v{v}_{s}' for v in (350, 450, 550, 650, 750)
    for s in ('logmed', 'dark', 'cover')) + ('enlil_dt_target',)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check_ref(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('artifact hash changed: ' + ref['path'])
    return Path(ref['path'])


def keyed(frame):
    frame = frame.copy()
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    if frame[['origin', 'h']].duplicated().any():
        raise ValueError('duplicate origin/horizon keys')
    if not frame.h.isin(HORIZONS).all():
        raise ValueError('horizon outside 1..72')
    return frame.sort_values(['origin', 'h']).reset_index(drop=True)


def widen(frame, features, *, expected_origins=None):
    """Never pivot label-filtered rows: require every feature row independently."""
    if not features or len(features) != len(set(features)):
        raise ValueError('features must be nonempty and unique')
    frame = keyed(frame)
    if {'y', 'target_kms', 'pred_kms'} & set(features):
        raise ValueError('target/prediction cannot be an input feature')
    origins = pd.DatetimeIndex(frame.origin.unique())
    if expected_origins is not None and not origins.equals(pd.DatetimeIndex(expected_origins)):
        raise ValueError('origin grid mismatch')
    expected = pd.MultiIndex.from_product([origins, HORIZONS], names=['origin', 'h'])
    indexed = frame.set_index(['origin', 'h'])
    if not indexed.index.equals(expected):
        raise ValueError('complete feature grid required; label-filtered rows cannot be widened')
    columns = {}
    for name in features:
        if name == 'h':
            continue  # output position defines horizon, pooled control adds it explicitly
        a = indexed[name].to_numpy(dtype=np.float32).reshape(len(origins), 72)
        if np.isinf(a).any():
            raise ValueError('infinite feature: ' + name)
        if name in VARYING:
            for h in HORIZONS:
                columns[f'{name}_h{h:02}'] = a[:, h - 1]
        else:
            if not np.all((a == a[:, :1]) | (np.isnan(a) & np.isnan(a[:, :1]))):
                raise ValueError('declared origin-invariant feature varies: ' + name)
            columns[name] = a[:, 0]
    return origins, pd.DataFrame(columns, index=origins)


def target_matrix(labels, origins):
    """Only preofficial, observed labels; absent labels never erase feature rows."""
    labels = keyed(labels)
    actual = labels.origin + pd.to_timedelta(labels.h, unit='h')
    if 'target_t' not in labels or not actual.equals(pd.to_datetime(labels.target_t, utc=True)):
        raise ValueError('target clock mismatch')
    if (actual >= CUTOFF).any():
        raise ValueError('official target in training labels')
    if not labels.origin.isin(origins).all():
        raise ValueError('label origin missing from complete feature grid')
    if not np.isfinite(labels.y).all():
        raise ValueError('label rows must contain observed finite targets')
    if 'was_missing' in labels and labels.was_missing.astype(bool).any():
        raise ValueError('unobserved targets must be absent from label rows')
    idx = pd.MultiIndex.from_product([origins, HORIZONS], names=['origin', 'h'])
    y = labels.set_index(['origin', 'h']).y.reindex(idx).to_numpy(float).reshape(-1, 72)
    return y, np.isfinite(y)


def split_masks(origins, mask):
    targets = (origins.tz_convert('UTC').tz_localize(None).to_numpy(dtype='datetime64[ns]')[:, None]
               + np.array(HORIZONS, dtype='timedelta64[h]')[None, :])
    training = mask & (targets < TRAIN_END.to_datetime64())
    validation_origins = (origins >= VALID_START) & (origins <= VALID_END)
    validation = mask & validation_origins[:, None]
    if not training.any(axis=0).all() or not validation.any(axis=0).all():
        raise ValueError('missing internal training/validation observations for a horizon')
    if (targets[validation] >= CUTOFF.to_datetime64()).any():
        raise ValueError('validation targets must precede final cutoff')
    return training, validation


def prediction_frame(origins, prediction):
    prediction = np.asarray(prediction)
    if prediction.shape != (len(origins), 72) or not np.isfinite(prediction).all():
        raise ValueError('invalid 72-horizon predictions')
    return pd.DataFrame({
        'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
        'horizon_hours': np.tile(HORIZONS, len(origins)),
        'pred_kms': prediction.ravel(),
    })


def input_audit(spec):
    """Check evidence, not just passed booleans; collect all recoverable gaps."""
    errors = []
    for ref in spec.get('artifacts', {}).values():
        try:
            check_ref(ref)
        except (OSError, ValueError) as e:
            errors.append(str(e))
    for ref in spec.get('dependencies', []):
        try:
            check_ref(ref)
        except (OSError, ValueError) as e:
            errors.append(str(e))
    if not spec.get('source_receipts'):
        errors.append('source admission and availability receipts are required')
    coverage = []
    checked_image_proofs = set()
    for ref in spec.get('source_receipts', []):
        try:
            r = json.loads(check_ref(ref).read_text())
            if not all(r.get(k) is True for k in ('passed', 'causal', 'byte_bound')):
                errors.append('source review incomplete: ' + ref['path'])
            # Size equality/local SHA are not remote object identity proof.
            covered = r.get('features_covered', [])
            coverage.extend(covered)
            proof_ref = r.get('exact_byte_identity')
            needs_image_proof = any(c.startswith(('strip_', 'ch_')) for c in covered)
            if needs_image_proof and proof_ref and proof_ref['sha256'] in checked_image_proofs:
                needs_image_proof = False  # Same bound proof covers both feature families.
            if needs_image_proof:
                if not proof_ref:
                    errors.append('SUVI exact local/remote content identity missing: ' + ref['path'])
                else:
                    checked_image_proofs.add(proof_ref['sha256'])
                    proof = json.loads(check_ref(proof_ref).read_text())
                    selected = proof.get('selected_objects', [])
                    if (not selected or proof.get('coverage') != 'all_selected_origins'
                            or proof.get('input_grid_sha256') != spec['artifacts']['train_features']['sha256']):
                        errors.append('SUVI identity proof does not cover the training grid')
                    if proof.get('official_grid_sha256') != spec['artifacts']['official_features']['sha256']:
                        errors.append('SUVI identity proof does not cover the official grid')
                    for obj in selected:
                        if obj.get('proof_method') in ('s3_single_part_sse_s3_etag_md5',
                                                       's3_single_part_plaintext_or_sse_s3_etag_md5'):
                            # AWS documents MD5 ETags for single-part SSE-S3 objects.
                            # Never infer this from multipart/KMS/size-only metadata.
                            from .source_proof import digests, validate
                            from urllib.parse import urlsplit
                            url = urlsplit(obj['url'])
                            if (url.scheme != 'https' or url.netloc not in
                                    ('noaa-goes16.s3.amazonaws.com', 'noaa-goes17.s3.amazonaws.com',
                                     'noaa-goes18.s3.amazonaws.com', 'noaa-goes19.s3.amazonaws.com')
                                    or url.path.lstrip('/') != obj['remote_key'] or not url.path.endswith('.nc')):
                                errors.append('untrusted SUVI object URL/key')
                            actual = digests(obj['local_path'])
                            if any(actual[k] != obj[k] for k in actual):
                                errors.append('SUVI local content changed: ' + obj['local_path'])
                            checked = validate(actual, obj['head_headers'], obj['earliest_origin'], obj['obs_end'])
                            if not checked['passed']:
                                errors.append('SUVI HEAD checksum/publication mismatch: ' + obj['local_path'])
                        elif (not obj.get('remote_sha256') or obj['remote_sha256'] != obj['local_sha256']
                                or sha(obj['path']) != obj['local_sha256']):
                            errors.append('SUVI content mismatch: ' + obj.get('path', 'unknown'))
                    if proof.get('proof_method') in ('s3_single_part_sse_s3_etag_md5',
                                                     's3_single_part_plaintext_or_sse_s3_etag_md5'):
                        mapping = pd.read_parquet(check_ref(proof['origin_mapping']))
                        obj_by_url = {obj['url']: obj for obj in selected}
                        for split, artifact in [('train', 'train_features'), ('official', 'official_features')]:
                            grid = pd.read_parquet(check_ref(spec['artifacts'][artifact]), columns=['origin'])
                            origins = pd.DatetimeIndex(sorted(pd.to_datetime(grid.origin, utc=True).unique()))
                            expected = pd.MultiIndex.from_product([origins, ['strip', 'ch']])
                            part = mapping[mapping.split == split].copy()
                            part['origin'] = pd.to_datetime(part.origin, utc=True)
                            got = pd.MultiIndex.from_frame(part[['origin', 'family']])
                            if got.has_duplicates or len(got) != len(expected) or len(expected.difference(got)):
                                errors.append('SUVI origin mapping incomplete: ' + split)
                            for row in part[part.used].itertuples():
                                obj = obj_by_url.get(row.url)
                                if (obj is None or obj['local_path'] != row.local_path
                                        or pd.Timestamp(obj['earliest_origin']) > row.origin
                                        or pd.Timestamp(obj['remote_last_modified']) > row.origin
                                        or pd.Timestamp(row.obs_end) > row.origin - pd.Timedelta('1h')):
                                    errors.append('SUVI selected-object coverage/publication mismatch')
            check_ref({'path': r['path'], 'sha256': r['sha256']})
            for evidence in r.get('evidence', {}).values():
                check_ref(evidence)
        except (OSError, ValueError, KeyError) as e:
            errors.append(f"source evidence {ref['path']}: {e}")
    try:
        features = json.loads(check_ref(spec['artifacts']['feature_names']).read_text())['features']
        missing = [c for c in features if c != 'h' and not any(fnmatch.fnmatchcase(c, p) for p in coverage)]
        if missing:
            errors.append('source receipts do not cover input features: ' + ', '.join(missing))
        train = pd.read_parquet(check_ref(spec['artifacts']['train_features']))
        origins, wide = widen(train, features)
        labels = pd.read_parquet(check_ref(spec['artifacts']['labels']))
        y, mask = target_matrix(labels, origins)
        split_masks(origins, mask)
        official = pd.read_parquet(check_ref(spec['artifacts']['official_features']))
        expected = pd.date_range('2026-05-31T23:00Z', '2026-09-26T23:00Z', freq='h')
        if {'y', 'target_kms', 'was_missing', 'pred_kms'} & set(official):
            raise ValueError('official feature grid must be targetless')
        _, ow = widen(official, features, expected_origins=expected)
        if list(wide) != list(ow):
            raise ValueError('training/official feature order mismatch')
    except (OSError, ValueError, KeyError) as e:
        errors.append('feature/target grid: ' + str(e))
    poison = spec.get('poison')
    if not poison:
        errors.append('bound raw future/past poison evidence required for full feature grid')
    else:
        try:
            receipt = json.loads(check_ref(poison).read_text())
            bound = {k: v for k, v in spec.items() if k != 'poison'}
            if receipt.get('input_sha256') != digest(bound):
                errors.append('poison evidence not bound to this full input envelope')
            groups = receipt.get('groups', {})
            if set(groups) != {'speed', 'plasma', 'recurrence', 'ch', 'strips', 'enlil'}:
                errors.append('poison evidence must cover all six feature families')
            for name, case in groups.items():
                if (case.get('future_delta') != 0 or case.get('past_delta', 0) <= 0
                        or case.get('future_rows_changed', 0) <= 0 or case.get('past_rows_changed', 0) <= 0):
                    errors.append('invalid raw poison/positive control: ' + name)
            if not receipt.get('builder_code_sha256'):
                errors.append('poison builder code hashes missing')
            for path, value in receipt.get('builder_code_sha256', {}).items():
                check_ref({'path': path, 'sha256': value})
        except (OSError, ValueError, KeyError) as e:
            errors.append('poison evidence: ' + str(e))
    return {'passed': not errors, 'errors': errors, 'input_sha256': digest(spec)}
