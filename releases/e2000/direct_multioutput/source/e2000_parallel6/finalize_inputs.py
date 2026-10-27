"""Bind rebuilt inputs to actual raw probes and complete HEAD checksum receipts."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .inputs import atomic_json, check_ref, digest, input_audit, reference
from .prepare import SHARED, V3, source_root
from .source_proof import digests, validate


def finalize(directory):
    directory = Path(directory).resolve()
    root = source_root()
    rebuild = json.loads((directory / 'rebuild-receipt.json').read_text())
    if not rebuild['parity']['passed'] or rebuild['parity']['mismatches']:
        raise ValueError('observed-row feature parity required')
    check_ref(rebuild['output'])
    # Original parity receipt is historical evidence after the user-requested
    # package rename. Fresh causal reconstruction below binds current code.
    # Do not relabel its historical code hashes as the new implementation.
    causal = json.loads((directory / 'availability-rebuild-receipt.json').read_text())
    check_ref(causal['preparation_code'])
    for path, value in causal['code_hashes'].items():
        check_ref({'path': path, 'sha256': value})
    check_ref(causal['corrected_metadata'])
    train = check_ref(causal['outputs']['train']['artifact'])
    official = check_ref(causal['outputs']['official']['artifact'])
    source_refs = causal['source_identity']['files']
    for ref in source_refs:
        check_ref(ref)
    proof_dir = directory / 'suvi-head-causal'
    selected = json.loads((proof_dir / 'selected_objects.json').read_text())
    objects = []
    for obj in selected:
        name = __import__('hashlib').sha256(obj['url'].encode()).hexdigest() + '.json'
        r = json.loads((proof_dir / 'objects' / name).read_text())
        actual = digests(obj['local_path'])
        if any(r.get(k) != v for k, v in {**obj, **actual}.items()):
            raise ValueError('HEAD receipt identity differs: ' + obj['url'])
        checked = validate(actual, r['head_headers'], obj['earliest_origin'], obj['obs_end'])
        if not checked['passed']:
            raise ValueError('HEAD receipt failed: ' + obj['url'] + ': ' + str(checked['errors']))
        objects.append({**r, **checked})
    # Never construct a successful proof from a partial list.
    mapping_path = proof_dir / 'origin_mapping.parquet'
    mapping = pd.read_parquet(mapping_path)
    if len(mapping) != 2 * (10688 + 2833):
        raise ValueError('incomplete origin selection coverage')
    if set(mapping[mapping.used].url) != {obj['url'] for obj in objects}:
        raise ValueError('selected object list does not cover actual origin mapping')
    exact_path = directory / 'suvi-exact-byte-proof.json'
    atomic_json(exact_path, {
        'coverage': 'all_selected_origins', 'proof_method': 's3_single_part_sse_s3_etag_md5',
        'input_grid_sha256': reference(train)['sha256'],
        'official_grid_sha256': reference(official)['sha256'],
        'origin_mapping': reference(mapping_path), 'selected_objects': objects,
        'aws_etag_rule': 'https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html',
        'note': 'HEAD-only checksum and exact .nc object publication checks; no new observation bodies collected.',
    })
    receipts = []
    for sid in ('SW-003', 'SW-004', 'SW-005', 'SW-006', 'SW-007', 'SW-010'):
        original = SHARED / f'{sid}.json'
        r = json.loads(original.read_text())
        r['original_review'] = reference(original)
        r['reviewer'] = 'codex-multioutput-s42 relocation and checksum verification'
        r['evidence'] = {'rebuild_parity': reference(directory / 'rebuild-receipt.json'),
                         'availability_correction': reference(directory / 'availability-rebuild-receipt.json'),
                         'raw_temporal_probes': reference(directory / 'raw-probes-causal.json')}
        if sid in ('SW-003', 'SW-007', 'SW-010'):
            if sid == 'SW-003':
                files = [ref for ref in source_refs if 'raw-ace-snapshot/' in ref['path']]
            elif sid == 'SW-007':
                files = [ref for ref in source_refs if '/stage/suvi_hourly/' in ref['path']]
            else:
                files = [ref for ref in source_refs if 'WSAEnlilSimulations_' in ref['path']]
            if not files:
                raise ValueError('empty source manifest: ' + sid)
            manifest = directory / 'sources' / f'{sid}.files.json'
            atomic_json(manifest, {'files': files, 'relocated_immutable_source_root': str(root)})
            r.update(reference(manifest))
        elif sid == 'SW-005':
            r.update(reference(root / 'ch-v1/ch-hourly.parquet'))
        elif sid == 'SW-006':
            r.update(reference(root / 'suvi-fusion/suvi-hourly-v2.parquet'))
        if sid in ('SW-005', 'SW-006', 'SW-007'):
            r['exact_byte_identity'] = reference(exact_path)
            r['limitations'] = ['Current exact .nc object ETag/MD5 and publication verified for all selected origins; historically exposed evaluation period remains diagnostic.']
        r.update(passed=True, causal=True, byte_bound=True)
        target = directory / 'sources' / f'{sid}.json'
        atomic_json(target, r)
        receipts.append(reference(target))
    from .runner import code_hashes
    dependencies = {ref['path']: ref for ref in source_refs}
    dependencies[causal['corrected_metadata']['path']] = causal['corrected_metadata']
    for path in causal['code_hashes']:
        dependencies[path] = reference(path)
    for path in (Path(__file__), Path(__file__).with_name('prepare.py'), Path(__file__).with_name('probes.py'),
                 Path(__file__).with_name('source_proof.py'), Path(__file__).with_name('availability_rebuild.py')):
        dependencies[str(path.resolve())] = reference(path)
    spec = {'version': 1, 'artifacts': {
        'train_features': reference(train), 'labels': reference(V3 / 'train_v3.parquet'),
        'official_features': reference(official),
        'feature_names': reference(V3 / 'features_v3.json')},
        'source_receipts': receipts, 'dependencies': list(dependencies.values()),
        'recipe': {'seed': 42, 'architecture': '72-independent-output-xgboost', 'target': 'absolute',
                   'E0003_D33_D45_mixture': False, 'output_corrections': False},
        'code_hashes': code_hashes(),
        'limitations': ['historical official interval previously exposed', 'single seed42 by explicit user request'],
    }
    probes = json.loads((directory / 'raw-probes-causal.json').read_text())
    probe_metadata = probes.get('corrected_metadata')
    if not probe_metadata or probe_metadata['sha256'] != causal['corrected_metadata']['sha256']:
        raise ValueError('raw probes did not use the corrected production metadata')
    check_ref(probe_metadata)
    probes['input_sha256'] = digest(spec)
    bound = directory / 'poison-bound.json'
    atomic_json(bound, probes)
    spec['poison'] = reference(bound)
    atomic_json(directory / 'inputs.json', spec)
    audit = input_audit(spec)
    atomic_json(directory / 'final-audit.json', audit)
    if not audit['passed']:
        raise ValueError(str(audit['errors']))
    return {'inputs': str(directory / 'inputs.json'), 'input_sha256': digest(spec),
            'verified_objects': len(objects), 'audit': audit}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.directory), indent=2))
