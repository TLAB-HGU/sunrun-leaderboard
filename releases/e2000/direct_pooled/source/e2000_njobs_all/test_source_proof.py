import json
from pathlib import Path
import tempfile
import unittest

from .source_proof import cache_matches, close_selection, digests, validate, verify_object


class SourceProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'image.nc'
        self.path.write_bytes(b'existing local observation')
        self.local = digests(self.path)
        self.headers = {'ETag': '"' + self.local['local_md5'] + '"',
                        'Content-Length': str(self.local['local_size']),
                        'Last-Modified': 'Sun, 05 Jan 2025 04:00:05 GMT',
                        'x-amz-server-side-encryption': 'AES256'}
        self.obj = {'url': 'https://example.com/image.nc', 'local_path': str(self.path),
                    'remote_key': 'image.nc', 'earliest_origin': '2025-01-05T06:00:00+00:00',
                    'obs_end': '2025-01-05T04:00:00+00:00', 'origin_family_uses': 1}

    def test_valid(self):
        self.assertTrue(validate(self.local, self.headers, self.obj['earliest_origin'], self.obj['obs_end'])['passed'])

    def test_plaintext_and_customer_encryption(self):
        headers = {k: v for k, v in self.headers.items() if k != 'x-amz-server-side-encryption'}
        self.assertTrue(validate(self.local, headers, self.obj['earliest_origin'], self.obj['obs_end'])['passed'])
        headers['x-amz-server-side-encryption-customer-algorithm'] = 'AES256'
        self.assertIn('unsupported_sse', validate(self.local, headers, self.obj['earliest_origin'], self.obj['obs_end'])['errors'])

    def test_reject_bad_proofs(self):
        for key, value, error in [('ETag', '0' * 32, 'md5_mismatch'),
                                  ('ETag', '0' * 32 + '-2', 'etag_not_single_part_md5'),
                                  ('x-amz-server-side-encryption', 'aws:kms', 'unsupported_sse'),
                                  ('Content-Length', '1', 'size_mismatch'),
                                  ('Last-Modified', 'Mon, 06 Jan 2025 04:00:05 GMT', 'publication_after_origin')]:
            with self.subTest(error=error):
                result = validate(self.local, self.headers | {key: value}, self.obj['earliest_origin'], self.obj['obs_end'])
                self.assertIn(error, result['errors'])
        result = validate(self.local, self.headers, self.obj['earliest_origin'], '2025-01-05T05:30:00+00:00')
        self.assertIn('observation_after_origin_minus_1h', result['errors'])

    def test_cache_bound_to_all_identity_and_bytes(self):
        cached = self.obj | self.local | {'passed': True}
        self.assertTrue(cache_matches(cached, self.obj, self.local))
        self.assertFalse(cache_matches(cached, self.obj | {'remote_key': 'other'}, self.local))
        self.assertFalse(cache_matches(cached, self.obj, self.local | {'local_sha256': 'other'}))
        self.assertFalse(cache_matches(cached | {'passed': False}, self.obj, self.local))

    def test_prior_plaintext_receipt_revalidated_without_head(self):
        import hashlib
        headers = {k: v for k, v in self.headers.items() if k != 'x-amz-server-side-encryption'}
        target = self.root / (hashlib.sha256(self.obj['url'].encode()).hexdigest() + '.json')
        target.write_text(json.dumps(self.obj | self.local | {'passed': False, 'errors': ['unsupported_sse'], 'head_headers': headers, 'checked_at': 'original time'}))
        def forbidden(*args, **kwargs):
            raise AssertionError('unexpected HTTP request')
        result = verify_object(self.obj, self.root, forbidden)
        self.assertTrue(result['passed'])
        self.assertEqual(result['checked_at'], 'original time')

    def test_closure_reselects_after_verifying_new_candidate(self):
        import hashlib
        import pandas as pd
        from unittest.mock import patch
        from . import source_proof, availability_rebuild
        source = self.root / 'cache'
        source.mkdir()
        output = self.root / 'proof'
        def verify(obj, destination):
            result = obj | {'passed': True, 'errors': []}
            target = destination / (hashlib.sha256(obj['url'].encode()).hexdigest() + '.json')
            target.write_text(json.dumps(result))
            return result
        metadata = pd.DataFrame({'slot': [pd.Timestamp('2025-01-05', tz='UTC')]})
        with patch.object(availability_rebuild, 'provisional_metadata', return_value=metadata), \
             patch.object(source_proof, 'selections', return_value=(metadata, [self.obj])) as select, \
             patch.object(source_proof, 'verify_object', side_effect=verify) as check:
            final = close_selection(output, source, 1)
        self.assertEqual(select.call_count, 2)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(final.name, 'closure_metadata_02.parquet')
        self.assertTrue(json.loads((output / 'closure.json').read_text())['passed'])

    def test_external_cache_preserved_when_mapping_changes(self):
        import hashlib
        cache = self.root / 'original'
        cache.mkdir()
        target = cache / (hashlib.sha256(self.obj['url'].encode()).hexdigest() + '.json')
        receipt = self.obj | self.local | {'passed': True, 'head_headers': self.headers}
        original = json.dumps(receipt)
        target.write_text(original)
        def forbidden(*args, **kwargs):
            raise AssertionError('unexpected HTTP request')
        earlier = self.obj | {'earliest_origin': '2025-01-05T04:00:00+00:00'}
        result = verify_object(earlier, self.root, forbidden, cache_directory=cache)
        self.assertFalse(result['passed'])
        self.assertEqual(target.read_text(), original)
        self.assertTrue((self.root / target.name).exists())

    def test_head_only_and_resume_rehashes(self):
        calls = []
        headers = self.headers
        class Response:
            def __init__(self):
                self.headers = headers
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        def opener(req, timeout):
            calls.append(req.get_method())
            return Response()
        result = verify_object(self.obj, self.root, opener)
        self.assertTrue(result['passed'])
        self.assertEqual(calls, ['HEAD'])
        self.assertEqual(verify_object(self.obj, self.root, opener), result)
        self.assertEqual(calls, ['HEAD'])
        # Reuse actual HEAD metadata, but recheck timing for changed origin mappings.
        earlier = self.obj | {'earliest_origin': '2025-01-05T04:00:00+00:00'}
        self.assertFalse(verify_object(earlier, self.root, opener)['passed'])
        self.assertEqual(calls, ['HEAD'])
        self.path.write_bytes(b'changed bytes')
        self.assertFalse(verify_object(self.obj, self.root, opener)['passed'])
        self.assertEqual(calls, ['HEAD', 'HEAD'])
