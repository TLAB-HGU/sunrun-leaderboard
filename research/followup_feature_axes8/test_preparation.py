from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import build_data
import prepare_controller


class PreparationTests(unittest.TestCase):
    def test_exact_append_preserves_targets_metadata_nulls_and_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = pa.table({'origin': pa.array(pd.date_range('2025-01-01', periods=4, tz='UTC')),
                                 'h': [1, 1, 1, 1], 'y': pa.array([1., float('nan'), None, 4.]),
                                 'existing': [1, 2, 3, 4], 'set': ['train'] * 4})
            cache = original.select(['origin', 'h']).append_column('new_feature', pa.array([5., 6., 7., 8.]))
            pq.write_table(original, root/'source.parquet', row_group_size=2)
            pq.write_table(cache, root/'cache.parquet', row_group_size=2)
            proof = build_data.augment(root/'source.parquet', root/'cache.parquet', root/'out.parquet')
            self.assertTrue(proof['original_columns_exact'])
            self.assertEqual(pq.read_table(root/'out.parquet')['new_feature'].to_pylist(), [5.,6.,7.,8.])
            self.assertTrue(build_data.equal_original(original, pq.read_table(root/'out.parquet', columns=original.column_names)))

    def test_feature_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = pa.table({'origin': [1], 'h': [1], 'existing': [2.]})
            for name in ('source', 'cache'):
                pq.write_table(table, root/f'{name}.parquet')
            with self.assertRaisesRegex(ValueError, 'replacement'):
                build_data.augment(root/'source.parquet', root/'cache.parquet', root/'out.parquet')

    def test_cache_uses_inputs_only_and_preserves_row_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = pd.DataFrame({'origin': pd.date_range('2025-01-01', periods=4, tz='UTC'),
                'h': 1, 'v_lag0': [410.,420.,430.,440.], 'v_lag24':400., 'v_lag48':390., 'v_lag72':380., 'y':np.inf})
            frame.to_parquet(root/'source.parquet', row_group_size=2)
            proof = build_data.build_cache(root/'source.parquet', root/'cache.parquet', 1)
            self.assertFalse(proof['target_columns_read'])
            self.assertEqual(proof['row_groups'], 2)
            self.assertEqual(pq.read_table(root/'cache.parquet')['v_diff24_48'].to_pylist(), [10.] * 4)

    def test_parent_is_frozen_by_full_gate_never_by_best_seed(self):
        report = dict(status='complete', seeds=[42,43,44], add_features=['v_lag24','v_lag48','v_lag72'],gate={'passed':False})
        self.assertEqual(prepare_controller.select_parent(report)['base_add_features'], [])
        report['gate']['passed'] = True
        self.assertEqual(prepare_controller.select_parent(report)['base_add_features'], report['add_features'])
        report['seeds'] = [42]
        with self.assertRaises(ValueError):
            prepare_controller.select_parent(report)


if __name__ == '__main__':
    unittest.main()
