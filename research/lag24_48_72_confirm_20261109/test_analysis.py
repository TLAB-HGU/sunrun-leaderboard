"""Joint-addition pairing, completion and decision-rule regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lag_joint_analysis_test', HERE / 'analyze_results.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def summary(value):
    return {scope: {metric: analysis.stats.describe([value] * 3)
                    for metric in analysis.stats.REPORT_METRICS}
            for scope in analysis.stats.SCOPES}


def recipe(seed):
    return {'experiment': f'lag24_48_72_s{seed}', 'seed': seed,
            'config': {'seed': seed, 'add_features': list(analysis.ADDITIONS)}}


class AnalysisTests(unittest.TestCase):
    def test_seed_order_is_explicit_and_pairing_not_file_order(self):
        rows = analysis.ordered_recipes({'experiments': [recipe(44), recipe(42), recipe(43)]})
        self.assertEqual([row['seed'] for row in rows], [42, 43, 44])
        self.assertEqual(analysis.matched_deltas({44: 110, 42: 100, 43: 105},
                                                {43: 103, 44: 107, 42: 99}), [-1., -2., -3.])
        with self.assertRaises(ValueError):
            analysis.matched_deltas({42: 100, 43: 105}, {42: 99, 43: 103, 44: 107})

    def test_reject_missing_duplicate_or_changed_joint_recipe(self):
        for seeds in ([42, 43], [42, 42, 44]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                analysis.ordered_recipes({'experiments': [recipe(seed) for seed in seeds]})
        rows = [recipe(seed) for seed in analysis.SEEDS]
        rows[-1]['config']['add_features'] = ['v_lag24']
        with self.assertRaises(ValueError):
            analysis.ordered_recipes({'experiments': rows})

    def test_gate_requires_all_safeguards_and_not_bh(self):
        base, trial = summary(100), summary(98)
        passed = analysis.improvement_gate(base, trial, [-2, -2, -2], [-3, -1], 2000)
        self.assertTrue(passed['passed'])
        self.assertFalse(any('bh' in check for check in passed['checks']))
        self.assertFalse(analysis.improvement_gate(base, trial, [-4, -3, 1], [-3, -1], 2000)['passed'])
        self.assertFalse(analysis.improvement_gate(base, trial, [-2] * 3, [-3, 0], 2000)['passed'])
        self.assertFalse(analysis.improvement_gate(base, trial, [-2] * 3, [-3, -1], 1999)['passed'])
        trial['total']['s']['mean'] = 99.1
        self.assertFalse(analysis.improvement_gate(base, trial, [-.9] * 3, [-2, -.1], 2000)['passed'])
        trial = summary(98)
        trial['total']['mae']['mean'] = 101.01
        self.assertFalse(analysis.improvement_gate(base, trial, [-2] * 3, [-3, -1], 2000)['passed'])
        trial = summary(98)
        trial['f3']['s']['mean'] = 102.01
        self.assertFalse(analysis.improvement_gate(base, trial, [-2] * 3, [-3, -1], 2000)['passed'])

    def test_partial_or_smoke_artifacts_cannot_be_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaisesRegex(ValueError, 'complete validated artifacts'):
                analysis.require_completed(run, 42, True)
            for name in ('training-validation.json', 'complete.json', 'local_score.json'):
                (run / name).write_text('{}')
            (run / 'train.exit').write_text('0\n')
            (run / 'predictions.parquet').touch()
            metadata = {'smoke': False, 'rounds': 900, 'seed': 42,
                        'config': {'add_features': list(analysis.ADDITIONS)},
                        'feature_count': 31, 'expanded_feature_count': 215}
            for period in ('F1', 'F2', 'F3'):
                (run / f'{period}_metadata.json').write_text(json.dumps(metadata))
            analysis.require_completed(run, 42, True)
            metadata['smoke'] = True
            (run / 'F3_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'full matching-seed training'):
                analysis.require_completed(run, 42, True)


if __name__ == '__main__':
    unittest.main()
