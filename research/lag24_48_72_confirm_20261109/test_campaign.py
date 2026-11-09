"""Frozen joint-addition scientific contract and resource-safe launch tests."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import baseline
import core
import host_scheduler
import launch_services
import prepare
import scheduler_support as support
import collector

ROOT = Path(__file__).resolve().parent


class JointCampaignTests(unittest.TestCase):
    def test_scientific_functions_identical_and_parent_prefix_preserved(self):
        names = prepare.validate_source()
        self.assertEqual(sum(map(len, names.values())), 31)
        rng = np.random.default_rng(42)
        old = {'C': rng.normal(size=(4, 20)).astype('float32'),
               'E': rng.normal(size=(4, 72, 8)).astype('float32')}
        extra = rng.normal(size=(4, 3)).astype('float32')
        new = {'C': np.column_stack([old['C'], extra]), 'E': old['E'].copy()}
        for start in (0, 24, 48):
            matrix = core.design(new, start, 24)
            self.assertEqual(matrix.shape, (4, 215))
            np.testing.assert_array_equal(matrix[:, :212], core.design(old, start, 24))
            np.testing.assert_array_equal(matrix[:, 212:], extra)
        self.assertEqual(len(baseline.PARENT_NAMES['constant']), 20)
        self.assertEqual(len(baseline.PARENT_NAMES['varying']), 8)

    def test_exact_three_fresh_remote_seeds_and_joint_config(self):
        recipes = prepare.recipes()
        self.assertEqual([r['seed'] for r in recipes], [42, 43, 44])
        for row in recipes:
            self.assertEqual(row['execution_host'], 'golab2')
            self.assertEqual(row['config']['add_features'], ['v_lag24', 'v_lag48', 'v_lag72'])
            config = support.expected_config(row)
            self.assertEqual(config['experiment'], row['experiment'])
            self.assertEqual(config['seed'], row['seed'])
        bad = dict(recipes[0], config={'seed': 42, 'add_features': ['v_lag24']})
        with self.assertRaises(ValueError):
            support.expected_config(bad)

    def test_admission_counts_existing_campaigns(self):
        jobs = [{'threads': 4}] * 3
        self.assertTrue(host_scheduler.capacity(jobs, 12, 12))
        self.assertFalse(host_scheduler.capacity(jobs * 2, 30, 12))
        self.assertFalse(host_scheduler.capacity(jobs, 10.9, 12))
        self.assertFalse(host_scheduler.capacity([{'threads': 22}], 30, 0))

    def test_launch_keeps_one_architecture_and_emits_initial_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('core.py', 'baseline.py', 'competition.py', 'inventory.json'):
                shutil.copy2(ROOT / name, root / name)
            data = root / 'input.parquet'
            data.write_bytes(b'test identity only')
            recipe = prepare.recipes()[0]
            assignment = dict(host='golab2', python='python', data=str(data), observations=str(data))
            with patch.object(support, 'session_alive', return_value=False), \
                 patch.object(support, 'session_id', return_value='$123'), \
                 patch.object(support.subprocess, 'run'):
                record = support.launch(root, assignment, recipe)
            out = root / 'runs' / recipe['experiment']
            compile((out / 'experiment.py').read_text(), 'experiment.py', 'exec')
            self.assertNotIn('SUPPORTED', (out / 'experiment.py').read_text())
            self.assertIn("'add_features': ['v_lag24', 'v_lag48', 'v_lag72']", (out / 'experiment.py').read_text())
            shell = (out / 'run.sh').read_text()
            self.assertIn('"unit": 0, "units": 6', shell)
            self.assertIn('CUDA_VISIBLE_DEVICES=', shell)
            self.assertIn(' >> ', shell)
            self.assertEqual(subprocess.run(['bash', '-n', str(out / 'run.sh')], capture_output=True).returncode, 0)
            progress = next(line for line in shell.splitlines() if line.startswith('printf'))
            subprocess.run(['bash', '-c', progress], check=True)
            self.assertEqual(json.loads((out / 'train.log').read_text())['unit'], 0)
            self.assertEqual(record['host'], 'golab2')

    def test_remote_process_identity_uses_actual_nul_delimiter(self):
        seen = []
        def reply(code, timeout):
            seen.append(code)
            compile(code, 'remote_status', 'exec')
            return subprocess.CompletedProcess([], 0, b'{"state": {}, "scheduler_alive": true}')
        with patch.object(collector, 'remote', side_effect=reply):
            self.assertTrue(collector.remote_status()['scheduler_alive'])
        self.assertIn('split(bytes([0]))', seen[0])
        argument_vector = b'python\0/r/host_scheduler.py\0--root\0/r\0'
        self.assertIn(b'/r/host_scheduler.py', argument_vector.split(bytes([0])))

    def test_live_service_resume_never_launches_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = dict(campaign_root=str(root), tmux_session='owned', tmux_session_id='$7')
            (root / 'collector.py.launch.json').write_text(json.dumps(receipt))
            with patch.object(launch_services, 'session_id', return_value='$7'), \
                 patch.object(launch_services.subprocess, 'run') as run:
                result = launch_services.start(root, 'python', 'collector.py', 'owned', True)
            self.assertEqual(result['state'], 'already_running')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
