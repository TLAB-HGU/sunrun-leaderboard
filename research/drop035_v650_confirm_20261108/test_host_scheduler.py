from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_scheduler import capacity, workloads
import core
from prepare import EXCLUDE


class CapacityTest(unittest.TestCase):
    def test_admission_budget_and_reserve(self):
        five = [{'threads': 4}] * 5
        self.assertTrue(capacity(five, 11, 20))
        self.assertFalse(capacity(five + [{'threads': 4}], 20, 20))
        self.assertFalse(capacity(five, 10.99, 20))
        self.assertFalse(capacity(five, 20, 20.01))
        self.assertFalse(capacity([{'threads': 22}], 20, 10))

    def test_process_inventory_ignores_shells_and_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, argv in enumerate([
                ['/venv/bin/python', 'experiment.py', '--out', '/a', '--device', 'cpu', '--threads', '4'],
                ['bash', '-c', 'python experiment.py --out /b --threads 4'],
                ['python3', 'experiment.py', '--out', '/c', '--device', 'cuda', '--threads', '4']
            ], 1):
                p = root / str(pid); p.mkdir(); (p / 'cmdline').write_bytes('\0'.join(argv).encode())
            self.assertEqual(workloads(root), [{'pid': 1, 'threads': 4, 'out': '/a'}])

    def test_exact_combination(self):
        config = core.normalize_config({'seed': 42, 'exclude_features': EXCLUDE})
        self.assertEqual(len(core.baseline.FEATURES) - len(EXCLUDE), 28)
        self.assertEqual(EXCLUDE, ['ch_ch_t035_equatorial_area', 'strip_v650_dark'])
        self.assertEqual(config['alpha'], .25)
        self.assertFalse(config['spectral_hessian_only'])
        self.assertNotIn('reuse_run', config)
        self.assertTrue(config['spectral'] and config['offset'] and config['clip_component'] and config['clip_final'])
        self.assertEqual((config['tail_weight'], config['block_size'], config['offset_window']), (4, 24, 336))
        names = core.feature_names(config)
        self.assertFalse(set(EXCLUDE) & set(names['constant'] + names['varying']))


if __name__ == '__main__':
    unittest.main()

class LifecycleTest(unittest.TestCase):
    def recipe(self):
        return dict(experiment='drop035_v650_s42', seed=42, config=dict(seed=42, exclude_features=EXCLUDE))

    def test_fresh_contract_rejects_reuse_and_other_changes(self):
        import scheduler_support as support
        for key, value in [('reuse_run', 'old'), ('alpha', 1), ('rounds', 2)]:
            recipe = self.recipe()
            recipe['config'][key] = value
            with self.assertRaises(ValueError):
                support.expected_config(recipe)
        config = support.expected_config(self.recipe())
        self.assertEqual(config['experiment'], 'drop035_v650_s42')
        self.assertEqual(config['alpha'], .25)

    def test_launch_snapshots_and_cpu_environment(self):
        import json
        from unittest.mock import patch
        import scheduler_support as support
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('core.py', 'baseline.py', 'competition.py'):
                (root/name).write_text('frozen science\n')
            data=root/'data';data.write_text('input')
            assignment=dict(python=sys.executable,data=str(data),observations=str(data),host='local')
            with patch.object(support,'session_alive',return_value=False), patch.object(support,'session_id',return_value='$123'), patch.object(support.subprocess,'run'):
                launch=support.launch(root,assignment,self.recipe())
            out=root/'runs'/'drop035_v650_s42'
            self.assertEqual((out/'experiment.py').read_bytes(),(out/'runner_at_launch.py').read_bytes())
            self.assertIn('CUDA_VISIBLE_DEVICES=',(out/'run.sh').read_text())
            self.assertEqual(launch['command'][-4:],['--device','cpu','--threads','4'])
            self.assertNotIn('reuse_run',json.loads((out/'config.json').read_text()))

    def test_lost_session_fails_without_retry(self):
        import json
        from unittest.mock import patch
        import scheduler_support as support
        import host_scheduler
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs'/'drop035_v650_s42';run.mkdir(parents=True)
            (run/'launch.json').write_text(json.dumps(dict(campaign_root=str(root),experiment='drop035_v650_s42',tmux_session='owned',tmux_session_id='$123')))
            state={}
            with patch.object(support,'session_id',return_value=None),patch.object(support,'launch') as launch:
                host_scheduler.tick(root,{'experiments':[self.recipe()]},state)
                launch.assert_not_called()
            self.assertEqual(state['drop035_v650_s42']['state'],'failed')

class ReservationTest(unittest.TestCase):
    def test_pending_launch_reserves_memory_before_second_admission(self):
        from unittest.mock import patch
        import host_scheduler
        import scheduler_support as support
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            recipes=[dict(experiment=f'drop035_v650_s{s}') for s in (43,44)]
            state={}
            with patch.object(host_scheduler,'workloads',return_value=[{'threads':4,'out':'/existing'}]), patch.object(support,'memory_gib',return_value=15), patch.object(support,'busy_cpu_cores',return_value=4), patch.object(support,'launch',return_value={'started_utc':'now'}) as launch:
                host_scheduler.tick(root,{'experiments':recipes,'max_jobs':2},state)
            self.assertEqual(launch.call_count,1)
            self.assertEqual(state['drop035_v650_s44']['state'],'waiting')
