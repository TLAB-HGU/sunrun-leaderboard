"""Scheduler safety and artifact acceptance checks; never start real jobs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location('ablation_scheduler', Path(__file__).with_name('ablation_scheduler.py'))
scheduler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheduler)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipes = [dict(experiment=f'baseline_s{s}', stage='baseline', config={'seed': s}) for s in (42, 43, 44)]
        self.recipes += [dict(experiment='arm', stage='ablation', config={'seed': 42}),
                         dict(experiment='replay', stage='ablation', config={'seed': 42, 'reuse_run': 'arm'})]
        self.assignment = dict(experiments=self.recipes, max_jobs=3, threads=4, max_cpu_threads_total=20,
                               reserve_memory_gib=6, estimated_job_memory_gib=5, python=sys.executable)

    def tick(self, state, memory=40, busy=0):
        launched = []
        def launch(root, assignment, recipe):
            launched.append(recipe['experiment'])
            return {'started_utc': 'test'}
        with patch.object(scheduler, 'memory_gib', return_value=memory), \
             patch.object(scheduler, 'busy_cpu_cores', return_value=busy), \
             patch.object(scheduler, 'launch', side_effect=launch):
            scheduler.tick(self.root, self.assignment, state)
        return launched

    def baseline_state(self):
        return {f'baseline_s{s}': {'state': 'trained_validated'} for s in (42, 43, 44)}

    def test_baseline_stages(self):
        state = {}
        self.assertEqual(self.tick(state), ['baseline_s42'])
        self.assertIn('seed42', state['arm']['reason'])
        state = {'baseline_s42': {'state': 'trained_validated'}}
        self.assertEqual(self.tick(state), ['baseline_s43', 'baseline_s44'])
        self.assertIn('all three', state['arm']['reason'])
        state = self.baseline_state()
        self.assertEqual(self.tick(state), ['arm'])
        self.assertIn('source run', state['replay']['reason'])

    def test_failed_baseline_blocks_everything(self):
        state = {'baseline_s42': {'state': 'failed'}}
        self.assertEqual(self.tick(state), [])
        self.assertTrue(all(e['state'] == 'waiting' for name, e in state.items() if name != 'baseline_s42'))

    def test_resources_and_lightweight_replay(self):
        self.assertEqual(self.tick({}, busy=18), [])
        self.assertEqual(self.tick({}, memory=10), [])
        state = self.baseline_state()
        state['arm'] = {'state': 'trained_validated'}
        self.assertEqual(self.tick(state, memory=7), ['replay'])

    def test_failed_dependency_is_terminal(self):
        state = self.baseline_state()
        state['arm'] = {'state': 'failed'}
        self.assertEqual(self.tick(state), [])
        self.assertEqual(state['replay']['state'], 'failed')

    def test_five_jobs_respect_total_cpu_and_memory(self):
        self.assignment.update(max_jobs=5)
        self.recipes.extend(dict(experiment=f'arm{i}', stage='ablation', config={'seed': 42})
                            for i in range(8))
        self.assertEqual(len(self.tick(self.baseline_state(), memory=40)), 5)
        self.assertEqual(len(self.tick(self.baseline_state(), memory=21)), 3)
        self.assertEqual(len(self.tick(self.baseline_state(), memory=40, busy=8)), 3)

    def test_host_reproduction_required_even_with_imported_baselines(self):
        data = self.root / 'data.parquet'
        data.write_bytes(b'fixture')
        self.assignment.update(host='local', host_reproduction_gate='host-reproduction.json',
                               smoke_proof='smoke-local.json', data=str(data))
        scheduler.atomic(self.root / 'smoke-local.json', dict(all_equal=True, data=scheduler.identity(data)))
        self.assertEqual(self.tick(self.baseline_state()), [])
        proof = dict(host='golab2', validated=True, baseline_max_abs_difference=0., baseline_score_difference=0., data=scheduler.identity(data))
        path = self.root / 'host-reproduction.json'
        path.write_text(json.dumps(proof))
        self.assertEqual(self.tick(self.baseline_state()), [])
        proof['host'] = 'local'
        proof['baseline_max_abs_difference'] = 1e-3
        path.write_text(json.dumps(proof))
        self.assertEqual(self.tick(self.baseline_state()), [])
        proof['baseline_max_abs_difference'] = 0.
        path.write_text(json.dumps(proof))
        self.assertEqual(self.tick(self.baseline_state()), ['arm'])
        data.write_bytes(b'changed')
        self.assertEqual(self.tick(self.baseline_state()), [])

    def test_host_completion_defers_global_analysis(self):
        self.assignment['run_analysis'] = False
        state = {r['experiment']: {'state': 'trained_validated'} for r in self.recipes}
        with patch.object(scheduler.subprocess, 'run') as runner:
            result = scheduler.analyze(self.root, self.assignment, state, final=True)
        runner.assert_not_called()
        self.assertTrue(result['analysis_deferred'])
        self.assertEqual(json.loads((self.root / 'host-complete.json').read_text())['state'], 'complete')
        self.assertFalse((self.root / 'campaign-complete.json').exists())

    def test_reassignment_only_before_launch(self):
        original = dict(self.assignment)
        updated = dict(original, experiments=self.recipes[:-1])
        state = {'replay': {'state': 'waiting'}}
        scheduler.validate_assignment_update(self.root, original, updated, state)
        self.assertNotIn('replay', state)
        state['replay'] = {'state': 'running'}
        with self.assertRaisesRegex(ValueError, 'started recipe'):
            scheduler.validate_assignment_update(self.root, original, updated, state)
        with self.assertRaisesRegex(ValueError, 'identity'):
            scheduler.validate_assignment_update(self.root, original, dict(original, host='local'), {})
        state['replay'] = {'state': 'waiting'}
        (self.root / 'runs' / 'replay').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'started recipe'):
            scheduler.validate_assignment_update(self.root, original, updated, state)

    def test_import_receipt_is_checked_before_dependency_use(self):
        state = self.baseline_state()
        state['baseline_s42']['imported_from'] = '/remote/baseline_s42'
        self.assertEqual(self.tick(state), [])
        self.assertEqual(state['baseline_s42']['state'], 'failed')
        out = self.root / 'runs' / 'baseline_s42'
        out.mkdir(parents=True)
        (out / 'train.exit').write_text('0\n')
        (out / 'imported.json').write_text('{}')
        state = self.baseline_state()
        state['baseline_s42']['imported_from'] = '/remote/baseline_s42'
        with patch.object(scheduler, 'validate_output', return_value={'rows': 454680}) as validate:
            self.assertEqual(self.tick(state), ['arm'])
            self.assertTrue(state['baseline_s42']['import_checked'])
            self.tick(state)
            self.assertEqual(validate.call_count, 1)

    def test_launch_failure_closes_baseline_gate(self):
        with patch.object(scheduler, 'memory_gib', return_value=40), \
             patch.object(scheduler, 'busy_cpu_cores', return_value=0), \
             patch.object(scheduler, 'launch', side_effect=OSError('test failure')):
            state = {}
            scheduler.tick(self.root, self.assignment, state)
        self.assertEqual(state['baseline_s42']['state'], 'failed')
        self.assertEqual(state['baseline_s43']['state'], 'waiting')

    def test_analysis_full_partial_and_idempotence(self):
        state = {r['experiment']: {'state': 'trained_validated'} for r in self.recipes}
        with patch.object(scheduler.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as runner:
            result = scheduler.analyze(self.root, self.assignment, state, final=True)
            command = runner.call_args.args[0]
            self.assertNotIn('--partial', command)
            self.assertEqual(command[command.index('--bootstrap') + 1], '2000')
            self.assertTrue(result['full'])
            scheduler.analyze(self.root, self.assignment, state, final=True)
            self.assertEqual(runner.call_count, 1)
        state['arm'] = {'state': 'failed'}
        with patch.object(scheduler.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as runner:
            scheduler.analyze(self.root, self.assignment, state, final=True)
            command = runner.call_args.args[0]
            self.assertIn('--partial', command)
            self.assertEqual(command[command.index('--bootstrap') + 1], '0')
        self.assertEqual(json.loads((self.root / 'campaign-complete.json').read_text())['state'], 'incomplete')

    def test_analysis_failure_is_preserved(self):
        state = {r['experiment']: {'state': 'trained_validated'} for r in self.recipes}
        with patch.object(scheduler.subprocess, 'run', return_value=subprocess.CompletedProcess([], 2)):
            result = scheduler.analyze(self.root, self.assignment, state, final=True)
        self.assertEqual(result['exit_code'], 2)
        self.assertEqual((self.root / 'analysis.exit').read_text().strip(), '2')
        self.assertEqual(json.loads((self.root / 'campaign-complete.json').read_text())['state'], 'incomplete')

    def test_launch_environment_and_receipts(self):
        for filename in ('core.py', 'baseline.py', 'competition.py'):
            (self.root / filename).write_text('print("fixture")\n')
        data = self.root / 'data.parquet'
        data.write_bytes(b'fixture')
        assignment = dict(self.assignment, data=str(data), observations=str(data), host='local')
        with patch.object(scheduler, 'session_alive', return_value=False), \
             patch.object(scheduler, 'session_id', return_value='$77'), \
             patch.object(scheduler.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '$77\n', '')):
            record = scheduler.launch(self.root, assignment, self.recipes[0])
        out = self.root / 'runs' / 'baseline_s42'
        script = (out / 'run.sh').read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES=', script)
        self.assertIn('--device cpu', script)
        self.assertIn('train.exit.tmp', script)
        self.assertEqual(record['training_host'], 'local')
        self.assertEqual(record['tmux_session_id'], '$77')
        self.assertIn(str(out / 'experiment.py'), record['command'])
        self.assertEqual(set(record['source_snapshots']), {'experiment.py', 'core.py', 'baseline.py', 'competition.py'})
        wrapper = (out / 'experiment.py').read_text()
        compile(wrapper, 'experiment.py', 'exec')
        self.assertIn("EXPERIMENT = 'baseline_s42'", wrapper)
        self.assertIn('locked_config=CONFIG', wrapper)
        self.assertNotIn('SUPPORTED', wrapper)
        with self.assertRaises(FileExistsError):
            scheduler.launch(self.root, assignment, self.recipes[0])

    def test_session_identity_uses_exact_name_and_nonempty_id(self):
        result = subprocess.CompletedProcess([], 0, 'other\t$4\nexpected\t$12\nempty\t\n', '')
        with patch.object(scheduler.subprocess, 'run', return_value=result) as run:
            self.assertEqual(scheduler.session_id('expected'), '$12')
            self.assertIsNone(scheduler.session_id('empty'))
            self.assertIsNone(scheduler.session_id('expect'))
            self.assertEqual(run.call_args.args[0][1], 'list-sessions')

    def test_cleanup_rejects_changed_session_identity(self):
        record = dict(tmux_session='expected', tmux_session_id='$12')
        with patch.object(scheduler, 'session_id', return_value='$13'), \
             patch.object(scheduler.subprocess, 'run') as run:
            scheduler.cleanup_terminal_session(record)
            run.assert_not_called()

    def test_scheduler_pid_and_lock_cleanup(self):
        with scheduler.scheduler_ownership(self.root):
            self.assertEqual((self.root / 'scheduler.pid').read_text().strip(), str(scheduler.os.getpid()))
            with self.assertRaises(BlockingIOError):
                with scheduler.scheduler_ownership(self.root):
                    self.fail('Concurrent scheduler acquired lock')
        self.assertFalse((self.root / 'scheduler.pid').exists())
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with scheduler.scheduler_ownership(self.root):
                raise RuntimeError('fixture')
        self.assertFalse((self.root / 'scheduler.pid').exists())


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'runs' / 'baseline_s42'
        self.out.mkdir(parents=True)
        frames = []
        for index, count in enumerate((2089, 2113, 2113), 1):
            period = f'F{index}'
            origins = pd.date_range(f'2026-{1 + (index - 1)*3:02d}-01', periods=count, freq='h', tz='UTC')
            values = np.full((count, 72), 400.)
            np.savez(self.out / f'{period}_predictions.npz', pred=values, origins=origins.asi8)
            (self.out / f'{period}_metadata.json').write_text(json.dumps({'period': period, 'smoke': False}))
            frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                                        'horizon_hours': np.tile(np.arange(1, 73), count), 'pred_kms': values.ravel()}))
        self.frame = pd.concat(frames, ignore_index=True)
        self.frame.to_parquet(self.out / 'predictions.parquet', index=False)
        self.reference = self.root / 'reference.parquet'
        self.frame.to_parquet(self.reference, index=False)
        for name in ('train.log', 'launch.json', 'runner_at_launch.py'):
            (self.out / name).write_text('fixture\n')
        (self.out / 'launch.json').write_text('{}')
        (self.out / 'local_score.json').write_text(json.dumps({s: {'s': 100., 'n_replaced': 0} for s in ('total', 'f1', 'f2', 'f3')}))
        self.assignment = dict(python=sys.executable, baseline_reference_predictions=str(self.reference), baseline_reference_s=100.)
        self.recipe = dict(experiment='baseline_s42', stage='baseline', config={'seed': 42})

    def validate(self):
        return scheduler.validate_output(self.root, self.assignment, self.recipe)

    def test_complete_artifacts_and_baseline_parity(self):
        proof = self.validate()
        self.assertEqual(proof['rows'], 454680)
        self.assertEqual(proof['baseline_max_abs_difference'], 0)
        self.assertEqual(proof['baseline_score_difference'], 0)
        self.assertEqual(json.loads((self.out / 'complete.json').read_text())['publish'], 'not_required')

    def test_score_and_prediction_reference_mismatch(self):
        self.assignment['baseline_reference_s'] = 101.
        with self.assertRaisesRegex(ValueError, 'score reproduction mismatch'):
            self.validate()
        self.assignment['baseline_reference_s'] = 100.
        self.frame.loc[0, 'pred_kms'] = 401.
        self.frame.to_parquet(self.reference, index=False)
        with self.assertRaisesRegex(ValueError, 'prediction reproduction mismatch'):
            self.validate()

    def test_incomplete_grid_and_npz_parquet_disagreement(self):
        with np.load(self.out / 'F1_predictions.npz') as z:
            origins, values = z['origins'].copy(), z['pred'].copy()
        values[0, 0] = 401.
        np.savez(self.out / 'F1_predictions.npz', pred=values, origins=origins)
        with self.assertRaisesRegex(ValueError, 'NPZ and parquet predictions disagree'):
            self.validate()
        np.savez(self.out / 'F1_predictions.npz', pred=values[:-1], origins=origins[:-1])
        with self.assertRaisesRegex(ValueError, 'Incomplete period prediction grid'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
