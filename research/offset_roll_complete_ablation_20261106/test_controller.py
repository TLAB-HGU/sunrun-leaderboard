"""Controller state transitions and archive safety without real processes."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('campaign_controller', ROOT / 'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'runs').mkdir()
        self.recipes = [dict(experiment='a', stage='baseline', seed=42)]
        controller.atomic(self.root / 'assignment.global.json', dict(experiments=self.recipes, python=sys.executable))
        controller.atomic(self.root / 'assignment.json', dict(experiments=self.recipes, python=sys.executable))
        controller.atomic(self.root / 'allocation.json', dict(owners={'a': 'local'}))
        controller.atomic(self.root / 'scheduler-state.json', {'a': {'state': 'trained_validated'}})

    def test_owner_specific_completion(self):
        self.assertTrue(controller.complete(self.recipes, {'a':'local'}, {'a':{'state':'trained_validated'}}, {}))
        self.assertFalse(controller.complete(self.recipes, {'a':'golab2'}, {'a':{'state':'trained_validated'}}, {}))

    def test_merge_final_idempotent_and_immutable(self):
        assignment = dict(experiments=[])
        controller.merge_recipes(assignment, self.recipes)
        controller.merge_recipes(assignment, self.recipes)
        self.assertEqual(assignment['experiments'], self.recipes)
        with self.assertRaisesRegex(ValueError, 'changed'):
            controller.merge_recipes(assignment, [dict(self.recipes[0], seed=43)])

    def test_lock_excludes_duplicate_controller(self):
        with controller.ownership(self.root):
            with self.assertRaises(BlockingIOError):
                with controller.ownership(self.root):
                    self.fail('Concurrent ownership')

    def test_generated_scheduler_code_compiles(self):
        def run(command, **kwargs):
            compile(command[2], '<remote scheduler>', 'exec')
            return subprocess.CompletedProcess(command, 0, '{"state":"running"}', '')
        with patch.object(controller.subprocess, 'run', side_effect=run):
            self.assertEqual(controller.ensure_scheduler(self.root, 'local', 'ablation')['state'], 'running')

    def test_collection_never_overwrites_local_run(self):
        (self.root / 'runs' / 'a').mkdir()
        with patch.object(controller, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                controller.collect_completed(self.root, {'a':'golab2'}, {'a':{'state':'trained_validated'}}, set())
            remote.assert_not_called()

    def test_collection_rejects_archive_traversal(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            member = tarfile.TarInfo('../bad');member.size=1
            archive.addfile(member, io.BytesIO(b'x'))
        with patch.object(controller, 'remote', return_value=subprocess.CompletedProcess([],0,stream.getvalue())):
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                controller.collect_completed(self.root, {'a':'golab2'}, {'a':{'state':'trained_validated'}}, set())

    def test_blocked_manifest_never_launches_final(self):
        (self.root / 'analysis').mkdir()
        controller.atomic(self.root / 'analysis' / 'removal_manifest.json', {'status':'blocked_no_retained_features'})
        with patch.object(controller,'ensure_scheduler',return_value={'state':'exited'}), \
             patch.object(controller,'remote_status',return_value={'state':{}}), \
             patch.object(controller,'collect_completed',return_value=[]), \
             patch.object(controller,'analyze') as analyze, \
             patch.object(controller,'append_final_recipes') as append, \
             patch.object(controller,'cleanup_scheduler_sessions'):
            status={'phase':'ablation'}
            controller.step(self.root,status)
            analyze.assert_called_once_with(self.root,final=False)
            append.assert_not_called()
            self.assertEqual(status['phase'],'blocked')

    def test_final_completion_does_not_refreeze(self):
        (self.root / 'analysis').mkdir()
        controller.atomic(self.root / 'analysis' / 'removal_manifest.json', {'status':'frozen','final_experiments':self.recipes})
        with patch.object(controller,'ensure_scheduler',return_value={'state':'exited'}), \
             patch.object(controller,'remote_status',return_value={'state':{}}), \
             patch.object(controller,'collect_completed',return_value=[]), \
             patch.object(controller,'analyze') as analyze, \
             patch.object(controller,'append_final_recipes') as append, \
             patch.object(controller,'cleanup_scheduler_sessions'):
            status={'phase':'final'}
            controller.step(self.root,status)
            analyze.assert_called_once_with(self.root,final=True)
            append.assert_not_called()
            self.assertEqual(status['phase'],'complete')
            self.assertEqual(controller.read(self.root/'campaign-complete.json')['state'],'complete')


if __name__=='__main__':unittest.main()
