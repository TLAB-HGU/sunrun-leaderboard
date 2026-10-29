"""Release replay and immutability checks; no external publication calls."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('tcn_publish_tested', Path(__file__).with_name('publish.py'))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


@pytest.fixture
def completed_run(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    origins = publish.P.expected_origins()
    frame = pd.DataFrame({
        'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
        'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
        'pred_kms': np.full(len(origins) * 72, 450., dtype=np.float64),
    })
    frame.to_parquet(out / 'predictions.parquet', index=False)
    manifest = dict(experiment=publish.EXPERIMENT, smoke=False, recipe={'seed': 42},
                    selected=[f'f{i}' for i in range(30)], varying=[f'e{i}' for i in range(9)],
                    input_identity={'test_fixture': True},
                    source_sha256={str(publish.HELPER): publish.P.sha(publish.HELPER)})
    publish.P.write_json(out / 'manifest.json', manifest)
    publish.P.write_json(out / 'results.json', {'combined': {'official': {
        'mse': 10., 'mse_f1': 11., 'mse_f2': 9., 'mse_ge550': 12.,
    }}})
    publish.P.write_json(out / 'timing.json', {'inference_seconds_per_fold': 0.002})
    for stage in ('internal', 'f1', 'final'):
        folder = out / stage
        folder.mkdir()
        (folder / 'best.pt').write_bytes(f'{stage} checkpoint fixture'.encode())
    return out, frame


def test_prepare_round_trip_is_offline_and_resumable(completed_run, monkeypatch):
    out, original = completed_run
    real_run = subprocess.run

    def local_replay_only(command, **kwargs):
        assert command[0] == str(publish.P.PYTHON)
        assert '--output' in command
        return real_run(command, **kwargs)

    monkeypatch.setattr(publish.subprocess, 'run', local_replay_only)
    item = publish.prepare(out)
    pd.testing.assert_frame_equal(pd.read_parquet(item['predictions']), original)
    assert publish.prepare(out) == item
    meta = json.loads(Path(item['meta']).read_text())
    assert item['config_sha256'] == publish.P.config_sha(meta['config'])
    assert meta['config']['data']['lookback_hours'] == 120
    assert len(meta['config']['data']['model_sha256']) == 3
    assert 'replays archived forecasts' in meta['description']


def test_prepared_release_rejects_tampering(completed_run):
    out, _ = completed_run
    item = publish.prepare(out)
    checkpoint = out / 'publication/package' / item['release'] / 'checkpoints/final/best.pt'
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='asset changed'):
        publish.prepare(out)


def test_changed_run_cannot_reuse_prepared_identity(completed_run):
    out, _ = completed_run
    publish.prepare(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['recipe']['seed'] = 43
    publish.P.write_json(out / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='run changed'):
        publish.prepare(out)


def test_changed_source_rejected_before_publication(completed_run):
    out, _ = completed_run
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['source_sha256'][str(Path(publish.__file__))] = '0' * 64
    publish.P.write_json(out / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='run source changed'):
        publish.prepare(out)


def test_copy_immutable_preserves_existing_different_file(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_text('new')
    target.write_text('original')
    with pytest.raises(ValueError, match='artifact differs'):
        publish.copy_immutable(source, target)
    assert target.read_text() == 'original'


def test_score_validation_checks_full_and_regional_metrics():
    scores = dict(mse=1., mse_f1=2., mse_f2=3., mse_ge550=4.)
    item = dict(config_sha256='abc', scores=scores)
    result = dict(config_sha256='abc', **scores)
    publish.Publisher.compare_scores(item, result)
    result['mse_ge550'] = 5.
    with pytest.raises(ValueError, match='mse_ge550'):
        publish.Publisher.compare_scores(item, result)
