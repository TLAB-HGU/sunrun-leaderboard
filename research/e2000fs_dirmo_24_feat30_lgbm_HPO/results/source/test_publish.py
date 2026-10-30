"""Publication boundaries and exact full-grid replay, without network writes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('hpo_publication_tests', Path(__file__).with_name('publish.py'))
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'runs/main'
    final = root / 'final'
    artifacts = final / 'artifacts'
    artifacts.mkdir(parents=True)
    source = tmp_path / 'train.py'
    source.write_text('# frozen test source\n')
    winner = {'study': 1, 'trial': 4, 'value': 12.5, 'params': {'n_estimators': 500}, 'completed': 50}
    best = dict(winner, study_winners=[dict(winner, study=i) for i in (1, 2, 3)])
    for index, item in enumerate(best['study_winners'], 1):
        directory = root / f'study{index}'
        publication.write(directory / 'best.json', item)
        publication.write(directory / 'identity.json', {'study': index})
        publication.write(directory / 'status.json', {'completed': 50})
        (directory / 'trials.csv').write_text('number,value\n4,12.5\n')
    manifest = {'experiment': publication.EXPERIMENT, 'selected': [f'f{i}' for i in range(30)],
                'posthoc_official_exposed': True, 'params': best['params'].copy(),
                'hpo_selection': best, 'evaluation_revision': 'frozen-revision',
                'hashes': {str(source): publication.sha(source)}}
    publication.write(root / 'best.json', best)
    publication.write(final / 'manifest.json', manifest)
    publication.write(final / 'status.json', {'status': 'completed'})
    for stage in ('f1', 'final'):
        for h in range(72):
            (artifacts / f'{stage}_b{h // 24:02}_h{h + 1:02}.txt').write_text('test model')
    origins = publication.existing.expected_origins()
    pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                  'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                  'pred_kms': np.full(len(origins) * 72, 400.)}).to_parquet(artifacts / 'predictions.parquet', index=False)
    publication.write(artifacts / 'scores.json', {k: 10. for k in publication.existing.METRICS.values()})
    return root


def test_frozen_results_require_completed_and_matching_winner(campaign):
    final, manifest, *_ = publication.frozen_inputs(campaign)
    manifest['params']['n_estimators'] = 501
    publication.write(final / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='parameter differs'):
        publication.frozen_inputs(campaign)
    publication.write(final / 'status.json', {'status': 'running'})
    with pytest.raises(ValueError, match='completed final'):
        publication.frozen_inputs(campaign)


def test_changed_input_is_rejected(campaign):
    manifest = publication.read(campaign / 'final/manifest.json')
    Path(next(iter(manifest['hashes']))).write_text('changed')
    with pytest.raises(ValueError, match='frozen input or source changed'):
        publication.frozen_inputs(campaign)


def test_full_grid_build_resume_and_tamper_detection(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    monkeypatch.setattr(publication, 'measure_timing', lambda *args: {
        'scope': 'synthetic unit-test timing only', 'inference_seconds_per_fold': 0.25})
    pub = tmp_path / 'publication'
    work = pub / 'worktree'
    work.mkdir(parents=True)
    (work / 'submit.py').write_text('# no network client used\n')
    publication.build(campaign, pub, work)
    publication.build(campaign, pub, work)
    item = publication.read(pub / 'build.json')['hpo']
    frame = pd.read_parquet(item['predictions'])
    assert len(frame) == 357840
    assert frame.equals(pd.read_parquet(campaign / 'final/artifacts/predictions.parquet'))
    release = work / item['release']
    config = publication.read(release / 'config.json')
    assert config['hyperparameters']['selection'] == publication.read(campaign / 'best.json')
    assert publication.existing.config_sha(config) == item['config_sha256']
    assert publication.read(release / 'meta.yaml')['config'] == config
    assert '1009' in config['hyperparameters']['sampler_seed_policy']
    (release / 'forecast.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='publication asset changed'):
        publication.build(campaign, pub, work)
