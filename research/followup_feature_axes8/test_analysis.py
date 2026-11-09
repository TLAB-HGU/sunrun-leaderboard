import json

import pytest

import analyze_results as a


def assignment():
    return {'experiments': [{'experiment': f'axis{axis}_s{seed}', 'axis': axis, 'seed': seed,
                            'config': {'seed': seed, 'add_features': [f'feature{axis}']}}
                           for axis in a.AXES for seed in a.SEEDS]}


def test_frozen_family_requires_every_seed_and_axis():
    data = assignment()
    assert len(a.conditions_from(data)) == 8
    data['experiments'][-1]['seed'] = 43
    with pytest.raises(ValueError, match='seeds'):
        a.conditions_from(data)


def test_partial_report_never_loads_official_targets(tmp_path, monkeypatch):
    data = assignment()
    (tmp_path / 'assignment.global.json').write_text(json.dumps(data))
    runtime = {r['experiment']: {'state': 'blocked_source'} for r in data['experiments'] if r['axis'] == 8}
    (tmp_path / 'runtime.json').write_text(json.dumps(runtime))
    monkeypatch.setattr(a, 'load_module', lambda *args: pytest.fail('must not load scorer before fits'))
    result = a.analyze(tmp_path)
    assert result['status'] == 'provisional'
    assert result['counts'] == {'waiting': 7, 'blocked_source': 1}
    assert not result['final_adoption_claims_allowed']
    assert all(r['family_p'] == 1 and not r['gate']['passed'] for r in result['rows'])
    assert (tmp_path / 'analysis/report.md').exists()


def test_fixed_bh_eight_and_incomplete_gate():
    rows = [dict(axis=i, state='waiting') for i in a.AXES]
    rows[0].update(state='complete', p_centered_bootstrap=.001)
    a.adjust_family(rows)
    assert rows[0]['q_bh'] == pytest.approx(.008)
    assert all(row['family_p'] == 1 for row in rows[1:])
    baseline = {p: {m: {'mean': 100} for m in ('s', 'mae', 'rmse')} for p in ('total', 'f1', 'f2', 'f3')}
    summary = {p: {m: {'mean': 98} for m in ('s', 'mae', 'rmse')} for p in baseline}
    row = dict(state='complete', summary=summary, per_seed_delta_s=[-2]*3, ci95=[-3, -1], q_bh=.008)
    assert not a.stats.gate(row, baseline, False, 2000)['passed']
    assert a.stats.gate(row, baseline, True, 2000)['passed']
    row['per_seed_delta_s'] = [-3, -3, .1]
    assert not a.stats.gate(row, baseline, True, 2000)['passed']


def test_terminal_status_requires_real_receipts(tmp_path):
    recipe = assignment()['experiments'][0]
    state = {recipe['experiment']: {'state': 'trained_validated'}}
    assert a.run_status(tmp_path, recipe, state)['state'] == 'failed'
    out = tmp_path / 'runs' / recipe['experiment']
    out.mkdir(parents=True)
    (out / 'train.exit').write_text('1')
    assert a.run_status(tmp_path, recipe, {})['state'] == 'failed'


def test_additive_union_locked_across_seeds():
    data = assignment()
    data['experiments'][1]['config']['add_features'] = ['replacement']
    with pytest.raises(ValueError, match='union'):
        a.conditions_from(data)


def test_completed_axis_scoring_cache_and_seed_pairing(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    data = assignment()
    observation = tmp_path / 'obs'
    observation.write_text('fixture')
    (tmp_path / 'competition.py').write_text('# fixture')
    data['observations'] = str(observation)
    parents = {}
    for seed in a.SEEDS:
        path = tmp_path / 'parents' / 'runs' / f'parent_s{seed}'
        parents[str(seed)] = str(path)
    data['baseline_runs'] = parents
    runtime = {}
    paths = [(a.Path(p), seed, None) for seed, p in zip(a.SEEDS, parents.values())]
    for recipe in data['experiments'][:3]:
        runtime[recipe['experiment']] = {'state': 'trained_validated'}
        paths.append((tmp_path / 'runs' / recipe['experiment'], recipe['seed'], ['feature1']))
    for path, seed, additions in paths:
        path.mkdir(parents=True)
        (path / 'train.exit').write_text('0')
        (path / 'predictions.parquet').write_text('fixture')
        for name, value in [('local_score.json', {}), ('training-validation.json', {'validated': True}),
                            ('complete.json', {'publish': 'published' if additions is None else 'not_required'})]:
            (path / name).write_text(json.dumps(value))
        for period in ('F1', 'F2', 'F3'):
            (path / f'{period}_metadata.json').write_text(json.dumps(
                {'seed': seed, 'rounds': 900, 'config': {'add_features': additions}}))
    (tmp_path / 'assignment.global.json').write_text(json.dumps(data))
    (tmp_path / 'runtime.json').write_text(json.dumps(runtime))
    calls = []
    scorer = SimpleNamespace(load_observations=lambda p: None, prepare_context=lambda x: {})
    monkeypatch.setattr(a, 'load_module', lambda *args: scorer)

    def score(root, recipe, scorer, context):
        calls.append(recipe['experiment'])
        val = 100. if recipe['experiment'].startswith('parent') else 98.
        summary = {p: {m: val for m in a.stats.REPORT_METRICS} for p in a.stats.SCOPES}
        return np.full((2, 2), val), summary

    monkeypatch.setattr(a.stats, 'score_run', score)
    monkeypatch.setattr(a.stats, 'shared_bootstrap_weights', lambda *args: (None, 0))
    monkeypatch.setattr(a.stats, 'bootstrap_scores_shared', lambda *args: (
        {'s': np.array([100.]*3+[98.]*3)}, {'s': np.tile([100.]*3+[98.]*3, (2000, 1))}, 0))
    result = a.analyze(tmp_path)
    row = result['rows'][0]
    assert row['improved_seeds'] == 3
    assert row['per_seed_delta_s'] == [-2.]*3
    assert row['ci95'] == [-2., -2.]
    assert row['summary']['f1']['mae']['mean'] == 98.
    assert not row['gate']['passed']
    assert len(calls) == 6
    calls.clear()
    result = a.analyze(tmp_path)
    assert len(calls) == 3  # Only parents rescore; completed-axis values come from identity cache.
    assert result['rows'][0]['per_seed_delta_s'] == [-2.]*3
