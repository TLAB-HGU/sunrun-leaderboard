"""Independent numerical checks for the ablation reporting kernel."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze_results as analysis


@pytest.fixture
def scoring():
    path = Path(__file__).parent / 'competition.py'
    if not path.exists():
        path = Path(__file__).parents[2] / 'leaderboard/scorer/competition.py'
    spec = importlib.util.spec_from_file_location('fixture_competition', path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.PERIODS = {'F1': ('2026-01-02', '2026-01-20'),
                      'F2': ('2026-01-21', '2026-02-08'),
                      'F3': ('2026-02-09', '2026-03-03')}
    rng = np.random.default_rng(8)
    times = pd.date_range('2026-01-01', '2026-03-05', freq='h', tz='UTC')
    observations = pd.DataFrame({'timestamp_utc': times, 'speed_kms': 400 + rng.normal(0, 40, len(times))})
    observations.loc[::13, 'speed_kms'] = np.nan
    context = scorer.prepare_context(observations)
    prediction = 400 + rng.normal(0, 30, context['truth'].shape)
    return scorer, observations, context, prediction


def frame(context, prediction):
    return pd.DataFrame({'origin_last_input_utc': np.repeat(context['origins'], 72),
                         'horizon_hours': np.tile(np.arange(1, 73), len(prediction)),
                         'pred_kms': prediction.ravel()})


def test_metric_kernel_matches_official_with_missing_truth(scoring):
    scorer, _, context, prediction = scoring
    score = scorer.score_competition(frame(context, prediction), context)
    exact, bootstrap, rejected = analysis.bootstrap_scores(context, np.stack([prediction, prediction + 5]), 7, 168)
    for metric in analysis.METRICS:
        np.testing.assert_allclose(exact[metric][0], score['total'][metric], rtol=1e-12)
        assert bootstrap[metric].shape == (7, 2)
    assert rejected == 0
    assert analysis.diagnostics(context['truth'], prediction, prediction + 5)['all']['identity_max_abs_error'] < 1e-8


def test_duplicate_resample_recomputes_empirical_timse(scoring):
    _, _, context, prediction = scoring
    ix = np.array([0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    truth = context['truth']
    error = np.where(np.isfinite(truth), (prediction - truth) ** 2, 0.)
    value = analysis.scores_from_weights(error[None], *analysis.metric_weights(truth, ix))
    expected = []
    for h in range(72):
        y = truth[ix, h]
        valid = np.isfinite(y)
        e, y = prediction[ix, h][valid] - y[valid], y[valid]
        q = np.quantile(y, np.arange(101) / 100)
        timse = sum((q[i + 1] - q[i]) * np.mean(e[y >= q[i]] ** 2) for i in range(100)) / (q[-1] - q[0])
        expected.append(np.sqrt(np.sqrt(np.mean(e ** 2)) * np.sqrt(timse)))
    np.testing.assert_allclose(value['s'][0], np.mean(expected), rtol=1e-12)


def test_bh_preserves_input_order_and_bounds():
    np.testing.assert_allclose(analysis.bh_adjust([.04, .01, .03]), [.04, .03, .04])
    np.testing.assert_allclose(analysis.bh_adjust([0., 1.]), [0., 1.])


def test_partial_zero_bootstrap_and_feature_grouping(tmp_path, monkeypatch, scoring):
    scorer, observations, context, prediction = scoring
    monkeypatch.setattr(analysis, 'scorer_at', lambda root: scorer)
    (tmp_path / 'competition.py').write_text('# scorer fixture identity\n')
    observations.to_parquet(tmp_path / 'observations.parquet')
    recipes = []
    for seed in (42, 43, 44):
        for name, category, shift in [('baseline', 'baseline', 0), ('feature_v', 'feature', 5), ('group_v', 'group', 5)]:
            recipe = dict(experiment=f'{name}_s{seed}', stage='baseline' if category == 'baseline' else 'ablation',
                          category=category, config=dict(seed=seed, exclude_features=[] if category == 'baseline' else ['v_lag0']))
            recipes.append(recipe)
            out = tmp_path / 'runs' / recipe['experiment']
            out.mkdir(parents=True)
            (out / 'train.exit').write_text('0\n')
            (out / 'complete.json').write_text(json.dumps({'publish': 'not_required'}))
            for label, offset in [('full', 1), ('no_offset', 0)]:
                pred = frame(context, prediction + shift + offset + seed - 42)
                pred.to_parquet(out / ('predictions.parquet' if label == 'full' else 'no_offset_predictions.parquet'))
                score = scorer.score_competition(pred, context)
                (out / ('local_score.json' if label == 'full' else 'paired_no_offset_score.json')).write_text(json.dumps({'competition': score}))
    assignment = dict(experiments=recipes, observations='observations.parquet')
    (tmp_path / 'assignment.json').write_text(json.dumps(assignment))
    report = analysis.analyze(tmp_path, count=0)
    assert report['status'] == 'complete'
    feature_rows = [r for r in report['contrasts'] if r['single_feature_removal']]
    assert len(feature_rows) == 1
    assert feature_rows[0]['variant'] == 'feature_v'
    assert feature_rows[0]['seeds'] == [42, 43, 44]
    assert feature_rows[0]['metrics']['s']['ci95'] is None
    text = (tmp_path / 'analysis/report.md').read_text()
    for section in ['성능 원인 판정', '기준선 성능', '일관되게 지지되는 구성의 기여',
                    '해로운 구성 후보와 불확실한 중복성', '핵심 손실·offset 상호작용과 기간 집중',
                    'offset이 오차를 바꾼 방식']:
        assert section in text
    assert '현재는 중간 진단입니다' in text
    assert '원인을 확정하지 않습니다' in text
    assignment['experiments'].append(dict(experiment='missing_s42', config=dict(seed=42)))
    (tmp_path / 'assignment.json').write_text(json.dumps(assignment))
    with pytest.raises(ValueError, match='incomplete'):
        analysis.analyze(tmp_path, count=0)
    assert analysis.analyze(tmp_path, count=0, partial=True)['status'] == 'provisional'


def test_validation_gate_requires_finished_supervised_run(tmp_path):
    name = 'candidate_s42'
    assert not analysis.validation_gate(tmp_path, name, {})[0]
    (tmp_path / 'train.exit').write_text('0\n')
    assert not analysis.validation_gate(tmp_path, name, {})[0]
    assert analysis.validation_gate(tmp_path, name, {name: {'state': 'trained_validated'}})[0]
    (tmp_path / 'complete.json').write_text(json.dumps({'publish': 'not_required'}))
    assert analysis.validation_gate(tmp_path, name, {})[0]
    assert not analysis.validation_gate(tmp_path, name, {name: {'state': 'failed'}})[0]
    (tmp_path / 'train.exit').write_text('1\n')
    assert not analysis.validation_gate(tmp_path, name, {})[0]


def test_partial_ignores_failed_artifact_even_when_files_exist(tmp_path, scoring):
    scorer, _, context, _ = scoring
    name = 'broken_s42'
    out = tmp_path / 'runs' / name
    out.mkdir(parents=True)
    (out / 'train.exit').write_text('0\n')
    (out / 'complete.json').write_text(json.dumps({'publish': 'not_required'}))
    for file in ['predictions.parquet', 'no_offset_predictions.parquet', 'local_score.json', 'paired_no_offset_score.json']:
        (out / file).write_text('invalid artifact')
    assignment = {'experiments': [{'experiment': name}]}
    (tmp_path / 'scheduler-state.json').write_text(json.dumps({name: {'state': 'failed'}}))
    runs, missing = analysis.load_runs(tmp_path, assignment, scorer, context, partial=True)
    assert runs == []
    assert missing == {name: 'failed'}
    # Even a stale completion marker must not make a corrupted arm stop a partial report.
    (tmp_path / 'scheduler-state.json').write_text('{}')
    runs, missing = analysis.load_runs(tmp_path, assignment, scorer, context, partial=True)
    assert runs == []
    assert name in missing


def test_narrative_requires_seed_ci_and_bh_agreement():
    def row(name, mean, ci, values, q):
        return dict(kind='versus_baseline', variant=name, seeds=[42, 43, 44],
                    single_feature_removal=True,
                    metrics={'s': dict(mean_delta=mean, ci95=ci, per_seed=values, bh_q=q)},
                    period_deltas={p: {'s': mean} for p in ('f1', 'f2', 'f3')})
    report = dict(status='complete', bootstrap={'replicates': 2000},
                  contrasts=[row('needed_feature', 2., [1., 3.], [1., 2., 3.], .01),
                             row('harmful_feature', -2., [-3., -1.], [-1., -2., -3.], .01),
                             row('uncertain_feature', 2., [1., 3.], [1., 2., 3.], .2)],
                  factorial_interactions=[], diagnostics={}, clipping={})
    text = '\n'.join(analysis.narrative(report, []))
    assert '가장 큰 조건부 기여는 `needed_feature`' in text
    assert '`harmful_feature`' in text and '자동 채택하지 않습니다' in text
    assert '미확정: `uncertain_feature`' in text


def decision_row(name, category='feature', values=(.2, .3, .4), ci=(.1, .5), p=.001):
    return dict(kind='versus_baseline', variant=name, category=category, seeds=[42, 43, 44],
                metrics={'s': dict(per_seed=list(values), mean_delta=float(np.mean(values)),
                                  ci95=list(ci), p_centered_bootstrap=p)})


def test_decisions_require_complete_evidence_and_strict_thresholds():
    row = decision_row('drop_feature')
    analysis.adjust_families([row], 2000)
    assert analysis.classify_rows([row], False)['drop_feature']['state'] == 'pending'
    assert analysis.classify_rows([row], True)['drop_feature']['supported']
    for field, value in [('mean_delta', .099), ('bh_q', .05), ('ci95', [0., .5]),
                         ('per_seed', [0., .3, .4])]:
        changed = json.loads(json.dumps(row))
        changed['metrics']['s'][field] = value
        assert not analysis.contribution_supported(changed)
    row['seeds'] = [42, 43]
    assert not analysis.contribution_supported(row)


def test_bh_families_are_separate():
    rows = [decision_row('feature', p=.04), decision_row('group', 'group', p=.001),
            decision_row('component1', 'loss', p=.02), decision_row('component2', 'mechanism', p=.8)]
    analysis.adjust_families(rows, 2000)
    assert [r['multiplicity_family'] for r in rows] == ['feature', 'group', 'other', 'other']
    np.testing.assert_allclose([r['metrics']['s']['bh_q'] for r in rows], [.04, .001, .04, .8])


def manifest_fixture():
    features = [f'f{i}' for i in range(30)]
    recipes = [dict(experiment='dropgroup_test_s42', variant='dropgroup_test', category='group',
                    config={'exclude_features': features[:3]})]
    components = ('loss_t1_sp1_offset', 'loss_t4_sp0_offset', 'loss_t4_sp1_nooffset',
                  'clip_component_off', 'clip_final_off')
    decisions = {n: dict(supported=False) for n in components}
    decisions['dropgroup_test'] = dict(supported=True)
    return dict(groups={'test': features[:3], 'other': features[3:]}, experiments=recipes), dict(decisions=decisions)


def test_manifest_preserves_proven_bundle_and_freezes_once(tmp_path):
    assignment, report = manifest_fixture()
    manifest = analysis.freeze_removal_manifest(tmp_path, assignment, report)
    assert manifest['status'] == 'frozen'
    assert manifest['retained_features'] == ['f0', 'f1', 'f2']
    assert manifest['individually_supported_features'] == []
    assert manifest['final_config']['exclude_features'] == [f'f{i}' for i in range(3, 30)]
    assert manifest['final_config']['tail_weight'] == 1
    assert not manifest['final_config']['spectral']
    assert [r['seed'] for r in manifest['final_experiments']] == [42, 43, 44]
    assert analysis.freeze_removal_manifest(tmp_path, assignment, report) == manifest
    report['decisions']['dropgroup_test']['supported'] = False
    with pytest.raises(ValueError, match='Refusing to revise'):
        analysis.freeze_removal_manifest(tmp_path, assignment, report)


def test_empty_retained_set_blocks_final_fit(tmp_path):
    assignment, report = manifest_fixture()
    report['decisions']['dropgroup_test']['supported'] = False
    manifest = analysis.freeze_removal_manifest(tmp_path, assignment, report)
    assert manifest['status'] == 'blocked_no_retained_features'
    assert manifest['final_experiments'] == []


def test_hessian_diagnostic_does_not_override_factorial_cells():
    recipes = []
    for seed in (42, 43, 44):
        for weight in (1, 4):
            for spectral in (False, True):
                recipes.append(dict(recipe=dict(experiment=f'loss_{weight}_{spectral}_s{seed}',
                    stage='baseline' if weight == 4 and spectral else 'ablation',
                    config=dict(seed=seed, tail_weight=weight, spectral=spectral))))
    _, expected = analysis.make_contrasts(recipes)
    recipes.append(dict(recipe=dict(experiment='hessian_s42', stage='ablation',
                                    config=dict(seed=42, spectral_hessian_only=True))))
    _, actual = analysis.make_contrasts(recipes)
    assert actual == expected
