"""Accounting must agree with the independent official scorer, including gaps/ties."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('current_diagnostic', Path(__file__).with_name('diagnose.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def fixture():
    index = pd.date_range('2025-12-31', '2026-01-21', freq='h', tz='UTC')
    # Repeated values at exact requested speed boundaries, and nonuniform gaps.
    y = np.resize([350., 400., 550., 700., 850., 400., 550.], len(index))
    y[np.arange(len(y)) % 11 == 0] = np.nan
    obs = pd.DataFrame({'timestamp_utc': index, 'speed_kms': y})
    periods = {f'F{i+1}': (f'2026-01-{1+7*i:02d}', f'2026-01-{8+7*i:02d}') for i in range(3)}
    context = audit.official.prepare_context(obs, periods=periods)
    pred = 480 + np.arange(len(context['origins']))[:, None] + np.arange(72)[None, :] / 10
    score = audit.official.score_competition(audit.from_array(context['origins'], pred), context)
    return context, pred, score


def test_energy_weights_match_official_all_scopes_with_ties_and_gaps(fixture):
    context, pred, score = fixture
    for name, population in context['scopes'].items():
        truth = context['truth'][population['selected']]
        sq = np.where(np.isfinite(truth), (pred[population['selected']] - truth)**2, 0.)
        mw, tw = audit.kernel.metric_weights(truth, np.arange(len(truth)))
        result = audit.kernel.scores_from_weights(sq[None], mw, tw)
        for metric in result:
            np.testing.assert_allclose(result[metric][0], score[name][metric], atol=1e-8, rtol=0)
        assert np.all(mw[~np.isfinite(truth)] == 0)
        assert np.all(tw[~np.isfinite(truth)] == 0)


def test_exhaustive_partitions_and_unique_target_accounting(fixture):
    context, pred, score = fixture
    result, table = audit.analyze(pred, context, score)
    for prefix in ('speed_', 'lead_'):
        groups = [v for k, v in result['groups'].items() if k.startswith(prefix)]
        assert sum(v['n_pairs'] for v in groups) == score['total']['n_pairs']
        for key in ('mse_fraction', 'timse_fraction'):
            np.testing.assert_allclose(sum(v[key] for v in groups), 1, atol=1e-12)
        assert all(0 <= v['restricted_error_removal_s'] <= score['total']['s'] for v in groups)
    assert table.forecast_pairs.sum() == score['total']['n_pairs']
    assert table.index.is_unique
    np.testing.assert_allclose(table.timse_contribution.sum(), score['total']['timse'], atol=1e-8)


def test_empty_stratum_is_explicit_and_missing_truth_has_no_energy(fixture):
    context, pred, _ = fixture
    truth = context['truth']
    weights = audit.kernel.metric_weights(truth, np.arange(len(truth)))
    empty = audit.decomposition(pred, truth, *weights, np.zeros_like(truth, bool))
    assert empty['n_pairs'] == 0
    assert empty['mae'] is None
    assert empty['mse_contribution'] == empty['timse_contribution'] == 0
    missing = audit.decomposition(pred, truth, *weights, ~np.isfinite(truth))
    assert missing['n_pairs'] == 0
