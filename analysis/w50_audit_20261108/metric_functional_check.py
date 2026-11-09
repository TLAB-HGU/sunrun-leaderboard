"""Check metric-functional algebra using synthetic observations only.

No scientific dataset, saved forecast, fitted model or runtime state is read.
This reproduces the bounded mathematical check; it does not fit any predictor
or assess achievable forecasting performance.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def check():
    scorer_path = REPO / 'leaderboard/scorer/competition.py'
    kernel_path = REPO / 'research/post_ablation_five_axes_20261108/analysis_kernel.py'
    scorer = module(scorer_path, 'synthetic_math_scorer')
    kernel = module(kernel_path, 'synthetic_math_kernel')
    index = pd.date_range('2020-01-01', '2020-01-23', freq='h', tz='UTC')
    speed = 300 + 25 * ((np.arange(len(index)) // 5) % 15).astype(float)
    speed[100] = np.nan
    observations = pd.DataFrame({'timestamp_utc': index, 'speed_kms': speed})
    periods = {'F1': ('2020-01-02', '2020-01-09'),
               'F2': ('2020-01-09', '2020-01-16'),
               'F3': ('2020-01-16', '2020-01-23')}
    context = scorer.prepare_context(observations, periods)
    truth = context['truth']
    predicted = (np.where(np.isfinite(truth), truth, context['persistence'][:, None])
                 + np.linspace(-8, 9, len(truth))[:, None] + 3)

    def frame(values):
        return pd.DataFrame({
            'origin_last_input_utc': np.repeat(context['origins'], 72),
            'horizon_hours': np.tile(np.arange(1, 73), len(values)),
            'pred_kms': values.ravel(),
        })

    actual = scorer.score_competition(frame(predicted), context)
    if actual['total']['n_replaced'] != 0:
        raise AssertionError('Synthetic finite-gradient fixture used replacement forecasts')
    mse_weights, tail_weights = kernel.metric_weights(truth, np.arange(len(truth)))
    error = np.where(np.isfinite(truth), predicted - truth, 0.)
    replayed = kernel.scores_from_weights(error[None] ** 2, mse_weights, tail_weights)
    score_difference = max(abs(replayed[key][0] - actual['total'][key]) for key in kernel.METRICS)
    weight_sum_error = float(np.max(abs(tail_weights.sum(axis=0) - 1)))

    column = 0
    valid = np.flatnonzero(np.isfinite(truth[:, column]))
    count = len(valid)
    mse = np.sum(mse_weights[:, column] * error[:, column] ** 2)
    timse = np.sum(tail_weights[:, column] * error[:, column] ** 2)
    score = (mse * timse) ** .25
    analytic = .5 * score * error[:, column] * (
        1 / (count * mse) + tail_weights[:, column] / timse)
    epsilon = 1e-4
    differences = []
    checked_rows = valid[::max(1, len(valid) // 6)]
    for row in checked_rows:
        plus, minus = predicted.copy(), predicted.copy()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        finite_difference = (
            scorer.score_competition(frame(plus), context)['total']['by_lead'][column]['s']
            - scorer.score_competition(frame(minus), context)['total']['by_lead'][column]['s']
        ) / (2 * epsilon)
        differences.append(abs(finite_difference - analytic[row]))
    tied_weights_equal = all(
        np.ptp(tail_weights[truth[:, column] == value, column]) == 0
        for value in np.unique(truth[valid, column]))
    shared_scale = ((.25 * mse) * (.25 * timse)) ** .25 / score
    tail_only_scale = (mse * timse / 16) ** .25 / score

    # Finite probability space: w_i=n*a_i, a fixed two-group predictor, and
    # a fixed perturbation direction. No optimization or fitting is performed.
    group = np.arange(len(valid)) % 2
    target = truth[valid, column]
    weights = count * tail_weights[valid, column]
    predictor = np.array([410., 470.])
    direction = np.array([.3, -.7])
    residual = predictor[group] - target
    population_mse = np.mean(residual ** 2)
    population_timse = np.mean(weights * residual ** 2)
    population_value = (population_mse * population_timse) ** .25
    analytic_direction = .5 * population_value * np.mean(
        residual * direction[group] * (1 / population_mse + weights / population_timse))

    def population_score(values):
        residual = values[group] - target
        return (np.mean(residual ** 2) * np.mean(weights * residual ** 2)) ** .25

    finite_difference = (population_score(predictor + epsilon * direction)
                         - population_score(predictor - epsilon * direction)) / (2 * epsilon)
    directional_error = abs(finite_difference - analytic_direction)
    metrics = {
        'kernel_scorer_total_max_difference': float(score_difference),
        'tail_weight_sum_max_error': weight_sum_error,
        'per_lead_gradient_finite_difference_max_error': float(max(differences)),
        'tied_target_coefficients_equal': bool(tied_weights_equal),
        'shared_mse_timse_quarter_scale': float(shared_scale),
        'fixed_mse_timse_sixteenth_scale': float(tail_only_scale),
        'population_directional_derivative_error': float(directional_error),
    }
    checks = {
        'kernel_scorer_parity': score_difference <= 1e-10,
        'normalized_tail_weights': weight_sum_error <= 1e-12,
        'per_lead_gradient': max(differences) <= 1e-8,
        'equal_tied_target_weights': tied_weights_equal,
        'shared_scale_factor': abs(shared_scale - .5) <= 1e-12,
        'tail_only_scale_factor': abs(tail_only_scale - .5) <= 1e-12,
        'population_directional_derivative': directional_error <= 1e-8,
    }
    return {
        'status': 'pass' if all(checks.values()) else 'fail',
        'checked_utc': datetime.now(UTC).isoformat(),
        'scope': 'synthetic mathematical verification only; not scientific performance evidence',
        'scientific_data_or_outcomes_read': False,
        'model_fitting_performed': False,
        'runtime_state_modified': False,
        'sources': [str(scorer_path.relative_to(REPO)), str(kernel_path.relative_to(REPO))],
        'scorer_version': scorer.VERSION,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'fixture': {'periods': periods, 'hourly_observations': len(index),
                    'missing_observations': 1, 'scored_origins': len(truth), 'leads': 72,
                    'finite_difference_epsilon': epsilon,
                    'gradient_rows_checked': checked_rows.tolist()},
        'metrics': metrics,
        'checks': {name: bool(passed) for name, passed in checks.items()},
        'limits': [
            'Valid positive scored predictions with no fallback; raw clipping needs a chain rule.',
            'Per-lead derivative; official total-S derivative additionally carries 1/72.',
            'Population derivative checked on a fixed finite distribution, not a fitted optimum.',
            'No statement about achievable gains, generalization, or current experiment outcomes.',
        ],
    }


def main():
    result = check()
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (ROOT / 'metric_functional_check.json').write_text(serialized)
    print(serialized, end='')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
