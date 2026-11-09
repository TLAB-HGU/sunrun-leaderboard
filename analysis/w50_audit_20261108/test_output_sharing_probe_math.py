import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def math_core():
    spec = importlib.util.spec_from_file_location('output_sharing_math_test', Path(__file__).with_name('output_sharing_probe_math.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(**changes):
    return dict({'reg_lambda': 0., 'reg_alpha': 0., 'min_child_weight': 0., 'max_delta_step': 0.}, **changes)


def test_aligned_positive_cuts_zero_regret_despite_concentration(math_core):
    result = math_core.sharing_diagnostics([[90., 1.], [30., .5]], [True, True])
    assert result['regret']['raw'] == 0
    assert result['regret']['normalized'] == 0
    assert result['regret']['informative']
    assert result['shared_candidate'] == 0
    assert result['all_outputs_compatible_with_shared']
    assert result['selected_candidate_agreement_fraction'] == 1
    assert result['positive_largest_share'] == pytest.approx(90 / 91)
    assert not result['leave_one_out_winner_changed'].any()


def test_conflicting_gains_no_split_and_deterministic_ties(math_core):
    result = math_core.sharing_diagnostics([[9., 1.], [1., 9.]], [True, True])
    assert result['regret']['raw'] == 8
    assert result['regret']['normalized'] == pytest.approx(8 / 18)
    assert result['shared_candidate'] == 0
    np.testing.assert_array_equal(result['preferred_candidates'], [0, 1])
    np.testing.assert_array_equal(result['shared_signed_contributions'], [9, 1])
    np.testing.assert_array_equal(result['output_regrets_at_selected'], [0, 8])
    np.testing.assert_array_equal(result['leave_one_out_candidates'], [1, 0])
    assert result['selected_candidate_agreement_fraction'] == .5


def test_output_gradient_signs_do_not_cancel(math_core):
    result = math_core.candidate_gains([0, 0], [2, 2], [[2, -2]], [[1, 1]], **params())
    np.testing.assert_array_equal(result['gains'], [[8, 8]])
    assert result['common_feasible'].tolist() == [True]
    diagnostic = math_core.sharing_diagnostics(result['gains'], result['common_feasible'])
    assert diagnostic['regret']['shared_maximum'] == 16
    assert diagnostic['regret']['raw'] == 0


def test_float32_vector_weight_precedes_float64_score(math_core):
    node = math_core.node_weight_score([1.], [3.], **params())
    assert node['weights'].dtype == np.float32
    assert node['scores'].dtype == np.float64
    expected = -float(np.float32(-1 / 3))
    assert node['scores'][0] == expected
    assert node['scores'][0] != 1 / 3


def test_l1_l2_clipped_vector_rule_and_minimum_hessian(math_core):
    node = math_core.node_weight_score([10., -10., 2., 5., 5.], [1., 1., 1., .5, 0.],
                                       **params(reg_alpha=2, reg_lambda=3, min_child_weight=1, max_delta_step=1))
    np.testing.assert_array_equal(node['weights'], [-1., 1., 0., 0., 0.])
    np.testing.assert_array_equal(node['scores'], [8., 8., 0., 0., 0.])
    np.testing.assert_array_equal(node['eligible_outputs'], [True, True, True, False, False])
    # Generic scalar quadratic clipping gain (with its L1 term) would be 18, not vector 8.
    scalar_clipped_gain = -(2 * 10 * -1 + (1 + 3) * (-1) ** 2) + 2 * abs(-1)
    assert scalar_clipped_gain == 18 and node['scores'][0] != scalar_clipped_gain
    unbounded = math_core.node_weight_score([10.], [1.], **params(reg_alpha=2, reg_lambda=3))
    assert unbounded['weights'][0] == -2 and unbounded['scores'][0] == 16


def test_explicit_common_feasibility_retains_infeasible_gains(math_core):
    result = math_core.candidate_gains([4., 4.], [4., 4.], [[1., 1.], [2., 2.]],
                                      [[.5, 2.], [2., 2.]], **params(min_child_weight=1))
    np.testing.assert_array_equal(result['child_hessian_feasible_by_output'], [[False, True], [True, True]])
    np.testing.assert_array_equal(result['common_feasible'], [False, True])
    assert not np.all(result['gains'][0] == 0)
    result = math_core.sharing_diagnostics([[100, 100], [2, 1]], [False, True])
    assert result['shared_candidate'] == 1 and result['infeasible_candidate_count'] == 1
    zero = math_core.candidate_gains([4.], [2.], [[1.]], [[0.]], **params())
    assert not zero['common_feasible'][0]
    assert zero['gains'][0, 0] == -3.5


@pytest.mark.parametrize('gains,mask', [([[-3., -2.], [-1., -9.]], [True, True]),
                                       ([[0., 0.]], [True]), ([[90., 90.]], [False]),
                                       (np.empty((0, 2)), np.empty(0, bool))])
def test_no_positive_gain_is_uninformative(math_core, gains, mask):
    result = math_core.sharing_diagnostics(gains, mask)
    assert result['shared_candidate'] == -1
    assert result['regret']['raw'] == 0
    assert result['regret']['normalized'] is None
    assert not result['regret']['informative']
    assert result['positive_concentration_hhi'] is None


def test_tolerance_tie_raw_maxima_and_negative_regret_guard(math_core):
    tiny = math_core.sharing_diagnostics([[1e-10, 0.]], [True])
    assert tiny['shared_candidate'] == -1
    assert tiny['regret']['raw'] == 0  # Exact maxima, not zero-tie representative, define R.
    assert tiny['shared_selection_gap'] == 1e-10
    assert not tiny['regret']['informative']
    almost = math_core.sharing_regret([1.], 1. + 1e-10)
    assert almost['raw'] < 0 and almost['nonnegative'] == 0
    assert almost['normalized_raw'] < 0 and almost['normalized'] == 0
    assert almost['tolerance'] == pytest.approx(1e-9 + 1e-8)
    with pytest.raises(ValueError, match='Negative sharing regret'):
        math_core.sharing_regret([1.], 1.001)


@pytest.mark.parametrize('gradient,hessian', [([1.], [-1.]), ([np.nan], [1.]), ([1.], [np.inf]),
                                            ([[1., 2.]], [1., 2.]), ([], []), (1., 2.)])
def test_statistics_reject_invalid_shapes_and_values(math_core, gradient, hessian):
    with pytest.raises(ValueError):
        math_core.node_weight_score(gradient, hessian, **params())


@pytest.mark.parametrize('parameter,value', [('reg_lambda', -1), ('reg_alpha', np.inf),
                                            ('min_child_weight', np.nan), ('max_delta_step', -1)])
def test_parameters_reject_invalid_values(math_core, parameter, value):
    with pytest.raises(ValueError):
        math_core.node_weight_score([1.], [1.], **params(**{parameter: value}))


def test_candidate_and_diagnostic_shape_finiteness_guards(math_core):
    with pytest.raises(ValueError, match='dimensions'):
        math_core.candidate_gains([1, 2], [1, 2], [[1]], [[1]], **params())
    with pytest.raises(ValueError, match='nonnegative Hessians'):
        math_core.candidate_gains([1], [1], [[0]], [[2]], **params())
    for gains, mask in [([[np.inf]], [False]), ([[np.nan]], [True]), ([[1]], [1]), ([[1]], [True, False]), ([], [])]:
        with pytest.raises(ValueError):
            math_core.sharing_diagnostics(gains, mask)
    with pytest.raises(ValueError, match='finite precision'):
        math_core.node_weight_score([1e300], [1.], **params())
    with pytest.raises(ValueError, match='nonfinite'):
        math_core.sharing_diagnostics([[1e308, 1e308]], [True])
    for maxima, shared in [([np.nan], 0), ([-1], 0), ([1], np.inf), ([], 0)]:
        with pytest.raises(ValueError):
            math_core.sharing_regret(maxima, shared)


def test_signed_sacrifice_is_separate_from_positive_gain_concentration(math_core):
    result = math_core.sharing_diagnostics([[12., -2.], [0., 5.]], [True, True])
    assert result['shared_candidate'] == 0
    assert result['regret']['raw'] == 7
    np.testing.assert_array_equal(result['shared_signed_contributions'], [12., -2.])
    np.testing.assert_array_equal(result['positive_contribution_shares'], [1., 0.])
    assert result['negative_contribution_mass'] == -2
    assert result['positive_concentration_hhi'] == 1
    np.testing.assert_array_equal(result['leave_one_out_candidates'], [1, 0])
