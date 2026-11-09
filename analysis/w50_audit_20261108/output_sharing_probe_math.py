"""NumPy-only XGBoost 2.1.4 vector-score/common-candidate diagnostics.

Pinned dmlc/xgboost 62e7923619352c4079b24303b367134486b1c84f,
param.h:248-257,299-319. G/H accumulations are float64; unshrunk vector
weights are stored float32 before computing -u*ThresholdL1(G) in float64.
Tree reg_alpha is L1 regularization, unrelated to calibration offset alpha.

This implements the declared stricter common child-Hessian set, not historical
vector/scalar split replay. No histogram generation, gamma/depth/leaf gating,
learning-rate shrinkage, sampling, fitted model or generalization claim occurs.
Candidate rows must already follow the caller's frozen feature/cut/missing order.
"""
import numpy as np


def _parameter(value, name):
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
        raise ValueError(f'{name} must be representable as finite nonnegative float32')
    return float(np.float32(value))


def _statistics(gradient, hessian):
    g, h = np.asarray(gradient, dtype=np.float64), np.asarray(hessian, dtype=np.float64)
    if g.ndim not in (1, 2) or g.shape != h.shape or not g.shape[-1]:
        raise ValueError('Gradient/Hessian require equal vector or candidate-by-output shapes')
    if not np.isfinite(g).all() or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError('Statistics must be finite with nonnegative Hessians')
    return g, h


def node_weight_score(gradient, hessian, *, reg_lambda, reg_alpha, min_child_weight, max_delta_step):
    """Return per-output float32 unshrunk weights, float64 scores, eligibility.

    Accept a vector or candidate-by-output matrix; no summing across outputs.
    Zero/low Hessians yield the pinned zero weight, explicitly marked ineligible.
    """
    g, h = _statistics(gradient, hessian)
    l2 = _parameter(reg_lambda, 'reg_lambda')
    l1 = _parameter(reg_alpha, 'reg_alpha')
    minimum = _parameter(min_child_weight, 'min_child_weight')
    limit = _parameter(max_delta_step, 'max_delta_step')
    eligible = (h > 0) & (h >= minimum)
    thresholded = np.sign(g) * np.maximum(np.abs(g) - l1, 0.)
    weights = np.zeros_like(g)
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            weights[eligible] = -thresholded[eligible] / (h[eligible] + l2)
            if limit > 0:
                weights = np.clip(weights, -limit, limit)
            weights = weights.astype(np.float32)
            scores = -weights.astype(np.float64) * thresholded
    except FloatingPointError as error:
        raise ValueError('Node weight/score arithmetic exceeded finite precision') from error
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite node scores')
    return {'weights': weights, 'scores': scores, 'eligible_outputs': eligible}


def candidate_gains(parent_gradient, parent_hessian, left_gradient, left_hessian, *,
                    reg_lambda, reg_alpha, min_child_weight, max_delta_step):
    """Per-candidate output gains plus explicit common-feasibility masks.

    Parent is (outputs,), left is (candidates, outputs); right = parent-left.
    Every raw gain is retained even for infeasible candidates. Do not zero or
    reinterpret those candidates: pass common_feasible to sharing_diagnostics.
    A zero-Hessian *entire child* is excluded even when min_child_weight is zero.
    """
    pg, ph = _statistics(parent_gradient, parent_hessian)
    lg, lh = _statistics(left_gradient, left_hessian)
    if pg.ndim != 1 or lg.ndim != 2 or lg.shape[1] != len(pg):
        raise ValueError('Parent vector and candidate output dimensions differ')
    try:
        with np.errstate(over='raise', invalid='raise'):
            rg, rh = pg[None, :] - lg, ph[None, :] - lh
    except FloatingPointError as error:
        raise ValueError('Right-child statistics exceeded finite precision') from error
    _statistics(rg, rh)
    params = {'reg_lambda': reg_lambda, 'reg_alpha': reg_alpha,
              'min_child_weight': min_child_weight, 'max_delta_step': max_delta_step}
    parent = node_weight_score(pg, ph, **params)
    left, right = node_weight_score(lg, lh, **params), node_weight_score(rg, rh, **params)
    minimum = _parameter(min_child_weight, 'min_child_weight')
    child_feasible = (lh >= minimum) & (rh >= minimum)
    nonzero_children = np.any(lh > 0, axis=1) & np.any(rh > 0, axis=1)
    common = child_feasible.all(axis=1) & nonzero_children
    try:
        with np.errstate(over='raise', invalid='raise'):
            gains = left['scores'] + right['scores'] - parent['scores'][None, :]
    except FloatingPointError as error:
        raise ValueError('Candidate gain arithmetic exceeded finite precision') from error
    return {'gains': gains, 'common_feasible': common,
            'child_hessian_feasible_by_output': child_feasible,
            'nonzero_hessian_children': nonzero_children,
            'parent': parent, 'left': left, 'right': right}


def sharing_regret(individual_maxima, shared_maximum):
    """Validate exact-maxima regret; retain raw values before tolerance clamping."""
    maxima = np.asarray(individual_maxima, dtype=np.float64)
    if (maxima.ndim != 1 or not maxima.size or not np.isfinite(maxima).all()
            or (maxima < 0).any() or np.ndim(shared_maximum) != 0):
        raise ValueError('Regret requires finite nonnegative per-output maxima and scalar shared maximum')
    shared = float(shared_maximum)
    if not np.isfinite(shared) or shared < 0:
        raise ValueError('Shared maximum must be finite and nonnegative (no split included)')
    with np.errstate(over='ignore'):
        available = float(maxima.sum(dtype=np.float64))
    if not np.isfinite(available):
        raise ValueError('Sum of output maxima is nonfinite')
    tolerance = 1e-9 + 1e-8 * max(1., available)
    raw = available - shared
    if raw < -tolerance:
        raise ValueError('Negative sharing regret exceeds the predefined numerical tolerance')
    return {'individual_maximum_sum': available, 'shared_maximum': shared,
            'raw': raw, 'nonnegative': max(0., raw), 'tolerance': tolerance,
            'normalized_raw': raw / available if available > 0 else None,
            'normalized': max(0., raw) / available if available > 0 else None,
            'informative': available > tolerance}


def _winner(values, tolerance):
    # Row zero is no split; remaining rows preserve the caller's frozen order.
    return int(np.flatnonzero(values >= values.max() - tolerance)[0])


def sharing_diagnostics(gains, common_feasible):
    """Compare shared/individual choices on exactly one common candidate set.

    Candidate IDs are original input row indices; -1 always means no split.
    Exact maxima define regret. A tolerance-aware deterministic representative
    may be slightly below the exact maximum; its selection gap and signed output
    contributions are reported separately rather than silently changing R.
    Concentration uses positive shared-winner contributions, with negative mass
    separately reported. It is not evidence of incompatible preferred cuts.
    """
    gains = np.asarray(gains, dtype=np.float64)
    feasible = np.asarray(common_feasible)
    if gains.ndim != 2 or gains.shape[1] == 0 or not np.isfinite(gains).all():
        raise ValueError('Gains must be a finite candidate-by-output matrix with outputs')
    if feasible.dtype != np.bool_ or feasible.shape != (len(gains),):
        raise ValueError('Common feasibility must be an explicit aligned boolean candidate mask')
    ids = np.r_[-1, np.flatnonzero(feasible)]
    matrix = np.vstack([np.zeros((1, gains.shape[1])), gains[feasible]])
    with np.errstate(over='ignore', invalid='ignore'):
        totals = matrix.sum(axis=1, dtype=np.float64)
    if not np.isfinite(totals).all():
        raise ValueError('Summed candidate gain is nonfinite')
    maxima = matrix.max(axis=0)
    regret = sharing_regret(maxima, totals.max())
    tolerance = regret['tolerance']
    winner = _winner(totals, tolerance)
    preferred = np.array([_winner(matrix[:, j], tolerance) for j in range(matrix.shape[1])])
    signed = matrix[winner].copy()
    output_regrets = maxima - signed
    compatible = output_regrets <= tolerance
    positive = np.maximum(signed, 0.)
    positive_mass = float(positive.sum())
    if not np.isfinite(positive_mass):
        raise ValueError('Positive contribution mass is nonfinite')
    shares = positive / positive_mass if positive_mass > 0 else np.zeros_like(positive)
    loo_winners, loo_maxima = [], []
    for j in range(matrix.shape[1]):
        without = np.delete(matrix, j, axis=1).sum(axis=1, dtype=np.float64)
        if not np.isfinite(without).all():
            raise ValueError('Leave-one-output-out gain is nonfinite')
        loo_winners.append(int(ids[_winner(without, tolerance)]))
        loo_maxima.append(float(without.max()))
    return {'regret': regret, 'feasible_candidate_count': int(feasible.sum()),
            'infeasible_candidate_count': int((~feasible).sum()), 'no_split_candidate': -1,
            'shared_candidate': int(ids[winner]), 'shared_selected_gain': float(totals[winner]),
            'shared_selection_gap': float(totals.max() - totals[winner]),
            'individual_maxima': maxima, 'preferred_candidates': ids[preferred],
            'shared_signed_contributions': signed, 'output_regrets_at_selected': output_regrets,
            'selected_candidate_agreement_fraction': float(np.mean(preferred == winner)),
            'shared_is_output_maximum_within_tolerance': compatible,
            'all_outputs_compatible_with_shared': bool(compatible.all()),
            'positive_contribution_shares': shares, 'positive_contribution_mass': positive_mass,
            'negative_contribution_mass': float(np.minimum(signed, 0.).sum()),
            'positive_concentration_hhi': float(np.sum(shares ** 2)) if positive_mass > 0 else None,
            'positive_largest_share': float(shares.max()) if positive_mass > 0 else None,
            'leave_one_out_candidates': np.asarray(loo_winners, dtype=int),
            'leave_one_out_maximum_gains': np.asarray(loo_maxima),
            'leave_one_out_winner_changed': np.asarray(loo_winners) != ids[winner]}
