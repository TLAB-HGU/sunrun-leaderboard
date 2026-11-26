"""E1: fixed depth3 A900 with depth5 S-cap proposals (CPU only)."""
from __future__ import annotations

import json
import time
import os
import sys
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import xgboost as xgb
from _module import calibration, continuation, gbm, objectives
from _module.table import load_grid  # noqa: E402

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_proposal_d5'
BASELINE_WEIGHTS = HERE.parent / 'e2000fs_calib_offset_roll_scap' / 'weights'
PARENT_CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
                   'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
                   'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
                   'suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area',
                   'suvi_t035_meridian30_equatorial_area', 'suvi_t055_meridian15_area',
                   'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area', 'img_age_h',
                   'enlil_n_pending', 'enlil_kp']
ADDED_CONSTANT = ['v_lag24', 'v_lag48', 'v_lag72']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark',
           'strip_v750_logmed']
SPECTRAL_COEFFICIENT = .1
OBJECTIVE = 'direct_s_spectral_continuation_mse_cap'
MSE_CAP = 1.01
CONTINUATION = continuation.Continuation(OBJECTIVE, spectral_coefficient=SPECTRAL_COEFFICIENT, mse_cap=MSE_CAP, proposal_max_depth=5)


def prepare_period(table, period):
    train, filled = load_grid(table, period, True, PARENT_CONSTANT + ADDED_CONSTANT, VARYING)
    rows = table.loc[table[f'fit_{period}'], ['origin', 'h', 'y_observed']].sort_values(['origin', 'h'], kind='stable')
    observed = rows.y_observed.to_numpy(np.float64).reshape(filled.shape)
    mask = np.isfinite(observed)
    if not np.array_equal(observed[mask], filled[mask]):
        raise ValueError('Original observed training labels differ from filled labels')
    prior, _, _ = calibration.prior_split(train['origins'])
    return dict(period=period, filled=filled, observed=observed, prior=prior, calls=0)


def actual_depth(model):
    return int(json.loads(model.save_config())['learner']['gradient_booster']['tree_train_param']['max_depth'])


def observed_bins(prediction, observed):
    result = {}
    for name, low, high in [('<400', -np.inf, 400), ('400-550', 400, 550), ('550-700', 550, 700), ('>=700', 700, np.inf)]:
        mask = np.isfinite(observed) & (observed >= low) & (observed < high)
        error = (prediction - observed)[mask]
        result[name] = dict(count=int(mask.sum()), rmse=float(np.sqrt(np.mean(error ** 2))) if error.size else None,
                            bias=float(error.mean()) if error.size else None)
    return result


def stage_diagnostics(observed, *, stage, loss, base, direction, alpha, budget, model, decision):
    accepted = base + alpha * direction
    error = base - loss.target
    bounds = [continuation.feasible_upper(error[:, j:j+1], direction[:, j:j+1], budget[j:j+1]) for j in range(base.shape[1])]
    change = (accepted - loss.target) ** 2 - error ** 2
    mask = np.isfinite(observed)
    observed_change = np.mean(np.where(mask, change, 0.), axis=0)
    imputed_change = np.mean(np.where(~mask, change, 0.), axis=0)
    total = change.mean(axis=0)
    np.testing.assert_allclose(observed_change + imputed_change, total, rtol=1e-10, atol=1e-9)
    unconstrained, _ = CONTINUATION.choose_step(loss, base, direction, None)
    depth = actual_depth(model)
    if depth != 5:
        raise ValueError('Proposal actual max_depth is not 5')
    return dict(stage=stage, alpha=alpha, proposal_max_depth=depth,
                proposal_rms=float(np.sqrt(np.mean(direction ** 2))), accepted_rms=float(np.sqrt(np.mean((alpha * direction) ** 2))),
                directional_derivative=float(np.sum(loss.gradient(base) * direction)), feasible_bounds=bounds,
                binding_lead=int(np.argmin(bounds)) if min(bounds) < 1. else None, bounds_domain='[0,1]; block-relative zero-based binding lead',
                unconstrained_alpha=unconstrained, decision=dict(decision), before=observed_bins(base, observed),
                proposal=observed_bins(base + direction, observed), accepted=observed_bins(accepted, observed),
                cap_consumption_observed=observed_change.tolist(), cap_consumption_imputed=imputed_change.tolist(),
                cap_consumption_total=total.tolist(), spectral_before=loss.spectral_value_gradient(error)[0],
                spectral_accepted=loss.spectral_value_gradient(accepted-loss.target)[0])


def fit_block(spec, x, xq, y, seed, threads, rounds, context, smoke):
    call = context['calls']
    if call >= 6:
        raise ValueError('Unexpected prior/full block call order')
    stage, block = ('prior' if call < 3 else 'full'), call % 3
    indices = context['prior'] if stage == 'prior' else np.arange(len(context['filled']))
    if smoke:
        indices = indices[np.linspace(0, len(indices)-1, min(len(indices), 128 if stage == 'prior' else 192)).astype(int)]
    columns = slice(block * spec.block, (block+1) * spec.block)
    if not np.array_equal(y, context['filled'][indices, columns]):
        raise ValueError('Diagnostic support does not match fitted training labels')
    observed = context['observed'][indices, columns]
    source = BASELINE_WEIGHTS / context['period'] / f's{seed}.npz'
    if smoke:
        matrix = xgb.DMatrix(x, label=y.astype(np.float32), nthread=threads)
        params = dict(gbm.PARAMS, seed=seed, device='cpu', nthread=threads, num_target=spec.block, base_score=float(y.mean()))
        model = xgb.train(params, matrix, num_boost_round=rounds, obj=spec.make_objective(y, context))
        reference = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        if actual_depth(model) != 3:
            raise ValueError('Smoke A900 depth is not 3')
        del matrix, model
    else:
        with np.load(source, allow_pickle=False) as saved:
            reference = saved[f'{stage}_b{block}'].copy()
            if not np.array_equal(reference, saved[f'{stage}_b{block}']):
                raise ValueError('Frozen reference copy differs')
    model = gbm.load_booster(reference, threads)
    if model.num_boosted_rounds() != rounds or model.num_features() != x.shape[1] or x.shape[1] != 215:
        raise ValueError('Frozen A900 rounds/features do not match')
    trees = json.loads(bytes(model.save_raw('json')))['learner']['gradient_booster']['model']['trees']
    maximum_depth = 0
    for tree in trees:
        if int(tree['tree_param']['size_leaf_vector']) != spec.block:
            raise ValueError('Frozen reference output width differs')
        pending = [(0, 0)]
        while pending:
            node, depth = pending.pop()
            maximum_depth = max(maximum_depth, depth)
            for child in (tree['left_children'][node], tree['right_children'][node]):
                if child != -1:
                    pending.append((child, depth + 1))
    if maximum_depth > 3 or (not smoke and maximum_depth != 3):
        raise ValueError('Frozen A900 serialized tree depth differs from 3')
    reference_prediction = model.inplace_predict(x).astype(np.float64).reshape(y.shape)
    if not np.isfinite(reference_prediction).all():
        raise ValueError('Nonfinite reference training predictions')
    records = []
    def diagnostic_callback(**data):
        records.append(stage_diagnostics(observed, **data))
    proposal_rounds = 2 if smoke else CONTINUATION.proposal_rounds
    forecast, stages, record = CONTINUATION.continue_block(model, x, xq, y.astype(np.float64), seed, threads,
                                                          proposal_rounds, diagnostic_callback=diagnostic_callback)
    alphas = [item['alpha'] for item in record['stages']]
    tick = time.perf_counter()
    replay = continuation.block_forecast(reference, stages, alphas, xq, threads)
    seconds = time.perf_counter() - tick
    if not np.array_equal(replay, forecast):
        raise ValueError('Checkpoint replay differs from fitted forecast')
    record.update(diagnostics=records, fixed_reference_reused=not smoke, reference_byte_equal=None if smoke else True,
                  reference_source=str(source), reference_key=f'{stage}_b{block}', reference_max_depth=3, reference_actual_tree_depth=maximum_depth,
                  proposal_max_depth=5, reference_check='smoke_not_applicable' if smoke else 'exact_array_equal',
                  observed_cells=int(np.isfinite(observed).sum()), imputed_cells=int(np.isnan(observed).sum()))
    json.dumps(record, allow_nan=False)
    context['calls'] += 1
    return forecast, dict(reference=reference, stages=stages, alphas=alphas), seconds, dict(
        proposal_rounds=proposal_rounds, continuation=record, replay_max_abs_error=0.)


SPEC = gbm.GBMSpec(
    experiment=EXPERIMENT, constant=PARENT_CONSTANT + ADDED_CONSTANT, varying=VARYING, parent_constant=PARENT_CONSTANT,
    objective=OBJECTIVE, component=OBJECTIVE, threads=5,
    make_objective=lambda target, context: objectives.point4_spectral(target, SPECTRAL_COEFFICIENT),
    fit_block=fit_block, predict_block=CONTINUATION.predict_block,
    prepare_period=prepare_period, keep_observed=True,
    launch=dict(tail=dict(threshold=objectives.TAIL_THRESHOLD, weight=objectives.TAIL_WEIGHT), spectral_coefficient=SPECTRAL_COEFFICIENT,
                continuation=CONTINUATION.launch_record()))


if __name__ == '__main__':
    gbm.main(SPEC, __doc__, HERE)
