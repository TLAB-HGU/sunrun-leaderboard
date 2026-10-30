"""Four frozen non-neural model-agnostic branches; no external side effects."""
import argparse
import json
from pathlib import Path
import traceback

import core as C
import numpy as np
import pandas as pd

METHODS = ['event', 'context', 'outputwave', 'quantile']
ARMS = {'event': ['event'], 'context': ['context_0.25', 'context_1.0'],
        'outputwave': ['wave_5', 'wave_20'], 'quantile': ['quantile']}
CONTROLS = {'event': ['raw72'], 'context': [],
            'outputwave': ['raw72', 'rawgroups', 'jointwave', 'zerodetail'], 'quantile': ['scalar_mse']}


def prepare(directory, data, smoke):
    from evaluation import metrics, aggregate_folds
    parity = {}
    for stage, grid in [('f1', 'f1'), ('final', 'official')]:
        p = C.parent_predict(data[grid], stage)
        np.testing.assert_array_equal(p, np.load(C.PARENT / 'artifacts' / f'{stage}_predictions.npy'))
        parity[stage] = dict(origins=len(p), max_abs_difference=0.)
    C.write(directory / 'parent_parity.json', parity)
    scores = []
    for stage, *_ in C.FOLDS:
        C.status(directory, stage)
        fit, valid = C.split(data, stage, smoke)
        tr, va = C.take(data['train'], fit), C.take(data['train'], valid)
        p = C.parent_predict(va, stage, directory, tr, va, smoke)
        C.save(directory / f'{stage}_parent.npz', pred=p, indices=valid)
        score = metrics(va['Y'], p, va['origins']); scores.append(score)
        C.write(directory / f'{stage}_metrics.json', score)
    C.write(directory / 'aggregate.json', aggregate_folds(scores))
    C.status(directory, 'prepare', 'completed', parity=parity)


def direct_model(folder, train, valid, query, y, yv, smoke, **overrides):
    model = C.fit_model(folder, C.design(train), y, C.design(valid), yv, smoke, overrides)
    return C.prediction(model, C.design(query))


def event_targets(directory, bundle, smoke):
    from transforms import fit_event_parameters
    folder = directory / 'target_parameters'; folder.mkdir(parents=True, exist_ok=True)
    values, diagnostics = [], []
    for i, (origin, truth) in enumerate(zip(bundle['origins'], bundle['Y'])):
        path = folder / f'{origin.value}.npz'
        if path.exists():
            a = np.load(path)
            values.append(a['params']); diagnostics.append([a['rmse'].item(), a['success'].item(), a['multiple'].item()])
        else:
            result = fit_event_parameters(truth[None, :], max_nfev=80)
            param = result['params'][0]
            C.save(path, params=param, rmse=result['reconstruction_rmse'][0],
                   success=result['success'][0], multiple=result['multiple_rises'][0])
            values.append(param); diagnostics.append([result['reconstruction_rmse'][0], result['success'][0], result['multiple_rises'][0]])
        if i % 256 == 0:
            print(f'Event target codec: {i}/{len(bundle["Y"])} training-only curves', flush=True)
    return np.asarray(values), np.asarray(diagnostics)


def event_prediction(directory, folder, tr, va, query, smoke):
    from transforms import event_curve
    y, diagnostics = event_targets(directory, tr, smoke)
    yv, vd = event_targets(directory, va, smoke)
    mean, scale = y.mean(0), np.maximum(y.std(0), 1.)
    C.write(folder / 'target_codec.json', dict(mean=mean, scale=scale, columns=['b', 'a', 'tau', 'duration', 'c', 'gap', 'decaydur'],
                                              training_reconstruction_rmse=float(diagnostics[:, 0].mean()),
                                              training_optimizer_success=float(diagnostics[:, 1].mean()),
                                              training_multiple_rises_fraction=float((diagnostics[:, 2] > 1).mean()),
                                              oracle_validation_reconstruction_rmse=float(vd[:, 0].mean())))
    p = direct_model(folder, tr, va, query, (y-mean)/scale, (yv-mean)/scale, smoke)
    return event_curve(p * scale + mean)


def gate_probabilities(directory, tr, va, block, smoke):
    from context import event_labels, purged_blocks
    folder = directory / f'gate_b{block:02}'
    path = folder / 'oof.npz'
    y = event_labels(tr['Y'][:, block*24:(block+1)*24])
    X = C.design(tr, block)
    if path.exists():
        p = np.load(path)['q']
        if len(p) != len(tr['Y']):
            raise RuntimeError('Gate cache row mismatch')
        return p
    probabilities = np.full(len(y), np.nan)
    fold_records = []
    for i, (fit, held) in enumerate(purged_blocks(tr['origins'].values)):
        if len(np.unique(y[fit])) < 2:
            probabilities[held] = y[fit].mean()
        else:
            model = C.fit_model(folder / f'fold{i}', X[fit], y[fit], X[held], y[held], smoke, classifier=True)
            probabilities[held] = C.prediction(model, X[held])
        fold_records.append(dict(fold=i, fit=fit, held=held,
                                 fit_target_maturity_before_outer_cutoff=True, purge_hours=72))
    if not np.isfinite(probabilities).all():
        raise RuntimeError('Incomplete cross-fitted gate')
    # The full fitted gate is an issuance-time classifier; its probabilities are diagnostic,
    # never an additive correction to the direct speed forecast.
    if len(np.unique(y)) > 1:
        model = C.fit_model(folder / 'full', X, y, C.design(va, block),
                            event_labels(va['Y'][:, block*24:(block+1)*24]), smoke, classifier=True)
        C.save(folder / 'validation_probabilities.npz', q=C.prediction(model, C.design(va, block)))
    C.write(folder / 'purge_receipt.json', fold_records)
    C.save(path, q=probabilities)
    return probabilities


def context_prediction(directory, folder, tr, va, query, strength, smoke):
    from context import context_objective
    outputs = []
    for b in range(3):
        q = gate_probabilities(directory, tr, va, b, smoke)
        model = C.fit_model(folder / f'b{b:02}', C.design(tr, b), tr['Y'][:, b*24:(b+1)*24],
                            C.design(va, b), va['Y'][:, b*24:(b+1)*24], smoke,
                            dict(objective=context_objective(q, strength)))
        outputs.append(C.prediction(model, C.design(query, b)))
    return np.hstack(outputs)


def wave_prediction(folder, tr, va, query, arm, smoke):
    from transforms import haar72, inverse_haar72
    y, yv = tr['Y'], va['Y']
    transform = arm != 'rawgroups'
    if transform:
        y, yv = haar72(y), haar72(yv)
    if arm == 'jointwave':
        return inverse_haar72(direct_model(folder, tr, va, query, y, yv, smoke))
    result = np.zeros((len(query['Y']), 72))
    for group, (s, e) in enumerate([(0, 9), (9, 18), (18, 36), (36, 72)]):
        if arm == 'zerodetail' and group > 0:
            continue
        regularization = float(arm.split('_')[1]) if arm.startswith('wave_') and group > 0 else 5.
        result[:, s:e] = direct_model(folder / f'g{group}', tr, va, query, y[:, s:e], yv[:, s:e], smoke,
                                      reg_lambda=regularization)
    return inverse_haar72(result) if transform else result


def scalar_predictions(folder, tr, va, query, smoke, quantile):
    levels = np.array([.1, .3, .5, .7, .9])
    output = np.empty((len(query['Y']), 72, 5)) if quantile else np.empty((len(query['Y']), 72))
    horizons = [0, 23, 24, 47, 48, 71] if smoke else list(range(72))
    for h in horizons:
        b = h // 24
        overrides = dict(multi_strategy='one_output_per_tree')
        if quantile:
            overrides.update(objective='reg:quantileerror', quantile_alpha=levels, eval_metric='quantile')
        model = C.fit_model(folder / f'h{h+1:02}', C.design(tr, b), tr['Y'][:, h],
                            C.design(va, b), va['Y'][:, h], smoke, overrides)
        output[:, h] = C.prediction(model, C.design(query, b))
    if smoke:
        # Smoke-only economical horizon sampling; scientific full runs fit all72.
        for i in range(len(output)):
            if quantile:
                for q in range(5):
                    output[i, :, q] = np.interp(np.arange(72), horizons, output[i, horizons, q])
            else:
                output[i] = np.interp(np.arange(72), horizons, output[i, horizons])
    return output


def quantile_prediction(folder, tr, va, query, smoke):
    from quantiles import quantile_samples, pit_ranks, couple_samples, independent_samples, probability_metrics
    # One chronological inner cross-prediction split: first80% fits marginal forecasts;
    # final20% supplies held-out PIT vectors. Both lie before the outer cutoff.
    boundary = max(1, int(.8 * len(tr['Y'])))
    calibration = np.arange(boundary, len(tr['Y']))
    fit = np.flatnonzero(tr['origins'] + pd.Timedelta(hours=72) < tr['origins'][boundary])
    if not len(fit) or not len(calibration):
        raise RuntimeError('Insufficient chronological PIT calibration split')
    cal = C.take(tr, calibration)
    qcal = scalar_predictions(folder / 'calibration', C.take(tr, fit), cal, cal, smoke, True)
    ranks = pit_ranks(cal['Y'], qcal)
    C.save(folder / 'pit_library.npz', ranks=ranks, calibration_origins=cal['origins'].asi8,
           fit_origins=tr['origins'][fit].asi8)
    qpred = scalar_predictions(folder / 'full', tr, va, query, smoke, True)
    marginal = quantile_samples(qpred)
    coupled = couple_samples(marginal, ranks)
    independent = independent_samples(marginal)
    C.save(folder / 'quantiles.npz', pred=qpred, levels=np.array([.1,.3,.5,.7,.9]))
    distributions = {}
    for label, samples in [('coupled', coupled), ('independent', independent)]:
        result = probability_metrics(query['Y'], samples)
        C.save(folder / f'{label}_distribution.npz', **result)
        distributions[label] = {k: result[k] for k in ['crps_mean', 'brier_mean', 'any550_brier_mean']}
    np.testing.assert_allclose(coupled.mean(1), independent.mean(1), rtol=1e-12)
    C.write(folder / 'probability_scores.json', distributions)
    C.write(folder / 'calibration_receipt.json', dict(chronological=True, purge_hours=72,
                                                    fit_rows=len(fit), calibration_rows=len(calibration),
                                                    endpoint_clamped=True, samples=128, smoke_sparse_horizons=smoke))
    return coupled.mean(1)


def arm_prediction(directory, data, stage, arm, smoke):
    fit, valid = C.split(data, stage, smoke)
    tr, va = C.take(data['train'], fit), C.take(data['train'], valid)
    if stage.startswith('fold'):
        query = va
    else:
        grid = data['f1' if stage == 'f1' else 'official']
        query = grid if not smoke else C.take(grid, np.arange(8))
    folder = directory / stage / arm
    path = folder / 'predictions.npz'
    receiptpath = folder / 'predictions.receipt.json'
    if path.exists() and receiptpath.exists():
        if C.digest(path) != json.loads(receiptpath.read_text())['sha256']:
            raise RuntimeError(f'Prediction checkpoint corruption: {path}')
        p = np.load(path)['pred']
        if p.shape != query['Y'].shape or not np.isfinite(p).all():
            raise RuntimeError(f'Invalid cached prediction {path}')
        return query, p
    C.status(directory, f'{stage}/{arm}')
    if arm == 'event':
        p = event_prediction(directory, folder, tr, va, query, smoke)
    elif arm.startswith('context_'):
        p = context_prediction(directory / stage, folder, tr, va, query, float(arm.split('_')[1]), smoke)
    elif arm == 'raw72':
        p = direct_model(folder, tr, va, query, tr['Y'], va['Y'], smoke)
    elif arm == 'quantile':
        p = quantile_prediction(folder, tr, va, query, smoke)
    elif arm == 'scalar_mse':
        p = scalar_predictions(folder, tr, va, query, smoke, False)
    else:
        p = wave_prediction(folder, tr, va, query, arm, smoke)
    p = np.clip(p, 1, 3000)
    if p.shape != query['Y'].shape or not np.isfinite(p).all():
        raise RuntimeError('Incomplete, nonfinite or misaligned output')
    C.save(path, pred=p, origins=query['origins'].asi8)
    C.write(receiptpath, dict(sha256=C.digest(path), stage=stage, arm=arm, rows=len(p)))
    return query, p


def run(directory, common, data, method, smoke):
    from evaluation import metrics, aggregate_folds, choose_candidate, selection_rank, frame, official_score
    C.wait_common(common, json.loads((directory/'manifest.json').read_text())['identity'])
    base = json.loads((common / 'aggregate.json').read_text())
    all_scores = {'parent': base}; fold_scores = {}
    for arm in ARMS[method] + CONTROLS[method]:
        values = []
        for stage, *_ in C.FOLDS:
            query, p = arm_prediction(directory, data, stage, arm, smoke)
            m = metrics(query['Y'], p, query['origins']); values.append(m)
            C.write(directory / stage / arm / 'metrics.json', m)
        all_scores[arm] = aggregate_folds(values); fold_scores[arm] = values
    eligible_choices = {'parent': base, **{a: all_scores[a] for a in ARMS[method]}}
    selected = choose_candidate(eligible_choices)
    # Research confirmation is preregistered even when the parent is retained;
    # this cannot promote a rejected arm or feed official scores back to selection.
    research_arm = min(ARMS[method], key=lambda a: selection_rank(all_scores[a], base) or (float('inf'),))
    C.write(directory / 'selection.json', dict(selected=selected, research_arm=research_arm,
                                             all_internal_scores=all_scores, eligible=list(a for a in eligible_choices if selection_rank(eligible_choices[a], base) is not None),
                                             mse_limit=1.03, frozen_before_official=True,
                                             research_only_if_parent_retained=selected=='parent'))
    results, frames = {}, []
    for stage in ['f1', 'final']:
        query, p = arm_prediction(directory, data, stage, research_arm, smoke)
        parent = C.parent_predict(query, stage)
        results[stage] = dict(candidate=metrics(query['Y'], p, query['origins']),
                              parent=metrics(query['Y'], parent, query['origins']))
        frames.append(frame(query['origins'], p))
        C.save(directory / f'{stage}_predictions.npz', pred=p, parent=parent, origins=query['origins'].asi8)
    joined = pd.concat(frames, ignore_index=True)
    joined.to_parquet(directory / 'candidate_predictions.parquet', index=False)
    if not smoke:
        assert len(joined) == 357840 and not joined.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
        results['official'] = official_score(joined, C.PARENT / 'manifest.json')
    results['combined'] = aggregate_folds([results[s]['candidate'] for s in ['f1','final']])
    results['parent_combined'] = aggregate_folds([results[s]['parent'] for s in ['f1','final']])
    results.update(selected=selected, research_arm=research_arm, posthoc_official_exposed=True)
    C.write(directory / 'results.json', results)
    from report import render
    render(directory, data, smoke)
    C.status(directory, 'completed', 'completed', selected=selected, research_arm=research_arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['prepare', *METHODS])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error('run-id must be a directory name')
    root = C.HERE / 'runs' / args.run_id
    directory = root / ('common' if args.method=='prepare' else args.method)
    try:
        m = C.manifest(args.smoke)
        C.freeze(directory, m | {'method': args.method})
        C.status(directory, 'starting')
        data = C.load_data()
        if args.method == 'prepare':
            prepare(directory, data, args.smoke)
        else:
            run(directory, root/'common', data, args.method, args.smoke)
    except BaseException as exc:
        traceback.print_exc()
        C.status(directory, 'exception', 'failed', error=str(exc), exit_code=1)
        raise


if __name__ == '__main__':
    main()
