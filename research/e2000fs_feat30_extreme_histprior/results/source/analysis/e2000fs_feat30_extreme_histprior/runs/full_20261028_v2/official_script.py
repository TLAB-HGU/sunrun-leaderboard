"""Official F1/final refit of the internal-gate-passing arm, for one dry-run score.

Run from outside the study directory so the study's recorded code hash stays exactly
the one that produced the internal results.  Imports the study modules in place.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

STUDY = Path('/home/t-lab01/sunrun/analysis/e2000fs_feat30_extreme_histprior')
sys.path.insert(0, str(STUDY))

import hist  # noqa: E402
import run  # noqa: E402


def fit_block(folder, X_fit, Y_fit, fit, X_query, factor, rounds):
    folder.mkdir(parents=True, exist_ok=True)
    params, _ = run.hist_params()
    registry = hist.LabelRegistry()
    reps = hist.representatives(Y_fit[fit].ravel())
    binw = hist.bin_weight(factor)
    dquery = xgb.DMatrix(X_query, nthread=3)
    raw = np.empty((len(X_query), run.BLOCK))
    corrected = np.empty((len(X_query), run.BLOCK))
    train_ce = np.empty((run.BLOCK, rounds))
    for h in range(run.BLOCK):
        target = Y_fit[:, h]
        q_fit = hist.soft_labels(target[fit])
        weight = hist.point_weights(target[fit], factor)
        dtrain = registry.add(xgb.DMatrix(X_fit[fit], label=np.arange(len(fit), dtype=np.float32),
                                         nthread=3), q_fit, weight)

        def metric(pred, dmat, registry=registry, reps=reps):
            truth, _ = registry(dmat)
            expected = hist.softmax(pred) @ reps
            return [('ce', hist.soft_cross_entropy(truth, pred)),
                    ('rmse_kms', float(np.sqrt(np.mean((expected - truth @ reps) ** 2))))]

        history = {}
        booster = xgb.train(params, dtrain, num_boost_round=rounds,
                            obj=hist.class_objective(registry), custom_metric=metric,
                            evals=[(dtrain, 'train')], evals_result=history, verbose_eval=False)
        train_ce[h] = history['train']['ce']
        booster.save_model(folder / f'h{h:02}.ubj')
        margin = booster.predict(dquery, output_margin=True)
        raw[:, h], _ = hist.histogram_mean(margin, reps)
        corrected[:, h], _ = hist.histogram_mean(margin, reps, binw)
    return raw, corrected, train_ce, reps


def frame(origins, pred):
    origins = pd.DatetimeIndex(origins).tz_convert('UTC')
    return pd.DataFrame({
        'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), run.HORIZON),
        'horizon_hours': np.tile(np.arange(1, run.HORIZON + 1), len(origins)),
        'pred_kms': np.asarray(pred, dtype=np.float64).ravel(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='full_20261028_v2')
    parser.add_argument('--arm', default='hist_weighted_raw')
    parser.add_argument('--factor', type=float, default=run.FACTOR)
    parser.add_argument('--rounds', type=int, default=run.ROUNDS)
    args = parser.parse_args()
    internal = STUDY / 'runs' / args.run_id
    gates = json.loads((internal / 'aggregate.json').read_text())['gates'][args.arm]
    if not (gates['gate1'] and gates['gate2'] and gates['confirmation_window_C']):
        raise SystemExit(f'internal gate not passed for {args.arm}: {gates}')
    out = internal / 'official'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((internal / 'manifest.json').read_text())
    print(json.dumps(dict(arm=args.arm, rounds=args.rounds, factor=args.factor,
                          study_identity=manifest['identity'],
                          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)
    _, data = run.load_data()
    frames, results = [], {}
    for stage, grid in (('f1', 'f1'), ('final', 'official')):
        fit = run.cutoff_split(data['train']['origins'], stage)
        bundle = data[grid]
        pred = np.empty((len(bundle['origins']), run.HORIZON))
        for block in range(3):
            raw, corrected, train_ce, reps = fit_block(
                out / 'models' / stage / f'b{block:02}',
                run.design(data['train'], block),
                data['train']['Y'][:, block * run.BLOCK:(block + 1) * run.BLOCK],
                fit, run.design(bundle, block), args.factor, args.rounds)
            key = 'corrected' if args.arm.endswith('corrected') else 'raw'
            pred[:, block * run.BLOCK:(block + 1) * run.BLOCK] = raw if key == 'raw' else corrected
            run.save_array(out / f'{stage}_b{block:02}_loss.npz', train_ce=train_ce,
                           representatives=reps)
            print(json.dumps(dict(stage=stage, block=block, fit_rows=int(len(fit)),
                                  query_rows=int(len(bundle['origins'])),
                                  mean_train_ce=float(train_ce[:, -1].mean()))), flush=True)
        pred = np.clip(pred, 1., 3000.)
        run.save_array(out / f'{stage}_predictions.npz', pred=pred, origins=bundle['origins'].asi8)
        results[stage] = run.band_metrics(bundle['Y'], pred)
        frames.append(frame(bundle['origins'], pred))
    combined = pd.concat(frames, ignore_index=True)
    keys = ['origin_last_input_utc', 'horizon_hours']
    if len(combined) != 357840 or combined.duplicated(keys).any() or not np.isfinite(combined.pred_kms).all():
        raise RuntimeError('combined submission grid is incomplete, duplicated or non-finite')
    combined.to_parquet(out / 'predictions.parquet', index=False)
    ys = np.vstack([data['f1']['Y'], data['official']['Y']])
    ps = np.vstack([np.load(out / 'f1_predictions.npz')['pred'],
                    np.load(out / 'final_predictions.npz')['pred']])
    results['combined'] = run.band_metrics(ys, ps)
    results['fold_weights'] = dict(f1=2137, final=2833,
                                   note='the official ranking is (2137*F1+2833*F2)/4970; '
                                        'results[combined] pools all 4970 origins equally')
    run.write(out / 'results.json', results)
    print(json.dumps(dict(stage_mse={k: v['mse'] for k, v in results.items() if isinstance(v, dict) and 'mse' in v}),
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
