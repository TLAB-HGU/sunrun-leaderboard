"""Round 11: validation-driven routing of frozen cluster experts (import of arXiv:2604.13748).

Single architecture: routing only.  No booster is trained; the three A+B1 cluster experts of
`offset_roll_cluster_experts` and the GLOBAL A+B1 model are rebuilt from their saved boosters and
reproduced exactly (<=1e-6) before any routing is applied.

Rules (each is one experiment, fixed in advance):
  hard           reproduce the round-9 C3 hard label routing (control)
  fallback       cluster k keeps its own expert only if the VAL Huber loss of the expert's PRIOR
                 booster beats GLOBAL on the same rows; otherwise every member of k uses GLOBAL
  table          cluster k is assigned to whichever of {e0, e1, e2, GLOBAL} wins on VAL (reassignment)
  recent_all     per-origin argmin of the matured short-lead error over the last LOOKBACK origins,
                 searched over all four candidates
  recent_cluster same trailing-error rule but restricted to {own expert, GLOBAL}

VAL = the calibration block minus the OFFSET_ROWS origins that fitted the offset, so the loss of a
prototype is out of sample for that prototype.  Official labels enter scoring only.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import time
import warnings

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

import baseline
import competition
import vd_common as vd

RULES = dict(
    b_c3=dict(experts='c3', rule='hard'),
    e1=dict(experts='c3', rule='fallback'),
    e2=dict(experts='c3', rule='recent_all'),
    e2b=dict(experts='c3', rule='recent_cluster'),
    e3=dict(experts='c3', rule='table'),
    r1=dict(experts='r3', rule='fallback'),
)
ELIGIBLE = ('e1', 'e2', 'e2b', 'e3')
IMPROVEMENT = 0.01
CANDIDATES = K = 3


def query_truth(data, period, origins):
    frame = pd.read_parquet(data, columns=['origin', 'h', 'y'],
                            filters=[('set', '==', 'query'), ('period', '==', period)])
    table = frame.pivot(index='origin', columns='h', values='y').reindex(
        index=pd.DatetimeIndex(origins, tz='UTC'), columns=range(1, baseline.H + 1))
    return table.to_numpy(np.float64)


def val_labels(train_origins, train_labels, origins):
    position = pd.DatetimeIndex(train_origins, tz='UTC').searchsorted(pd.DatetimeIndex(origins, tz='UTC'))
    selected = train_labels[position]
    if not np.isin(selected, np.arange(K)).all():
        raise ValueError('VAL origins must all carry a causal cluster label')
    return selected


def route(spec, candidates, query_labels, labels, losses, truth):
    """Return (weights (m, 4), decision detail)."""
    m = len(query_labels)
    n_candidates = candidates.shape[0]
    weights = np.zeros((m, n_candidates))
    detail = {}
    if spec == 'hard':
        weights[np.arange(m), query_labels] = 1.
        detail['kind'] = 'hard_label'
    elif spec in ('fallback', 'table'):
        restricted = spec == 'fallback'
        table, info = vd.cluster_routing_table(labels, losses, n_candidates, restricted=restricted)
        mapping = np.array([table[k] for k in range(K)])
        weights[np.arange(m), mapping[query_labels]] = 1.
        detail.update(kind=spec, table={str(k): int(mapping[k]) for k in range(K)}, val=info)
    elif spec in ('recent_all', 'recent_cluster'):
        choice, count = vd.recent_error_weights(candidates, truth)
        if spec == 'recent_cluster':
            choice, info = vd.recent_error_table(query_labels, choice, restricted=True)
            detail.update(info)
        choice = np.where(choice < 0, n_candidates - 1, choice)
        weights[np.arange(m), choice] = 1.
        detail.update(kind=spec, lookback=vd.LOOKBACK, leads=vd.SHORT_LEADS,
                      share=[float((choice == c).mean()) for c in range(n_candidates)],
                      mean_pairs=float(np.mean(count)))
    else:
        raise ValueError(spec)
    return weights, detail


def run_period(name, spec, seed, data, out, period, threads, identity):
    began = time.perf_counter()
    meta_cfg = RULES[spec]
    run = vd.EXPERT_ROOT / f"cluster_{meta_cfg['experts']}_s{seed}"
    names = baseline.selected_names(list(baseline.FEATURES))
    query, _ = baseline.load_grid(data, period, names, False)
    if len(query['origins']) != baseline.QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')

    forecasts, expert_meta = vd.expert_forecasts(run, data, period, query, threads)
    reproduction = vd.check_expert_reproduction(run, period, forecasts, query['origins'].asi8)
    global_pred = vd.global_query_forecast(seed, period, query['origins'].asi8)
    candidates = np.concatenate([forecasts, global_pred[None]], 0)

    with np.load(run / f'{period}_labels.npz') as labels:
        train_origins, train_labels = labels['train_origins'], labels['train_labels']
        query_labels = labels['query_labels']

    block = vd.val_block(seed, period)
    grid = vd.val_grid(data, period, block['origins'])
    prior = vd.prior_forecasts(run, data, period, grid, threads)
    prior_global = vd.global_prior_forecast(seed, period, grid, threads)
    losses = np.stack([vd.huber_rows(prior[k], block['truth']) for k in range(K)]
                      + [vd.huber_rows(prior_global, block['truth'])])
    labels_val = val_labels(train_origins, train_labels, block['origins'])

    truth = query_truth(data, period, query['origins']) if meta_cfg['rule'].startswith('recent') else None
    weights, detail = route(meta_cfg['rule'], candidates, query_labels, labels_val, losses, truth)
    if not (np.isfinite(weights).all() and np.allclose(weights.sum(1), 1.)):
        raise ValueError('Routing weights must be one-hot per row')
    pred = np.clip(np.einsum('mc,cmh->mh', weights, candidates), 1., 3000.)

    origins = query['origins'].asi8
    out = Path(out)
    np.savez_compressed(out / f'{period}_predictions.npz', pred=pred.astype(np.float32), origins=origins)
    np.savez_compressed(out / f'{period}_gate.npz', weights=weights.astype(np.float32),
                        candidates=candidates.astype(np.float32), origins=origins)
    share = weights.mean(0)
    report = dict(experiment=name, seed=seed, period=period, smoke=False, policy=spec,
                  config=meta_cfg, rule=detail,
                  expert_run=str(run), expert_train_cutoff=expert_meta['training_cutoff'],
                  expert_reproduction_max_abs_diff=reproduction,
                  val_block=dict(start=block['start'], end=block['end'], rows=int(len(block['origins'])),
                                 dropped_offset_rows=block['dropped'],
                                 cluster_rows=[int((labels_val == k).sum()) for k in range(K)],
                                 huber_delta=vd.HUBER_DELTA,
                                 mean_loss=[None if not np.isfinite(x) else float(x) for x in
                                            np.nanmean(losses, 1)]),
                  candidate_share=[float(x) for x in share], query_origins=len(pred),
                  training_cutoff=baseline.CUTOFFS[period],
                  eligible_for_adoption=spec in ELIGIBLE,
                  end_to_end_seconds=time.perf_counter() - began, identity=identity,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, xgboost=xgb.__version__, sklearn=sklearn.__version__))
    vd.write_json(out / f'{period}_metadata.json', report)
    print(json.dumps(dict(period=period, stage='route', component=spec, unit=list(baseline.CUTOFFS).index(period) + 1,
                          units=len(baseline.CUTOFFS), candidate_share=[round(float(x), 4) for x in share])),
          flush=True)
    return report


def prediction_frame(out):
    frames = []
    for period in baseline.CUTOFFS:
        with np.load(Path(out) / f'{period}_predictions.npz') as arrays:
            origins = pd.DatetimeIndex(arrays['origins'], tz='UTC')
            pred = arrays['pred']
        if len(origins) != baseline.QUERY_COUNTS[period] or pred.shape != (len(origins), baseline.H):
            raise ValueError(f'{period}: incomplete full prediction grid')
        if origins.has_duplicates or not origins.is_monotonic_increasing or not np.isfinite(pred).all():
            raise ValueError(f'{period}: invalid prediction grid')
        frames.append(pd.DataFrame(dict(origin_last_input_utc=np.repeat(
            origins.strftime('%Y-%m-%dT%H:%M:%SZ'), baseline.H),
            horizon_hours=np.tile(np.arange(1, baseline.H + 1), len(origins)), pred_kms=pred.ravel())))
    return pd.concat(frames, ignore_index=True)


def score_outputs(out, observations, seed):
    out = Path(out)
    frame = prediction_frame(out)
    observed = competition.load_observations(observations)
    score = competition.score_competition(frame, competition.prepare_context(observed))
    if score['total']['n_replaced']:
        raise ValueError('Scorer replaced forecasts; incomplete runs are not accepted')
    result = dict(competition=score, observations=vd.file_identity(observations),
                  observation_rows=len(observed), scorer=vd.file_identity(competition.__file__),
                  scorer_version=competition.VERSION)
    vd.write_json(out / 'local_score.json', result)
    total = float(score['total']['s'])
    baseline_total = float(vd.global_state(seed)['reproduction']['total_s'])
    threshold = baseline_total * (1 - IMPROVEMENT)
    candidate = total <= threshold
    vd.write_json(out / 'complete.json', dict(
        experiment=out.name, total_s=total, rows=int(len(frame)),
        global_baseline_s=baseline_total, adoption_threshold_s=threshold,
        improvement_fraction=(baseline_total - total) / baseline_total,
        publish='candidate' if candidate else 'not_required',
        publish_reason=('beats the seed GLOBAL baseline by >=1%' if candidate
                        else 'routing-only run; below the 1% adoption rule against the seed GLOBAL baseline')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=sorted(RULES))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, choices=(42, 43, 44))
    parser.add_argument('--period', choices=sorted(baseline.CUTOFFS))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only runs require CUDA_VISIBLE_DEVICES="" before startup')
    warnings.filterwarnings('ignore', category=FutureWarning)
    args.out.mkdir(parents=True, exist_ok=True)
    import subprocess
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    identity = dict(data=vd.file_identity(args.data), experiment=vd.file_identity(__file__),
                    common=vd.file_identity(vd.__file__), baseline=vd.file_identity(baseline.__file__),
                    git_commit=git.stdout.strip(), seed=args.seed)
    periods = [args.period] if args.period else list(baseline.CUTOFFS)
    for period in periods:
        run_period(args.experiment, args.experiment, args.seed, args.data, args.out, period, args.threads, identity)
    if args.period:
        return 0
    score_outputs(args.out, args.observations, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
