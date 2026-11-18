"""Block-seam smoothing on top of e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec.

No retraining. The base row's 3 blocks x 24 leads forecast jumps at the 24|25 and 48|49 seams (about 10x the
within-block step). Nine fixed transforms are pre-registered:
  none, ma{1,2,3,4,6}: centred moving average (window 2m+1) on the m leads on each side of a seam
  xf{3,6,12}:          remove the seam jump beyond the local slope, half from each side, linear taper over k leads
Per period the transform is chosen on that period's training-internal calibration window (last 3 months before
the cutoff, the base row's prior models + frozen offset, mean of seeds 42/43/44, training labels, official per-lead
S formula). The chosen transform is applied to each seed forecast, then seeds 42/43/44 are averaged with fixed equal
weights. LB F1/F2/F3 truth is used only to score the result.

CPU only: CUDA_VISIBLE_DEVICES= python experiment.py --threads 4 --donki DONKI.json
Default base = the board row (research/result/<base>/results + research/<base>/weights);
--base DIR reads DIR/results and DIR/weights (a base-row run with --out DIR).
Board run writes research/result/<experiment>/{results,logs}; --out DIR writes DIR/{results,logs}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from _module import calibration as cal
from _module import common, gbm, public_events
from _module.table import load_grid, load_table, subset

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam'
BASE = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec'
_loader = importlib.util.spec_from_file_location(f'{BASE}_experiment', HERE.parent / BASE / 'experiment.py')
BASEMOD = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(BASEMOD)
SPEC = BASEMOD.SPEC

H, SEAMS, PERIODS, SEEDS = common.H, (24, 48), ('F1', 'F2', 'F3'), common.SEEDS
VARIANTS = ['none', 'ma1', 'ma2', 'ma3', 'ma4', 'ma6', 'xf3', 'xf6', 'xf12']


def smooth(pred, variant):
    """pred: (n, 72). Transformed copy clipped to [1, 3000]; corrections are computed on the original curve."""
    p = np.asarray(pred, np.float64)
    q = p.copy()
    if variant == 'none':
        return np.clip(q, 1., 3000.)
    kind, width = variant[:2], int(variant[2:])
    for b in SEAMS:
        if kind == 'ma':
            for j in range(b - width, b + width):
                q[:, j] = p[:, max(0, j - width):min(H, j + width + 1)].mean(1)
        else:
            slope = .5 * ((p[:, b - 1] - p[:, b - 2]) + (p[:, b + 1] - p[:, b]))
            jump = (p[:, b] - p[:, b - 1]) - slope
            for i in range(width):
                taper = .5 * jump * (width - i) / width
                q[:, b - 1 - i] += taper
                q[:, b + i] -= taper
    return np.clip(q, 1., 3000.)


def lead_s(pred, truth):
    """Official per-lead S = sqrt(RMSE * sqrt(TIMSE)), TIMSE = 101-quantile tail MSE (scorer formula), lead mean."""
    values = []
    for c in range(H):
        valid = np.isfinite(truth[:, c])
        observed, error = truth[valid, c], pred[valid, c] - truth[valid, c]
        squared = error ** 2
        quantiles = np.quantile(observed, np.arange(101) / 100, method='linear')
        order = np.argsort(observed, kind='stable')
        positions = np.searchsorted(observed[order], quantiles[:-1], side='left')
        suffix = np.cumsum(squared[order][::-1])[::-1]
        tail = suffix[positions] / (len(observed) - positions)
        timse = float(np.dot(np.diff(quantiles), tail) / (quantiles[-1] - quantiles[0]))
        values.append(np.sqrt(np.sqrt(squared.mean()) * np.sqrt(timse)))
    return float(np.mean(values))


def seam_stats(pred):
    d = np.diff(pred, axis=1)
    interior = np.abs(np.delete(d, [s - 1 for s in SEAMS], axis=1)).mean()
    return dict(interior_abs_step=float(interior),
                **{f'seam{s}_ratio': float(np.abs(d[:, s - 1]).mean() / interior) for s in SEAMS})


def select(table, weights, threads, logs):
    """Per period: the variant with the lowest proxy S on the training-internal calibration window."""
    chosen = {}
    for period in PERIODS:
        train, y = load_grid(table, period, True, SPEC.constant, SPEC.varying)
        _, rows, boundary = cal.prior_split(train['origins'])
        grid, truth = subset(train, rows), y[rows]
        origins = grid['origins']
        six = np.asarray(origins.hour % 6 == 0)
        preds = []
        for seed in SEEDS:
            blocks, state = gbm.load_checkpoint(weights / period / f's{seed}.npz', 'prior', H // SPEC.block)
            raw = np.empty((len(origins), H))
            for b, start in enumerate(range(0, H, SPEC.block)):
                raw[:, start:start + SPEC.block] = SPEC.predict_block(blocks[b], SPEC.design(grid, start, SPEC.block),
                                                                      threads)
            offset = np.asarray(state['offset'])
            again, _ = cal.estimate_offset(raw, truth, SPEC.offset_window, SPEC.alpha)
            if not np.array_equal(again, offset):
                raise ValueError(f'{period} s{seed}: prior checkpoint does not reproduce the stored offset')
            preds.append(cal.apply_offset(raw, offset))
        ensemble = np.mean(preds, axis=0)
        proxy = {v: lead_s(smooth(ensemble, v)[six], truth[six]) for v in VARIANTS}
        chosen[period] = min(VARIANTS, key=lambda v: (proxy[v], v != 'none'))
        record = dict(period=period, training_cutoff=common.CUTOFFS[period], calibration_rows=len(origins),
                      calibration_scored_origins=int(six.sum()), calibration_start=str(origins.min()),
                      calibration_end=str(origins.max()), boundary=str(boundary), proxy_s=proxy,
                      chosen=chosen[period], seam_stats=seam_stats(ensemble))
        logs('run', 'period_selected', **record)
        common.write_json(logs.folder / f'{period}_metadata.json', record)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--base', type=Path, help='base-row run folder with results/ and weights/')
    parser.add_argument('--data', type=Path, default=common.DATA)
    parser.add_argument('--donki', type=Path, required=True, help='saved DONKI CME notifications JSON')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GBM inference is CPU only: start with CUDA_VISIBLE_DEVICES=""')
    common.output_paths(args, HERE, EXPERIMENT)
    base_results = (args.base / 'results' if args.base else common.RESULT / BASE / 'results').resolve()
    weights = (args.base / 'weights' if args.base else HERE.parent / BASE / 'weights').resolve()
    logs = common.run_logs(args, EXPERIMENT)
    common.launch(logs, args, base=BASE, base_results=str(base_results), weights=str(weights), variants=VARIANTS,
                  seams=list(SEAMS), versions=common.versions(xgboost=gbm.xgb.__version__),
                  data=common.file_identity(args.data))

    with tempfile.TemporaryDirectory(prefix=f'{EXPERIMENT}-') as work:
        table_path = Path(work) / 'table.parquet'
        donki = public_events.augment(args.data, args.donki, table_path)
        logs('data', 'donki_augmented', **{k: v for k, v in donki.items() if not isinstance(v, (list, dict))})
        table = load_table(table_path, SPEC.constant + SPEC.varying, logs, keep_observed=SPEC.keep_observed)
    chosen = select(table, weights, args.threads, logs)
    del table

    observations = common.observations_path(args.observations)
    frames = {s: pd.read_parquet(base_results / f's{s}_predictions.parquet') for s in SEEDS}
    keys = frames[SEEDS[0]][common.KEYS]
    counts = [common.QUERY_COUNTS[p] for p in PERIODS]
    if len(keys) != H * sum(counts) or not np.array_equal(
            keys['horizon_hours'].to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (sum(counts), H))):
        raise ValueError('base predictions are not the origin-major F1/F2/F3 query grid')
    period_of = np.repeat(PERIODS, counts)
    smoothed = []
    for seed in SEEDS:
        frame = frames[seed]
        if not frame[common.KEYS].equals(keys):
            raise ValueError('seed key mismatch')
        matrix = frame['pred_kms'].to_numpy(np.float64).reshape(-1, H)
        out = matrix.copy()
        for period in PERIODS:
            rows = period_of == period
            out[rows] = smooth(matrix[rows], chosen[period])
        result = keys.copy()
        result['pred_kms'] = out.ravel()
        scores = common.score(result, observations)
        common.write_results(args.out / 'results', f's{seed}', result, scores)
        logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
        smoothed.append(result)
    ensemble = common.mean3(smoothed)
    scores = common.score(ensemble, observations)
    common.write_results(args.out / 'results', 'mean3', ensemble, scores)
    logs('run', 'complete', chosen=chosen, total_s=scores['competition']['total']['s'],
         seam_stats=seam_stats(ensemble['pred_kms'].to_numpy(np.float64).reshape(-1, H)))


if __name__ == '__main__':
    main()
