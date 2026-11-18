"""Block-seam smoothing xf8 on top of e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec.

No retraining. The base row's 3 blocks x 24 leads forecast jumps at the 24|25 and 48|49 seams (about 10x the
within-block step). At each seam the jump beyond the local slope is removed half from each side, with a linear
taper over k = 8 leads (xf8), on every seed forecast and every period; then clip [1, 3000] and the fixed
equal-weight mean of seeds 42/43/44.

k = 8 was set on explicit user request from a leaderboard sweep of xf1..xf24 (LB-selected hyperparameter; no
training-window selection). Model inputs and fitting are unchanged and use only pre-cutoff data.

CPU only: CUDA_VISIBLE_DEVICES= python experiment.py
Default base = the board row's seed predictions (research/result/<base>/results);
--base DIR reads DIR/results (a base-row run with --out DIR).
Board run writes research/result/<experiment>/{results,logs}; --out DIR writes DIR/{results,logs}.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from _module import common

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam'
BASE = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec'
H, SEAMS, SEEDS, TAPER = common.H, (24, 48), common.SEEDS, 8


def smooth(pred, width=TAPER):
    """pred: (n, 72). xf{width} seam correction computed on the original curve, then clip [1, 3000]."""
    p = np.asarray(pred, np.float64)
    q = p.copy()
    for b in SEAMS:
        slope = .5 * ((p[:, b - 1] - p[:, b - 2]) + (p[:, b + 1] - p[:, b]))
        jump = (p[:, b] - p[:, b - 1]) - slope
        for i in range(width):
            taper = .5 * jump * (width - i) / width
            q[:, b - 1 - i] += taper
            q[:, b + i] -= taper
    return np.clip(q, 1., 3000.)


def seam_stats(pred):
    d = np.diff(pred, axis=1)
    interior = np.abs(np.delete(d, [s - 1 for s in SEAMS], axis=1)).mean()
    return dict(interior_abs_step=float(interior),
                **{f'seam{s}_ratio': float(np.abs(d[:, s - 1]).mean() / interior) for s in SEAMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    parser.add_argument('--base', type=Path, help='base-row run folder with results/')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only: start with CUDA_VISIBLE_DEVICES=""')
    common.output_paths(args, HERE, EXPERIMENT)
    base_results = (args.base / 'results' if args.base else common.RESULT / BASE / 'results').resolve()
    logs = common.run_logs(args, EXPERIMENT)
    common.launch(logs, args, base=BASE, base_results=str(base_results), seams=list(SEAMS), transform=f'xf{TAPER}',
                  selection='k=8 from a leaderboard sweep of xf1..xf24 on explicit user request',
                  versions=common.versions())

    observations = common.observations_path(args.observations)
    frames = {s: pd.read_parquet(base_results / f's{s}_predictions.parquet') for s in SEEDS}
    keys = frames[SEEDS[0]][common.KEYS]
    origins = sum(common.QUERY_COUNTS.values())
    if len(keys) != H * origins or not np.array_equal(
            keys['horizon_hours'].to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (origins, H))):
        raise ValueError('base predictions are not the origin-major F1/F2/F3 query grid')
    smoothed = []
    for seed in SEEDS:
        frame = frames[seed]
        if not frame[common.KEYS].equals(keys):
            raise ValueError('seed key mismatch')
        result = keys.copy()
        result['pred_kms'] = smooth(frame['pred_kms'].to_numpy(np.float64).reshape(-1, H)).ravel()
        scores = common.score(result, observations)
        common.write_results(args.out / 'results', f's{seed}', result, scores)
        logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
        smoothed.append(result)
    ensemble = common.mean3(smoothed)
    scores = common.score(ensemble, observations)
    common.write_results(args.out / 'results', 'mean3', ensemble, scores)
    logs('run', 'complete', transform=f'xf{TAPER}', total_s=scores['competition']['total']['s'],
         seam_stats=seam_stats(ensemble['pred_kms'].to_numpy(np.float64).reshape(-1, H)))


if __name__ == '__main__':
    main()
