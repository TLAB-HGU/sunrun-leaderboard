"""Arrival-CME inputs + smooth-tail A900 + spectral-free S-cap continuation.

Fixed mean of seeds 42/43/44 from the registered smooth_tail_scap_nospec arm.
This is a two-factor replacement ablation: replace point4 + spectral in the
first stage with smooth-tail MSE, and remove spectral from the continuation.
All 223 ordered inputs, 900 + 180 trees, training-only offset and clipping
are retained. No leaderboard targets determine the loss or calibration.

CPU only: CUDA_VISIBLE_DEVICES= python experiment.py --threads 1
Use --predict-only for saved weights, --donki for the saved notification corpus,
or --smoke --out DIR for a wiring check. The shared GBM runner writes the usual
research/result/<experiment>/{results,logs} and local weights/F{n}/s{seed}.npz;
--out DIR instead writes DIR/{results,logs,weights}.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from _module import common, continuation, gbm, public_events
from scipy.special import expit

EXPERIMENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec'
PARENT = 'e2000fs_calib_offset_roll_scap_arrival_cme'
_loader = importlib.util.spec_from_file_location(f'{PARENT}_experiment', HERE.parent / PARENT / 'experiment.py')
ANCHOR = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(ANCHOR)


class SmoothTailMSE:
    """Exact smooth_tail_mse callback from the preregistered campaign."""

    def __init__(self, target):
        self.target = np.array(target, dtype=np.float64, copy=True)
        if self.target.ndim != 2 or len(self.target) < 2 or not np.isfinite(self.target).all():
            raise ValueError('Expected finite two-dimensional fitted training targets')
        if np.any(self.target <= 0):
            raise ValueError('Speed targets must be positive')
        self.target.setflags(write=False)
        delta = max(float(self.target.std()), 10.)
        center = np.quantile(self.target, .8, axis=0)
        self.weights = 1. + 3. * expit((self.target - center) / (.2 * delta))
        self.weights /= self.weights.mean(0)
        self.fixed_scale = 1. / float(self.weights.mean())
        if not np.isfinite(self.fixed_scale) or self.fixed_scale <= 0:
            raise ValueError('Invalid initial curvature scale')

    def __call__(self, prediction, matrix):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(self.target.shape)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite predictions')
        # Keep the frozen callback's operation order, including its scalar normalization.
        gradient = self.weights * (prediction - self.target)
        gradient, hessian = self.fixed_scale * gradient, self.fixed_scale * self.weights
        if not (np.isfinite(gradient).all() and np.isfinite(hessian).all() and np.all(hessian >= 0)):
            raise FloatingPointError('Invalid exact derivatives')
        with np.errstate(over='raise', invalid='raise'):
            gradient, hessian = gradient.astype(np.float32), hessian.astype(np.float32)
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise FloatingPointError('Nonfinite float32 callback')
        return gradient, hessian


CONTINUATION = continuation.Continuation('direct_s_continuation_mse_cap', spectral_coefficient=0., mse_cap=1.01)
SPEC = dataclasses.replace(
    ANCHOR.SPEC, experiment=EXPERIMENT, objective='smooth_tail_scap_nospec', component='smooth_tail_scap_nospec',
    threads=1, rounds=900, make_objective=lambda target, context: SmoothTailMSE(target),
    fit_block=CONTINUATION.fit_block, predict_block=CONTINUATION.predict_block,
    launch={
        **{key: value for key, value in ANCHOR.SPEC.launch.items()
           if key not in ('tail', 'spectral_coefficient', 'continuation', 'source_ablation')},
        'parent_experiment': PARENT,
        'classification': 'two-factor replacement ablation: A-stage loss and continuation spectral removal',
        'added_constant': [], 'added_varying': [], 'removed_features': [],
        'a_stage': {'loss': 'smooth_tail_mse', 'rounds': 900,
                    'weight': '1 + 3 * sigmoid((y - per-lead training q80) / (0.2 * max(training std, 10)))',
                    'normalization': 'per-lead mean weight, then reciprocal global mean Hessian',
                    'spectral_coefficient': 0.},
        'spectral_coefficient': 0.,
        'continuation': {**CONTINUATION.launch_record(), 'spectral_coefficient': 0.},
        'tree_budget': 1080,
        'source_ablation': '~/sunrun-continuous-loss-suite/runs/smooth_tail_scap_nospec_s{42,43,44}',
    })


def main():
    if {'-h', '--help'}.intersection(sys.argv[1:]):
        gbm.main(SPEC, __doc__, HERE)
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--donki', type=Path, help='saved DONKI CME notifications JSON (default: download)')
    parser.add_argument('--data', type=Path, default=common.DATA)
    known, rest = parser.parse_known_args()
    with tempfile.TemporaryDirectory(prefix=f'{EXPERIMENT}-') as work:
        corpus = known.donki or public_events.fetch_notifications(Path(work) / 'donki_notifications.json')
        table = Path(work) / 'table.parquet'
        SPEC.launch['donki'] = {**public_events.augment(known.data, corpus, table),
                                'df_source': common.file_identity(known.data)}
        sys.argv = [sys.argv[0], *rest, '--data', str(table)]
        gbm.main(SPEC, __doc__, HERE)


if __name__ == '__main__':
    main()
