"""Stage drivers used by the notebook.

select  fit targets < 2025-12-01, validate on origins 2025-12-04..2026-02-22 -> RectifyMO sigma / Stratify choice
f1      fit targets < 2026-02-26 (leaderboard F1 cutoff)                     -> F1 predictions (2,137 origins)
final   fit targets < 2026-05-29                                             -> F2 predictions (2,833 origins)
Predictions are checkpointed per strategy so an interrupted notebook resumes without refitting.
Selection uses the select stage only; leaderboard scores are never used to choose anything.
"""
import json
import time
from pathlib import Path

import numpy as np

import data
import evaluate
import strategies as S

HERE = Path(__file__).resolve().parent
STAGES = ('select', 'f1', 'final')


def canonical(name):
    """Stratify-RecMO-DirMO-s is RectifyMO-s; fit it once."""
    sp = S.spec(name)
    if sp['kind'] == 'rectify' and sp['base'][0] == 'RecMO' and sp['residual'][0] == 'DirMO' \
            and sp['base'][1] == sp['residual'][1] and sp['base'][1] > 1:
        return f'RectifyMO-{sp["base"][1]}'
    return name


class Stage:
    def __init__(self, name, train, f1, official, series, out, cfg, subsample=1):
        if name not in STAGES:
            raise ValueError('unknown stage ' + name)
        self.name, self.cfg, self.out = name, cfg, Path(out)
        fit, val = data.select_split(train)
        if name == 'f1':
            fit = data.f1_split(train)
        elif name == 'final':
            fit = data.final_split(train)
        fit = fit[::subsample]
        self.origins = train.origins[fit]
        self.fit, self.Y = S.Inputs.of(train.take(fit)), data.future_targets(series, self.origins)
        vo = train.origins[val]
        self.valid, self.Yv = S.Inputs.of(train.take(val)), data.future_targets(series, vo)
        self.val_observed = evaluate.observed_mask(vo)
        self.target = {'select': 'val', 'f1': 'f1', 'final': 'official'}[name]
        self.target_origins = {'select': vo, 'f1': f1.origins, 'final': official.origins}[name]
        bundle = {'select': None, 'f1': f1, 'final': official}[name]
        self.targets = {self.target: self.valid if bundle is None else S.Inputs.of(bundle)}
        self.base_cache = {}
        self.meta = {}
        (self.out / 'predictions' / name).mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.out / 'predictions' / self.name / f'{self.target}_{name}.npy'

    def run(self, name):
        source = canonical(name)
        path = self.path(source)
        meta = path.with_suffix('.json')
        if path.exists():
            self.meta[name] = json.loads(meta.read_text())
            return np.load(path)
        start = time.perf_counter()
        preds, fit_s, per_origin = S.fit_predict(
            source, self.cfg, self.fit, self.Y, self.valid, self.Yv, self.targets, origins=self.origins,
            log_root=self.out / 'loss_history' / self.name, base_cache=self.base_cache)
        np.save(path, preds[self.target])
        self.meta[name] = {'strategy': source, 'stage': self.name, 'fit_seconds': fit_s,
                           'inference_seconds_per_fold': per_origin[self.target],
                           'wall_seconds': time.perf_counter() - start, 'n_fit_origins': len(self.origins),
                           'fit_target_end_utc': {'select': str(data.SELECT_TRAIN_END), 'f1': str(data.F1_TRAIN_END),
                                                  'final': str(data.CUTOFF)}[self.name],
                           'rounds': self.cfg.n_estimators}
        meta.write_text(json.dumps(self.meta[name]))
        return preds[self.target]


def combined_file(out, name, f1_stage, final_stage, f1_pred, f2_pred):
    """F1 + F2 leaderboard file (357,840 rows)."""
    origins = f1_stage.target_origins.append(final_stage.target_origins)
    path = Path(out) / 'predictions' / 'combined' / f'{name}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluate.prediction_frame(origins, np.vstack([f1_pred, f2_pred])).to_parquet(path, index=False)
    return path


def select(rows, names):
    """Lowest selection-stage observed validation MSE among names."""
    return min(names, key=lambda n: rows[n]['val_mse_obs'])


def family(name):
    if name in ('Recursive', 'DirRec', 'MIMO'):
        return name
    return name.split('-')[0]
