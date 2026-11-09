"""Frozen-booster, single-input stale-image-age sensitivity experiment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PARENT = ROOT.parent / 'drop035_v650_confirm_20261108'
DATA = ROOT.parent / 'rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
RAW = REPO / 'data-v2/input/df.parquet'
OBS = ROOT.parent / 'offset_roll_complete_ablation_20261106/competition-observations.parquet'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def frame(origins, pred):
    return pd.DataFrame({'origin_last_input_utc': origins.repeat(72),
                         'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                         'pred_kms': pred.ravel()})


def repaired_age():
    raw = pd.read_parquet(RAW, columns=['origin', 'img_age_h'], filters=[('h', '==', 1)])
    raw = raw.drop_duplicates('origin').sort_values('origin').set_index('origin')
    age = raw.img_age_h
    source_time = pd.Series(raw.index, index=raw.index) - pd.to_timedelta(age, unit='h')
    source_time = source_time.ffill()
    fixed = (pd.Series(raw.index, index=raw.index) - source_time).dt.total_seconds() / 3600
    return fixed, age


def main(seed, experiment):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=[experiment], default=experiment)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu'], required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    if args.threads != 1:
        raise ValueError('This bounded inference audit is frozen to one CPU thread')
    out = args.out
    original = PARENT / 'runs' / f'drop035_v650_s{seed}'
    baseline = module(PARENT / 'baseline.py', 'age_baseline')
    scorer = module(REPO / 'leaderboard/scorer/competition.py', 'age_scorer')
    fixed_age, raw_age = repaired_age()
    control_frames, treatment_frames, receipts = [], [], []
    for period in ('F1', 'F2', 'F3'):
        meta = json.loads((original / f'{period}_metadata.json').read_text())
        names = meta['features']
        grid, labels = baseline.load_grid(DATA, period, names, False)
        assert labels is None
        changed = {key: value.copy() for key, value in grid.items()}
        col = names['constant'].index('img_age_h')
        ages = fixed_age.reindex(grid['origins']).to_numpy(np.float32)
        if not np.isfinite(ages).all() or (ages < 0).any():
            raise ValueError('No causal source frame for an evaluated input')
        changed['C'][:, col] = ages
        assert np.array_equal(grid['E'], changed['E'], equal_nan=True)
        keep = [i for i in range(grid['C'].shape[1]) if i != col]
        assert np.array_equal(grid['C'][:, keep], changed['C'][:, keep], equal_nan=True)
        controls, treatments = [], []
        model_root = original if seed == 42 else ROOT / 'frozen_models' / original.name
        for block in range(3):
            start = 24 * block
            model = xgb.Booster(params={'nthread': args.threads, 'device': 'cpu'})
            model.load_model(model_root / f'{period}_models/full/b{block}.ubj')
            model.set_param({'nthread': args.threads, 'device': 'cpu'})
            for inputs, predictions in ((grid, controls), (changed, treatments)):
                x = np.hstack([inputs['C'], inputs['E'][:, start:start + 24].reshape(len(ages), -1)])
                assert x.shape[1] == len(meta['expanded_features'][str(start)])
                predictions.append(model.inplace_predict(x).astype(np.float64))
            print(json.dumps({'period': period, 'block': block, 'unit': block + 1,
                              'units': 3, 'stage': 'frozen_model_age_repair'}), flush=True)
        with np.load(original / f'{period}_raw_predictions.npz') as state:
            offset = state['offset'].copy()
        a = np.clip(np.clip(np.hstack(controls), 1, 3000) + offset, 1, 3000)
        b = np.clip(np.clip(np.hstack(treatments), 1, 3000) + offset, 1, 3000)
        with np.load(original / f'{period}_predictions.npz') as stored:
            difference = float(np.max(np.abs(a - stored['pred'])))
            assert np.array_equal(grid['origins'].asi8, pd.DatetimeIndex(stored['origins']).asi8)
        if difference > 1e-6:
            raise ValueError(f'{period}: frozen parent reproduction failed: {difference}')
        origins = grid['origins']
        np.savez_compressed(out / f'{period}_predictions.npz', origins=origins.asi8, pred=b)
        np.savez_compressed(out / f'{period}_control.npz', origins=origins.asi8, pred=a)
        receipt = {'period': period, 'seed': seed, 'features': names,
                   'expanded_features': meta['expanded_features'], 'cutoff': baseline.CUTOFFS[period],
                   'changed_column': 'img_age_h', 'unchanged_other_inputs': True,
                   'max_parent_prediction_difference': difference,
                   'changed_origins': int(np.count_nonzero(np.abs(ages - grid['C'][:, col]) > 1e-5)),
                   'missing_raw_frame_origins': int(raw_age.reindex(origins).isna().sum()),
                   'max_repaired_age_h': float(ages.max()),
                   'mean_absolute_prediction_change': float(np.abs(b - a).mean()),
                   'max_absolute_prediction_change': float(np.abs(b - a).max()),
                   'no_fit': True, 'frozen_original_metadata': str(original / f'{period}_metadata.json')}
        write(out / f'{period}_metadata.json', receipt)
        receipts.append(receipt)
        control_frames.append(frame(origins, a))
        treatment_frames.append(frame(origins, b))
    # All treatment predictions are frozen before official truth is read.
    treatment = pd.concat(treatment_frames, ignore_index=True)
    control = pd.concat(control_frames, ignore_index=True)
    treatment.to_parquet(out / 'predictions.parquet', index=False)
    control.to_parquet(out / 'control.parquet', index=False)
    context = scorer.prepare_context(scorer.load_observations(OBS))
    scores = {}
    for name, table in (('control', control), ('treatment', treatment)):
        scores[name] = scorer.score_competition(table, context)
        array = scorer.align_predictions(table, context)
        for scope in ('total', 'f1', 'f2', 'f3'):
            assert scores[name][scope]['n_replaced'] == 0
            mask = np.ones(len(array), bool) if scope == 'total' else context['period'] == scope.upper()
            scores[name][scope]['mae'] = float(np.nanmean(np.abs(array[mask] - context['truth'][mask])))
    write(out / 'local_score.json', scores)
    write(out / 'submission.json', {'competition': scores['treatment'], 'local_only': True, 'published': False})
    write(out / 'provenance.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                  'python': platform.python_version(), 'numpy': np.__version__,
                                  'pandas': pd.__version__, 'xgboost': xgb.__version__, 'seed': seed,
                                  'source_models': str(model_root), 'data': str(DATA), 'observations': str(OBS),
                                  'classification': 'inference-only replacement bugfix diagnostic',
                                  'reproduction': receipts, 'completed_utc': datetime.now(UTC).isoformat()})
    write(out / 'complete.json', {'status': 'complete', 'publish': 'not_required',
                                'publish_reason': 'fixed-model diagnostic; corrected training untested',
                                'exit_code': 0, 'total_s': scores['treatment']['total']['s']})
    print(json.dumps({'stage': 'complete', 'seed': seed, 'control_s': scores['control']['total']['s'],
                      'treatment_s': scores['treatment']['total']['s']}), flush=True)
