"""Frozen E1711 historical fold3 diagnostics; never fits or selects a model."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import gc
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
sys.path.insert(0, str(ROOT/'leaderboard'))
from experiments.solar_full_anchor_late_head_20261023 import run as R

OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'research/expos/runs/E1711'


def run_seed(seed):
    archived = next(r for r in json.loads((SOURCE/f'receipt_s{seed}.json').read_text()) if r['fold'] == 3)
    path = SOURCE/f'model_f3_s{seed}.joblib'
    assert R.clock.sha(path) == archived['model_sha256']
    model = joblib.load(path)
    booster = model.get_booster()
    booster.set_param({'nthread': 1, 'device': 'cpu'})
    assert booster.num_boosted_rounds() == 300
    config = json.loads(booster.save_config())
    base = float(np.asarray(json.loads(config['learner']['learner_model_param']['base_score'])).reshape(-1)[0])
    ace, channels, grids = R.load_inputs(seed)
    rows, train_x = R.materialize(grids, 3, ace, channels)
    train_hash = R.digest(rows, train_x)
    assert train_hash == archived['training_sha256']
    assert len(rows) == archived['rows'] and train_x.shape[1] == 125
    assert ((rows.target + 72*R.HOUR) <= R.clock.cutoff(3)).all()
    train_y = (rows.y - rows.control).to_numpy(dtype=np.float64)
    org, control = grids[3]
    parts = []
    for start in range(0, len(org), 256):
        stop = min(start + 256, len(org))
        x, names = R.design(org[start:stop], control[start:stop], channels, ace, 'A')
        assert names == archived['feature_names']
        parts.append(x.reshape(stop-start, 72, 125)[:, 24:, :].reshape(-1, 125))
    val_x = np.concatenate(parts)
    del parts
    targets = pd.DatetimeIndex(np.repeat(org, 48)) + pd.to_timedelta(np.tile(np.arange(25, 73), len(org)), unit='h')
    actual = ace.reindex(targets).to_numpy(dtype=np.float64)
    mask = np.isfinite(actual) & (targets < R.CAP)
    val_y = actual[mask] - control[:, 24:].ravel()[mask]
    train_dm = xgb.DMatrix(train_x, nthread=1)
    val_dm = xgb.DMatrix(val_x, nthread=1)
    records = []
    for stage in range(301):
        if stage == 0:
            train_pred = np.full(len(train_y), base, dtype=np.float32)
            val_pred = np.full(len(val_x), base, dtype=np.float32)
        else:
            train_pred = booster.predict(train_dm, iteration_range=(0, stage))
            val_pred = booster.predict(val_dm, iteration_range=(0, stage))
        records.append(dict(round=stage, train_mse=float(np.mean((train_pred.astype(np.float64)-train_y)**2)),
                            validation_mse=float(np.mean((val_pred[mask].astype(np.float64)-val_y)**2))))
    assert np.array_equal(train_pred, model.predict(train_x))
    raw_path = SOURCE/'raw/roll3'/f'f3_s{seed}.parquet'
    assert R.clock.sha(raw_path) == archived['raw_sha256']
    saved = R.clock.load_grid(raw_path, org)
    replay = control.copy()
    replay[:, 24:] += val_pred.reshape(len(org), 48)
    assert np.array_equal(replay, saved), float(np.max(np.abs(replay-saved)))
    frame = pd.DataFrame(records)
    assert np.isfinite(frame.to_numpy()).all() and len(frame) == 301
    csv_path = OUT/f'e1711_fold3_seed{seed}_paired_loss.csv'
    frame.to_csv(csv_path.with_suffix('.tmp'), index=False)
    csv_path.with_suffix('.tmp').replace(csv_path)
    receipt = dict(experiment='E1711', fold=3, seed=seed, diagnostic='newly reconstructed frozen-model staged inference; not original logged history',
                   historical_validation_exposed=True, fitting=False, tuning=False, official_evaluation=False,
                   target='raw residual y-control, before calibration', metric='unweighted observed-row MSE in (km/s)^2',
                   stages=301, boosted_rounds=300, stage0_base_score=base, leads=[25,72],
                   train_rows=len(rows), validation_rows=int(mask.sum()), validation_origins=len(org),
                   train_cutoff=str(R.clock.cutoff(3)), train_max_target=str(rows.target.max()),
                   validation_origin_start=str(org.min()), validation_origin_end=str(org.max()),
                   validation_target_cap_exclusive=str(R.CAP), training_sha256=train_hash,
                   original_training_sha256_match=True, model_sha256=R.clock.sha(path),
                   raw_sha256=R.clock.sha(raw_path), final_raw_grid_exact=True,
                   final_train_wrapper_prediction_exact=True, raw_max_abs_error=float(np.max(np.abs(replay-saved))),
                   csv_sha256=R.clock.sha(csv_path), script_sha256=R.clock.sha(__file__),
                   xgboost_version=xgb.__version__, final_losses=records[-1])
    receipt_path = OUT/f'e1711_fold3_seed{seed}_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    for seed in range(3):
        run_seed(seed)
        gc.collect()
