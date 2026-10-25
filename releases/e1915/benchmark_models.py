"""Measure actual E1915 model prediction after loading, excluding data preparation.

The leaderboard inference time covers the real prediction path: nine loaded CPU
stable-UWR ensembles (three seeds x three horizon bands) predicting the frozen
official grid, plus the equal mean of the three seeds.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

LB = Path('/home/t-lab01/sunrun/leaderboard')
CODE = LB / 'experiments/stable_uwr_worker4_20261025'
CAMPAIGN = LB.parent / 'research/literature-transfer-r4-20261025T102156Z/worker-4/campaign'
BUNDLE = CAMPAIGN / 'bundle-uwr'
OUT = LB.parent / 'research/expos/runs/E1915'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(LB))
sys.path.insert(0, str(CODE))
import gate
import runner
from contracts import TargetSpec, load_checkpoint

SEEDS = [0, 1, 2]
BANDS = ((1, 6), (7, 24), (25, 72))


def main():
    manifest = gate.read(BUNDLE / 'manifest.json')
    inputs = gate.read(BUNDLE / 'inputs.json')
    receipt = gate.read(OUT / 'user-publication-execution.json')
    assert receipt['status'] == 'completed'
    assert all(runner.file_hash(p) == sha for p, sha in receipt['outputs'].items())
    recipe = manifest['recipe']
    target = TargetSpec(recipe['target'])
    test = runner._read_artifact(inputs['grids']['official'])
    features = inputs['features']
    loaded = {}
    records = []
    for seed in SEEDS:
        loaded[seed] = []
        for lo, hi in BANDS:
            method = gate.read(OUT / 'method-receipts' / f'official_band{lo}_{hi}_s{seed}.json')
            path = OUT / 'checkpoints' / (method['identity'] + '.model')
            model = load_checkpoint(path, method['identity'])
            mask = test.h.between(lo, hi).to_numpy()
            X = test.loc[mask, list(features)]
            speed = test.loc[mask, 'v_lag0'].to_numpy()
            loaded[seed].append((mask, X, speed, model, [lo, hi]))
            records.append(dict(seed=seed, band=[lo, hi], path=str(path),
                                sha256=runner.file_hash(path), identity=method['identity'],
                                actual_rounds=method['training_curve'][-1]['trees'],
                                prior_rounds=method['base_curve'][-1]['trees']))
    original = [pd.read_parquet(OUT / f'predictions_official_s{s}.parquet').pred_kms.to_numpy()
                for s in SEEDS]

    def predict():
        seeds = []
        for seed in SEEDS:
            values = np.empty(len(test), dtype=np.float64)
            for mask, X, speed, model, band in loaded[seed]:
                values[mask] = model.predict(X, current_speed=speed, target=target)
            seeds.append(values)
        return seeds, np.mean(seeds, axis=0)

    seeds, _ = predict()
    assert all(np.array_equal(a, b) for a, b in zip(seeds, original)), 'loaded models do not reproduce outputs'
    elapsed = []
    for _ in range(3):
        start = time.perf_counter()
        seeds, mean = predict()
        elapsed.append(time.perf_counter() - start)
    assert all(np.array_equal(a, b) for a, b in zip(seeds, original))
    result = dict(candidate='E1915', publication_execution=receipt['publication_execution'],
                  n_origins=2833, n_rows=len(test), seeds=SEEDS,
                  batch_seconds=elapsed, inference_seconds_per_fold=float(np.mean(elapsed) / 2833),
                  scope='Actual predictions of nine loaded CPU stable-UWR ensembles and the equal mean of three seeds',
                  excluded=['data loading', 'training', 'model initialization/loading',
                            'feature preparation', 'output writing'],
                  n_jobs_per_model=2, checkpoints=records, prediction_reproduction_max_abs_diff=0.0,
                  per_stage_rounds={'earlier_snapshot': 900, 'final_single_learner': 900})
    (HERE / 'inference-timing.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}))


if __name__ == '__main__':
    main()
