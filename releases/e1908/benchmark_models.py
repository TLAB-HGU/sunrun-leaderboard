"""Measure actual E1908 model prediction after loading, excluding data preparation."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

LB = Path('/home/t-lab01/sunrun/leaderboard')
CAMPAIGN = LB.parent / 'research/literature-transfer-r3-20261025T082601Z/worker-3/tsaa'
sys.path.insert(0, str(CAMPAIGN / 'code'))
import gate
import runner
from contracts import TargetSpec, checkpoint_identity, load_checkpoint

def main():
    bundle = CAMPAIGN / 'campaign/candidate'
    manifest = gate.read(bundle / 'manifest.json')
    inputs = gate.read(bundle / 'inputs.json')
    out = LB.parent / 'research/expos/runs/E1908'
    receipt = gate.read(out / 'user-publication-execution.json')
    assert receipt['status'] == 'completed'
    assert all(runner.file_hash(p) == sha for p, sha in receipt['outputs'].items())
    recipe = manifest['recipe']
    cutoff = pd.Timestamp(manifest['contract']['cutoff'])
    train = runner._read_artifact(inputs['train'])
    train = train[train.target_t < cutoff]
    ref = inputs['grids']['official']
    test = runner._read_artifact(ref)
    features = inputs['features']
    target = TargetSpec(recipe['target'])
    grid = dict(name='official', start=pd.Timestamp(ref['start']).isoformat(),
                end=pd.Timestamp(ref['end']).isoformat())
    records = []
    models = {}
    for seed in [0, 1, 2]:
        models[seed] = []
        for lo, hi in runner._head_bands(recipe):
            pool = runner._pool(train, recipe, lo, hi)
            params = {k:recipe[k] for k in ('max_depth','n_estimators','learning_rate','subsample',
                                          'colsample_bytree','min_child_weight','reg_lambda')}
            params.update(objective='reg:squarederror',tree_method='hist',device='cpu',random_state=seed,n_jobs=2)
            hyper = dict(recipe=recipe,estimator=params,stage='official',head_band=[lo,hi],
                         manifest_sha256=gate.digest(manifest),mask_sha256=manifest['contract']['mask_sha256'],
                         training_rows_hash=runner.mask_hash(pool))
            identity = checkpoint_identity(code_hashes=runner.code_hashes(),
                data_hashes={'train':inputs['train']['sha256'],'grid':ref['sha256']},
                target=target,seed=seed,hyperparameters=hyper,feature_names=features,
                cutoff=cutoff.isoformat(),grid=grid,source_availability=inputs['source_availability'])
            path = out / 'checkpoints' / (identity + '.model')
            model = load_checkpoint(path, identity)
            assert model.estimator.get_booster().num_boosted_rounds() == 900
            mask = test.h.between(lo,hi).to_numpy()
            # All feature preparation is performed before the measured section.
            X = test.loc[mask, features]
            speed = test.loc[mask, 'v_lag0'].to_numpy()
            models[seed].append((mask, X, speed, model))
            records.append(dict(seed=seed,band=[lo,hi],path=str(path),sha256=runner.file_hash(path),
                                identity=identity,actual_rounds=900))
    def predict():
        seeds = []
        for seed in [0, 1, 2]:
            values = np.empty(len(test), dtype=np.float64)
            for mask, X, speed, model in models[seed]:
                values[mask] = model.predict(X, current_speed=speed, target=target)
            seeds.append(values)
        return seeds, np.mean(seeds, axis=0)
    original = [pd.read_parquet(out / f'predictions_official_s{s}.parquet').pred_kms.to_numpy()
                for s in [0,1,2]]
    seeds, _ = predict()
    assert all(np.array_equal(a,b) for a,b in zip(seeds,original))
    elapsed=[]
    for _ in range(3):
        start=time.perf_counter()
        seeds, mean=predict()
        elapsed.append(time.perf_counter()-start)
    assert all(np.array_equal(a,b) for a,b in zip(seeds,original))
    result=dict(candidate='E1908',n_origins=2833,n_rows=len(test),seeds=[0,1,2],
        batch_seconds=elapsed,inference_seconds_per_fold=float(np.mean(elapsed)/2833),
        scope='Actual predictions of nine loaded CPU XGBoost heads and equal mean of three seeds',
        excluded=['data loading','training','model initialization/loading','feature preparation','output writing'],
        n_jobs_per_model=2,checkpoints=records,prediction_reproduction_max_abs_diff=0.0)
    Path(__file__).with_name('inference-timing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'}))
if __name__=='__main__':
    main()
