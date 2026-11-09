"""Prove host-specific saved-model inference and official score parity before admission."""
import argparse
import json
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import xgboost as xgb
import core
import competition
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args()
    root=args.root.resolve();assignment=json.loads((root/'assignment.json').read_text());data=Path(assignment['data']);out=root/'runs/baseline_s42'
    frames=[];difference=0.
    for period in core.baseline.CUTOFFS:
        query,_=core.baseline.load_grid(data,period,core.feature_names(core.normalize_config({})),False)
        raw=np.empty((len(query['origins']),72))
        for block,start in enumerate((0,24,48)):
            model=xgb.Booster(params=dict(nthread=4,device='cpu'));model.load_model(out/f'{period}_models/full/b{block}.ubj')
            raw[:,start:start+24]=model.inplace_predict(core.design(query,start,24))
        state=json.loads((out/f'{period}_models/frozen_calibration.json').read_text())
        pred,_=core.apply_offset(raw,np.asarray(state['offset']),core.normalize_config({}))
        with np.load(out/f'{period}_predictions.npz') as z:
            if not np.array_equal(query['origins'].asi8,z['origins']):raise ValueError('Origin mismatch')
            difference=max(difference,float(np.max(np.abs(pred-z['pred']))))
        frames.append(pd.DataFrame(dict(origin_last_input_utc=np.repeat(query['origins'].strftime('%Y-%m-%dT%H:%M:%SZ'),72),horizon_hours=np.tile(np.arange(1,73),len(pred)),pred_kms=pred.ravel())))
        print(json.dumps(dict(period=period,stage='host_reproduction',max_abs=difference)),flush=True)
    score=competition.score_competition(pd.concat(frames,ignore_index=True),competition.prepare_context(competition.load_observations(assignment['observations'])))
    stored=json.loads((out/'local_score.json').read_text());stored=stored.get('competition',stored)
    delta=abs(score['total']['s']-stored['total']['s'])
    if difference>1e-6 or delta>1e-6:raise ValueError('Host parity failed')
    core.write_json(root/'host-reproduction.json',dict(host=assignment['host'],validated=True,baseline_max_abs_difference=difference,baseline_score_difference=delta,total_s=score['total']['s'],data=core.file_identity(data),versions=dict(numpy=np.__version__,xgboost=xgb.__version__),method='full F1/F2/F3 saved-model CPU inference, frozen state, official scoring; separate smoke validates training path'))


if __name__=='__main__':main()
