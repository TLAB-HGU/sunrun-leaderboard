"""Reproduce E1908's fixed leaderboard predictions from hash-pinned seed forecasts.

This is archived forecast replay, not new-origin model inference or model training.
The separately recorded leaderboard timing measures actual loaded-model prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

CONFIG_SHA256 = '4e583eee5ebd4be9efc5519464cbf48fbdd21cab62d3585768f5e7279d44e3c7'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,default=Path(__file__).resolve().parents[3]/'releases/e1908')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-timing',type=Path)
    args=parser.parse_args()
    config=json.loads((args.release/'config.json').read_text())
    canonical=json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest()!=CONFIG_SHA256:
        raise ValueError('Configuration checksum differs')
    origins=pd.date_range('2026-05-31T23:00:00Z','2026-09-26T23:00:00Z',freq='h').as_unit('ns')
    seeds=[]
    records=config['data']['assets']
    if [x['seed'] for x in records]!=[0,1,2]:
        raise ValueError('Exactly original seeds 0/1/2 required')
    for record in records:
        name=record['filename']
        if Path(name).name!=name:
            raise ValueError('Asset must be a local basename')
        path=args.release/name
        if sha(path)!=record['sha256']:
            raise ValueError('Seed asset checksum differs')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'origins_ns','predictions'}:
                raise ValueError('Unexpected archive contents')
            if not np.array_equal(data['origins_ns'],origins.asi8):
                raise ValueError('Official origin grid differs')
            values=data['predictions'].copy()
        if values.shape!=(2833,72) or values.dtype!=np.float64 or not np.isfinite(values).all():
            raise ValueError('Invalid seed predictions')
        seeds.append(values)
    start=time.perf_counter()
    mean=np.mean(seeds,axis=0)
    seconds=time.perf_counter()-start
    frame=pd.DataFrame({'origin_last_input_utc':np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'),72),
                        'horizon_hours':np.tile(np.arange(1,73),2833),'pred_kms':mean.ravel()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.output,index=False)
    if args.replay_timing:
        args.replay_timing.write_text(json.dumps({'cached_averaging_seconds':seconds,
            'not_model_inference':True,'not_the_leaderboard_inference_timing':True},indent=2)+'\n')
    print(json.dumps({'candidate':'E1908','rows':len(frame),'config_sha256':CONFIG_SHA256,
                      'scope':'Fixed-grid archived seed-forecast replay'}))

if __name__=='__main__':
    main()
