"""E1532 three-seed residual-head ensemble on frozen causal forecast features.
Run from a checkout: python <this_script> --output predictions.parquet --timing timing.json
Requires numpy, pandas, pyarrow. Artifacts live in releases/e1532/.
This reproduces the final head from prepared upstream forecasts, not upstream training.
Timing covers all three residual heads and averaging; excludes frozen upstream products.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
CONFIG = {'data': {'experiment': 'E1532', 'sources': ['SW-001 ACE obs+2h', 'SW-012 GONG/PFSS/WSA/HUXt frozen E1466 channel', 'SW-062 OMNI2 training-only frozen E1506 expert', 'E1149 frozen champion forecasts'], 'feature_contract': 'Frozen causal forecast trajectories as prepared features; no upstream model inference in this submission', 'train_targets_before_utc': '2026-05-29T00:00:00Z', 'input_sha256': {'receipt_s1.json': 'f1a1b4e56804ec4a3e1a64ab03f71ab17efeef060e4e75e4eefa9d0afddcb50d', 'receipt_s0.json': 'c36795ddad63ed71e954d21791dddf7bd7974cb8919f1df16d2337b6c8dad612', 'prepared_inputs.npz': 'eac0e3b10ec6e335a624ff2ff44fce7632d0590e58be59f667df08335f027656', 'model.npz': '8fde388d19e45b40c6f663bba2af1648451f0ae186a1f1aebd918fd8e8406778', 'receipt_s2.json': '15fccb6b4c51cd5d3cb089a66b8d5b3ea7eea1b2db9ed3c4fd86e73093a691bf'}}, 'hyperparameters': {'model': 'horizon-wise champion-anchored ridge residual head', 'ridge_lambda_times_n': 0.01, 'horizon_pool_halfwidth': 3, 'head_fit': 'all six preofficial folds; observed-only labels; cutoff mask', 'seeds': [0, 1, 2], 'ensemble': 'equal prediction mean', 'timing_scope': '3 residual-head predictions plus seed mean; upstream E1149/E1506/HUXt products are frozen input features'}}
CONFIG_SHA256 = 'fecbdfb00fb28692c85c88c7b8bc3d2b0f6bc1fcfe70247a5ba777d4e62e7512'

def predict(champion, design, beta):
    return (champion + np.einsum('shk,shk->sh',design,beta)).mean(axis=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--timing',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[3];release=root/'releases/e1532'
    canonical=json.dumps(CONFIG,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest()==CONFIG_SHA256==Path(__file__).stem
    for name,digest in CONFIG['data']['input_sha256'].items():
        assert hashlib.sha256((release/name).read_bytes()).hexdigest()==digest,name
    with np.load(release/'prepared_inputs.npz') as z:
        origins=pd.to_datetime(z['origins_ns'],utc=True);c=z['champion'];x=z['design']
    with np.load(release/'model.npz') as z: beta=z['beta']
    assert c.shape==(3,2833,72) and x.shape==(3,2833,72,4) and beta.shape==(3,72,4)
    # Each call issues one full 72h ensemble forecast; no file I/O inside timing.
    for i in range(20): predict(c[:,i],x[:,i],beta)
    pred=np.empty((len(origins),72));t0=time.perf_counter()
    for i in range(len(origins)): pred[i]=predict(c[:,i],x[:,i],beta)
    seconds=(time.perf_counter()-t0)/len(origins)
    assert np.isfinite(pred).all() and (pred>0).all() and (pred<=3000).all()
    pd.DataFrame({'origin_last_input_utc':np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'),72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),'pred_kms':pred.ravel()}).to_parquet(a.output,index=False)
    report={'inference_seconds_per_fold':seconds,'origins_measured':len(origins),'seeds':[0,1,2],'scope':CONFIG['hyperparameters']['timing_scope'],'rows':int(pred.size)}
    Path(a.timing).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
