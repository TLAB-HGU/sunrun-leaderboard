"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0512', 'direction': 'D148', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2; training cutoff May29. Robustness/sensitivity bound, not an adoption candidate: ENLIL completion+12h availability. Successful run2 and poison passes. Retrospective comparator.', 'provenance': 'd151_e0472_audit/lean_enlil_lag.py; E0512 run2/poison; registered output hashes', 'source_code_sha256': {'experiments/d147_e0421_lean/lean.py': '8f2ba0b9c9dba52879683ff770810e5272a216e9eba67422a08b6bf5f8c5972b', 'experiments/d148_enlil_live/lean_enlil.py': '20ad7d4ee78ae762b6781aea6fd8e88169c95fd67770a7680ccea49f72e5be96', 'experiments/d151_e0472_audit/lean_enlil_lag.py': '87f06b56a1c01007825c6127aaa5d582dd3cf0e5effe026f5c5a70000ffdf089'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '160917d1f96563413a2970e5d8aeb96dfb70a22e072c8e6862210b921071776c', 'predictions_official_s1.parquet': '40a89ae43a77b788ff8060968030ee3c4bb9d620a8825037595f23c4dce0d0aa', 'predictions_official_s2.parquet': '32d9ad449837f81902639500d2f40c1b3a5503695565ac774e4ef4b240311e4d'}, 'input_sha256': {'member_forecasts.npz': 'ffdcf0273ed71a2cd0d4ede9ffa22fd9c00669ff6a167bd94dace7001d50ac59'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'E0472 audit, availability-lag sensitivity (reference E0472 3281.31, CPU): identical recipe (D148 lean+ENLIL s3_std: per-lead mixture + std_equal residual-analog + 5 DONKI WSA-Enlil features plus enlil_dt_target) with ONE change: every ENLIL run becomes available at modelCompletionTime + 12 h instead of modelCompletionTime, for training and official origins alike. DONKI posts a run some time after completion and one API snapshot cannot show that lag; the loss against E0472 measures how much the E0472 gain depends on prompt posting. Not a selection candidate; judged as a robustness bound. [axes: feat=enlil-lag]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'fc7ca30c893f42ad18acf23b7e3c25736e9915325b1217da9bcf52bfd0410e49'


def predict(members):
    return members.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--timing', required=True)
    a = ap.parse_args()
    canonical = json.dumps(CONFIG,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest()!=CONFIG_SHA256 or Path(__file__).stem!=CONFIG_SHA256:
        raise ValueError('configuration identity mismatch')
    release = Path(__file__).resolve().parents[3]/'releases'/CONFIG['data']['expos_experiment'].lower()
    for name, expected in CONFIG['data']['input_sha256'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('input identity mismatch: '+name)
    with np.load(release/'member_forecasts.npz',allow_pickle=False) as z:
        members, origins = z['forecasts'], pd.to_datetime(z['origins_ns'],utc=True)
    if members.shape!=(3,2833,72) or not np.isfinite(members).all():
        raise ValueError('invalid archived forecast matrix')
    for i in range(20):
        predict(members[:,i,:])
    pred = np.empty((len(origins),72),dtype=float)
    start = time.perf_counter()
    for i in range(len(origins)):
        pred[i] = predict(members[:,i,:])
    elapsed = time.perf_counter()-start
    if not np.array_equal(pred,members.mean(axis=0)):
        raise ValueError('ensemble reproduction mismatch')
    pd.DataFrame({'origin_last_input_utc':np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'),72),
                  'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                  'pred_kms':pred.ravel()}).to_parquet(a.output,index=False)
    report = {'inference_seconds_per_fold':elapsed/len(origins),'origins_measured':len(origins),
              'rows':int(pred.size),'scope':CONFIG['hyperparameters']['timing_scope']}
    Path(a.timing).write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
