"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0524', 'direction': 'D151', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2; training cutoff May29. ENLIL completed by origin, arrival correction fit pre-cutoff shock pairs and bounded +/-48h. Original poison passes. Retrospective comparator.', 'provenance': 'd151_enlil_ext/ext_lib.py and run.py; E0524/poison.json; registered output hashes', 'source_code_sha256': {'experiments/d151_enlil_ext/ext_lib.py': '5ac9a6768051f72864529112597586a62a860fe7c6be3a85949d7ea3fd94f8e2', 'experiments/d151_enlil_ext/run.py': '4d52c3f36966653946d26e017ef58a8804143d24622ff649f0bf6ab5d166aea7', 'experiments/d151_enlil_ext/spec.py': '713836f568574a4442cda2fe2002749bd8bfd3fdbef6c90dcd5e7aeb9a610470'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '39a1d5277aadcd3310510e8a6dd35c1f51df88d51a280421414632fcf02d782b', 'predictions_official_s1.parquet': '007003d1aafc3bfc4a5514082ac1572d9e3b40d7d54cdc1e7d9ee33e24cc964f', 'predictions_official_s2.parquet': 'dbf4caff929d2f363e29725b92dced6a3da91ae440a4ceec2988497c87721064'}, 'input_sha256': {'member_forecasts.npz': '44a8fc32efe257ae62fa65550ab361c0fbe41c387d5d38b7ca3e446cbf51827a'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=enlil-ext] Arm E learned arrival correction on E0472: pre-cutoff matched pairs only with blocked time-ordered validation plus drag-regime set from real-time inputs only; new columns enlil_corr_arr_h enlil_corr_dt enlil_drag_fast enlil_dbm_disagree; close E if out-of-sample MAE does not improve on 11-12h; CPU base identical otherwise.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'bb4b24d1969cd15ae9478e947f7d72b6dc5472cafb346a4114520826f4e9f7e0'


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
