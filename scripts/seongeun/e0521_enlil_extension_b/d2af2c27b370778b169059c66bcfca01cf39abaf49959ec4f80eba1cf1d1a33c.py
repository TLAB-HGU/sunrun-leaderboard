"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0521', 'direction': 'D151', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2; May29 cutoff. Nearest pending ENLIL run duration/end, completion <= origin. Original poison passes. Retrospective comparator.', 'provenance': 'd151_enlil_ext source; E0521/poison.json; registered output hashes', 'source_code_sha256': {'experiments/d151_enlil_ext/ext_lib.py': '5ac9a6768051f72864529112597586a62a860fe7c6be3a85949d7ea3fd94f8e2', 'experiments/d151_enlil_ext/run.py': '4d52c3f36966653946d26e017ef58a8804143d24622ff649f0bf6ab5d166aea7', 'experiments/d151_enlil_ext/spec.py': '713836f568574a4442cda2fe2002749bd8bfd3fdbef6c90dcd5e7aeb9a610470'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '9f15205c46c64f830f625c51b0e59ec8c7c3e752ee8065101d4c5e12099cb23c', 'predictions_official_s1.parquet': 'c7219a57d11ffef769ae8fbf3b2dfa55edf9b59201124a2a085f4bb6ceb6aa7a', 'predictions_official_s2.parquet': '0267482ceff718a490fe6614f865f6c762683ea5d5811fa0d9e5236535139aa0'}, 'input_sha256': {'member_forecasts.npz': '222f16db27f8ba7a8e42765b07436ce98e7a9c78ace997b5c6171dec63207aef'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=enlil-ext] Arm B duration on E0472: enlil_dur_h of the nearest pending run (NaN kept, no imputation) and implied enlil_end_h; vintage modelCompletionTime<=origin joined by CME key; new columns enlil_dur_h enlil_end_h; CPU base identical otherwise.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'd2af2c27b370778b169059c66bcfca01cf39abaf49959ec4f80eba1cf1d1a33c'


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
