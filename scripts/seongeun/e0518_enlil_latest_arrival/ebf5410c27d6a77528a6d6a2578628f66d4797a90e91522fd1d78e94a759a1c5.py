"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0518', 'direction': 'D151', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2; training cutoff May29. Latest same-CME ENLIL arrival revision/run count; completion <= origin. Original poison passes. Retrospective comparator.', 'provenance': 'd151_enlil_ext/ext_lib.py; E0518/poison.json; registered output hashes', 'source_code_sha256': {'experiments/d151_enlil_ext/ext_lib.py': '256b7c5667b360b0c58cfd1268213839f36722e783f9497232831ea838e43dec', 'experiments/d151_enlil_ext/run.py': 'a75f36cf4cbb24d23a6035561900f84cf75b01d389d93e37e0acaab72646d786', 'experiments/d151_enlil_ext/spec.py': '713836f568574a4442cda2fe2002749bd8bfd3fdbef6c90dcd5e7aeb9a610470'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '4562ea08d1180ff5c94f6e547bdd267254b156688540fce41acdb856aa01961c', 'predictions_official_s1.parquet': '829a39ebac474f87f45d69c3534b733c1de8a28b714ef296463aa8b1b3ad2ebf', 'predictions_official_s2.parquet': 'eddb22208f1045015846bec5e8bd202e6abf7af6b99d19487edcd372b2815ce7'}, 'input_sha256': {'member_forecasts.npz': 'aa73b0b157800f3e8a0d866d47903efcc0814cf1d177e9c0598cfc9f2d016f7e'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=enlil-ext] Arm A latest-run arrival on E0472: keep earliest in enlil_next_arr_h and add enlil_next_arr_latest_h plus enlil_n_runs plus enlil_revision_h (latest minus first) for the pending CME with vintage modelCompletionTime<=origin joined by CME key; new columns enlil_next_arr_latest_h enlil_n_runs enlil_revision_h; CPU base identical otherwise.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'ebf5410c27d6a77528a6d6a2578628f66d4797a90e91522fd1d78e94a759a1c5'


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
