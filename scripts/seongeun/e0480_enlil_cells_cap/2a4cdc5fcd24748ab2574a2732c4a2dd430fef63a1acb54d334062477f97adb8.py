"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0480', 'direction': 'D149', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2. Residual band caps selected on preofficial OOF, not official regime labels. Initial poison positive-control probe failed (moved=0); repaired E0482 passes poison and all 3 prediction files are byte-identical. Retrospective comparator.', 'provenance': 'd149_enlil_cells/run_ec.py; E0480/poison.json; E0482 successful rerun poison and identical prediction hashes', 'source_code_sha256': {'experiments/d149_enlil_cells/cells_lib.py': '2ed6fae2e843e2e961054f081689bc40ff0589dc0bc1da18566d4269e8e98e23', 'experiments/d149_enlil_cells/run_ec.py': '2154302e6b9d557e6eeec081b06a18ee641f5f3adf2b57cc9f9126fc70673b48', 'experiments/d149_enlil_cells/spec_ec.py': '5b6c5317b511b610a15f99e11380ea80874c88fc6b4cdfd5721c8c29285df345'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': 'f9147f6917533efc0e5fab1ab765f5d46ccaf7bb596f858f48dfb0eaa20685bf', 'predictions_official_s1.parquet': '18b7b1904450729f545ea5412de3b64462b650586a6e9216879135835282aa15', 'predictions_official_s2.parquet': '6f9cf0882ecda4600150e1d8df80d03892c15184742e16b5da9a4d0c85fdd353'}, 'input_sha256': {'member_forecasts.npz': '950854de20603cf53be9fe6fc0782a7659fcee66630b6c6af93dc7302268123b'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'E0472 recipe identical (D148 lean+ENLIL s3_std: per-lead mixture + std_equal residual-analog + 5 ENLIL plus dt_target, vintage modelCompletionTime<=origin) plus residual bandcap only: per-band clip w*dv to [-CAP,+CAP] with CAP OOF-selected among 25/50/75/150 by masked pooled OOF MSE and band mask ON iff capped beats base in >=2/3 OOF windows (W0/W1/W2, OOF-only, frozen before official). Judge by lb-eval vs E0472 3281.31 (3 seeds, mean+every-seed+F1+4 cells). [axes: post=clipping/monotone]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '2a4cdc5fcd24748ab2574a2732c4a2dd430fef63a1acb54d334062477f97adb8'


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
