"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1339', 'direction': 'D_L72COND_20261014', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison; masked mature-cell likelihood requires target+72h<=cutoff, chronological Ridge. Pinned v1 snapshot. Retrospective archive.', 'provenance': 'solar_pivot_masklevel72_20261015_r2/run.py; E1339 receipts/poison', 'source_code_sha256': {'/home/t-lab01/sunrun/research/e0472-ablation-hpo/campaign/configs/baseline.json': 'a8904153f8d828332629b064f0087a5a31b9f86eb328597c3a797dfafba941b4', 'experiments/e0472_ablation_hpo/configs.py': '46fca7bb80fdbd87f4c6eca284c18ffc51aea84b6d1e3bf59beaed646120e02c', 'experiments/e0472_ablation_hpo/recipe.py': '8f9aa430352bf671b32afb9cbbec4a47ba174c956620d0635a30d0b711e0b67e', 'experiments/solar_pivot_masklevel72_20261015_r2/__init__.py': '0c4cadb10545cd6619ad36e3008d993b7e1e2b6a2ad8b5d33efb9232b725eafd', 'experiments/solar_pivot_masklevel72_20261015_r2/lean_enlil_pinned.py': '87c5feb96a2c47ab5d505c11be80b969d43f202568dd77ed2fe92f52885549d8', 'experiments/solar_pivot_masklevel72_20261015_r2/preregistration.json': '0a1479b1d9236a7e6e3afeeb966a2fd821ccc4871300d7db7f83cb7d01847df3', 'experiments/solar_pivot_masklevel72_20261015_r2/run.py': '8bbbb2e08d992ab3978ac6722533671f4837f0d124532a7a709858a50d9b7be6'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '4b91b261e6bb8f967741174ecaf343a0a25a22c86b7b3ac9873a282881ebb545', 'predictions_official_s1.parquet': '8f321de5683c2a0de8b0ca8597ac01094e9e504112c9030bda9274e37d8d04fe', 'predictions_official_s2.parquet': '95546a206824218be847b0ff414e00d08e3e853ae6695e552acfb1e9dd5f9b3c'}, 'input_sha256': {'member_forecasts.npz': '79cf9ebb84f34803b46895b9695736c00c393a318a35893754c18c26d25df1b5'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: target=masked-latent-level] Replace only E1149 54/72 thresholded observed-mean Ridge branch with a masked mature-cell likelihood; profile lead residuals under sum(m*s)=0, weight by observed-cell count, choose Ridge regularization via train-only expanding chronological CV, retain all downstream rows with OOF NaN; preserve E1149 E1022 replay, ACE+2h, CH/SUVI, DONKI WSA-Enlil, XGB mixture/K15, seeds 0/1/2.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '919f4077c275d625ad8274e6f3b2728c4362ebd68e560041670919db89f8966e'


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
