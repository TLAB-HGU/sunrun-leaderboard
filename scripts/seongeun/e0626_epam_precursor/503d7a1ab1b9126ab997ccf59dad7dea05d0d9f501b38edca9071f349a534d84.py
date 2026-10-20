"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0626', 'direction': 'D156', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 base ACE contract, not ACE+2. EPAM markers available bin-close+4min, no interpolation. Latest training target May28 23:00 UTC; future invariance and positive controls pass. Retrospective comparator.', 'provenance': 'd156_nm57_feature/run.py; E0626/training_receipt.json and poison.json; registered output hashes', 'source_code_sha256': {'experiments/d143_axisA_analog/a2b/resid_lib_a2b.py': 'f3fbb52216280ac1a158699a60f74f6bc8541a01c9d207f61b163de892ba92c4', 'experiments/d143_axisA_analog/a2b/run_official_a2b.py': '63abb966de263697e0d6bdef51dfd2cf717d8565bb24a222a03b046e6f2f4575', 'experiments/d143_axisA_analog/a2b/spec_a2b.py': '7d6a131138b6f9e17348e269d695a3ffae37428e2398ca0dd18eb65af6b710d2', 'experiments/d147_e0421_lean/lean.py': '8f2ba0b9c9dba52879683ff770810e5272a216e9eba67422a08b6bf5f8c5972b', 'experiments/d148_enlil_live/lean_enlil.py': '20ad7d4ee78ae762b6781aea6fd8e88169c95fd67770a7680ccea49f72e5be96', 'experiments/d156_nm57_feature/run.py': '17d95c2caf495579dd4cd567b7f656717726c9cbc740d0ae26b19a9ed6e60616', 'experiments/d156_nm57_hourly/hourly.py': '494c1ca5df4e029142cccd7b22d4179db0d84eb7e0a8d988bcb281a6ea20b2d0', 'experiments/d156_nm57_hourly/markers.py': '779199fd14549873b663dcccbffa04261b3badbafd60aae6bb29d7025d78f4a7', 'experiments/d156_nm57_hourly/spec.json': '9f951fc03fedff7c74458a7b1a307509a7b40e27e368c32f817dbeb80b93b4b1', 'experiments/d156_nm57_zero_count/run_zero_count.py': '965aa9a3c4257dddf6b6bc46441829cfd9708eeb44732f1cee753a7cb87761a4', 'experiments/d1_longitude_strips/strips.py': '50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023', 'experiments/headroom_audit/headroom.py': '2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f', 'experiments/headroom_audit/info_ablation.py': '85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '35a35d2ea61907a7f5501e9301a27458bcd74381a2a049856cd51a72ed5ba5b5', 'predictions_official_s1.parquet': '7488486e9e5d783975acbc9803e991030e9f111bde6d5313bdbf667b1605fc2a', 'predictions_official_s2.parquet': '8b9df86793216d6a9e3c5d7b98d9d8bd905923ad6f9780ab892ef5ce3a25d2a7'}, 'input_sha256': {'member_forecasts.npz': '4e16d694f2fdf08d0c1c36b0392c9b1bfb195ef22b8996b576dd3a92006ada4d'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=hourly-epam-precursor] E0472 fresh-fit recipe plus 10 causal hourly EPAM proton-channel1 marker columns: 5 verified 24/48h entropy/fractal EWMA markers and 1h differences. Existing raw/logged EPAM features retained, so this is derived-precursor control for the separate hourly NM increment. Hourly bin-close+4min, no interpolation, stage1 scales or thresholds. Three varying seeds and all2833 origins; targets before2026-05-29. No tuning. Compare E0472 3281.3; differences below90 noise.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '503d7a1ab1b9126ab997ccf59dad7dea05d0d9f501b38edca9071f349a534d84'


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
