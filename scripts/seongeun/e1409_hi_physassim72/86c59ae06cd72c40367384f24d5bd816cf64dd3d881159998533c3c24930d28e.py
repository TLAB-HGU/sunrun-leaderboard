"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1409', 'direction': 'D_HI_PHYSASSIM72_20261015', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison. Raw-DN HI sequence residual proxy with DATE-END+1h availability; not physical arrival posterior/density inversion/brightness likelihood (CALFAC=0,DISTCORR=false). All 3 HI poison positive controls pass. Retrospective archive.', 'provenance': 'solar_pivot_hiphysassim72_20261015/run_official.py; E1409 receipts/poison', 'source_code_sha256': {'experiments/solar_pivot_hiphysassim72_20261015/anchors.json': 'dea3bcb097beec998f764f193e260daf89d6e8f486a87621ee8f3d95cdbdd786', 'experiments/solar_pivot_hiphysassim72_20261015/baseline.json': 'a8904153f8d828332629b064f0087a5a31b9f86eb328597c3a797dfafba941b4', 'experiments/solar_pivot_hiphysassim72_20261015/e1149_recipe.py': 'cb8d6f1af0b1befe3938696766fd9402b42f2fdce71f4e18a11a4a6e202e8623', 'experiments/solar_pivot_hiphysassim72_20261015/model.py': 'e7b86396a217bc118fdb608785367a29974617062bbeb611c61c8aac6eecd588', 'experiments/solar_pivot_hiphysassim72_20261015/resume_manifest.json': '4a984933879600259881abe868d1ba1a066e7d5c6876ef13d06c61690a57188b', 'experiments/solar_pivot_hiphysassim72_20261015/run_official.py': '14b7f2163b2d140fb8bfee5e47d4f34156579285900c0cac6bdcbe3a3cf2c1bc', 'experiments/solar_pivot_hiphysassim72_20261015/source.py': 'ffaef88f78326b8ef097af57837e06a9ece937a6ce325871539ccdcf313d2d8a'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '0d5383af6b485e7f661d54d5053d9d6072dc3f3e2ec844905094a38ba966adc0', 'predictions_official_s1.parquet': '845736437ea270bae336be0e2f034ea85c818a5425d7c656cc4785faaa4ce20e', 'predictions_official_s2.parquet': '207de828c465bfdbd8809d9096b748a52c32a15542be35679127e084ea596319'}, 'input_sha256': {'member_forecasts.npz': '371f81d5186d52ab69e4addf868089b532b2874ab42ca3839387200a72a25a9c'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=HI-chronological-image-residual-state] Resume the same causal raw-DN chronological residual-state proxy on E1149. E1408 failed before ExpOS loaded because the system Python lacks numpy; no scientific runner started. Use the existing .venv-fusion interpreter for the required GPU1 helper, reuse only hash-validated seed0 partial artifacts, and complete all 3 official seeds, poison, and lb-eval. No physical arrival posterior is claimed.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '86c59ae06cd72c40367384f24d5bd816cf64dd3d881159998533c3c24930d28e'


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
