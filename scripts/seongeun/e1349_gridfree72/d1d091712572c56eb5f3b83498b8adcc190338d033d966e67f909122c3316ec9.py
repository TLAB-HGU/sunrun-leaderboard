"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1349', 'direction': 'D_GRIDFREE72_20261015', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison; train-only native-hourly entropy/Gibbs weights. E1367 repairs receipt schema, not a distinct scientific recipe. Extra normalized-weight poison future delta0 and positive>0, seed0 scope. Retrospective archive.', 'provenance': 'solar_pivot_gridfree72_20261015/run.py; E1349/weight_audit/normalized_weight_poison.json; E1367 repair', 'source_code_sha256': {'experiments/e0472_ablation_hpo/configs.py': '46fca7bb80fdbd87f4c6eca284c18ffc51aea84b6d1e3bf59beaed646120e02c', 'experiments/e0472_ablation_hpo/recipe.py': '8f9aa430352bf671b32afb9cbbec4a47ba174c956620d0635a30d0b711e0b67e', 'experiments/solar_pivot_gridfree72_20261015/preregistration.json': '1029efa5b56443a004973ac8bbd9878ad77ef1f792296c69c76a9403859395cd', 'experiments/solar_pivot_gridfree72_20261015/run.py': '02d1b1a68d5470865bd979d373e1d38ed7becdc718bfec40f2d04d2289bbcb07', 'experiments/solar_pivot_gridfree72_20261015/training.py': '5577042ba55d08d99405353271999f863b8e428b58c7c1d7815464be4ab0abfb'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '5aa6503b9b4ecbcf47eea77c8f994a9a1df5bd4c187e8c51406ff2e013b1fb31', 'predictions_official_s1.parquet': '994f4dbe4963f2ea3532e6b1d17bc94285ed980ede9acbedc351b7fc710908b7', 'predictions_official_s2.parquet': '122055d8f254c5f9f34eed157ecf2b7ef747f7567c5edd3887e6d6ced934e006'}, 'input_sha256': {'member_forecasts.npz': 'f0e300494b31d048b277fddb594289dbb661e9b859cf291c51a1b680960d45f9'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'GRIDFREE72 E1149 all native hourly origins; positive Gibbs weights with entropy temperature selected by chronological Ridge implicit-gradient validation, full72 varying state retained; XGB/analog/L72 graph unchanged; recovery E1348 [axes: data=native-hourly, loss=learned-origin-measure]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'd1d091712572c56eb5f3b83498b8adcc190338d033d966e67f909122c3316ec9'


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
