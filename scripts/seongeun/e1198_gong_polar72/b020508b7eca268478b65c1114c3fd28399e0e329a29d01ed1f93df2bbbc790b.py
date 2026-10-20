"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1198', 'direction': 'D_GONG_POLAR72_20261014', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison; GONG availability max(issue+2h,Last-Modified). Original seeds 0-2 only. Later six-seed replication did not confirm improvement. Retrospective archive.', 'provenance': 'solar_inc_gongpolar72_20261014_r2/run_official.py; E1198 receipts/poison; replication record', 'source_code_sha256': {'experiments/d143_axisA_analog/a2b/resid_lib_a2b.py': 'f3fbb52216280ac1a158699a60f74f6bc8541a01c9d207f61b163de892ba92c4', 'experiments/d143_axisA_analog/a2b/run_official_a2b.py': '63abb966de263697e0d6bdef51dfd2cf717d8565bb24a222a03b046e6f2f4575', 'experiments/d143_axisA_analog/a2b/spec_a2b.py': '7d6a131138b6f9e17348e269d695a3ffae37428e2398ca0dd18eb65af6b710d2', 'experiments/d147_e0421_lean/lean.py': '8f2ba0b9c9dba52879683ff770810e5272a216e9eba67422a08b6bf5f8c5972b', 'experiments/d148_enlil_live/lean_enlil.py': '20ad7d4ee78ae762b6781aea6fd8e88169c95fd67770a7680ccea49f72e5be96', 'experiments/d1_longitude_strips/strips.py': '50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023', 'experiments/headroom_audit/headroom.py': '2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f', 'experiments/headroom_audit/info_ablation.py': '85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199', 'experiments/solar_inc_gongpolar72_20261014_r2/__init__.py': '273c96bdb6f1d34ec24d425eda659eb7c9faeb95001754ba6de0775b3f2d52c0', 'experiments/solar_inc_gongpolar72_20261014_r2/config.json': 'a8904153f8d828332629b064f0087a5a31b9f86eb328597c3a797dfafba941b4', 'experiments/solar_inc_gongpolar72_20261014_r2/configs.py': '46fca7bb80fdbd87f4c6eca284c18ffc51aea84b6d1e3bf59beaed646120e02c', 'experiments/solar_inc_gongpolar72_20261014_r2/gong_polar.py': '12cebf2e72deba3d933e406cc335af0471d5d12de84e900443ed4216aefcf2d8', 'experiments/solar_inc_gongpolar72_20261014_r2/metrics.py': 'e198cd32c136035413fee0469e25ae36aa48e93e2d68a28981b2eb9b4f5c8f4c', 'experiments/solar_inc_gongpolar72_20261014_r2/recipe.py': '05b86e380218383e47ddc439a567865299f05845c4bf62ac7f43508c798ad863', 'experiments/solar_inc_gongpolar72_20261014_r2/run_official.py': '8dca985748aeb4aa70537bf2dec70623bc31256d722f38e12214c42f2f020e65', 'experiments/solar_inc_gongpolar72_20261014_r2/runner.py': '630563ea7bef40ff4c1ae329d266bf6eca3d8e8fa3f44f16cecc6a343b75a3f9'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': 'a86d68594280e97307789cc8adccf4b0419bdc73ae76e81b1ffe17f99d8aedec', 'predictions_official_s1.parquet': 'c8684a18a23f75a76e6c291c62aae240ca9c4fb3772b07807e7a3ecafedd348b', 'predictions_official_s2.parquet': '6c511e78ab2fbbce14a4df72b2463748d298f812430c61188695afe7295eae23'}, 'input_sha256': {'member_forecasts.npz': '877bcfdd3075fb6b56ccd7ddef26b2e25451148df96074f1d2c9e1e51b43d2b8'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'E1022 exact ACE obs+2h replay plus eight origin-causal GONG polar-cap field summaries; three official seeds; [axes: feat=gong-polar-boundary-state]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'b020508b7eca268478b65c1114c3fd28399e0e329a29d01ed1f93df2bbbc790b'


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
