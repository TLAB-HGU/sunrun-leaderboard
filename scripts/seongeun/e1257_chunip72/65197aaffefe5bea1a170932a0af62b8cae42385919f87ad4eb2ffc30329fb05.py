"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1257', 'direction': 'D196_CHUNIP72_20261014', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison; causal occupied-bin CH magnetic imbalance summaries. E1198 parent improvement failed later replication. Retrospective archive.', 'provenance': 'solar_inc_chunip72_20261014/run_official.py; E1257 receipts/poison; E1198 replication record', 'source_code_sha256': {'experiments/d143_axisA_analog/a2b/resid_lib_a2b.py': 'f3fbb52216280ac1a158699a60f74f6bc8541a01c9d207f61b163de892ba92c4', 'experiments/d143_axisA_analog/a2b/run_official_a2b.py': '63abb966de263697e0d6bdef51dfd2cf717d8565bb24a222a03b046e6f2f4575', 'experiments/d143_axisA_analog/a2b/spec_a2b.py': '7d6a131138b6f9e17348e269d695a3ffae37428e2398ca0dd18eb65af6b710d2', 'experiments/d147_e0421_lean/lean.py': '8f2ba0b9c9dba52879683ff770810e5272a216e9eba67422a08b6bf5f8c5972b', 'experiments/d148_enlil_live/lean_enlil.py': '20ad7d4ee78ae762b6781aea6fd8e88169c95fd67770a7680ccea49f72e5be96', 'experiments/d1_longitude_strips/strips.py': '50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023', 'experiments/headroom_audit/headroom.py': '2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f', 'experiments/headroom_audit/info_ablation.py': '85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199', 'experiments/solar_inc_chmorph72_20261014/morph_features.py': '0cb177cfce634d9b1d18baa0fb1fef8390095ac93245f1d750f46316770d198e', 'experiments/solar_inc_chmorph72_20261014/spec.py': '97f9deb27d25cd6837bd35d03d5d36bf590c47d5bd408b45278289207adaef35', 'experiments/solar_inc_chunip72_20261014/__init__.py': '273c96bdb6f1d34ec24d425eda659eb7c9faeb95001754ba6de0775b3f2d52c0', 'experiments/solar_inc_chunip72_20261014/ch_unip_features.py': '9a493ba8f2d035c0f1b6ccd0250a237652f550d822fa6fc25e368d857183c070', 'experiments/solar_inc_chunip72_20261014/config.json': 'a8904153f8d828332629b064f0087a5a31b9f86eb328597c3a797dfafba941b4', 'experiments/solar_inc_chunip72_20261014/configs.py': '46fca7bb80fdbd87f4c6eca284c18ffc51aea84b6d1e3bf59beaed646120e02c', 'experiments/solar_inc_chunip72_20261014/gong_manifest.json': '531f882b5d22c01d8661673dd84f5e73c27891506a8d2c115c3739e2356e9963', 'experiments/solar_inc_chunip72_20261014/gong_polar.py': '181b1d02ca3a1a23034f13e3a179162762a430fa7e6f447a4f823a0f4b24e27b', 'experiments/solar_inc_chunip72_20261014/metrics.py': 'e198cd32c136035413fee0469e25ae36aa48e93e2d68a28981b2eb9b4f5c8f4c', 'experiments/solar_inc_chunip72_20261014/recipe.py': '8417aeb3d21e0608d725eec7759232c6f58669e554865e93e09f19d06b798e7b', 'experiments/solar_inc_chunip72_20261014/run_official.py': 'd362598fe8af79a5c7a9c96924785295db539580af32d2f2128227869dae4441', 'experiments/solar_inc_chunip72_20261014/runner.py': 'f390c5ca35240fc1754929100da639fb1b55cfa3829d015a948eeee996275742', 'experiments/solar_inc_cmechip72_20261014/chip_features.py': 'f9929ca8576f0ec3f6cfc515aa216c17be7929815b32d63f3ee9dde8b4a80540'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '0e0f6ec44cc49bbafe60b960d58af629fd1594e181ea401917720e4a16a824ea', 'predictions_official_s1.parquet': '4f8683c0c9fed486fb29ac90d5864b68e55a9e2288d44e7fdbdbbabb17032f61', 'predictions_official_s2.parquet': '73efcab142fea571bc3d5076aee5888f48687b2a93f0be874e4253fa514ef9cc'}, 'input_sha256': {'member_forecasts.npz': '99be5b9fefe95fd759f4cfc8d3bb2ba76a9448d808603479ff3b0ca49eac1585'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'E1198 exact ACE+2h/GONG-polar recipe plus only 14 magnetic-flux imbalance summaries from the already selected t035 12x3 causal CH area grid; occupied-bin center samples from the origin-admissible E1198 GONG map, with all model/analog/postprocessing fixed; [axes: feat=ch-unipolarity]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '65197aaffefe5bea1a170932a0af62b8cae42385919f87ad4eb2ffc30329fb05'


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
