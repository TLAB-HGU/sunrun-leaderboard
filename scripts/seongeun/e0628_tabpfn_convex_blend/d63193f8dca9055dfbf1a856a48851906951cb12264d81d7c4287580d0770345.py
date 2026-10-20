"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E0628', 'direction': 'D156', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE family, not ACE+2; May29 cutoff. Frozen E0605 TabPFN convex blend with preofficial per-lead OOF weights; no neural model is executed by this archive. Future/other-seed exclusion, purge and positive controls pass. Retrospective comparator.', 'provenance': 'fm_tabpfn/run_blend.py; E0628/poison.json; E0605/E0619 upstream poison; registered output hashes', 'source_code_sha256': {'experiments/fm_tabpfn/blend_spec.json': '27f2d4728b87d2621c364d1a14e744cda2ae39a7c026a4851204bbe9c1440115', 'experiments/fm_tabpfn/run_blend.py': '74b282ac948a2050b16064da4bc04decef1f79acf8954f4189504da57b93007f', 'experiments/fm_tabpfn/run_official.py': '5487e6d4f6a046a0b1ea4cfaaaca7ecdc1f119a03963c94336543814e779509a'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '94447ae2c59082722dcea0de5e6cf61ab3664e926a8297081c9dc567b7fbadda', 'predictions_official_s1.parquet': 'a1666ab95d65fcb9361058a4ec5f7c2998ffc8ce14334a6a0de3dd6ac81e3d74', 'predictions_official_s2.parquet': 'f5bf89b4490853acce78e4aa772b25db29d1ef5c421466c0e423646e730bdfdd'}, 'input_sha256': {'member_forecasts.npz': 'f10a5b81b232b93d13e2cd3203926dda6f74289f88428455d9ef014e97f3f22e'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'fm-BLEND arm A correction: preserve every shared OOF key while excluding rows with missing ACE y only from weight fitting and labeled-window MSE; fit the preregistered bounded per-lead convex weights with purge, then apply to E0472/E0605 official predictions. This handles observed-label gaps without changing the recipe.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'd63193f8dca9055dfbf1a856a48851906951cb12264d81d7c4287580d0770345'


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
