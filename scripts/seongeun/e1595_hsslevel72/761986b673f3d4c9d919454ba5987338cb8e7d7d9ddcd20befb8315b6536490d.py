"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1595', 'direction': 'D_HSSLEVEL72_20261019', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical origin-hour ACE, not ACE+2. Last training target May31 23:00 UTC equals first official origin; fails stricter May29 target cutoff. Separate poison absent. No later-at-origin target found. Retrospective leaderboard-selected comparator.', 'provenance': 'solar_hsslevel72_20261019/official.py and run.py; E1595/receipt.json; registered output hashes', 'source_code_sha256': {'experiments/solar_hsslevel72_20261019/PREREG.md': '44a8cdaab6b947d5592cfe77b32c6d466307a91f9d16a411e226cb2af8f0ebe2', 'experiments/solar_hsslevel72_20261019/official.py': '2ceb778077c1e434f1f5c8f1cbdaff9748f15c9ec663c9127032ced1b9ce503f', 'experiments/solar_hsslevel72_20261019/run.py': 'd8a7ad1b63a0a596973294c66bc19554e7943e8022e16382abc3121344f0cd63'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': 'b43bbe630f5566e68c86dbfbdcc9246b15217f63337703dc2dcef83b80e88053', 'predictions_official_s1.parquet': 'f7d6d7f0cb2affa1f488cc08afe2e80c3a39d0474037fe2c9bcf3fabdeb47f38', 'predictions_official_s2.parquet': 'acae5a1f3b38cc2227fca8aae903992a9599dfcd72dec621b96fa0dbc785704f'}, 'input_sha256': {'member_forecasts.npz': '9dfa095f0fab52a5f638d344a09fdc114819a63b6e964eb5646846e13e356e8f'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': "Official-window confirmation of the registered clipped level combination. The champion's 25-72h mean and the E1466 TD-HUXt 25-72h mean are two biased estimates of the same true level; their errors are only partly correlated (s=0.935, rho=0.632), so a weight-fitted clipped combination lowers level RMSE. Weights fitted on all six roll6 folds (last origin 2026-05-28, strictly before the official window 2026-05-31..09-26). Shift clipped to +/-40 km/s and applied only to leads 25-72; leads 1-24 stay bit-identical. The runner refuses to run if the official window overlaps training. No truth, fold or score at any official time is read.", 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '761986b673f3d4c9d919454ba5987338cb8e7d7d9ddcd20befb8315b6536490d'


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
