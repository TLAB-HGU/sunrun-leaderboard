"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1584', 'direction': 'D_HUXTAMPL15_20261019', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Historical E0472 origin-hour ACE contract, not ACE+2. Frozen base plus fixed 0.15 causal E1466 HUXt Earth-path amplitude restoration. No official truth/cell routing in production. No standalone poison file; upstream lineage has poison evidence. Retrospective comparator.', 'provenance': 'solar_huxtblend15_20261019/official.py and run.py; E1584/receipt.json; huxt_extract.py; E1466 upstream', 'source_code_sha256': {'experiments/solar_huxtblend15_20261019/PREREG.md': '37424351845fa04434180675ddd83383228a27fb231cc6256684cdc8ddab0ed8', 'experiments/solar_huxtblend15_20261019/official.py': 'b02bcf105d47c55eae9b2dc3f79d31152c293016a3a9634dfdeb7fe605d29c35', 'experiments/solar_huxtblend15_20261019/run.py': 'ec65dde113cd6d1e6f555f358cea4d4ddfdb9bcce4d1bb24bd681e37b8354319'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': 'f43d319b68c212e30b1b560fcb4b3e05651f4f1c9d8a268d0c3c015437303334', 'predictions_official_s1.parquet': '26d2d23b1ffcb0f010e6b9baa30da71d88be215c5459069ecf8c8a84c2e30b9c', 'predictions_official_s2.parquet': '8be7bffd478a7a016cfa97184c2539dc977e4124910b2ba993a168f6df582b7b'}, 'input_sha256': {'member_forecasts.npz': '07a17e1a0eeb58fe47f3c05f41b737fd2a0e3190f01f8b2ae9b018ad424cfabf'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': "Official-window confirmation of the registered HUXT-AMPL15 blend: identical transform (fixed weight 0.15, per-origin amplitude restore) applied to the frozen official predictions of the current LB#1 recipe E0472 (seed-mean 3281.31), whose own preofficial OOF windows showed the same sign of gain. E1466 TD-HUXt uses no ACE history and is the only store asset whose error correlation rho is well below its skill ratio s against this lineage. Runner reads no truth, folds or score; it preserves the champion's exact origin key representation so the published validator accepts the file.", 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '4e06b2df6c8b520f2444dd296adc0a37fec83efd9adc9cd6ea3db1b77d0dfa45'


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
