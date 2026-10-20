"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1466', 'direction': 'D_TDWSAHUXT72_20261016', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2 and May29 fit cutoff; replay latest training target May26. GONG source availability uses max(obs+2h,Last-Modified), with documented fallback; all-seed poison/positive controls pass. Seed1 regression prevented adoption. Retrospective leaderboard-selected archive, not independent validation.', 'provenance': 'solar_pivot_tdwsahuxt72_20261016/run_tdwsahuxt.py; E1466/receipt_s0.json and poison.json; registered output hashes', 'source_code_sha256': {'experiments/solar_pivot_tdwsahuxt72_20261016/config.json': '6274cf3c6ffeb9d8bff0aa5e44ccf72a2db75c7066998c781eaf290067eb2710', 'experiments/solar_pivot_tdwsahuxt72_20261016/prepare_sources.py': 'a4571d09af7aeb23d16d0176e518910605da7527cef27593db66427807739feb', 'experiments/solar_pivot_tdwsahuxt72_20261016/run_tdwsahuxt.py': '8e062c947bc6826d42f175866ba16e1a82c884305e096edd73894e3b3164aa29', 'experiments/solar_pivot_tdwsahuxt72_20261016/td_field_encoder.py': '4098be5a3ce728e6907f1849358cdffbeb5c08d930626f82c3a380adcc409837', 'experiments/solar_pivot_tdwsahuxt72_20261016/tdwsahuxt_state.py': '57f8781a05453b5a0e25d0a19ff0ad2fcd727078483afc7e8977b133f7ceeb9d'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '5bb3aaf8391f6cd65efe7886df240c436267d1c4528c26c897ab2df6b7d0a194', 'predictions_official_s1.parquet': 'b600a5a98e8b8a32f6c0051300fe2bfb3981f55099f23e55bdb829a074cbb51a', 'predictions_official_s2.parquet': 'fa3566bf4486f7d6dd4eea7a931f5e657c65903e4faf7340acc6c490554a554c'}, 'input_sha256': {'member_forecasts.npz': 'af75003f35102373813dba11fbfe677f017cc642efd32bd61c4f3483e23bbc5b'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': '[axes: feat=causal-per-longitude-time-dependent-HUXt-state, model=lead-conditioned-field-encoder] Continue the E1149 L72-COND replay with the origin-causal SW-012 GONG map sequence as time-dependent PFSS/WSA boundaries propagated through HUXt; a trained lead-conditioned encoder reads the full radial-longitude field and adds only its latent block. E1455 ended before manifest creation and E1464 failed on its preparer environment. E1465 pinned all 5,956 local v2 maps into its run folder but was stopped during profile preparation after inspection found the verified profile cache index uses basenames while source rows use maps/... paths. This recovery matches by basename, then still requires each profile cache key and body SHA to match before reuse; missing profiles are built with the same pinned local implementation. Source, state, fields, and encoder artifacts stay in the new EID run output. Maps without per-file HTTP Last-Modified use observation+2h conservative availability; E1149 features and recipe stay fixed.', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = '188f51bde06c971861d7401068350ece94f17ea91f9c8350da14e6eade94ee67'


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
