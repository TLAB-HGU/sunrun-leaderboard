"""Archive ensemble reproduction; timing covers cached averaging, not upstream inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {'data': {'expos_experiment': 'E1428', 'direction': 'D_BOOST_MHD72_20261015', 'input_contract': 'archived original-contract member forecasts; no upstream model is rerun', 'contract_notes': 'Documented ACE+2, May29 cutoff and passed poison. BOOST-MHD72 implements HD-only finite-volume state, not full MHD Eq19 or full-history assimilation. Stale start event exists but completed lbeval, receipts and predictions exist. Retrospective archive.', 'provenance': 'solar_enablephys_boostmhd72_20261015/run_boostmhd72.py; E1428 receipts/poison; lbeval', 'source_code_sha256': {'../data-v2/derived/vendor/pfss_wsa.py': '0a95d991cf80f1c39f7469141d5d0c85e50fa01ec60bbf0abb7e1845f9ffead6', '../research/e0472-ablation-hpo/campaign/configs/baseline.json': 'a8904153f8d828332629b064f0087a5a31b9f86eb328597c3a797dfafba941b4', 'experiments/e0472_ablation_hpo/configs.py': '46fca7bb80fdbd87f4c6eca284c18ffc51aea84b6d1e3bf59beaed646120e02c', 'experiments/e0472_ablation_hpo/recipe.py': '8f9aa430352bf671b32afb9cbbec4a47ba174c956620d0635a30d0b711e0b67e', 'experiments/gjc_inc_l72cond_20261014/run_proxy.py': 'd8271c4c9bbb02734abaf1399393183143cfe318e7551e35f6119e0086105eb9', 'experiments/solar_enablephys_boostmhd72_20261015/audit_official_manifest.py': 'eb3058a72701880c1cab41e2d2d735cd80243dea7a41c59d6afbab61fc0c94e4', 'experiments/solar_enablephys_boostmhd72_20261015/boost_operator.py': 'dd18048607712c17342aa8dc48e719b88cdb25829b675d3eba4270fda3bd308b', 'experiments/solar_enablephys_boostmhd72_20261015/boost_operator_config.json': '1dd26fb6be5c542033445b63fdc7b981ae28818d761c2e9d0f371d9540d45871', 'experiments/solar_enablephys_boostmhd72_20261015/boosted_regression_tests.json': 'a44080a6cc712ae7b678bbc3d0b9345076f9f82f4efba202f42b3c351d597797', 'experiments/solar_enablephys_boostmhd72_20261015/build_official_profiles.py': 'ac0620360a8d30cdb1782d265136530f31117959e359c84212e64371ccd7c13f', 'experiments/solar_enablephys_boostmhd72_20261015/build_official_state_cache.py': '7362164c8cf3c877473bb69227f7fb57463659a7b5daec1ef9c79e66d4f65cd8', 'experiments/solar_enablephys_boostmhd72_20261015/build_profiles.py': 'aedbfe620d57363bd4225f7a3170a880d34c884e41675f68cc202698e3dd1809', 'experiments/solar_enablephys_boostmhd72_20261015/build_state_cache.py': 'f5e8e2547fea8c19997083234c6fddfb6425656daf1a5878ad3de03a3cb14ace', 'experiments/solar_enablephys_boostmhd72_20261015/earth_query_contract.json': '2079ed99346b89c8b1dd0839044dbb9ed636dfb4dc8fbcb149378dcabf153b90', 'experiments/solar_enablephys_boostmhd72_20261015/encode_state.py': 'f24f1522c095e7dff4bc16c058f918931dceb7bbf461da99a1e0bda57bdda558', 'experiments/solar_enablephys_boostmhd72_20261015/extend_preofficial_horizon.py': 'db0fcf1b04e023441d4d0eaa8dd99b7ec885eab61ca9bfe5a523e61114268b96', 'experiments/solar_enablephys_boostmhd72_20261015/finalize_metadata.py': '3b42c6210d4c59ee2a8cbfc2e29afa2db5696699e8d85fa86d43fac8c5555a53', 'experiments/solar_enablephys_boostmhd72_20261015/official_input/profile_manifest.json': 'ad2bee162ed5582a918442e4fcfd0714ab9f0e139334e677e1261b41b65395bb', 'experiments/solar_enablephys_boostmhd72_20261015/official_input/source_receipt.json': '1d3db3a43efb6d39f73d6711352e59a5047fddf613a4764d967cd1f7b6f1b8f4', 'experiments/solar_enablephys_boostmhd72_20261015/poison.json': '225018f949b76ef6c773f35476c36dc042032829898f517b3f662de3a3f921cb', 'experiments/solar_enablephys_boostmhd72_20261015/prepare_official_source.py': '81d8db94202ef75e8b0ac1ff2925172aa87b191af56af82d7c5edfd5b78ebe5d', 'experiments/solar_enablephys_boostmhd72_20261015/prepare_source.py': '2a01dde3bf6c283b521b25495f78f945fde7888db3e3d447d8eabfb0c48fa48c', 'experiments/solar_enablephys_boostmhd72_20261015/profile_manifest.json': 'dcc91c8dbab7fd28dfee0e8486591368f54a6b6740c63f9156d6d6c7765057e4', 'experiments/solar_enablephys_boostmhd72_20261015/propagation.py': 'ba191c12c53f71085cabd41c9790013223fe351a13507f7bf6eaabb60eec5689', 'experiments/solar_enablephys_boostmhd72_20261015/recover_operator_exceptions.py': '29116f630655eeeef28d3f3161e114d3e1950a1e160cfb8f299c3a3fe6962136', 'experiments/solar_enablephys_boostmhd72_20261015/reuse_existing_profiles.py': '1c07418ea4df2521e0d987280f492c10ad9bd69d459cd3115f920b3638853fa9', 'experiments/solar_enablephys_boostmhd72_20261015/run_boostmhd72.py': '89cbd395f08c1b67556000b0f330cdcd0522d5fef3119cf0a65019bb8776d000', 'experiments/solar_enablephys_boostmhd72_20261015/run_official_state_cache.py': '26314be4304885519dea86eb21a029e93e072585ffe14ae7c920d2281f0fe0cd', 'experiments/solar_enablephys_boostmhd72_20261015/solver_inventory.json': 'dfcfc3cdae89b3bdb12f12cc377dc43422f4db4151ae67572290bc4aafb0af1c', 'experiments/solar_enablephys_boostmhd72_20261015/source_byte_pin_audit.json': '5b3d08177fa4f85bedef5e6d0825168f33cc9eacf929fb90f156fed614085014', 'experiments/solar_enablephys_boostmhd72_20261015/source_manifest_pin_audit.json': 'ce448b71294f1776ebb2c32b44c716c90ffc18f40b6ba3435b93fcd01aa1c7f8', 'experiments/solar_enablephys_boostmhd72_20261015/source_receipt.json': '6cf4557a5fdf32450e5ff3afebe96e578bfb0afa17908078e6ea1ad7c31a26b3', 'experiments/solar_enablephys_boostmhd72_20261015/state_feature_store.py': 'fac1b18ac353850747328e3a46da4a4bd0fb52f5bcaf0120523a641338bb43e2', 'experiments/solar_enablephys_boostmhd72_20261015/test_operator.py': '1976d438e804337e1003bbc3fb252024ac4b723be439bb8be5f03c6c75a34a5c', 'experiments/solar_enablephys_boostmhd72_20261015/test_source_poison.py': '87e11b4bbeb7c1da189998d2882e96bc4b07038b839e8b5020ed3c83afd39e00', 'store/surf77-source/repo/surf/surf_solvers.py': 'd3016ad65ab76f849bedda06a05688a429dda32bc6e45b51091ee0b9ae416f25'}, 'original_prediction_sha256': {'predictions_official_s0.parquet': '896f9ae69669a0864404805b05132d46fe4abf6aa413229982e0bc7b54e7d40a', 'predictions_official_s1.parquet': 'd03b9d91d69b11595c12c4cd49a3db4d7a36db66223d5df3fe4b8bd61494984d', 'predictions_official_s2.parquet': '9a9549c9f20ce8aeb81282f795e502ae84417006723d732b0ead5b5dc6a31950'}, 'input_sha256': {'member_forecasts.npz': 'e7a644b7e3ebb4dcf643c915aba9510ed147261709c248eefe3033ae819c4d57'}, 'reproduction_scope': 'reproduce equal-weight ensemble from archived member forecasts, not upstream training or new-origin inference'}, 'hyperparameters': {'seeds': [0, 1, 2], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 'ensemble': 'equal prediction mean', 'timing_scope': 'cached 3-member forecast averaging only; excludes ALL upstream model inference, training, feature generation and I/O', 'original_hypothesis': 'Closest causal implementation of BOOST-MHD72; E1428 rerun includes a writable LevelBank inference buffer after E1427 read-only reindex output caused ValueError. Add one SW-012-derived boosted finite-volume HD native-field block to E1149 with preofficial-only learned encoder, causal-valid mask and continuous lead; preserve E1149 experts, L72-COND and analog. Local core is HD-only; requested MHD Eq.19 remains unsupported. [axes: data=SW-012-boosted-HD-field]', 'internal_adoption_gate_pass': False}}
CONFIG_SHA256 = 'b8dc440d67cb73ed1d74b02b5c924006cf2653339551d2f2653125b6b4ed43db'


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
