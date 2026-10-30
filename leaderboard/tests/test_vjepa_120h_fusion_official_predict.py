"""Immutable official prediction proofs and causal fixed-PCA saved replay."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.vjepa_120h_fusion import numeric, official_predict
from test_vjepa_frozen_fusion_numeric import frames, config
from test_vjepa_120h_fusion_numeric import write_embeddings


def predictions(origins):
    return pd.DataFrame({'origin_last_input_utc': np.repeat(origins,72),
        'horizon_hours': np.tile(np.arange(1,73),len(origins)), 'pred_kms': 450.})


def test_full_grid_exact_baseline_equality_rejects_changed_value_or_key():
    origins = pd.date_range('2026-06-01',periods=3,freq='h',tz='UTC')
    original = predictions(origins)
    canonical = official_predict.same_predictions(original,original.sample(frac=1,random_state=42),origins)
    assert len(canonical)==216 and canonical.origin_last_input_utc.str.endswith('Z').all()
    changed=original.copy();changed.loc[0,'pred_kms']+=1e-9
    with pytest.raises(AssertionError):
        official_predict.same_predictions(changed,original,origins)
    with pytest.raises(ValueError,match='incomplete'):
        official_predict.same_predictions(original.iloc[:-1],original,origins)


def test_prediction_checkpoint_is_immutable(tmp_path):
    origins=pd.date_range('2026-06-01',periods=2,freq='h',tz='UTC')
    path=tmp_path/'predictions.parquet';data=predictions(origins)
    first=official_predict.immutable_predictions(path,data,origins)
    original=path.read_bytes()
    assert official_predict.immutable_predictions(path,data,origins)==first
    assert path.read_bytes()==original
    data.loc[0,'pred_kms']+=1
    with pytest.raises(AssertionError):
        official_predict.immutable_predictions(path,data,origins)
    assert path.read_bytes()==original


def test_evidence_restart_preserves_frozen_hash_and_rejects_changed_claim(tmp_path):
    path=tmp_path/'validation.json'
    first={'passed':True,'created_utc':'first','wall_seconds':1.,'sha256':'original'}
    assert official_predict.immutable_evidence(path,first)==first
    original=path.read_bytes()
    second={**first,'created_utc':'later','wall_seconds':99.}
    assert official_predict.immutable_evidence(path,second)==first
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='evidence drift'):
        official_predict.immutable_evidence(path,{**second,'sha256':'changed'})


def test_saved_official_extension_future_numeric_and_image_rows_are_invariant(tmp_path):
    f=frames();index=f['speed'].index;cutoff=index[900];origins=index[[980,990,1000]]
    cfg=config();pool=numeric.train_origins(f,cutoff,cfg['max_train_origins'])
    requested=pool.union(origins)
    embeddings=write_embeddings(tmp_path/'features',requested)
    baseline=tmp_path/'baseline';visual=tmp_path/'pca32'
    numeric.fit_predict(f,cutoff,origins,cfg,'cpu',baseline)
    expected,meta=numeric.fit_predict(f,cutoff,origins,cfg,'cpu',visual,'pca32',embeddings)
    replay=numeric.predict_saved(f,origins,visual,'cpu',embeddings)
    official_predict.same_predictions(replay,expected,origins)
    checks=official_predict.causal_replay(f,origins,{'baseline':baseline,'pca32':visual},embeddings,tmp_path/'scratch')
    assert len(checks)==12 and all(c['bitwise_equal'] for c in checks)
    with np.load(visual/'visual_transform.npz') as z:
        assert z['components'].shape==(32,30720)
        assert np.array_equal(z['train_origins'],pool.asi8)
        assert z['train_origins'].max()+72*3600*10**9<=cutoff.value
    assert meta['config']==cfg
    benchmark=official_predict.pure_inference_benchmark(f,origins,visual,embeddings,repeats=2)
    assert benchmark['inference_seconds_per_fold']>0 and len(benchmark['samples_seconds'])==6


def test_recipe_guard_rejects_replacing_fixed_pca32(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'baseline':{'config_sha256':official_predict.BASELINE_SHA},'arm':'pca16'}))
    with pytest.raises(ValueError,match='fixed PCA32'):
        official_predict.validate_contract(tmp_path)
