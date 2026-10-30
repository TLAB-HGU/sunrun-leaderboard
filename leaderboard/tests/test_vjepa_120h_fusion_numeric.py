"""Matched numerical recipe, causal PCA and immutable saved inference checks."""
import copy

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_four_regimes import tree
from experiments.vjepa_120h_fusion import numeric, compare
from test_vjepa_frozen_fusion_numeric import frames, config


def write_embeddings(directory,index,values=None):
    directory.mkdir(parents=True,exist_ok=True)
    values = np.random.default_rng(4).normal(size=(len(index),30720)).astype('float32') if values is None else values
    path = directory / 'embeddings.npy'
    np.save(path,values)
    np.savez(directory / 'embedding_metadata.npz',origins=index.asi8,source_end_ns=index.asi8[:,None] + np.asarray([-104,-89,-74,-59,-44,-29,-14,0])*3600*10**9,quality=np.ones((len(index),45),dtype='float32'))
    return path


def test_baseline_parity_saved_grid_and_future_numeric(tmp_path):
    f = frames(); index = f['speed'].index; cutoff = index[900]; origins = index[[980,990]]
    a,ma = numeric.fit_predict(f,cutoff,origins,config(),'cpu',tmp_path / 'adapter')
    b,mb = tree.fit_predict(f,cutoff,origins,config(),'cpu',tmp_path / 'original')
    np.testing.assert_array_equal(a.pred_kms,b.pred_kms)
    assert ma['train_origin_sha256'] == mb['train_origin_sha256']
    expected = numeric.predict_saved(f,origins[:1],tmp_path / 'adapter','cpu')
    poisoned = copy.deepcopy(f)
    for key,data in poisoned.items():
        data.loc[data.index > origins[0]] = False if key == 'observed' else 999999
    deleted = {k:v.loc[:origins[0]] for k,v in f.items()}
    for changed in (poisoned,deleted):
        actual = numeric.predict_saved(changed,origins[:1],tmp_path / 'adapter','cpu')
        np.testing.assert_array_equal(actual.pred_kms,expected.pred_kms)
    assert len(a) == 144 and np.isfinite(a.pred_kms).all()


@pytest.mark.parametrize('components',[16,32])
def test_train_only_pca_and_future_embedding_invariance(tmp_path,components):
    f = frames(); index = f['speed'].index; cutoff = index[900]; origins = index[[980,990]]
    pool = numeric.train_origins(f,cutoff,32)
    requested = pool.union(origins)
    values = np.random.default_rng(4).normal(size=(len(requested),30720)).astype('float32')
    values[0,:10] = np.nan
    path = write_embeddings(tmp_path / 'source',requested,values)
    out = tmp_path / 'model'
    pred,meta = numeric.fit_predict(f,cutoff,origins,config(),'cpu',out,f'pca{components}',path)
    replay = numeric.predict_saved(f,origins,out,'cpu',path)
    np.testing.assert_allclose(pred.pred_kms,replay.pred_kms,atol=1e-4)
    assert len(pred) == 144 and np.isfinite(pred.pred_kms).all()
    assert meta['train_origin_sha256'] == __import__('hashlib').sha256(pool.asi8.tobytes()).hexdigest()
    with np.load(out / 'visual_transform.npz') as z:
        original = z['components'].copy()
        assert z['components'].dtype == np.float32
        assert z['train_origins'].max() + 72*3600*10**9 <= cutoff.value
    altered = values.copy(); altered[requested > cutoff] = 1e9
    changed = write_embeddings(tmp_path / 'changed',requested,altered)
    numeric.visual_features(changed,pool,requested,tmp_path / 'transform',components)
    with np.load(tmp_path / 'transform/visual_transform.npz') as z:
        np.testing.assert_array_equal(original,z['components'])
    expected = numeric.predict_saved(f,origins[:1],out,'cpu',path)
    for truncate in (False,True):
        kept = requested <= origins[0] if truncate else np.ones(len(requested),bool)
        altered = values.copy(); altered[requested > origins[0]] = 1e9
        changed = write_embeddings(tmp_path / f'future{truncate}',requested[kept],altered[kept])
        actual = numeric.predict_saved(f,origins[:1],out,'cpu',changed)
        np.testing.assert_array_equal(actual.pred_kms,expected.pred_kms)
    altered = values.copy(); altered[-2] += 1
    changed = write_embeddings(tmp_path / 'corrupt',requested,altered)
    with pytest.raises(ValueError,match='identity mismatch'):
        numeric.predict_saved(f,origins[:1],out,'cpu',changed)


def test_future_source_and_missing_embeddings_rejected(tmp_path):
    index = pd.date_range('2024-01-01',periods=32,freq='h',tz='UTC')
    path = write_embeddings(tmp_path,index)
    with pytest.raises(ValueError,match='missing planned train'):
        numeric.visual_features(path,index+pd.Timedelta(days=2),index,tmp_path / 'model')
    np.savez(tmp_path / 'embedding_metadata.npz',origins=index.asi8,source_end_ns=index.asi8+1,quality=np.ones((32,45),dtype='float32'))
    with pytest.raises(ValueError,match='future image'):
        numeric.read_embeddings(path)


def test_dev_choice_and_selection_gate():
    rows = {a:{'metrics':dict(mse=m,precision=.5,recall=.5)} for a,m in [('baseline',100),('pca16',90),('pca32',90)]}
    assert compare.development_choice(rows) == 'pca16'
    assert compare.operational_choice(rows,'pca16') == 'pca16'
    rows['pca16']['metrics']['recall'] = .4
    assert compare.operational_choice(rows,'pca16') == 'baseline'
    # Better selection PCA32 does not permit replacing the development choice.
    rows['pca32']['metrics']['mse'] = 1
    assert compare.operational_choice(rows,'pca16') == 'baseline'


def test_scoring_same_support_and_masks():
    f = frames(); origins = f['speed'].index[[980,990]]
    pred = pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),2),'pred_kms':450.})
    f['observed'].iloc[995] = False
    a = compare.score(pred,f,origins)
    pred.pred_kms += 20
    b = compare.score(pred,f,origins)
    for key in ('prediction_keys_sha256','target_observation_mask_sha256'):
        assert a[key] == b[key]
    assert a['n_observed'] == 142
