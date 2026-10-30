"""CPU-only numerical controls on synthetic inputs; no official target reads."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from test_mse20_adaptive_features import frames
from experiments.mse20_adaptive import neural


def config(kind='linear',target='mr'):
    c=neural.candidate_grid()[0]
    c.update(kind=kind,width=0 if kind=='linear' else 128,target=target,
             max_epochs=2,patience=1,max_train_origins=12)
    return c


def test_grid():
    assert len(neural.candidate_grid())==8
    assert neural.candidate_grid()==neural.candidate_grid()
    for c in neural.candidate_grid():neural.validate_config(c)


@pytest.mark.parametrize('kind',['linear','mlp'])
@pytest.mark.parametrize('target',['mr','persistence'])
def test_restore_subset_causality_and_cutoff_poison(tmp_path,kind,target):
    f=frames();cutoff=f['speed'].index[850];origins=f['speed'].index[[1000,1001]]
    c=config(kind,target)
    pred,meta=neural.fit_predict(f,cutoff,origins,c,'cpu',tmp_path/'fit',fixed_epochs=2)
    assert len(pred)==144 and pd.Timestamp(meta['train_label_max'])<=cutoff
    restored=neural.predict_saved(f,origins,tmp_path/'fit','cpu')
    np.testing.assert_allclose(pred.pred_kms,restored.pred_kms,rtol=0,atol=1e-3)
    subset=neural.predict_saved(f,origins[:1],tmp_path/'fit','cpu')
    np.testing.assert_allclose(pred.pred_kms[:72],subset.pred_kms,rtol=0,atol=1e-3)
    for drop in (True,False):
        altered=copy.deepcopy(f)
        for key,value in altered.items():
            if drop:altered[key]=value.loc[:origins[0]].copy()
            elif key=='observed':value.loc[value.index>origins[0]]=False
            else:value.loc[value.index>origins[0]]=1800.
        actual=neural.predict_saved(altered,origins[:1],tmp_path/'fit','cpu')
        np.testing.assert_allclose(subset.pred_kms,actual.pred_kms,rtol=0,atol=1e-3)
    altered=copy.deepcopy(f)
    for key,value in altered.items():
        if key=='observed':value.loc[value.index>cutoff]=False
        else:value.loc[value.index>cutoff]=1800.
    _,new=neural.fit_predict(altered,cutoff,origins,c,'cpu',tmp_path/'poison',fixed_epochs=2)
    assert meta['model_sha256']==new['model_sha256']
    assert meta['scaler']==new['scaler']
    assert meta['train_origin_sha256']==new['train_origin_sha256']


def test_purged_early_stopping_and_full_refit(tmp_path):
    f=frames(3500);cutoff=f['speed'].index[3300]
    c=config();c['max_train_origins']=100
    _,meta=neural.fit_predict(f,cutoff,f['speed'].index[[3450]],c,'cpu',tmp_path/'fit')
    valid=meta['validation']
    assert pd.Timestamp(valid['train_label_max'])+pd.Timedelta(hours=72)<=pd.Timestamp(valid['validation_origin_min'])
    assert pd.Timestamp(valid['validation_label_max'])<=cutoff
    assert meta['n_train_origins']>valid['n_train']
    assert len(meta['history'])==meta['selected_epochs']
    assert 1<=meta['selected_epochs']<=2
    changed=copy.deepcopy(f)
    for key,value in changed.items():
        if key=='observed':value.loc[value.index>cutoff]=False
        else:value.loc[value.index>cutoff]=1800.
    _,other=neural.fit_predict(changed,cutoff,f['speed'].index[[3450]],c,'cpu',tmp_path/'poison')
    assert other['selected_epochs']==meta['selected_epochs']
    assert other['model_sha256']==meta['model_sha256']
    assert other['scaler']==meta['scaler']


@pytest.mark.parametrize('tamper',['metadata','model','schema'])
def test_tamper_nooverwrite(tmp_path,tamper):
    f=frames();origin=f['speed'].index[[1000]]
    neural.fit_predict(f,f['speed'].index[850],origin,config(),'cpu',tmp_path,fixed_epochs=1)
    with pytest.raises(FileExistsError):neural.fit_predict(f,f['speed'].index[850],origin,config(),'cpu',tmp_path,fixed_epochs=1)
    if tamper=='metadata':
        with (tmp_path/'metadata.json').open('a') as fh:fh.write(' ')
    elif tamper=='model':
        with (tmp_path/'model.pt').open('ab') as fh:fh.write(b'x')
    else:
        m=json.loads((tmp_path/'metadata.json').read_text());m['schema']['history_hours']=999
        (tmp_path/'metadata.json').write_text(json.dumps(m));(tmp_path/'metadata.sha256').write_text(neural.digest(tmp_path/'metadata.json'))
    with pytest.raises(ValueError):neural.predict_saved(f,origin,tmp_path,'cpu')


def test_current_demean_mask_and_schema():
    f=frames();origin=f['speed'].index[[1000]]
    inputs=neural.inputs(f,origin,'persistence')
    assert inputs[0][0,-1]==0
    np.testing.assert_array_equal(inputs[-1],np.full((1,72),f['speed'].loc[origin[0]]))
    f['observed']=f['observed'].astype(object);f['observed'].iloc[0]=None
    with pytest.raises(ValueError,match='observed'):neural.inputs(f,origin,'mr')
    f=frames();f['base']['future_regime']=1
    with pytest.raises(ValueError,match='schema'):neural.inputs(f,origin,'mr')


def test_rangeindex_ace_extra_is_cutoff_bounded(tmp_path,monkeypatch):
    f=frames();cutoff=f['speed'].index[850]
    f['ace']=pd.DataFrame({'timestamp_utc':f['speed'].index,'filled_speed_kms':f['speed'].to_numpy()})
    original=neural.training_origins
    def checked(bounded,end,cap):
        assert bounded['ace'].timestamp_utc.max()<=end
        return original(bounded,end,cap)
    monkeypatch.setattr(neural,'training_origins',checked)
    pred,_=neural.fit_predict(f,cutoff,f['speed'].index[[1000]],config(),'cpu',tmp_path,fixed_epochs=1)
    assert len(pred)==72 and f['ace'].timestamp_utc.max()>cutoff
