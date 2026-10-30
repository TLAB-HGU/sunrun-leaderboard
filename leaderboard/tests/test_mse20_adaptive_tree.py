import copy
import json
import numpy as np
import pandas as pd
import pytest
from test_mse20_adaptive_features import frames
from experiments.mse20_adaptive import tree
from experiments.mse20_four_regimes import tree as old


def config(**kwargs):
    c=tree.candidate_grid()[1]
    c['params']={'n_estimators':3,'max_depth':2,'learning_rate':.05}
    c['max_train_origins']=12
    return c|kwargs


def test_grid_and_bands():
    grid=tree.candidate_grid()
    assert len(grid)==24 and grid==tree.candidate_grid()
    for c in grid: tree.validate_config(c)
    masks=tree.band_masks(2,3)
    assert all(x.sum()==48 for x in masks)
    np.testing.assert_array_equal(sum(x.astype(int) for x in masks),np.ones(144))
    assert masks[0][23] and not masks[0][24] and masks[1][24] and masks[1][47] and masks[2][48]


@pytest.mark.parametrize('target',['mr','direct','persistence'])
@pytest.mark.parametrize('bands',[1,3])
def test_saved_full_subset_and_label_poison(tmp_path,target,bands):
    f=frames();cutoff=f['speed'].index[850];origins=f['speed'].index[[1000,1001]]
    c=config(target=target,bands=bands)
    preds,meta=tree.fit_predict(f,cutoff,origins,c,'cpu',tmp_path/'fit')
    assert pd.Timestamp(meta['train_label_max'])<=cutoff
    pd.testing.assert_frame_equal(preds,tree.predict_saved(f,origins,tmp_path/'fit','cpu'),check_exact=True)
    pd.testing.assert_frame_equal(preds.iloc[:72].reset_index(drop=True),tree.predict_saved(f,origins[:1],tmp_path/'fit','cpu'),check_exact=True)
    poisoned=copy.deepcopy(f)
    for key, value in poisoned.items():
        if key=='observed': value.loc[value.index>cutoff]=False
        else: value.loc[value.index>cutoff]=1600.
    _,new=tree.fit_predict(poisoned,cutoff,origins,c,'cpu',tmp_path/'poison')
    assert meta['model_files']==new['model_files']
    assert meta['train_origin_sha256']==new['train_origin_sha256']


def test_baseline_model_equivalence(tmp_path):
    f=frames();cutoff=f['speed'].index[850];origins=f['speed'].index[[1000]]
    c=config(features='base')
    predicted,_=tree.fit_predict(f,cutoff,origins,c,'cpu',tmp_path/'new')
    oc={k:c[k] for k in ('id','params','seed','max_train_origins')}
    f['speed'].attrs['observed']=f['observed']
    prior,_=old.fit_predict(f,cutoff,origins,oc,'cpu',tmp_path/'old')
    pd.testing.assert_frame_equal(predicted,prior,check_exact=True)


@pytest.mark.parametrize('tamper',['metadata','model','schema'])
def test_tamper_and_no_overwrite(tmp_path,tamper):
    f=frames();origin=f['speed'].index[[1000]]
    tree.fit_predict(f,f['speed'].index[850],origin,config(),'cpu',tmp_path)
    with pytest.raises(FileExistsError):tree.fit_predict(f,f['speed'].index[850],origin,config(),'cpu',tmp_path)
    if tamper=='metadata':
        with (tmp_path/'metadata.json').open('a') as fh:fh.write(' ')
    elif tamper=='model':
        with (tmp_path/'model-band0.ubj').open('ab') as fh:fh.write(b'x')
    else:
        m=json.loads((tmp_path/'metadata.json').read_text());m['columns'][0]='future_regime'
        (tmp_path/'metadata.json').write_text(json.dumps(m))
        (tmp_path/'metadata.sha256').write_text(tree.digest(tmp_path/'metadata.json'))
    with pytest.raises(ValueError):tree.predict_saved(f,origin,tmp_path,'cpu')


def test_full_label_pool_and_weights(tmp_path):
    f=frames();f['speed'].iloc[800]=np.nan
    pool=tree.training_origins(f,f['speed'].index[850],config())
    assert not any((pool<f['speed'].index[800]) & (pool+pd.Timedelta(hours=72)>=f['speed'].index[800]))
    f=frames();f['observed'].iloc[750:800]=False
    pred,_=tree.fit_predict(f,f['speed'].index[850],f['speed'].index[[1000]],config(observed_weights=True,peak_weights=True),'cpu',tmp_path)
    assert len(pred)==72


def test_dynamic_saved_prediction_future_poison_and_drop(tmp_path):
    f=frames();origin=f['speed'].index[[1050]]
    c=config(features='dynamic',bands=3,observed_weights=True)
    expected,_=tree.fit_predict(f,f['speed'].index[850],origin,c,'cpu',tmp_path)
    for drop in (False,True):
        changed=copy.deepcopy(f)
        for key,value in changed.items():
            if drop:changed[key]=value.loc[:origin[0]].copy()
            elif key=='observed':value.loc[value.index>origin[0]]=False
            else:value.loc[value.index>origin[0]]=1800.
        pd.testing.assert_frame_equal(expected,tree.predict_saved(changed,origin,tmp_path,'cpu'),check_exact=True)


def test_history_window_changes_pool_without_future_labels():
    f=frames(19000);cutoff=f['speed'].index[-1]
    full=tree.training_origins(f,cutoff,config(max_train_origins=12000))
    recent=tree.training_origins(f,cutoff,config(history_days=365,max_train_origins=12000))
    assert full.min()<recent.min()
    assert recent.min()>=cutoff-pd.Timedelta(days=365)
    assert recent.max()+pd.Timedelta(hours=72)<=cutoff


def test_rangeindex_ace_extra_is_cutoff_bounded(tmp_path,monkeypatch):
    f=frames();cutoff=f['speed'].index[850]
    f['ace']=pd.DataFrame({'timestamp_utc':f['speed'].index,'filled_speed_kms':f['speed'].to_numpy()})
    original=tree.training_origins
    def checked(bounded,end,c):
        assert bounded['ace'].timestamp_utc.max()<=end
        assert isinstance(bounded['ace'].index,pd.Index)
        return original(bounded,end,c)
    monkeypatch.setattr(tree,'training_origins',checked)
    pred,_=tree.fit_predict(f,cutoff,f['speed'].index[[1000]],config(),'cpu',tmp_path)
    assert len(pred)==72 and f['ace'].timestamp_utc.max()>cutoff
