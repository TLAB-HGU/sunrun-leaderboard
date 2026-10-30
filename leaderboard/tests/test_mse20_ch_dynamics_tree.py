import copy
import json
import numpy as np
import pandas as pd
import pytest
from test_mse20_ch_dynamics_inputs import frames
from experiments.mse20_ch_dynamics import tree, inputs


def config(**kwargs):
    c=tree.candidate_grid()[0]
    c['params']={**c['params'],'n_estimators':3,'max_depth':2}
    c['max_train_origins']=8
    return c|kwargs


def test_grid_layout_memory_and_band_allocations():
    grid=tree.candidate_grid()
    assert len(grid)==36 and grid==tree.candidate_grid()
    assert len({json.dumps(tree.semantic_config(c),sort_keys=True) for c in grid})==36
    assert len(grid[:12])==12 and all(c['bands']==3 for c in grid[:12])
    for c in grid:
        tree.validate_config(c)
        layout=tree.feature_layout(c)
        assert layout['width']<=3000
        estimate=tree.estimate_memory(6000,c)
        assert estimate['matrix_bytes']==6000*(72//c['bands'])*layout['width']*4
        assert estimate['additional_bytes']>=2.5*estimate['matrix_bytes']+4*1024**3
    f=frames();c=config(features='summary+arrival')
    data=inputs.make_inputs(f,f['speed'].index[[1000,1001]])
    bands=[tree.band_matrix(data,c,lo,hi).reshape(2,24,-1) for lo,hi in ((1,24),(25,48),(49,72))]
    full=tree.band_matrix(data,c,1,72).reshape(2,72,-1)
    assert full.dtype==np.float32
    np.testing.assert_array_equal(np.concatenate(bands,axis=1),full)


@pytest.mark.parametrize('variant', tree.VARIANTS)
@pytest.mark.parametrize('target', ['mr','persistence'])
def test_roundtrip_full_subset_and_future_training_poison(tmp_path,variant,target):
    f=frames(); cutoff=f['speed'].index[850];origins=f['speed'].index[[1000,1001]]
    c=config(features=variant,target=target)
    p,m=tree.fit_predict(f,cutoff,origins,c,'cpu',tmp_path/'fit')
    pd.testing.assert_frame_equal(p,tree.predict_saved(f,origins,tmp_path/'fit','cpu'),check_exact=True)
    pd.testing.assert_frame_equal(p.iloc[72:].reset_index(drop=True),tree.predict_saved(f,origins[1:],tmp_path/'fit','cpu'),check_exact=True)
    changed=copy.deepcopy(f)
    for key,value in changed.items():
        value.loc[value.index>cutoff]=False if key=='observed' else 1800
    _,pm=tree.fit_predict(changed,cutoff,origins,c,'cpu',tmp_path/'poison')
    assert m['model_files']==pm['model_files'] and m['train_origins']==pm['train_origins']
    assert m['scaler']=={} and pd.Timestamp(m['train_label_max'])<=cutoff
    with pytest.raises(FileExistsError):tree.fit_predict(f,cutoff,origins,c,'cpu',tmp_path/'fit')


@pytest.mark.parametrize('what', ['metadata','model','schema','origins','scaler','source'])
def test_tamper_fail(tmp_path,what):
    f=frames();origins=f['speed'].index[[1000]]
    tree.fit_predict(f,f['speed'].index[850],origins,config(features='arrival',bands=1),'cpu',tmp_path)
    if what=='model':
        with (tmp_path/'model-band0.ubj').open('ab') as h:h.write(b'bad')
    elif what=='metadata':
        with (tmp_path/'metadata.json').open('a') as h:h.write(' ')
    else:
        meta=json.loads((tmp_path/'metadata.json').read_text())
        if what=='schema':meta['schema']['ch_columns'][0]='future_T'
        if what=='origins':meta['train_origins'][0]=meta['train_origins'][-1]
        if what=='scaler':meta['scaler']={'illegal':1}
        if what=='source':meta['source_sha256']={}
        (tmp_path/'metadata.json').write_text(json.dumps(meta))
        (tmp_path/'metadata.sha256').write_text(tree.digest(tmp_path/'metadata.json'))
    with pytest.raises(ValueError):tree.predict_saved(f,origins,tmp_path,'cpu')


def test_prediction_chunks_never_exceed_128_and_subsets(tmp_path,monkeypatch):
    f=frames(1200);c=config(features='arrival',bands=1)
    origin=f['speed'].index[[1000]]
    tree.fit_predict(f,f['speed'].index[850],origin,c,'cpu',tmp_path)
    calls=[];original=inputs.make_inputs
    def wrapped(frames,origins,target):
        calls.append(len(origins));return original(frames,origins,target)
    monkeypatch.setattr(inputs,'make_inputs',wrapped)
    origins=f['speed'].index[1000:1129]
    prediction=tree.predict_saved(f,origins,tmp_path,'cpu')
    assert calls==[128,1]
    for i in (0,64,128):
        actual=tree.predict_saved(f,origins[[i]],tmp_path,'cpu')
        pd.testing.assert_frame_equal(prediction.iloc[i*72:(i+1)*72].reset_index(drop=True),actual,check_exact=True)


def test_actual_training_label_identity_reconstructed(tmp_path):
    f=frames();origins=f['speed'].index[[1000]]
    _,meta=tree.fit_predict(f,f['speed'].index[850],origins,config(features='arrival'),'cpu',tmp_path)
    proof=tree.verify_training_provenance(f,meta)
    assert proof['train_label_sha256']==meta['train_label_sha256']
    changed=copy.deepcopy(f)
    first_label=pd.Timestamp(meta['train_origins'][0])+pd.Timedelta(hours=1)
    changed['speed'].loc[first_label]+=1
    with pytest.raises(ValueError,match='label array'):
        tree.predict_saved(changed,origins,tmp_path,'cpu')
