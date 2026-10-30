"""P1 synthetic CPU tests only; no scientific or official fits."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
from experiments.mse20_hss_arrival import model as m


def frames(n=1150):
    index=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC')
    x=np.arange(n)
    speed=380+10*np.sin(x/20)+250/(1+np.exp(np.clip(-(x-820)/3,-60,60)))/(1+np.exp(np.clip((x-900)/6,-60,60)))
    ace=pd.DataFrame({'filled_speed_kms':speed,'was_missing':np.zeros(n)},index=index)
    ch=pd.DataFrame({c:.01+.004*np.sin(x/40+j) for j,c in enumerate(m.inputs.CH_COLUMNS)},index=index)
    raw=pd.DataFrame({'ace_density_cm3':5+np.cos(x/10),'ace_temperature_k':1e5+x,
                     'ace_bt_nt':5+np.sin(x/17),'ace_speed_kms':speed},index=index)
    return m.inputs.rebuild_frames(ace,ch,raw)


def test_six_structural_configs_no_hpo_domains():
    configs=m.candidate_configs()
    assert len(configs)==6 and len({json.dumps(c,sort_keys=True) for c in configs})==6
    assert {(c['bin_hours'],c['slow_hours']) for c in configs}=={(a,b) for a in (3,6,12) for b in (72,168)}
    for config in configs:m.validate_config(config)
    with pytest.raises(ValueError):m.validate_config(dict(configs[0],epochs=1))


def test_masked_normalizer_and_importance_prior_preservation():
    raw=np.array([[1.,np.nan,0.],[3.,np.nan,0.]],dtype=np.float32)
    stats=m.fit_normalizer(raw)
    transformed=m.transform(raw,stats)
    assert stats['mean']==[2.,0.,0.] and stats['count']==[2,0,2]
    assert np.isfinite(transformed).all() and np.all(transformed[:,1]==0)
    assert transformed[:,4].sum()==0
    labels=np.array([0,0,0,1])
    q,weight=m.balanced_probabilities(labels)
    np.testing.assert_allclose(q*weight,np.full(4,.25))
    assert q[-1]==.5


@pytest.mark.parametrize('config',m.candidate_configs())
def test_paired_posterior_profiles_same_scenarios_and_model(config):
    m.setup()
    network=m.ArrivalPlateau(config,8).eval()
    x=torch.randn(4,8)
    output=network(x,torch.full((4,),500.),torch.full((4,),380.))
    torch.testing.assert_close(output['probabilities'].sum(1),torch.ones(4))
    torch.testing.assert_close(output['mse'],(output['probabilities'][:,:,None]*output['scenarios']).sum(1))
    torch.testing.assert_close(output['peak'],output['scenarios'][torch.arange(4),output['mode']])
    assert output['mse'].shape==(4,72) and torch.isfinite(output['mse']).all()
    # Force one arrival scenario, not oracle/highest-amplitude/horizonwise picking.
    with torch.no_grad():network.arrival.bias[2]=10
    modal=network(x,torch.full((4,),500.),torch.full((4,),380.))
    assert (modal['mode']==2).all()
    torch.testing.assert_close(modal['peak'],modal['scenarios'][:,2])
    loss=output['mse'].square().mean()+output['logits'].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())


def test_feature_priors_use_actual_issuance_prefix_and_future_invariance(monkeypatch):
    f=frames();origins=f['speed'].index[[1000,1002]];config=m.candidate_configs()[0]
    helper=m.onset_times;seen=[]
    def recording(speed,observed=None):
        seen.append(speed.index.max())
        return helper(speed,observed)
    monkeypatch.setattr(m,'onset_times',recording)
    before=m.make_features(f,origins,config)
    assert seen==list(origins)
    for source in ('ace','ch','raw'):
        raw={k:f[k].copy() for k in ('ace','ch','raw')}
        raw[source].loc[raw[source].index>origins.max()]=1e7
        changed=m.make_features(m.inputs.rebuild_frames(**raw),origins,config)
        np.testing.assert_array_equal(before['values'],changed['values'])
        np.testing.assert_array_equal(before['slow'],changed['slow'])
    assert before['values'].shape[1]>5000
    bad=copy.deepcopy(f);bad['ch']['official_regime']=1
    with pytest.raises(ValueError,match='schema'):m.make_features(bad,origins,config)


def test_label_horizon_dependency_and_onset_bins():
    f=frames();config=m.candidate_configs()[0];cutoff=f['speed'].index[1050]
    pool=m.training_origins(f,cutoff,config)
    assert (pool+pd.Timedelta(hours=96)<=cutoff).all()
    origins=f['speed'].index[[760,780,930]]
    features=m.make_features(f,origins,config)
    labels=m.make_labels(f,origins,cutoff,config,features['slow'])
    assert (labels['arrival']>0).any() and labels['arrival'][-1]==0
    for h,bin_index in zip(labels['onset_hours'],labels['arrival']):
        if np.isfinite(h):assert bin_index==int((h-1)//config['bin_hours'])+1
    with pytest.raises(ValueError,match='crosses cutoff'):
        m.make_labels(f,f['speed'].index[[1000]],cutoff,config,np.array([380.]))


def test_saved_cpu_pair_roundtrip_future_poison_and_training_provenance(tmp_path):
    f=frames();config=m.candidate_configs()[1]
    cutoff=f['speed'].index[1040];origins=f['speed'].index[[1041,1042,1043]]
    predictions,meta=m.fit_predict(f,cutoff,origins,config,'cpu',tmp_path/'first')
    restored=m.predict_saved(f,origins,tmp_path/'first','cpu')
    for name in ('mse','peak'):
        assert len(predictions[name])==216
        np.testing.assert_allclose(predictions[name].pred_kms,restored[name].pred_kms,atol=1e-3,rtol=0)
        subset=m.predict_saved(f,origins[1:2],tmp_path/'first','cpu')[name]
        np.testing.assert_allclose(restored[name].pred_kms.to_numpy().reshape(3,72)[1],subset.pred_kms,atol=1e-3,rtol=0)
    assert m.verify_training_provenance(f,tmp_path/'first')['normalizer']==meta['normalizer']
    poison=copy.deepcopy(f)
    for key,value in poison.items():
        if isinstance(value,(pd.Series,pd.DataFrame)) and isinstance(value.index,pd.DatetimeIndex):
            mask=value.index>cutoff
            if isinstance(value,pd.Series):value.loc[mask]=False if value.dtype==bool else 1e6
            else:
                for col in value:
                    if pd.api.types.is_numeric_dtype(value[col]):value.loc[mask,col]=1e6
    _,other=m.fit_predict(poison,cutoff,origins,config,'cpu',tmp_path/'second')
    assert meta['model_sha256']==other['model_sha256'] and meta['normalizer']==other['normalizer']
    assert pd.Timestamp(meta['train_label_max'])<=cutoff
    with pytest.raises(FileExistsError):m.fit_predict(f,cutoff,origins,config,'cpu',tmp_path/'first')
    path=tmp_path/'first/model.pt';path.write_bytes(path.read_bytes()+b'broken')
    with pytest.raises(ValueError,match='model hash'):m.predict_saved(f,origins,tmp_path/'first','cpu')


def test_declared_cutoff_guard_and_restore_metadata_tamper(tmp_path):
    config=m.candidate_configs()[0]
    with pytest.raises(ValueError,match='post-May31'):
        m.training_origins({},pd.Timestamp('2026-06-01T00:00:00Z'),config)
    with pytest.raises(ValueError,match='post-May31'):
        m.fit_predict({},pd.Timestamp('2026-06-01T00:00:00Z'),[],config,'cpu',tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
    directory=tmp_path/'metadata';directory.mkdir()
    (directory/'metadata.json').write_text('{}')
    (directory/'metadata.sha256').write_text('0'*64)
    with pytest.raises(ValueError,match='metadata hash'):
        m.load_metadata(directory)


@pytest.mark.parametrize('bins',[6,12,24])
def test_hierarchical_mode_avoids_diffuse_event_joint_map_artifact(bins):
    probabilities=torch.tensor([[.3]+[.7/bins]*bins,[.6]+[.4/bins]*bins,
                                [.5]+[.5/bins]*bins],dtype=torch.float64)
    event_probability,mode=m.hierarchical_mode(probabilities)
    # Joint MAP says no-event in all three rows despite .7 total event mass.
    assert probabilities.argmax(1).tolist()==[0,0,0]
    torch.testing.assert_close(event_probability,torch.tensor([.7,.4,.5],dtype=torch.float64))
    assert mode.tolist()==[1,0,1]  # .5 event tie and equal-bin tie are fixed prospectively.
    probabilities[0,1]=.7/bins-.01
    probabilities[0,3]=.7/bins+.01
    assert m.hierarchical_mode(probabilities)[1][0]==3


def test_hierarchical_decoder_changes_only_peak_choice_not_expectation_or_scenarios():
    config=m.candidate_configs()[0]
    m.setup();network=m.ArrivalPlateau(config,8).eval()
    bins=72//config['bin_hours']
    weights=torch.tensor([.3]+[.7/bins]*bins)
    with torch.no_grad():network.arrival.bias.copy_(weights.log())
    output=network(torch.zeros(1,8),torch.tensor([400.]),torch.tensor([380.]))
    assert output['probabilities'].argmax(1).item()==0
    assert output['mode'].item()==1 and output['event_probability'].item()>.5
    torch.testing.assert_close(output['peak'],output['scenarios'][:,1])
    torch.testing.assert_close(output['mse'],(output['probabilities'][:,:,None]*output['scenarios']).sum(1))
    assert (output['peak']>output['scenarios'][:,0]).all()
    definition=m.decoder_definition()
    assert definition['event_threshold']==.5 and definition['threshold_tuning'] is False
    assert 'not joint MAP' in definition['peak']
