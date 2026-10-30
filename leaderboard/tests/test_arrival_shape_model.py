"""Synthetic CPU checks; frozen P1 sources are imported read-only."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
from experiments.mse20_hss_shape import model as m
from test_mse20_hss_arrival_model import frames


def test_four_configs_fixed_mechanism_losses():
    configs=m.candidate_configs()
    assert len(configs)==4
    assert {(c['bin_hours'],c['slow_hours']) for c in configs}=={(6,72),(6,168),(12,72),(12,168)}
    for c in configs:
        m.validate_config(c)
        assert (c['shape_loss_weight'],c['timing_loss_weight'],c['arrival_loss_weight'])==(3.,1.,.2)
    with pytest.raises(ValueError):m.validate_config(dict(configs[0],shape_loss_weight=1.))


def test_trapezoid_anchor_rise_plateau_fall_and_no72closure():
    h=torch.tensor([-10.,0.,9.,10.,15.,20.,30.,40.,45.,50.,72.,96.])
    output=m.trapezoid_pulse(h,10.,200.,10.,20.,10.)
    torch.testing.assert_close(output,torch.tensor([0.,0.,0.,0.,100.,200.,200.,200.,100.,0.,0.,0.]))
    still_high=m.trapezoid_pulse(torch.tensor([0.,24.,72.,96.]),24.,300.,12.,80.,24.)
    assert still_high[0]==0 and still_high[1]==0 and still_high[2]==300 and still_high[3]==300
    for onset in (1.,36.,72.):
        for rise in (1.,24.):
            for duration in (6.,96.):
                for fall in (1.,48.):
                    amplitude=torch.tensor(400.,requires_grad=True)
                    pulse=m.trapezoid_pulse(torch.arange(0,97),onset,amplitude,rise,duration,fall)
                    assert torch.isfinite(pulse).all() and (pulse>=0).all() and (pulse<=amplitude).all()
                    assert pulse[0]==0
                    pulse.sum().backward();assert torch.isfinite(amplitude.grad)


def test_exact_closed_shape_targets_and_conservative_censors():
    h=torch.arange(96,dtype=torch.float32)
    segment=(400+m.trapezoid_pulse(h,0.,200.,20.,20.,20.)).numpy()
    target,mask,lower,censored=m.shape_targets(segment,400.)
    values=target*np.array([400,96,24,48])
    assert mask.all() and not lower.any() and not censored.any()
    np.testing.assert_allclose(values,[200,25,18,15],atol=1e-5)
    # End while still rising: amplitude is a lower bound; phase lengths unknown.
    target,mask,lower,censored=m.shape_targets(segment[:12],400.)
    assert mask.tolist()==[True,False,False,False]
    assert lower[0] and lower[1] and lower[3] and censored.all()
    assert target[1]==0  # Not a false exact duration equal to truncated support.
    # Peak/plateau already seen, falling tail not yet complete: one-sided fall.
    target,mask,lower,censored=m.shape_targets(segment[:49],400.)
    assert mask.all() and lower.tolist()==[False,False,False,True]
    assert censored.tolist()==[False,False,False,True]
    assert target[3]*48==pytest.approx(5.)


def test_censored_loss_one_sided_and_masked_unknowns():
    target=torch.tensor([[.5,.8,.3,.5]])
    mask=torch.tensor([[True,False,False,True]])
    lower=torch.tensor([[True,True,False,True]])
    above=torch.tensor([[.6,0.,0.,.6]],requires_grad=True)
    assert m.censored_shape_loss(above,target,mask,lower).item()==0
    below=torch.tensor([[.4,1.,1.,.4]],requires_grad=True)
    loss=m.censored_shape_loss(below,target,mask,lower);loss.sum().backward()
    assert below.grad[0,0]<0 and below.grad[0,3]<0
    assert below.grad[0,1]==0 and below.grad[0,2]==0


def test_conditional_cdf_respects_timing_distance_and_noevent_probability():
    target=torch.tensor([2])
    near=torch.tensor([[-2.,-10.,10.,-10.,-10.]])
    far=torch.tensor([[-2.,-10.,-10.,-10.,10.]])
    assert m.conditional_cdf_loss(near,target)<m.conditional_cdf_loss(far,target)
    changed=near.clone();changed[:,0]=20
    torch.testing.assert_close(m.conditional_cdf_loss(near,target),m.conditional_cdf_loss(changed,target))
    labels=np.array([0,0,0,1,2]);q,ht=m.balanced_probabilities(labels)
    per_positive=np.array([0.,0.,0.,2.,4.]);fraction=np.mean(labels>0)
    assert np.sum(q*ht*per_positive/fraction)==pytest.approx(3.)


def test_labels96h_cutoff_hash_and_tail_changes(monkeypatch):
    index=pd.date_range('2024-01-01',periods=200,freq='h',tz='UTC')
    origin=index[50];onset=origin+pd.Timedelta(hours=12)
    speed=pd.Series(400.,index=index)
    hours=np.arange(len(index))-62
    speed[:]=400+m.trapezoid_pulse(torch.tensor(hours),0.,200.,12.,45.,30.).numpy()
    f={'speed':speed,'observed':pd.Series(True,index=index)}
    monkeypatch.setattr(m,'onset_times',lambda *a:pd.DatetimeIndex([onset]))
    origins=pd.DatetimeIndex([origin]);cutoff=origin+pd.Timedelta(hours=96);config=m.candidate_configs()[0]
    before=m.make_labels(f,origins,cutoff,config,np.array([400.]))
    assert before['profile'].shape==(1,72) and before['profile96'].shape==(1,96)
    later=copy.deepcopy(f);later['speed'].loc[origin+pd.Timedelta(hours=80)]+=25
    after=m.make_labels(later,origins,cutoff,config,np.array([400.]))
    np.testing.assert_array_equal(before['profile'],after['profile'])
    assert m.shape_label_digest(before)!=m.shape_label_digest(after)
    beyond=copy.deepcopy(f);beyond['speed'].loc[beyond['speed'].index>cutoff]=1e9
    assert m.label_hashes(before)==m.label_hashes(m.make_labels(beyond,origins,cutoff,config,np.array([400.])))
    with pytest.raises(ValueError,match='crosses cutoff'):
        m.make_labels(f,origins,cutoff-pd.Timedelta(hours=1),config,np.array([400.]))


def test_paired_same_scenarios_fixedhierarchy_and_backward():
    m.setup();config=m.candidate_configs()[0];net=m.ArrivalPlateau(config,10)
    x=torch.randn(3,10)
    output=net(x,torch.full((3,),400.),torch.full((3,),380.))
    torch.testing.assert_close(output['mse'],(output['probabilities'][:,:,None]*output['scenarios']).sum(1))
    torch.testing.assert_close(output['peak'],output['scenarios'][torch.arange(3),output['mode']])
    assert (output['scenarios'][:,1:,:3]-output['scenarios'][:,:1,:3]==0).all()
    loss=output['mse'].square().mean()+m.conditional_cdf_loss(output['logits'],torch.tensor([1,2,3])).mean()
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    event,mode=m.hierarchical_mode(torch.tensor([[.3]+[.7/12]*12,[.5]+[.5/12]*12]))
    assert mode.tolist()==[1,1] and (event>=.5).all()


def test_cpu_fit_pair_provenance_and_postcutoff_poison(tmp_path):
    f=frames();cutoff=f['speed'].index[1040];origins=f['speed'].index[[1041,1042,1043]];config=m.candidate_configs()[0]
    pair,meta=m.fit_predict(f,cutoff,origins,config,'cpu',tmp_path/'first')
    assert meta['family']=='hss_shape_trapezoid' and meta['schema']['version']==2
    assert meta['train_shape_label_sha256'] and set(meta['train_label_hashes'])=={'profile96','arrival','shape','shape_mask','shape_lower_bound','shape_censored','onset_hours'}
    restored=m.predict_saved(f,origins,tmp_path/'first')
    for key in ('mse','peak'):
        np.testing.assert_allclose(pair[key].pred_kms,restored[key].pred_kms,rtol=0,atol=1e-3)
        subset=m.predict_saved(f,origins[1:2],tmp_path/'first')[key]
        np.testing.assert_allclose(restored[key].pred_kms.to_numpy().reshape(3,72)[1],subset.pred_kms,rtol=0,atol=1e-3)
    provenance=m.verify_training_provenance(f,tmp_path/'first')
    assert provenance['train_shape_label_sha256']==meta['train_shape_label_sha256']
    poisoned=copy.deepcopy(f);poisoned['speed'].loc[poisoned['speed'].index>cutoff]=1e6
    _,other=m.fit_predict(poisoned,cutoff,origins,config,'cpu',tmp_path/'second')
    assert other['model_sha256']==meta['model_sha256'] and other['normalizer']==meta['normalizer']
    path=tmp_path/'first/metadata.json';record=json.loads(path.read_text());record['train_shape_label_sha256']='0'*64
    path.write_text(json.dumps(record));(path.parent/'metadata.sha256').write_text(m.digest(path))
    with pytest.raises(ValueError,match='shape/censor'):m.verify_training_provenance(f,tmp_path/'first')
