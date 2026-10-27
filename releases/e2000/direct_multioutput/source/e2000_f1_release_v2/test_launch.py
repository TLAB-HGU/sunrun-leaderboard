import numpy as np
import pandas as pd
from experiments.e2000_f1_release_v2.launch import cutoff_masks,F1_CUTOFF,F1_ORIGINS

def test_strict_cutoff_and_diagnostic():
    origins=pd.date_range('2026-01-20','2026-02-27',freq='h',tz='UTC')
    observed=np.ones((len(origins),72),dtype=bool)
    mask,diagnostic=cutoff_masks(origins,observed)
    targets=origins.tz_localize(None).to_numpy()[:,None]+np.arange(1,73,dtype='timedelta64[h]')
    assert np.all(targets[mask]<F1_CUTOFF.to_datetime64())
    assert not mask[targets==F1_CUTOFF.to_datetime64()].any()
    assert (diagnostic<=mask).all()
    assert targets[diagnostic].min()>=(F1_CUTOFF-pd.Timedelta(days=28)).to_datetime64()

def test_grid_and_observation_mask():
    assert len(F1_ORIGINS)*72==153864
    origins=pd.date_range('2026-01-20','2026-02-27',freq='h',tz='UTC')
    observed=np.ones((len(origins),72),dtype=bool);observed[::2,3]=False
    mask,diagnostic=cutoff_masks(origins,observed)
    assert not mask[::2,3].any()
    assert not diagnostic[::2,3].any()


def test_production_context_actual_fit_replay_both(tmp_path):
    from experiments.e2000_f1_release_v2 import launch
    from experiments.e2000_parallel6.model import LoggedMultiOutputRegressor,ModelConfig
    from experiments.e2000_njobs_all.model import PooledRegressor,ModelConfig as PooledConfig
    feature_file=tmp_path/'features';feature_file.write_text('fixture')
    from experiments.e2000_njobs_all.inputs import reference
    spec={'source_receipts':[reference(feature_file)]}
    receipt={'features':reference(feature_file)}
    origins=pd.date_range('2026-02-28T23:00Z',periods=4,freq='h')
    context=launch.fit_context(spec,receipt,['a','b'],origins)
    assert context['stage']=='official' and context['grid']['evaluation_period']=='F1'
    rng=np.random.default_rng(42);x=rng.normal(size=(40,2));y=rng.normal(400,20,size=(40,72));mask=np.ones_like(y,dtype=bool)
    mask[::3,0]=False;y[~mask]=np.nan
    for arm,make in [('direct',lambda:LoggedMultiOutputRegressor(ModelConfig(n_estimators=2),workers=2)),
                     ('pooled',lambda:PooledRegressor(PooledConfig(n_estimators=2,n_jobs=1)))]:
        kwargs=dict(observed_mask=mask,validation=(x[-10:],y[-10:],mask[-10:]),output_dir=tmp_path/arm,
                    context=context,validation_role='overlapping_diagnostic')
        model=make().fit(x,y,**kwargs);pred=model.predict(x[:4])
        restored=make().fit(x,y,**kwargs)
        np.testing.assert_array_equal(pred,restored.predict(x[:4]))
        assert pred.shape==(4,72) and np.isfinite(pred).all()
        losses=list((tmp_path/arm).glob('loss_history*.csv'))
        assert len(losses)==(72 if arm=='direct' else 1)
        assert all(len(pd.read_csv(p))==2 for p in losses)
