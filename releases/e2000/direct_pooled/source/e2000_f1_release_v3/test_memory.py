import json
import numpy as np
import pandas as pd
from experiments.e2000_njobs_all.model import PooledRegressor,ModelConfig
from experiments.e2000_f1_release_v3.model import DiskBackedPooledRegressor
from experiments.e2000_f1_release_v3.launch import fit_context
from experiments.e2000_njobs_all.inputs import reference


def test_disk_backed_fit_exact_original_and_cleanup(tmp_path):
    rng=np.random.default_rng(42);x=rng.normal(size=(128,5)).astype(np.float32)
    x[::7,1]=np.nan
    y=rng.normal(450,80,size=(128,72));mask=np.ones_like(y,dtype=bool);mask[::3,1]=False;y[~mask]=np.nan
    source=tmp_path/'source';source.write_text('test')
    context=fit_context({'source_receipts':[reference(source)]},{'features':reference(source)},list('abcde'),pd.date_range('2026-03-01',periods=4,tz='UTC'))
    results=[];losses=[]
    for name,cls in [('original',PooledRegressor),('mapped',DiskBackedPooledRegressor)]:
        out=tmp_path/name
        kwargs=dict(observed_mask=mask,validation=(x[-16:],y[-16:],mask[-16:]),output_dir=out,context=context,validation_role='overlapping_diagnostic')
        m=cls(ModelConfig(n_estimators=5,n_jobs=1)).fit(x,y,**kwargs)
        results.append(m.predict(x[:12]));losses.append(pd.read_csv(out/'loss_history_s42_pooled.csv'))
        replay=cls(ModelConfig(n_estimators=5,n_jobs=1)).fit(x,y,**kwargs)
        np.testing.assert_array_equal(results[-1],replay.predict(x[:12]))
        assert not list(out.glob('pooled-expansion-*'))
    np.testing.assert_array_equal(*results)
    pd.testing.assert_frame_equal(*losses)
