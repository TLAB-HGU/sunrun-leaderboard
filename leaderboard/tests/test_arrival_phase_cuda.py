"""Synthetic CUDA head readiness and strict paired CPU restoration."""
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
from experiments.mse20_hss_phase import model as m
from test_arrival_phase_model import frames,_fake_backbone


@pytest.mark.parametrize('device',['cuda:0','cuda:1'])
@pytest.mark.parametrize('variant',[0,1])
def test_synthetic_composite_cuda_cpu_restore(tmp_path,monkeypatch,device,variant):
    if torch.cuda.device_count()<2:pytest.skip('two CUDA devices unavailable')
    f=frames(5900);cutoff=f['speed'].index[5700];origins=f['speed'].index[[5701,5702]]
    pool=pd.DatetimeIndex([cutoff-pd.Timedelta(days=120),cutoff-pd.Timedelta(days=30)])
    monkeypatch.setattr(m,'training_origins',lambda *args:pool)
    _fake_backbone(monkeypatch)
    (tmp_path/'registered.json').write_text('{}')
    pair,_=m.fit_predict(f,cutoff,origins,m.candidate_configs()[variant],device,tmp_path/'head/model')
    restored=m.predict_saved(f,origins,tmp_path/'head/model','cpu')
    for mode in ('mse','peak'):
        np.testing.assert_allclose(pair[mode].pred_kms,restored[mode].pred_kms,rtol=0,atol=1e-3)
