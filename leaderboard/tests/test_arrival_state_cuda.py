"""Synthetic paired restore readiness, never a scientific data fit."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from experiments.mse20_hss_state import model as m
from test_mse20_hss_arrival_model import frames


@pytest.mark.parametrize('device', ['cuda:0', 'cuda:1'])
@pytest.mark.parametrize('prior', [0, 1])
def test_cuda_same_model_cpu_restore(tmp_path, device, prior):
    if torch.cuda.device_count() < 2:
        pytest.skip('two CUDA devices unavailable')
    f = frames()
    cutoff = f['speed'].index[1040]
    origins = f['speed'].index[[1041, 1042, 1043]]
    config = m.candidate_configs()[prior]
    pair, _ = m.fit_predict(f, cutoff, origins, config, device, tmp_path/'model')
    restored = m.predict_saved(f, origins, tmp_path/'model', 'cpu')
    for mode in ('mse', 'peak'):
        np.testing.assert_allclose(pair[mode].pred_kms, restored[mode].pred_kms,
                                   rtol=0, atol=1e-3)
