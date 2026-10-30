from experiments.mse20_adaptive import search as s, protocol as p
from experiments.mse20_adaptive import tree
from tests.test_mse20_adaptive_protocol import metric


def test_same_configuration_retried_only_for_gpu_capacity(monkeypatch):
    attempts=[]
    def run(family,cfg,*a):
        attempts.append(cfg.copy())
        if len(attempts)<3:raise p.GPUBusy('busy')
        return {'status':'ok'}
    monkeypatch.setattr(s,'run_trial',run)
    monkeypatch.setattr(p,'deadline',lambda *a:1000)
    monkeypatch.setattr(s.time,'sleep',lambda _:None)
    cfg={'id':'same'}
    assert s.run_with_capacity_wait('tree',cfg,None,None,None)=={'status':'ok'}
    assert attempts==[cfg]*3


def test_config_hash_and_registered_grid():
    grid=tree.candidate_grid()
    assert len(grid)==24 and len({s.config_hash(v) for v in grid})==24
    for cfg in grid:
        tree.validate_config(cfg)
        assert s.config_hash(cfg)==s.config_hash(dict(reversed(list(cfg.items()))))


def test_all_gate_and_peak_failures_remain_distinct():
    assert s.objective_value(metric(status='pass',mse=3200,cell=3700))<s.objective_value(metric())
    assert s.objective_value(metric())<s.objective_value(metric(mse=3000,cell=3500,precision=.9))
