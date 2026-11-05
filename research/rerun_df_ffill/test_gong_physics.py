import ast
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('gong_reproduction',Path(__file__).with_name('gong_physics.py'))
g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)


def test_wsa_original_equation():
    fs=np.array([1.,10.,100.,500.]);theta=np.array([0.,1.,5.,40.])
    expected=np.clip(267.5+410*np.clip(fs,1,300)**-.4*(1-.8*np.exp(-(np.clip(theta,0,30)/2)**2))**3,240,800)
    np.testing.assert_array_equal(g.wsa_speed(fs,theta),expected)


def test_all_original_huxt_numerical_functions_unchanged():
    original=(g.ROOT/'leaderboard/lb-f1f2/docs/visualizations/e1466_td_wsa_huxt72/epoch_repro_seed0/code/tdwsahuxt_state.py').read_text()
    current=Path(g.__file__).read_text()
    def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(original),functions(current)
    for name in ['_new_model','_standard_buffer','_rotated_profiles','_boundary_matrix','_boundary_at_time','_fractional_step','_step_function','_forecast_worker_init','_forecast_one']:
        assert a[name]==b[name],name


def test_summary_preserves_one_based_hour():
    trajectory=np.arange(72,dtype=np.float32)+300
    np.testing.assert_array_equal(g.summarize(trajectory),[371,72,300,323,347,371])


def test_causal_profile_late_old_map_does_not_roll_back(tmp_path):
    rows=[]
    for i,(obs,available,value) in enumerate([('2024-01-01','2024-01-01T02:00Z',300.),('2024-01-02','2024-01-02T02:00Z',500.),('2024-01-01T12:00Z','2024-01-03',100.)]):
        p=tmp_path/f'{i}.npz';np.savez(p,v_wsa_kms=np.full(36,value))
        rows.append(dict(file=str(i),observation_utc=pd.Timestamp(obs,tz='UTC').isoformat() if 'Z' not in obs else pd.Timestamp(obs).isoformat(),available_utc=pd.Timestamp(available).tz_localize('UTC').isoformat() if 'Z' not in available else pd.Timestamp(available).isoformat(),cache_path=str(p)))
    updates,available,speeds=g.causal_profiles(rows)
    assert len(updates)==2
    assert (speeds[-1]==500.).all()
    origin=pd.Timestamp('2024-01-01T10:00Z').value
    assert (speeds[np.searchsorted(available,origin,side='right')-1]==300.).all()


def test_original_six_hour_inventory_coverage():
    records=g.inventory(pd.Timestamp('2026-09-26T23:00Z'),'/tmp/nonexistent-gong-test-download')
    assert len(records)==5811
    assert all('/maps-hourly/' not in row['url'] for row in records)
    assert all(pd.Timestamp(row['available_utc'])>=pd.Timestamp(row['observation_utc'])+pd.Timedelta(hours=2) for row in records)
    assert sum(row['exists'] for row in records)==5339


def test_profile_source_rejects_wrong_size():
    record=g.inventory(pd.Timestamp('2022-10-02T00:00Z'),'/tmp/nonexistent')[0]
    record['bytes']+=1
    with pytest.raises(ValueError,match='missing/changed'):
        g.source_record(record)


def test_restored_huxt_static_mapping_matches_original_equivalence_gate(monkeypatch,tmp_path):
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path))
    import astropy.units as u
    from astropy.utils import iers
    iers.conf.auto_download=False
    profile=np.linspace(300.,600.,36)
    _,model,_=g._new_model(pd.Timestamp('2024-07-01T00:00Z').value,profile,73)
    model.ts_from_vlong()
    times=np.asarray(model.model_time.to_value(u.s))
    actual=g._boundary_matrix(model,times,np.zeros(len(times),np.int32),g._rotated_profiles(model,profile[None]),float(model.simtime.to_value(u.s)))
    np.testing.assert_allclose(actual,np.asarray(model.input_v_ts,np.float32),rtol=0,atol=1e-5)


def test_legacy_source_and_configuration_gate():
    row=g.source_record(g.inventory(pd.Timestamp('2022-10-02T00:00Z'),'/tmp/nonexistent')[0])
    legacy=g.legacy_inventory()[row['file']]
    assert g.legacy_profile(row,legacy)['v_wsa_kms'].shape==(36,)
    changed=dict(row,available_utc=(pd.Timestamp(row['available_utc'])+pd.Timedelta(hours=1)).isoformat())
    with pytest.raises(ValueError,match='source/availability'):
        g.legacy_profile(changed,legacy)


def test_forecast_resume_never_crosses_changed_history_identity(monkeypatch,tmp_path):
    from types import SimpleNamespace
    revision=[1];calls=[];origin=np.array([pd.Timestamp('2024-06-01',tz='UTC').value])
    def history(requested,available,speeds,cache):
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        state=cache/'state_at_origin.npy'
        np.save(state,np.ones((1,1,1),np.float16));np.save(cache/'origins_ns.npy',origin);np.save(cache/'state_floor_ns.npy',origin)
        g.write_json(cache/'history.json',{'revision':revision[0]})
        return origin,origin,state,SimpleNamespace(nr=1,nlon=1)
    def forecast(index):
        calls.append(revision[0]);trajectory=np.full(72,300.+revision[0],np.float32)
        return index,trajectory,g.summarize(trajectory)
    monkeypatch.setattr(g,'build_history',history);monkeypatch.setattr(g,'forecast_summary',forecast)
    args=(origin,[],origin,np.full((1,36),400.),tmp_path)
    first=g.build_forecasts(*args);second=g.build_forecasts(*args)
    assert calls==[1]
    pd.testing.assert_frame_equal(first,second)
    revision[0]=2
    third=g.build_forecasts(*args)
    assert calls==[1,2] and third.huxt_h1.iloc[0]==302
