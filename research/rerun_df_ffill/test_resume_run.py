"""Small real boosters and fake immutable snapshots exercise checkpoint boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
SPEC=importlib.util.spec_from_file_location('resume_run',Path(__file__).with_name('resume_run.py'))
R=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(R)


def booster(path,rounds=2):
    rng=np.random.default_rng(1);x=rng.normal(size=(80,3));y=rng.normal(400,30,(80,24))
    m=xgb.train(dict(max_depth=2,nthread=1,tree_method='hist',multi_strategy='multi_output_tree',num_target=24),xgb.DMatrix(x,label=y,nthread=1),num_boost_round=rounds)
    m.save_model(path);return m


def test_model_rounds_corruption_and_loss_completeness(tmp_path):
    path=tmp_path/'model.ubj';booster(path)
    R.xgb_complete(path,2,3,24)
    with pytest.raises(ValueError):R.xgb_complete(path,3,3,24)
    loss=tmp_path/'model.loss.json';loss.write_text(json.dumps({'train':{'observed_rmse':[3.,2.]}}))
    R.loss_complete(loss,2,True)
    loss.write_text(json.dumps({'train':{'observed_rmse':[3.]}}))
    with pytest.raises(ValueError):R.loss_complete(loss,2,True)
    path.write_bytes(b'interrupted')
    ok,_=R.valid(lambda:R.xgb_complete(path,2,3,24));assert not ok


def test_source_and_input_identity_rejected(tmp_path):
    source=tmp_path/'experiment.py';launch=tmp_path/'runner_at_launch.py';data=tmp_path/'df_ffill_trainy.parquet'
    source.write_text('x=1\n');launch.write_bytes(source.read_bytes());data.write_bytes(b'dataset')
    stat=data.stat();args=(source,launch,data,tmp_path,stat.st_size,stat.st_mtime_ns,'top30','recipe')
    assert R.identity(*args)['data']['bytes']==7
    launch.write_text('x=2\n')
    with pytest.raises(ValueError,match='source differs'):R.identity(*args)
    launch.write_bytes(source.read_bytes());data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Data differs'):R.identity(*args)


def fake_top_snapshot(tmp_path,monkeypatch):
    source=tmp_path/'experiment.py';launch=tmp_path/'runner_at_launch.py';data=tmp_path/'df_ffill_trainy.parquet'
    data.write_bytes(b'current training-y snapshot');source.write_text('immutable code\n');launch.write_bytes(source.read_bytes())
    origins=pd.date_range('2025-01-01',periods=80,freq='h',tz='UTC');query=pd.date_range('2026-01-01',periods=4,freq='h',tz='UTC')
    rng=np.random.default_rng(6);y=rng.normal(400,20,(80,24));fits=[];merges=[]
    params=dict(max_depth=2,nthread=1,tree_method='hist',multi_strategy='multi_output_tree',num_target=24)
    module=SimpleNamespace(RECIPES={'recipe':('xgb','plain',3,True)},PERIODS={'F1':4,'F2':4,'F3':4},CUTOFFS={p:'2025-12-29' for p in ['F1','F2','F3']},XGB_PARAMS=params,
                           feature_names=lambda b,w:['a','b','c'],load_block=lambda *args:((origins,None,None,y),(query,None,None,None)))
    def fit(experiment,data,out,period,block,threads,smoke):
        fits.append((period,block));stem=out/f'{period}_b{block}';rounds=2 if smoke else 900
        # This synthetic test uses 900 tiny depth-zero rounds to exercise the full
        # nine-block completion contract without running production training.
        x=np.zeros((80,3),np.float32);actual=dict(params,nthread=threads,base_score=float(y.mean()),disable_default_eval_metric=1)
        model=xgb.train(actual,xgb.DMatrix(x,label=y,nthread=1),num_boost_round=rounds)
        model.save_model(stem.with_suffix('.ubj'))
        pd.DataFrame({'round':range(1,rounds+1),'train_observed_rmse':np.ones(rounds)}).to_csv(stem.with_name(stem.name+'_loss.csv'),index=False)
        leads=list(range(block*24+1,(block+1)*24+1));np.savez(stem.with_suffix('.npz'),origins=query.asi8,pred=np.ones((4,24))*400,leads=leads)
        report=dict(status='complete',experiment=experiment,period=period,block=block,leads=leads,smoke=smoke,source=str(data),feature_names=['a','b','c'],feature_count=3,maturity_hours=144,training_cutoff='2025-12-29',train_origins_before_target_mask=80,training_origin_first=str(origins.min()),training_origin_last=str(origins.max()),fits=[dict(rounds=rounds,parameters=actual,loss='plain',train_origins=80)],model_files=[str(stem.with_suffix('.ubj'))])
        stem.with_suffix('.json').write_text(json.dumps(report))
    module.fit_block=fit;module.merge=lambda out:merges.append(out)
    monkeypatch.setattr(R,'import_snapshot',lambda path:module)
    return source,launch,data,module,fits,merges


def test_completed_blocks_skip_and_corrupt_checkpoint_refits(tmp_path,monkeypatch):
    source,launch,data,module,fits,merges=fake_top_snapshot(tmp_path,monkeypatch);stat=data.stat()
    args=(source,launch,data,tmp_path,stat.st_size,stat.st_mtime_ns,'top30','recipe',1,True)
    assert R.run(*args)==1;assert fits==[('F1',0)]
    assert R.run(*args)==1;assert len(fits)==1
    (tmp_path/'F1_b0.ubj').write_bytes(b'corrupt')
    assert R.run(*args)==1;assert len(fits)==2
    (tmp_path/'F1_b0_loss.csv').write_text('round,train_observed_rmse\n1,2\n')
    assert R.run(*args)==1;assert len(fits)==3
    assert not merges
    assert json.loads((tmp_path/'resume_complete.json').read_text())['status']=='smoke_complete'


def test_outside_owned_artifacts_rejected(tmp_path):
    with pytest.raises(ValueError,match='outside'):R.owned(tmp_path.parent/'alien.ubj',tmp_path)


def test_calibration_models_reused_when_weight_phase_interrupted(tmp_path,monkeypatch):
    data=tmp_path/'df_ffill_trainy.parquet';data.write_bytes(b'ffilled fixture')
    source=tmp_path/'experiment.py';source.write_bytes(Path(__file__).with_name('calib11.py').read_bytes())
    launch=tmp_path/'runner_at_launch.py';launch.write_bytes(source.read_bytes())
    module=R.import_snapshot(source)
    experiment='e2000fs_calib11b_metricwt';rng=np.random.default_rng(8)
    origins=pd.date_range('2025-01-01',periods=96,freq='24h',tz='UTC');qo=pd.date_range('2026-01-01',periods=4,freq='h',tz='UTC')
    def grid(origins):
        n=len(origins)
        return (origins,rng.uniform(300,700,(n,len(module.CONSTANT))).astype('float32'),rng.uniform(300,700,(n,72,len(module.VARYING))).astype('float32'),rng.uniform(300,700,(n,72)))
    training,query=grid(origins),grid(qo)
    module.load=lambda *args:(training,query)
    monkeypatch.setattr(R,'import_snapshot',lambda path:module)
    original=module.fit;fit_calls=[]
    def counting(*args,**kwargs):
        fit_calls.append(Path(args[-1]).name)
        return original(*args,**kwargs)
    module.fit=counting
    stat=data.stat();args=(source,launch,data,tmp_path,stat.st_size,stat.st_mtime_ns,'calib11',experiment,1,True)
    R.run(*args)
    assert len(fit_calls)==4
    # Simulate reboot after three complete calibration heads but before final
    # weight metadata was durably written. The verified heads must not refit.
    (tmp_path/'F1_weights.json').unlink()
    R.run(*args)
    assert len(fit_calls)==4
    # Incomplete calibration log invalidates that head only, then rebuilds weights.
    (tmp_path/'F1_cal_b1.loss.json').write_text('{"train":{"observed_rmse":[1.]}}')
    R.run(*args)
    assert fit_calls.count('F1_cal_b1.ubj')==2
    assert fit_calls.count('F1_cal_b0.ubj')==1 and fit_calls.count('F1_cal_b2.ubj')==1
    # Damaged weight archives rebuild weights without refitting intact heads.
    for broken in (b'', b'PK\x03\x04truncated'):
        before = list(fit_calls)
        (tmp_path / 'F1_w.npz').write_bytes(broken)
        R.run(*args)
        assert fit_calls == before


def test_only_all_nine_verified_blocks_allow_complete_and_merge(tmp_path,monkeypatch):
    source,launch,data,module,fits,merges=fake_top_snapshot(tmp_path,monkeypatch)
    def fit(experiment,data,out,period,block,threads,smoke):
        fits.append((period,block));(out/f'{period}_b{block}.json').write_text('{"valid":true}')
    def check(module,record,out,period,block,threads,smoke):
        assert json.loads((out/f'{period}_b{block}.json').read_text())['valid']
    module.fit_block=fit;monkeypatch.setattr(R,'validate_top_block',check)
    stat=data.stat();args=(source,launch,data,tmp_path,stat.st_size,stat.st_mtime_ns,'top30','recipe',1,False)
    assert R.run(*args)==9
    assert len(fits)==9 and len(merges)==1
    assert json.loads((tmp_path/'resume_complete.json').read_text())['blocks']==9
    assert R.run(*args)==9
    assert len(fits)==9 and len(merges)==2


@pytest.mark.parametrize("broken", [b"", b"PK\x03\x04truncated"])
def test_corrupt_npz_rebuilds_only_affected_block(tmp_path, monkeypatch, broken):
    source, launch, data, module, fits, merges = fake_top_snapshot(tmp_path, monkeypatch)
    stat = data.stat()
    args = (source, launch, data, tmp_path, stat.st_size, stat.st_mtime_ns,
            'top30', 'recipe', 1, False)
    assert R.run(*args) == 9
    assert len(fits) == 9
    (tmp_path / 'F2_b1.npz').write_bytes(broken)
    assert R.run(*args) == 9
    assert fits[9:] == [('F2', 1)]
    assert len(merges) == 2
    backups = list((tmp_path / 'resume_invalid').glob('F2_b1_*/F2_b1.npz'))
    assert len(backups) == 1 and backups[0].read_bytes() == broken
