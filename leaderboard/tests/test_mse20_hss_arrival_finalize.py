"""Synthetic producer tests; no real fits, GPU allocation, or official targets."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_hss_arrival import finalize as f


def write(path,data):
    f.p.write_new(path,data);return f.p.file_ref(path)


@pytest.fixture
def producer(tmp_path,monkeypatch):
    from experiments import mse20_hss_arrival as package
    index=pd.date_range('2026-05-01',periods=1800,freq='h',tz='UTC');x=np.arange(len(index))
    ace=pd.DataFrame({'filled_speed_kms':400+np.sin(x/10),'was_missing':0},index=index)
    ch=pd.DataFrame({col:.01 for col in f.inputs.CH_COLUMNS},index=index)
    raw=pd.DataFrame({'ace_density_cm3':5.,'ace_temperature_k':1e5,'ace_bt_nt':5.,'ace_speed_kms':ace.filled_speed_kms},index=index)
    frames=f.inputs.rebuild_frames(ace,ch,raw);origins=index[[900,1100,1400]]
    config={'id':'chosen'};registration={'sources':{'source':{'path':'synthetic','sha256':'synthetic'}}}
    selection={'frozen_utc':'2026-10-03T10:00:00Z','selected_config':config,'selection_uses_official':False}
    write(tmp_path/'registered.json',registration);write(tmp_path/'dev-selection.json',selection);write(tmp_path/'authorization.json',{'regime_cells_final_gate':True})
    monkeypatch.setattr(f,'final_context',lambda run:(tmp_path,registration,selection,config,origins,{}))
    monkeypatch.setattr(f,'issued_frames',lambda *a:frames)
    monkeypatch.setattr(f.p,'load_frames',lambda *a:frames)
    monkeypatch.setattr(f.p,'verify_registration',lambda *a:registration)
    monkeypatch.setattr(f.p,'deadline',lambda *a,**k:10000)
    monkeypatch.setattr(f.p,'memory_admission',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(f.p,'gpu_lease',lambda *a,**k:contextlib.nullcontext('cuda:0'))
    monkeypatch.setattr(f.p,'sources',lambda:registration['sources'])
    monkeypatch.setattr(f.p,'setup',lambda *a:{'regime_cells_final_gate':True})
    calls=[]
    def predict(frames,origins,directory,device):
        calls.append(('predict',device,len(origins)))
        values=frames['speed'].reindex(origins).to_numpy()
        pred=pd.DataFrame({f.p.KEY[0]:np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                           'pred_kms':np.repeat(values,72)})
        return {'mse':pred,'peak':pred.copy()}
    def fit(frames,cutoff,origins,config,device,directory):
        calls.append(('fit',device,len(origins)))
        assert (tmp_path/'final-fit-claim.json').exists()
        assert origins.tolist()==[pd.Timestamp(f.p.CUTOFF)]
        assert not (tmp_path/'prediction-pair.json').exists()
        directory.mkdir(parents=True)
        (directory/'model.pt').write_bytes(b'synthetic model')
        meta=dict(config=config,cutoff=cutoff.isoformat(),train_label_max=cutoff.isoformat())
        (directory/'metadata.json').write_text(json.dumps(meta));(directory/'metadata.sha256').write_text(f.p.digest(directory/'metadata.json'))
        return {},meta
    fake=types.SimpleNamespace(fit_predict=fit,predict_saved=predict,verify_training_provenance=lambda *a:{'normalizer':{'mean':[1.]}})
    monkeypatch.setattr(package,'model',fake,raising=False)
    return tmp_path,frames,origins,calls,registration


def test_single_fit_claim_precedes_fit_and_pair_resume_never_refits(producer):
    run,frames,origins,calls,registration=producer
    receipt=f.fit_final(run)
    assert receipt['claim']==f.p.file_ref(run/'final-fit-claim.json')
    pair=f.p.read_json(run/'prediction-pair.json')
    assert pair['model_metadata']==receipt['model_metadata']
    assert len(f.read_pair(pair,origins)['mse'])==216
    f.fit_final(run)
    assert sum(row[0]=='fit' for row in calls)==1


def test_interrupted_fit_claim_cannot_be_retried(producer):
    run,*_=producer
    write(run/'final-fit-claim.json',{'already':'consumed'})
    with pytest.raises(FileExistsError):f.fit_final(run)


def test_paired_proof_actual_raw18_derived6_inventory_and_subsets(producer):
    run,frames,origins,calls,registration=producer
    f.fit_final(run);proof=f.replay(run)
    assert len(proof['checks'])==24 and len(proof['subset_checks'])==3
    assert sum(row['kind']=='raw' for row in proof['checks'])==18
    assert sum(row['kind']=='derived' for row in proof['checks'])==6
    for row in proof['checks']:
        assert set(row['baseline'])==set(row['altered'])=={'mse','peak'}
        assert row['max_abs_diff_kms']=={'mse':0.,'peak':0.}
        assert all(value>0 for value in row['mutation_counts'].values())
        if row['kind']=='raw':assert row['rebuild']['function']=='inputs.rebuild_frames'
    with np.load(proof['inventory']['path']) as data:
        assert set(data.files)==set(f.RAW+f.DERIVED)
        assert all(data[key].dtype==np.dtype('int64') for key in data.files)
    assert any(row[0]=='predict' and row[1]=='cpu' and row[2]==3 for row in calls)
    assert any(row[0]=='predict' and row[1]=='cuda:0' and row[2]==3 for row in calls)
    with pytest.raises(FileExistsError):f.replay(run)


def test_two_scores_only_after_pair_replay_and_regression(producer,monkeypatch):
    from experiments.mse20_hss_arrival import evaluate,diagnostics
    run,frames,origins,calls,registration=producer
    f.fit_final(run);f.replay(run)
    (run/'regression.log').write_text('synthetic regression passed')
    write(run/'regression-summary.json',dict(passed=True,exit_code=0,sources=registration['sources'],files={'log':f.p.file_ref(run/'regression.log')}))
    observed=[]
    def benchmark():
        assert (run/'prediction-pair.json').exists() and (run/'replay-proof.json').exists()
        assert (run/'regression-summary.json').exists()
        return {}
    monkeypatch.setattr(diagnostics,'official_benchmark',benchmark)
    def score(mse,peak,benchmark,cells):
        observed.append(1)
        assert len(mse)==len(peak)==216 and cells is True
        return {'status':'unmet','test':True}
    monkeypatch.setattr(evaluate,'score_pair',score)
    manifest=f.finish(run)
    assert len(observed)==2
    assert manifest['files']['final_scores_1']!=manifest['files']['final_scores_2']
    assert set(manifest['files'])>={'prediction_pair','final_fit','replay_proof','mse_predictions','peak_predictions','regression'}


def test_pair_delta_rejects_one_changed_output():
    frame=pd.DataFrame({'origin_last_input_utc':['2026-06-01T00:00:00Z'],'horizon_hours':[1],'pred_kms':[400.]})
    other=frame.copy();other['pred_kms']=401.
    with pytest.raises(ValueError,match='tolerance'):
        f.paired_difference({'mse':frame,'peak':frame},{'mse':frame,'peak':other})


def test_producer_import_is_torch_free():
    result=subprocess.run([str(f.p.ROOT/'.venv-fusion/bin/python'),'-c',
        'import sys;from experiments.mse20_hss_arrival import finalize;assert "torch" not in sys.modules'],
        cwd=f.p.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_regression_executes_fresh_numeric_suite_binds_log_and_sources(producer,monkeypatch):
    run,frames,origins,calls,registration=producer
    f.fit_final(run)
    root=run/'synthetic-suite';(root/'tests').mkdir(parents=True)
    (root/'tests/test_numeric.py').write_text('def test_numeric():\n    assert 2+2==4\n')
    (root/'tests/test_vjepa_excluded.py').write_text('raise AssertionError("independent image test must not run")\n')
    monkeypatch.setattr(f.p,'ROOT',root)
    proof=f.regression(run)
    assert proof['suite']=='full_numeric_regression' and proof['passed'] and proof['exit_code']==0
    assert proof['log']==proof['files']['log']
    assert 'tests/test_numeric.py' in proof['files']
    assert not any('vjepa' in name for name in proof['files'])
    assert pd.Timestamp(proof['started_utc'])>=pd.Timestamp(f.p.read_json(run/'final-fit.json')['fit_completed_utc'])
    assert Path(proof['log']['path']).read_text().strip()


def test_issued_loader_rejects_empty_or_changed_partition_inventory(tmp_path,monkeypatch):
    root=tmp_path/'root';folder=root/'store/ch-breakthrough-v2/upload/raw-ace-snapshot'
    year=folder/'year=2026';year.mkdir(parents=True)
    part=year/'part.parquet';part.write_bytes(b'synthetic input')
    monkeypatch.setattr(f.p,'ROOT',root)
    with pytest.raises(ValueError,match='inventory'):f.issued_frames('2026-09-30T23:00:00Z',{})
    def load(_):
        extra=folder/'year=2027';extra.mkdir();(extra/'part.parquet').write_bytes(b'new unregistered partition')
        return {}
    monkeypatch.setattr(f.p.legacy.original,'load_frames',load)
    with pytest.raises(ValueError,match='while loading'):
        f.issued_frames('2026-09-30T23:00:00Z',{'raw_year=2026':f.p.file_ref(part)})
