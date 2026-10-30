"""Synthetic-only causal folds, all-event ranking and exclusive attempt accounting."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_hss_arrival import dev


def synthetic_frames(start='2025-05-01',end='2025-08-31T23:00:00Z'):
    index=pd.date_range(pd.to_datetime(start,utc=True),pd.to_datetime(end,utc=True),freq='h')
    speed=pd.Series(380.+20*np.sin(np.arange(len(index))/100),index=index)
    # Finite filled series; explicit observation mask retained.
    return dict(speed=speed,observed=pd.Series(True,index=index),base=pd.DataFrame({'x':1.},index=index),
                ch=pd.DataFrame({'area':.1},index=index))


def config(i):
    return dict(id=f'c{i}',epochs=20,max_train_origins=4096)


def test_folds_causal_complete_and_early_month_not_dropped():
    folds=dev.fold_grid()
    assert len(folds)==14 and len({f['id'] for f in folds})==14
    assert [f['id'] for f in folds[:4]]==['2023Q4','2024Q2','2024Q4','2025Q2']
    for f in folds:
        origins=dev.fold_origins(f);start=pd.Timestamp(f['start_utc']);end=pd.Timestamp(f['end_utc'])
        assert origins.min()==start
        assert origins.max()+pd.Timedelta(hours=72)<end
        assert np.all(np.diff(origins.asi8)==6*3600*10**9)
        assert start-pd.Timedelta(hours=1)<origins.min()
    assert dev.registration_payload()['folds']==folds


def test_support_skips_unavailable_ch_without_fabricating():
    f=synthetic_frames();fold=dev.fold_grid()[4]
    assert dev.source_support(f,fold)['supported']
    f['ch'].loc[:]=np.nan
    assert not dev.source_support(f,fold)['supported']
    f=synthetic_frames('2025-07-30','2025-08-31T23:00:00Z')
    assert not dev.source_support(f,fold)['supported']


def test_onset_penalty_counts_every_miss_and_false_alarm():
    t=pd.Timestamp('2025-01-01T00:00:00Z')
    actual=[t,t+pd.Timedelta(days=6),t+pd.Timedelta(days=12)]
    predicted=[t+pd.Timedelta(hours=2),t+pd.Timedelta(days=20)]
    result=dev.onset_penalty(actual,predicted)
    assert (result['tp'],result['fn'],result['fp'])==(1,2,1)
    assert result['penalty_hours']==74 and result['penalized_timing_hours']==74/3
    missed=dev.onset_penalty(actual,[])
    assert missed['penalized_timing_hours']==24


def test_onset_merge_bounded_not_transitive_and_nearest_unique():
    t=pd.Timestamp('2025-01-01T00:00:00Z')
    stamps=[t+pd.Timedelta(hours=h) for h in (0,10,20,30)]
    assert dev.merge_onsets(stamps)==[stamps[0],stamps[2]]
    result=dev.onset_penalty([t,t+pd.Timedelta(hours=2)],[t+pd.Timedelta(hours=1)])
    assert result['tp']==1 and result['fn']==1


def metrics(tp=1,penalty=1,se=10):
    return dict(hss=dict(actual_count=2,predicted_count=1,tp=tp,fp=0,fn=2-tp,penalty_hours=penalty),
        modal_unique_events=dict(tp=1,fp=0,fn=1),squared_error_sum=se,naive_squared_error_sum=100,n_pairs=72)


def records(configs,folds,tp=1):
    return [dict(config_sha256=dev.config_hash(c),fold=f,status='ok',details={'metrics':metrics(tp=tp)}) for c in configs for f in folds]


def test_ranking_primary_timing_not_mse_and_allmiss_fallback():
    configs=[config(0),config(1)];folds=dev.fold_grid()[:1];rows=records(configs,folds)
    rows[0]['details']['metrics']=metrics(penalty=2,se=1)
    rows[1]['details']['metrics']=metrics(penalty=1,se=10)
    result=dev.rank_results(rows,configs,folds)
    assert result['selected_config']==configs[1] and not result['fallback_dev_mse']
    for row in rows:row['details']['metrics']['hss']['tp']=0
    result=dev.rank_results(rows,configs,folds)
    assert result['selected_config']==configs[0] and result['fallback_dev_mse'] and result['warning']


def test_failed_fold_not_silently_dropped_for_better_ranking():
    configs=[config(0),config(1)];folds=dev.fold_grid()[:2];rows=records(configs,folds)
    rows[0]['status']='failed'
    result=dev.rank_results(rows,configs,folds)
    assert len(result['candidates'])==1 and result['selected_config']==configs[1]


def test_exclusive_claim_failed_attempt_consumed_and_cap(tmp_path):
    dev.p.write_new(tmp_path/'registered.json',{'test':True})
    c=config(0);fold=dev.fold_grid()[0]
    claim=dev.claim(tmp_path,c,fold)
    assert claim is not None and dev.claim(tmp_path,c,fold) is None
    dev.terminal(tmp_path,claim,'failed',{'reason':'synthetic'})
    assert dev.claim(tmp_path,c,fold) is None
    assert len((tmp_path/'ledger.jsonl').read_text().splitlines())==1
    for i in range(1,84):dev.p.write_new(tmp_path/'attempts'/f'fake{i}.json',{})
    with pytest.raises(ValueError,match='cap'):dev.claim(tmp_path,config(1),fold)


def test_score_saved_pair_counts_complete_support_and_baselines():
    f=synthetic_frames();fold=dev.fold_grid()[4];origins=dev.fold_origins(fold)
    pairs=pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                       'pred_kms':np.repeat(f['speed'].reindex(origins).to_numpy(),72)})
    result=dev.score_pair({'mse':pairs,'peak':pairs},f,fold)
    assert result['n_pairs']==len(origins)*72
    assert result['mse']==pytest.approx(result['naive_mse'])
    assert np.isfinite(result['mr_mse']) and np.isfinite(result['recurrence27_mse'])
    assert result['official_informed_selection'] is False


def test_import_does_not_import_torch():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([str(root/'.venv-fusion/bin/python'),'-c',
        'import sys;from experiments.mse20_hss_arrival import dev;assert "torch" not in sys.modules;assert len(dev.registration_payload()["folds"])==14'],
        cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_worker_persists_pair_before_scoring_and_never_retries(tmp_path,monkeypatch):
    from experiments import mse20_hss_arrival as package
    fold=dev.fold_grid()[4];c=config(0);registration={'configs':[c],'folds':[fold]}
    dev.p.write_new(tmp_path/'registered.json',registration)
    f=synthetic_frames();calls=[]
    monkeypatch.setattr(dev,'verify_registration',lambda _:registration)
    monkeypatch.setattr(dev.p,'require_legacy_complete',lambda:None)
    monkeypatch.setattr(dev.p,'load_frames',lambda end,clock:f)
    monkeypatch.setattr(dev.p,'deadline',lambda *a,**kw:10000)
    monkeypatch.setattr(dev.p,'memory_admission',lambda *a,**kw:contextlib.nullcontext())
    monkeypatch.setattr(dev.p,'gpu_lease',lambda *a,**kw:contextlib.nullcontext('cpu'))
    def fit(bounded,cutoff,origins,config,device,out):
        assert bounded['speed'].index.max()<pd.Timestamp(fold['end_utc'])
        assert cutoff==pd.Timestamp(fold['start_utc'])-pd.Timedelta(hours=1)
        calls.append(out);out.mkdir(parents=True)
        meta=dict(config=config,cutoff=cutoff.isoformat(),train_label_max=cutoff.isoformat())
        (out/'metadata.json').write_text(json.dumps(meta));(out/'model.pt').write_bytes(b'synthetic')
        pred=pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                           'pred_kms':400.})
        return {'mse':pred,'peak':pred},meta
    fake=types.SimpleNamespace(candidate_configs=lambda:[c],fit_predict=fit)
    monkeypatch.setattr(package,'model',fake,raising=False)
    def score(pair,frames,fold):
        receipts=list((tmp_path/'outputs').glob('*/prediction-pair.json'))
        assert len(receipts)==1
        receipt=json.loads(receipts[0].read_text())
        for ref in receipt['predictions'].values():dev.p.verify_ref(ref)
        return metrics()
    monkeypatch.setattr(dev,'score_pair',score)
    dev.run_worker(tmp_path,tmp_path/'clock',0)
    dev.run_worker(tmp_path,tmp_path/'clock',0)
    assert len(calls)==1 and len(list((tmp_path/'attempts').glob('*.json')))==1
    result=json.loads(next((tmp_path/'results').glob('*.json')).read_text())
    assert result['status']=='ok'
    selection=dev.freeze(tmp_path,tmp_path/'clock')
    assert selection['selected_config']==c and selection['selection_uses_official'] is False


def test_dead_owner_claim_is_terminal_not_retried(tmp_path,monkeypatch):
    dev.p.write_new(tmp_path/'registered.json',{})
    c=config(0);fold=dev.fold_grid()[0];path=dev.claim(tmp_path,c,fold)
    monkeypatch.setattr(dev.p,'process_birth',lambda _:None)
    dev.recover_dead_claims(tmp_path)
    assert json.loads((tmp_path/'results'/path.name).read_text())['status']=='interrupted'
    assert dev.claim(tmp_path,c,fold) is None
