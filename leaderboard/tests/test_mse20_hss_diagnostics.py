import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_hss_arrival import diagnostics as d


def frame(origins,values=None):
    origins=pd.DatetimeIndex(origins)
    result=pd.DataFrame({d.KEY[0]:np.repeat(origins,72),d.KEY[1]:np.tile(np.arange(1,73),len(origins)),
                         'pred_kms':np.full(len(origins)*72,400.)})
    if values is not None:result['pred_kms']=np.asarray(values).reshape(-1)
    return result


def fixture():
    origins=pd.date_range('2026-01-01',periods=3,freq='12h',tz='UTC')
    index=pd.date_range(origins.min()+pd.Timedelta(hours=1),origins.max()+pd.Timedelta(hours=72),freq='h')
    target=pd.Series(400.,index=index);target.loc[origins[0]+pd.Timedelta(hours=32)]=700.
    observed=pd.Series(True,index=index)
    p=frame(origins)
    stamps=pd.to_datetime(p[d.KEY[0]],utc=True)+pd.to_timedelta(p.horizon_hours,unit='h')
    actual=target.reindex(stamps).to_numpy()
    truth=p.rename(columns={'pred_kms':'target_kms'});truth['target_kms']=actual;truth['was_missing']=0
    benchmark=dict(origins=origins,target=target,observed=observed,truth=truth,naive=frame(origins),
        actual_events=d.actual_unique_events(target,observed,origins),source_files={})
    return benchmark


def test_prediction_merge_is_truth_independent_bounded_not_transitive():
    benchmark=fixture();p=frame(benchmark['origins'])
    for i in range(3):p.loc[i*72+19,'pred_kms']=700+i*10
    events=d.predicted_unique_events(p,benchmark['origins'])
    assert len(events)==2  # Absolute peaks20/32/44 must not all chain together.
    assert events[0]['window_occurrences']==2 and events[0]['peak_span_hours']==12
    assert pd.Timestamp(events[0]['peak_utc'])==benchmark['origins'][0]+pd.Timedelta(hours=20)
    assert events[0]['height_kms']==705
    permuted=p.sample(frac=1,random_state=3)
    assert d.predicted_unique_events(permuted,benchmark['origins'])==events


def test_actual_exact_observed_dedup_one_to_one_f1():
    benchmark=fixture();p=frame(benchmark['origins'])
    for i in range(3):p.loc[i*72+19,'pred_kms']=700.
    result=d.unique_event_metrics(p,benchmark['target'],benchmark['observed'],benchmark['origins'])
    assert result['unique_actual_events']==1
    assert (result['tp'],result['fp'],result['fn'])==(1,1,0)
    assert result['f1']==pytest.approx(2/3)
    no_events=d.unique_event_metrics(frame(benchmark['origins']),benchmark['target'],benchmark['observed'],benchmark['origins'])
    assert no_events['f1']==0 and no_events['precision'] is None
    obs=benchmark['observed'].copy();obs.loc[benchmark['target'].idxmax()]=False
    assert d.actual_unique_events(benchmark['target'],obs,benchmark['origins'])==[]


def test_hss_labels_exact_future_support_and_prefix_invariance():
    index=pd.date_range('2020-01-01',periods=400,freq='h',tz='UTC')
    speed=pd.Series(400.,index=index);speed.iloc[70:130]=650;speed.iloc[270:340]=700
    events=d.hss_onsets(speed)
    assert isinstance(events,pd.DatetimeIndex) and len(events)==2
    assert events.tz is not None and d.HSS_FUTURE_HOURS==24
    first=events[0]
    assert len(d.hss_onsets(speed.loc[:first+pd.Timedelta(hours=23)]))==0
    assert d.hss_onsets(speed.loc[:first+pd.Timedelta(hours=24)]).equals(events[:1])
    cutoff=index[200];changed=speed.copy();changed.loc[changed.index>cutoff]=1000
    prefix=d.hss_onsets(speed.loc[:cutoff])
    valid=d.hss_onsets(changed);valid=valid[valid+pd.Timedelta(hours=24)<=cutoff]
    assert valid.equals(prefix)
    assert len(d.hss_onsets(pd.Series(400.,index=index)))==0


def test_oracle_is_bound_only_no_shifted_forecast_outputs():
    actual=np.full((2,72),400.);actual[:,30:40]=700
    predicted=np.full((2,72),400.);predicted[:,35:45]=700
    original=predicted.copy()
    result=d.oracle_shift_diagnostic(predicted,actual)
    assert result['oracle_mse']==0 and result['gap_mse']>0
    assert result['best_shift_histogram']['-5']==2
    assert result['diagnostic_only'] and result['not_for_selection_or_predictions']
    assert set(result).isdisjoint({'predictions','shifted_predictions','corrected_values'})
    np.testing.assert_array_equal(predicted,original)


def test_diagnose_bands_jump_boundaries_full_keys_and_naive(tmp_path):
    benchmark=fixture();p=frame(benchmark['origins'])
    reference=tmp_path/'prediction.parquet';p.to_parquet(reference,index=False)
    result=d.diagnose(d._ref(reference),benchmark,baseline_f1=0.)
    assert sum(row['rows'] for row in result['horizon_bands'].values())==216
    assert sum(row['rows'] for row in result['jump_buckets'].values())==216
    assert sum(row['squared_error_fraction'] for row in result['horizon_bands'].values())==pytest.approx(1.)
    assert result['short_1_6h']['mse_difference']==0
    assert result['unique_f1_nonregression'] is True
    assert result['legacy_window_events']['not_a_gate'] and result['eligible_for_p1_selection'] is False
    assert result['jump_buckets']['>=150']['rows']==3
    for bad in (p.iloc[:-1],pd.concat([p,p.iloc[:1]],ignore_index=True),p.assign(horizon_hours=p.horizon_hours.astype(float))):
        bad.to_parquet(reference,index=False)
        with pytest.raises(ValueError):d.diagnose(d._ref(reference),benchmark)


def test_forward_fill_never_uses_future_to_fill_leading_truth():
    benchmark=fixture();obs=benchmark['observed'].copy();obs.iloc[0]=False
    with pytest.raises(ValueError,match='leading'):
        d.actual_unique_events(benchmark['target'],obs,benchmark['origins'])


def test_hss_lead48_censoring_explicit():
    index=pd.date_range('2020-01-01',periods=180,freq='h',tz='UTC')
    target=pd.Series(400.,index=index);target.iloc[110:170]=700
    actual=d.hss_onsets(target)[0]
    origins=pd.DatetimeIndex([actual-pd.Timedelta(hours=48),actual-pd.Timedelta(hours=24)])
    p=frame(origins)
    times=pd.to_datetime(p[d.KEY[0]],utc=True)+pd.to_timedelta(p.horizon_hours,unit='h')
    p['pred_kms']=target.reindex(times).to_numpy()
    output=d.hss_onset_diagnostics(p,target,pd.Series(True,index=index),origins)
    assert output['by_lead_hours']['48']['hits']==1
    assert output['by_lead_hours']['48']['events'][0]['search_right_censored'] is True
    assert output['by_lead_hours']['24']['events'][0]['search_right_censored'] is False


def test_all_attempt_summary_waits_for72_and_never_fabricates_failures(tmp_path,monkeypatch):
    benchmark=fixture();base=tmp_path/'old';run=base/'ch-dynamics24';out=run/'alignment'
    champion=tmp_path/'champion.parquet';frame(benchmark['origins']).to_parquet(champion,index=False)
    benchmark['contract']={'files':{'reference':d._ref(champion)}}
    monkeypatch.setattr(d,'official_benchmark',lambda:benchmark)
    monkeypatch.setattr(d,'hss_count_comparison',lambda benchmark:{'canonical_counts':{}})
    monkeypatch.setattr(d.original,'RUN',base)
    hpo=base/'adaptive/trials/tree/hpo_015/predictions.parquet';hpo.parent.mkdir(parents=True)
    frame(benchmark['origins']).to_parquet(hpo,index=False)
    grid={family:[{'id':f'network{i:03d}' if family=='neural' else f'tree{i:02d}','value':i} for i in range(36)] for family in ('tree','neural')}
    d._json_new(run/'registered.json',{'grid':grid})
    def terminal(family,index,status,saved_forecast=False,invalid_forecast=False):
        config=grid[family][index];sha=d._semantic(family,config);name=f'{family}__{sha}.json'
        claim=run/'attempts'/name
        d._json_new(claim,dict(family=family,config=config,config_sha256=sha))
        details={'family':family,'config':config}
        if status=='ok' or saved_forecast:details['prediction']=d._ref(champion)
        if status!='ok':details['reason']='synthetic counted rejection'
        if invalid_forecast:
            invalid=tmp_path/f'incomplete-{index}.parquet'
            frame(benchmark['origins']).iloc[:-1].to_parquet(invalid,index=False)
            details['prediction']=d._ref(invalid)
        d._json_new(run/'attempt-results'/name,dict(attempt=d._ref(claim),family=family,config_sha256=sha,status=status,details=details))
    terminal('tree',0,'ok');terminal('tree',1,'admission_rejected')
    terminal('tree',2,'failed',saved_forecast=True)
    terminal('tree',3,'failed',invalid_forecast=True)
    original=d.diagnose
    def diagnose(*args,**kwargs):
        assert (out/'definition.json').exists()  # Definition precedes ANY comparison.
        return original(*args,**kwargs)
    monkeypatch.setattr(d,'diagnose',diagnose)
    partial=d.write_all_attempts(run,out)
    assert partial['terminal_attempts']==4 and partial['status'].startswith('partial')
    assert not (out/'all-attempts-final.json').exists()
    assert partial['attempts'][1]['status']=='admission_rejected' and partial['attempts'][1]['report'] is None
    failed=partial['attempts'][2]
    assert failed['status']=='failed' and failed['report'] is not None and failed['comparison']['mse']>=0
    assert failed['eligible_for_model_selection'] is False and 'integrity NOT established' in failed['integrity_warning']
    report=json.loads(Path(failed['report']['path']).read_text())
    assert report['attempt_status']=='failed' and report['model_integrity_claim'] is False
    assert 'integrity NOT established' in report['integrity_warning']
    assert partial['attempts'][3]['report'] is None and 'saved forecast invalid' in partial['attempts'][3]['integrity_warning']
    assert Path(report['diagnostics_source']['path']).parent.name=='sources'
    for family in ('tree','neural'):
        for i in range(4 if family=='tree' else 0,36):terminal(family,i,'failed')
    final=d.write_all_attempts(run,out)
    assert final['terminal_attempts']==72 and final['status']=='complete_all72_terminal'
    assert (out/'all-attempts-final.json').exists()
    assert sum(row['report'] is not None for row in final['attempts'])==2
    assert Path(final['final_report_path']).exists()
    assert final['objective_achieved'] is False
