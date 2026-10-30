import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_ch_dynamics import evaluate as e


def saved(path, data):
    if isinstance(data,pd.DataFrame):data.to_parquet(path,index=False)
    else:path.write_text(json.dumps(data))
    return e.p.file_ref(path)


def frame(origins):
    return pd.DataFrame({e.p.KEY[0]:np.repeat(origins,72),e.p.KEY[1]:np.tile(np.arange(1,73),len(origins)),'pred_kms':400.})


def causal(tmp_path):
    origins=pd.date_range('2026-06-01',periods=3,freq='100h',tz='UTC')
    full=frame(origins)
    index=pd.date_range(origins[0]-pd.Timedelta(hours=10),origins[-1]+pd.Timedelta(hours=80),freq='h')
    inventory={k:index for k in (*e.RAW,*e.DERIVED)}
    source=tmp_path/'inputs.py';source.write_text('pass\n')
    sources={'experiments/mse20_ch_dynamics/inputs.py':e.p.file_ref(source)}
    checks=[];subsets=[]
    for i,o in enumerate(origins):
        ref=saved(tmp_path/f'base{i}.parquet',frame(pd.DatetimeIndex([o])))
        subsets.append(dict(origin=o.isoformat(),baseline=ref,max_abs_diff_kms=0.))
        for kind,src in [('raw',s) for s in e.RAW]+[('derived','joint')]:
            for action in ('poison','drop'):
                keys=[src] if kind=='raw' else e.DERIVED
                checks.append(dict(kind=kind,source=src,origin=o.isoformat(),action=action,
                    mutation_counts={k:72 for k in keys},baseline=ref,altered=ref,max_abs_diff_kms=0.,
                    rebuild={'function':'inputs.rebuild_frames','source':e.p.file_ref(source),'cache_bypass':True}))
    return dict(checks=checks,subset_checks=subsets),full,origins,inventory,sources


def test_causal_snapshots_complete_and_independently_counted(tmp_path):
    args=causal(tmp_path)
    e.verify_causality(*args)
    for mutation in ('count','missing','duplicate','source','delta','altered','subset','rebuild'):
        replay=copy.deepcopy(args[0]);check=replay['checks'][0]
        if mutation=='count':check['mutation_counts']['ace']=71
        if mutation=='missing':replay['checks'].pop()
        if mutation=='duplicate':replay['checks'][-1]=copy.deepcopy(check)
        if mutation=='source':check['source']='quality'
        if mutation=='delta':check['max_abs_diff_kms']=.0001
        if mutation=='altered':
            data=frame(pd.DatetimeIndex([check['origin']]));data['pred_kms']+=1
            check['altered']=saved(tmp_path/'bad.parquet',data)
        if mutation=='subset':replay['subset_checks'].pop()
        if mutation=='rebuild':check['rebuild']['cache_bypass']=False
        with pytest.raises((ValueError,e.old.IntegrityError)):
            e.verify_causality(replay,*args[1:])


def test_prediction_strict_support_and_file_tamper(tmp_path):
    origins=pd.date_range('2026-06-01',periods=1,tz='UTC')
    ref=saved(tmp_path/'full.parquet',frame(origins))
    valid=e.predictions(ref,origins=origins)
    assert len(valid)==72
    missing=saved(tmp_path/'missing.parquet',frame(origins).iloc[:-1])
    with pytest.raises(ValueError):e.predictions(missing,origins=origins)
    with (tmp_path/'full.parquet').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError):e.predictions(ref,origins=origins)


def test_actual_normalizer_arrays():
    sizes={'ace':4,'ch':180,'recurrence':3,'transport':75,'age':180,'current':1}
    stats={key:dict(mean=[0.]*n,scale=[1.]*n,count=[10]*n) for key,n in sizes.items()}
    metadata=dict(normalizer=stats,selection_normalizer=stats)
    e.verify_normalizers(metadata,metadata)
    bad=copy.deepcopy(metadata);bad['normalizer']['ace']['mean'][0]=1
    with pytest.raises(e.old.IntegrityError):e.verify_normalizers(metadata,bad)
    bad=copy.deepcopy(metadata);bad['normalizer']['ace']['scale'][0]=0
    with pytest.raises(e.old.IntegrityError):e.verify_normalizers(bad,bad)


def test_import_remains_torch_free():
    code='import sys;from experiments.mse20_ch_dynamics import evaluate;assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)


def test_inventory_is_actual_pinned_content_and_endpoint(tmp_path,monkeypatch):
    index=pd.date_range('2026-01-01',periods=100,freq='h',tz='UTC')
    inventory={key:index for key in (*e.RAW,*e.DERIVED)}
    monkeypatch.setattr(e,'actual_source_indices',lambda contract:inventory)
    path=tmp_path/'index.npz';np.savez(path,**{key:times[:90].asi8 for key,times in inventory.items()})
    proof=dict(source_inventory=e.p.file_ref(path),source_files={},frames_end_utc=index[89].isoformat())
    result=e.verify_inventory(proof,{'files':{}})
    assert all(len(x)==90 for x in result.values())
    np.savez(path,**{key:times[:89].asi8 for key,times in inventory.items()})
    proof['source_inventory']=e.p.file_ref(path)
    with pytest.raises(e.old.IntegrityError):e.verify_inventory(proof,{'files':{}})


def test_ledger_rejects_orphans_even_if_inner_checker_missed_it(tmp_path,monkeypatch):
    monkeypatch.setattr(e.p,'verify_attempts',lambda *a,**kw:[])
    (tmp_path/'attempt-results').mkdir();(tmp_path/'attempt-results/orphan.json').write_text('{}')
    with pytest.raises(e.old.IntegrityError,match='orphan'):
        e.verify_ledger(tmp_path,{}, {})


def test_cli_missing_integrity_is_exit2_append_only(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'evaluate',lambda run:(_ for _ in ()).throw(ValueError('missing mandatory proof')))
    assert e.main(['--run-dir',str(tmp_path)])==2
    files=list(tmp_path.glob('acceptance-*.json'))
    assert len(files)==1
    result=json.loads(files[0].read_text())
    assert result['objective_achieved'] is False and result['status']=='indeterminate'


def test_seven_gates_delegate_original_boundaries():
    official={'mse':e.old.TOTAL_MAX,'by_regime':{key:{'mse':3799.,'reliable':True} for key in e.old.BASELINE_CELLS}}
    event={'precision':1.,'recall':1/818}
    status,gates=e.old.acceptance_gates(official,event,event)
    assert status=='pass' and len(gates)==7
    official['by_regime'][next(iter(e.old.BASELINE_CELLS))]['mse']=3800.
    assert e.old.acceptance_gates(official,event,event)[0]=='unmet'
    official['mse']=e.old.TOTAL_MAX+1
    assert e.old.acceptance_gates(official,event,event)[0]=='unmet'
    event['precision']=None
    assert e.old.acceptance_gates(official,event,{'precision':1.,'recall':1/818})[0]!='pass'


def test_compact_score_exactly_excludes_naive_only_diagnostics():
    official={'mse':100.,'by_regime':{key:{'mse':100.,'reliable':True} for key in e.old.BASELINE_CELLS},
              'skill_vs_naive':.5,'vs_naive':{'diagnostic':True},'mse_by_horizon':{1:100.}}
    event={'precision':1.,'recall':1/818}
    report={'official':official,'diagnostics':{'events':event}}
    compact=e.compact_metrics(report,report)
    assert compact['status']=='pass'
    assert compact['official']['mse_by_horizon']=={'1':100.}
    assert 'skill_vs_naive' not in compact['official'] and 'vs_naive' not in compact['official']


def blend_fixture(tmp_path,monkeypatch,partial=False):
    monkeypatch.setattr(e.p,'rank',lambda m:(m['rank'],m['official']['mse'],m['official']['mse']))
    monkeypatch.setattr(e.p,'OLD',tmp_path/'old')
    origins=pd.date_range('2026-06-01',periods=3,freq='h',tz='UTC')
    metadata=saved(tmp_path/'metadata.json',{})
    model=saved(tmp_path/'model.json',{})
    config={'id':'new','parameter':1}
    digest=e.p.semantic_digest('tree',config)
    new={'key':'tree__'+digest,'family':'tree','config':config,'metadata':metadata,'model_files':[model],
         'prediction':saved(tmp_path/'new.parquet',frame(origins)), 'metrics':{'rank':10,'official':{'mse':10}}}
    reference=dict(new,key='reference__champion',family='reference',config={'id':'champion'})
    prediction=frame(origins);prediction['pred_kms']=500.
    reference['prediction']=saved(tmp_path/'reference.parquet',prediction)
    (tmp_path/'old/adaptive').mkdir(parents=True)
    saved(tmp_path/'old/adaptive/recipe.json',{'components':[reference]})
    result={'status':'ok','details':{**new,'score_completed_utc':'2026-10-03T13:00:00Z'}}
    result_ref=saved(tmp_path/'result.json',result)
    weights=e._simplex(2)
    grid={'number':0,'registered_utc':'2026-10-03T14:00:00Z','components':[new,reference],
          'weights':weights,'identity':sorted([e._prediction_identity(new),e._prediction_identity(reference)])}
    directory=tmp_path/'blends/triple0';directory.mkdir(parents=True)
    grid_ref=saved(directory/'grid.json',grid)
    rows=[]
    for i,weight in enumerate(weights[:2] if partial else weights):
        pred=frame(origins);pred['pred_kms']=400*weight[0]+500*weight[1]
        row={'index':i,'weights':weight,'prediction':saved(tmp_path/f'blend{i}.parquet',pred),
             'metrics':{'rank':20,'official':{'mse':20}},'completed_utc':'2026-10-03T15:00:00Z'}
        rows.append(saved(directory/f'{i:02d}.json',row))
    summary={'number':0,'status':'partial_diagnostic_only' if partial else 'complete','grid':grid_ref,'results':rows}
    chosen={'kind':'standalone','components':[dict(new,weight=1.)],'prediction':new['prediction'],
            'metrics':new['metrics'],'key':new['key']}
    selection={'completed_utc':'2026-10-03T16:00:00Z','records':[result_ref], 'triples':[summary],
               'chosen':chosen,'official_informed_selection':True,'confirmatory':False}
    selection_ref=saved(tmp_path/'selection.json',selection)
    recipe={'frozen_utc':'2026-10-03T17:00:00Z','components':chosen['components'],'selection':new['metrics'],
            'prediction':saved(tmp_path/'final-predictions.parquet',frame(origins))}
    return recipe,{digest:(result_ref,result)}, {'search_stop_utc':'2026-10-04T06:00:00Z'},selection_ref


@pytest.mark.parametrize('partial',[False,True])
def test_whole_static_grid_and_eligibility(tmp_path,monkeypatch,partial):
    recipe,results,runtime,selection_ref=blend_fixture(tmp_path,monkeypatch,partial)
    e.verify_blends(tmp_path,recipe,results,runtime,selection_ref)
    selection=json.loads(Path(selection_ref['path']).read_text())
    if partial:
        selection['triples'][0]['status']='complete'
    else:
        selection['triples'][0]['results'].pop()
    selection_ref=saved(tmp_path/'bad-selection.json',selection)
    with pytest.raises(e.old.IntegrityError,match='incomplete|uncounted'):
        e.verify_blends(tmp_path,recipe,results,runtime,selection_ref)
