import numpy as np
import pandas as pd
import pytest
from experiments.e0472_ablation_hpo import official_format_recovery as F

@pytest.fixture
def grid():
    origins = pd.date_range('2026-05-31 23:00', '2026-09-26 23:00', freq='h')
    return pd.DataFrame({F.KEY: origins.repeat(72), 'horizon_hours': np.tile(np.arange(1,73),len(origins)),
                         'pred_kms': np.linspace(300,700,len(origins)*72)})

def test_canonical_roundtrip_preserves_bits_order_and_utc(grid,tmp_path):
    grid.loc[0,'pred_kms'] = -0.0
    result = F.canonical_frame(grid)
    assert result.iloc[0][F.KEY] == '2026-05-31T23:00:00Z'
    path=tmp_path/'output.parquet';result.to_parquet(path,index=False)
    F.verify_values(grid,pd.read_parquet(path))
    assert str(grid[F.KEY].dtype).startswith('datetime64')

@pytest.mark.parametrize('defect',['duplicate','missing','nonfinite','horizon','fraction','timeline'])
def test_invalid_grid_fails_closed(grid,defect):
    if defect=='duplicate': grid.iloc[1]=grid.iloc[0]
    elif defect=='missing': grid=grid.iloc[:-1]
    elif defect=='nonfinite': grid.loc[3,'pred_kms']=np.inf
    elif defect=='horizon': grid.loc[3,'horizon_hours']=73
    elif defect=='fraction': grid.loc[3,F.KEY]+=pd.Timedelta(nanoseconds=1)
    elif defect=='timeline': grid[F.KEY]+=pd.Timedelta(hours=1)
    with pytest.raises(ValueError): F.canonical_frame(grid)

@pytest.mark.parametrize('defect',['value','signed_zero','dtype','order','horizon'])
def test_numeric_and_row_changes_rejected(grid,defect):
    after=grid.copy()
    if defect=='value': after.loc[1,'pred_kms']+=1
    elif defect=='signed_zero': grid.loc[0,'pred_kms']=0.;after.loc[0,'pred_kms']=-0.
    elif defect=='dtype': after['pred_kms']=after.pred_kms.astype('float32')
    elif defect=='order': after=after.iloc[::-1].reset_index(drop=True)
    elif defect=='horizon': after.loc[1,'horizon_hours']=70
    with pytest.raises(ValueError): F.verify_values(grid,after)

def test_no_adapter_execution_without_expos(monkeypatch,tmp_path):
    monkeypatch.delenv('EXPOS_OUT',raising=False)
    with pytest.raises(RuntimeError,match='requires ExpOS'):F.execute(tmp_path/'absent.json')

def test_manifest_change_and_deadline_fail(monkeypatch):
    manifest={'name':'baseline','source_root':'source','deadline':100}
    monkeypatch.setattr(F,'source_manifest',lambda *a:dict(manifest))
    monkeypatch.setattr(F.time,'time',lambda:101)
    with pytest.raises(TimeoutError): F.validate_manifest(manifest)
    monkeypatch.setattr(F,'source_manifest',lambda *a:{**manifest,'deadline':200})
    with pytest.raises(RuntimeError,match='changed'): F.validate_manifest(manifest)

def test_original_scored_source_rejected(monkeypatch,tmp_path):
    from expos import core
    state={'units':{'baseline:official:official':{'experiment':'E0849','status':'complete'}}}
    monkeypatch.setattr(F.C,'read_json',lambda p:state)
    monkeypatch.setattr(core,'state',lambda:{'experiments':{'E0849':{'lbeval':[{'score':1}]}}})
    with pytest.raises(RuntimeError,match='already has a score'):F.source_manifest('baseline',tmp_path)

@pytest.mark.parametrize('case',['duplicate','failed','changed'])
def test_recorded_source_lineage_or_hash_rejected(monkeypatch,tmp_path,case):
    from expos import core
    output=tmp_path/'E0849';output.mkdir();(output/'receipt.json').write_text('{}')
    entry={'runs':[{'rc':1 if case=='failed' else 0,'outputs':{'receipt.json':'bad'}}]}
    monkeypatch.setattr(core,'RUNS',tmp_path)
    monkeypatch.setattr(core,'state',lambda:{'experiments':{'E0849':entry}})
    monkeypatch.setattr(core,'_lines',lambda:[{'event':'start','id':'E0849'}]*(2 if case=='duplicate' else 1))
    monkeypatch.setattr(F.C,'experiment_processes',lambda eid:[])
    with pytest.raises(RuntimeError):F.O.verify_recorded_files('E0849',['receipt.json'])

def test_execute_preserves_source_and_receipts_across_three_seeds(grid,monkeypatch,tmp_path):
    from expos import core
    src=tmp_path/'E0849';src.mkdir();out=tmp_path/'new';out.mkdir()
    for seed in F.O.SEEDS:
        own=grid.copy();own['pred_kms']+=seed
        own.to_parquet(src/f'predictions_official_s{seed}.parquet',index=False)
        (src/f'fit_audit_s{seed}.json').write_text('{"source":"original"}')
        (src/f'poison_official_s{seed}.json').write_text('{"passed":true}')
    original={p.name:F.C.file_hash(p) for p in src.iterdir()}
    manifest={'source_eid':'E0849','source_outputs':original}
    path=tmp_path/'unit.json';F.C.write_json(path,manifest)
    monkeypatch.setenv('EXPOS_OUT',str(out));monkeypatch.setattr(core,'RUNS',tmp_path)
    monkeypatch.setattr(F,'validate_manifest',lambda value:None)
    F.execute(path)
    assert original=={p.name:F.C.file_hash(p) for p in src.iterdir()}
    audit=F.C.read_json(out/'format_recovery_audit.json')
    assert len(audit['seeds'])==3 and audit['scientific_changes'] is False
    for seed in F.O.SEEDS:
        F.verify_values(pd.read_parquet(src/f'predictions_official_s{seed}.parquet'),
                        pd.read_parquet(out/f'predictions_official_s{seed}.parquet'))
        assert F.C.file_hash(out/f'fit_audit_s{seed}.json')==original[f'fit_audit_s{seed}.json']
        assert F.C.file_hash(out/f'poison_official_s{seed}.json')==original[f'poison_official_s{seed}.json']

def test_review_requires_exact_code_and_manifests(monkeypatch,tmp_path):
    import argparse
    recovery=F.Recovery(argparse.Namespace(root=tmp_path))
    recovery.state={'manifests':{'baseline':{'sha256':'frozen'}}}
    monkeypatch.setattr(F,'adapter_hashes',lambda:{'adapter':'new'})
    F.C.write_json(tmp_path/'review.json',{'verdict':'accept','reviewer':'independent',
                       'code_hashes':{'adapter':'old'},'manifest_hashes':{'baseline':'frozen'}})
    with pytest.raises(RuntimeError,match='review'):recovery.check_review()
    review=F.C.read_json(tmp_path/'review.json');review['code_hashes']={'adapter':'new'};review['independent']=True;review['signature']=F.C.digest(review)
    F.C.write_json(tmp_path/'review.json',review);recovery.check_review()


def test_exact_protected_validator_key_contract(grid):
    from scorer.scoring import validate
    canonical = F.canonical_frame(grid)
    folds = canonical[[F.KEY]].drop_duplicates()
    assert any('unexpected rows' in e for e in validate(grid, folds))
    assert validate(canonical, folds) == []

@pytest.mark.parametrize('interrupted',[False,True])
def test_recovery_score_request_is_durable_and_never_repeated(tmp_path,monkeypatch,interrupted):
    import argparse
    from expos import core
    recovery=F.Recovery(argparse.Namespace(root=tmp_path))
    recovery.state={'started_at':F.time.time(),'max_hours':72,'official_evaluations':{},
                    'units':{'baseline:official:official':{'status':'complete','experiment':'NEW'}}}
    graph={'experiments':{'NEW':{'lbeval':[]}}}
    monkeypatch.setattr(core,'state',lambda:graph)
    monkeypatch.setattr(recovery,'verified_result',lambda record:{})
    monkeypatch.setattr(recovery,'event',lambda *args:None)
    calls=[]
    def cli(*args,**kwargs):
        assert F.C.read_json(recovery.path)['official_evaluations']['baseline']['status']=='requested'
        calls.append(args)
        if interrupted:raise RuntimeError('uncertain connection')
        graph['experiments']['NEW']['lbeval']=[{'summary':{}}]
    monkeypatch.setattr(recovery,'cli',cli)
    recovery.evaluate_once('baseline')
    recovery.state=F.C.read_json(recovery.path)
    recovery.evaluate_once('baseline')
    assert calls==[('lb-eval','NEW')]
