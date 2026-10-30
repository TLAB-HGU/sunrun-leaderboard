"""Small readiness checks; no science fitting or official truth reads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_hss_arrival import protocol as p


def auth_fixture(tmp_path,monkeypatch):
    user=tmp_path/'user.md'; user.write_text('actual instruction')
    clock=tmp_path/'clock'; clock.mkdir()
    (clock/'runtime.json').write_text('{}')
    monkeypatch.setattr(p,'USER_DOCUMENT',user)
    monkeypatch.setattr(p,'USER_SHA',p.digest(user))
    monkeypatch.setattr(p,'CLOCK',clock)
    return dict(user_document=p.file_ref(user),runtime=p.file_ref(clock/'runtime.json'),files={},
        peak_gate='unique_event_f1_nonregression_vs_48c28',total_mse_max=p.TOTAL_MAX,
        training_cutoff_utc=p.CUTOFF,official_for_selection=False,dual_output_same_model=True,
        additional_tree_hpo=False,simplex_blending=False,regime_cells_final_gate=True)


@pytest.mark.parametrize('value',[True,False])
def test_final_cell_policy_must_be_prospectively_explicit(tmp_path,monkeypatch,value):
    auth=auth_fixture(tmp_path,monkeypatch);auth['regime_cells_final_gate']=value
    (tmp_path/'authorization.json').write_text(json.dumps(auth))
    assert p.setup(tmp_path)['regime_cells_final_gate'] is value


@pytest.mark.parametrize('field,value',[('regime_cells_final_gate',None),('regime_cells_final_gate',1),
    ('official_for_selection',True),('additional_tree_hpo',True),('simplex_blending',True),
    ('dual_output_same_model',False),('total_mse_max',4000)])
def test_contract_boundary_fails_closed(tmp_path,monkeypatch,field,value):
    auth=auth_fixture(tmp_path,monkeypatch);auth[field]=value
    (tmp_path/'authorization.json').write_text(json.dumps(auth))
    with pytest.raises(ValueError):p.setup(tmp_path)


def test_deadline_always_uses_original_clock(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(p.legacy,'deadline',lambda *a:calls.append(a) or 1)
    assert p.deadline(tmp_path,'final',75,'now')==1
    assert calls==[(p.CLOCK,'final',75,'now')]


def test_dev_loader_refuses_official_sources_before_read(monkeypatch):
    monkeypatch.setattr(p,'read_json',lambda *a:pytest.fail('read must not happen'))
    with pytest.raises(ValueError,match='post-May31'):p.load_frames('2026-06-01T00:00:00Z')


def test_current_sources_require_finalizer_and_evaluator(monkeypatch):
    # Even when a performer has not finished writing its module yet, this
    # explicit gate prevents accidental preregistration/optimization.
    text=Path(p.__file__).read_text()
    assert "'evaluate','finalize'" in text
    monkeypatch.setattr(p.legacy,'sources',lambda:{})
    package=Path(p.__file__).parent
    if not (package/'evaluate.py').exists() or not (package/'finalize.py').exists():
        with pytest.raises(ValueError,match='incomplete'):p.sources()
    else:
        assert 'experiments/mse20_hss_arrival/finalize.py' in p.sources()


def test_readiness_revalidates_nested_log_hash(tmp_path):
    logfile=tmp_path/'run.log';logfile.write_text('passed')
    files={path.name:p.file_ref(path) for path in (p.ROOT/'tests').glob('test_mse20_hss*.py')}
    files['log']=p.file_ref(logfile)
    proof=dict(passed=True,exit_code=0,sources={},files=files)
    p.verify_readiness(proof,{})
    logfile.write_text('mutated')
    with pytest.raises(ValueError):p.verify_readiness(proof,{})


def test_readiness_rejects_incomplete_test_inventory():
    with pytest.raises(ValueError,match='inventory'):
        p.verify_readiness(dict(passed=True,exit_code=0,sources={},files={}),{})


def test_gate_declarations_match_auth_without_dev_cell_selection():
    assert len(p.final_gates({'regime_cells_final_gate':True}))==6
    assert len(p.final_gates({'regime_cells_final_gate':False}))==2
