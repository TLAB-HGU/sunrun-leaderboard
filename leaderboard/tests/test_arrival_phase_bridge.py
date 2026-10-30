"""No fits or official truth: cross-cycle dev eligibility and clock boundaries."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.mse20_hss_phase import selection_bridge as g


@pytest.fixture
def pairs(monkeypatch,tmp_path):
    old=tmp_path/'state';new=tmp_path/'phase';old.mkdir();new.mkdir()
    monkeypatch.setattr(g.state,'RUN',old)
    monkeypatch.setattr(g.p,'verify_registration',lambda _: {})
    monkeypatch.setattr(g.p,'setup',lambda _: {'regime_cells_final_gate':True})
    folds=g.dev.fold_grid();records={};selections={}
    for directory,name,penalty in ((old,'state',5.),(new,'phase',2.)):
        config={'id':name}
        rows=[dict(config_sha256=g.dev.config_hash(config),fold=fold,status='ok',
            details={'metrics':dict(hss=dict(actual_count=1,predicted_count=1,tp=1,fp=0,fn=0,penalty_hours=penalty),
                modal_unique_events=dict(tp=1,fp=0,fn=0),squared_error_sum=100.,naive_squared_error_sum=200.,n_pairs=72)})
            for fold in folds]
        records[directory]=rows
        selections[directory]=g.state_dev.rank_results(rows,[config],folds)
        for name in ('registered.json','dev-selection.json','authorization.json'):
            (directory/name).write_text('{}')
    def snapshot(directory,protocol,verifier,independent=False):
        local=selections[directory]
        return dict(configs=[local['selected_config']],folds=folds,ranking=g.dev.RANKING),local,records[directory]
    monkeypatch.setattr(g,'snapshot',snapshot)
    return old,new,records,selections


def test_bridge_chooses_only_local_winner(pairs):
    old,new,records,selections=pairs
    extra=deepcopy(records[new][0]);extra['config_sha256']='unselected'
    records[new].append(extra)
    result=g.decision_payload(new)
    assert result['family']=='phase' and result['selected_config']==selections[new]['selected_config']
    assert len(result['reported_all_config_candidates'])==2 and not result['selection_uses_official']


def test_bridge_retains_all_four_caps(pairs,monkeypatch):
    old,new,*_=pairs
    monkeypatch.setattr(g.p,'setup',lambda _: {'regime_cells_final_gate':False})
    with pytest.raises(ValueError,match='caps'):g.decision_payload(new)


def test_bridge_must_precede_final_fit(pairs,monkeypatch):
    old,new,*_=pairs
    monkeypatch.setattr(g.p,'deadline',lambda *args:3600)
    (old/'final-fit-claim.json').write_text('{}')
    with pytest.raises(ValueError,match='precede'):g.select(new)


def test_bridge_cannot_compare_different_support(pairs,monkeypatch):
    old,new,records,selections=pairs
    original=g.snapshot
    def bad(directory,*args,**kwargs):
        registration,selection,rows=original(directory,*args,**kwargs)
        if directory==new:registration['folds']=registration['folds'][:-1]
        return registration,selection,rows
    monkeypatch.setattr(g,'snapshot',bad)
    with pytest.raises(ValueError,match='folds'):g.decision_payload(new)
