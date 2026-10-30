"""Global comparison honors frozen local winners; no final fits or truth."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_hss_shape import global_selection as g


def fixture(monkeypatch,tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr(g.original,'RUN',old)
    monkeypatch.setattr(g.p,'verify_registration',lambda run:{})
    monkeypatch.setattr(g.p,'setup',lambda run:dict(regime_cells_final_gate=True))
    folds=g.dev.fold_grid();configs=[{'id':'old'},{'id':'new'}]
    records={};selections={}
    for directory,config,penalty in ((old,configs[0],5.),(new,configs[1],2.)):
        rows=[dict(config_sha256=g.original_dev.config_hash(config),fold=f,status='ok',details={'metrics':dict(
            hss=dict(actual_count=1,predicted_count=1,tp=1,fp=0,fn=0,penalty_hours=penalty),
            modal_unique_events=dict(tp=1,fp=0,fn=0),squared_error_sum=100.,naive_squared_error_sum=200.,n_pairs=72)}) for f in folds]
        records[directory]=rows
        selections[directory]=g.original_dev.rank_results(rows,[config],folds)
        for name in ('registered.json','dev-selection.json','authorization.json'):(directory/name).write_text('{}')
    def snapshot(directory,protocol):
        return dict(configs=[configs[0] if directory==old else configs[1]],folds=folds,ranking=g.dev.RANKING),selections[directory],records[directory]
    monkeypatch.setattr(g,'snapshot',snapshot)
    return old,new,records,selections


def test_shared_timing_rank_chooses_frozen_cycle_winner(monkeypatch,tmp_path):
    old,new,records,selections=fixture(monkeypatch,tmp_path)
    result=g.decision_payload(new)
    assert result['cycle']=='shape' and result['selected_run']==str(new)
    assert result['selected_config']==selections[new]['selected_config']
    assert len(result['reported_all_config_candidates'])==2
    assert result['selection_uses_official'] is False


def test_mandatory_caps_cannot_be_dropped(monkeypatch,tmp_path):
    old,new,*_=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(g.p,'setup',lambda run:dict(regime_cells_final_gate=False))
    with pytest.raises(ValueError,match='mandatory'):g.decision_payload(new)


def test_global_selection_rejects_already_started_fit(monkeypatch,tmp_path):
    old,new,*_=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(g.p,'deadline',lambda *args:1)
    (old/'final-fit-claim.json').write_text('{}')
    with pytest.raises(ValueError,match='precede'):g.select(new)


def test_comparison_does_not_promote_nonwinning_local_configuration(monkeypatch,tmp_path):
    old,new,records,selections=fixture(monkeypatch,tmp_path)
    # This diagnostic-only local alternative is intentionally not eligible
    # after its own immutable cycle chose another configuration.
    other=deepcopy(records[new][0]);other['config_sha256']='nonwinner'
    records[new].append(other)
    result=g.decision_payload(new)
    assert result['selected_config']==selections[new]['selected_config']
