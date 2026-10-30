"""Actual new deadline authority, without mutating registered old run clocks."""
import pytest
from experiments.mse20_hss_phase import protocol as p


@pytest.fixture
def clock(tmp_path,monkeypatch):
    directive=tmp_path/'directive.json'
    p.write_new(directive,{'search_time_limit_removed':True})
    p.write_new(tmp_path/'runtime.json',dict(resumed_utc='2026-10-04T07:05:19Z',
        search_time_limit_removed=True,search_stop_utc=None,optimization_stop_utc=None,
        deadline_utc=None,user_resume=p.file_ref(directive)))
    monkeypatch.setattr(p,'CLOCK',tmp_path)
    return tmp_path,directive


def test_unlimited_search_has_finite_operation_timeout(clock):
    for stage in ('search','optimization','final'):
        assert p.deadline(stage=stage,now='2028-01-01T00:00:00Z')==3600.


def test_operation_before_actual_resume_or_unknown_stage_rejected(clock):
    with pytest.raises(ValueError,match='precedes'):
        p.deadline(now='2026-10-04T06:00:00Z')
    with pytest.raises(ValueError,match='unknown'):
        p.deadline(stage='fake')


def test_unbounded_authority_is_hash_verified(clock):
    directory,directive=clock
    directive.write_text('{"search_time_limit_removed":false}')
    with pytest.raises(ValueError,match='hash|changed|stale artifact'):
        p.deadline(now='2028-01-01T00:00:00Z')


def test_ordered_actual_timestamps_with_no_expiry():
    p.check_clock(['2026-10-03T06:26:29Z','2026-10-04T01:38:12Z',None],'search')
    with pytest.raises(ValueError,match='nonmonotonic'):
        p.check_clock(['2026-10-04T01:38:12Z','2026-10-03T06:26:29Z',None],'search')
    with pytest.raises(ValueError,match='future'):
        p.check_clock(['2100-01-01T00:00:00Z',None],'search')
