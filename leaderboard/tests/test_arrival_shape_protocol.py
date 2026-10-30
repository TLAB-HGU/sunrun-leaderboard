"""Small new-cycle contract tests without scientific fits."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mse20_hss_shape import protocol as p


def test_fixed_four_config_56_claims_and_shared_clock():
    assert p.CAPS==dict(dev_claimed_attempts=56,structural_configs=4,final_refits=1)
    assert p.CLOCK==p.previous.CLOCK
    assert p.RUN!=p.previous.RUN
    assert len(p.final_gates({'regime_cells_final_gate':True}))==6


def test_cycle_reads_keep_previous_registration_unchanged():
    assert "result=previous.sources()" in Path(p.__file__).read_text()
    assert "test_arrival_shape*.py" in Path(p.__file__).read_text()
    # New test naming cannot invalidate the previous frozen readiness glob.
    assert not Path(__file__).name.startswith('test_mse20_hss')


def test_dev_still_refuses_postcutoff_before_source_access(monkeypatch):
    monkeypatch.setattr(p,'read_json',lambda *args:pytest.fail('must not access sources'))
    with pytest.raises(ValueError):p.load_frames('2026-06-01T00:00:00Z')
