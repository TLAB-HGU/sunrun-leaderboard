import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('external_features', Path(__file__).with_name('external_features.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
O = pd.Timestamp('2026-01-01T01:00Z')


def ace(minutes, speeds, status=None):
    return pd.DataFrame({'S': status or [0] * len(minutes), 'spd': speeds},
                        index=[O + pd.Timedelta(minutes=i) for i in minutes])


def record(issue, speed=500, message='m1'):
    return dict(message_id=message, conservative_issue_time=str(issue),
                single_cme_forecast_summary=True, event_ids=['2026-01-01T00:00:00-CME-001'],
                speed_kms=speed, half_angle_deg=30, longitude_deg=4, latitude_deg=-5)


def test_publication_boundary_status_and_future_poison():
    frame = ace([-60, -30, -15, -4, -3, 0], [999, 400, 500, 600, 9999, 9999])
    got = m.build(6, [O], ace_minutes=frame).iloc[0]
    np.testing.assert_allclose(got, [np.std([400, 500, 600]), 100, 200])
    polluted = pd.concat([frame, ace([1, 2], [1e9, 1e9])])
    pd.testing.assert_frame_equal(m.build(6, [O], ace_minutes=frame), m.build(6, [O], ace_minutes=polluted))
    filtered = ace([-30, -20, -10, -4], [400, 999, -1, 600], [0, 1, 0, 0])
    assert m.build(6, [O], ace_minutes=filtered).iloc[0, 2] == 200


def test_trend_order_and_missing():
    frame = ace([-45, -15, -5], [415, 445, 455])
    got = m.build(7, [O, O - pd.Timedelta(days=1), O], ace_minutes=frame)
    assert got.iloc[0, 0] == pytest.approx(1)
    assert got.iloc[0, 1] == 25
    assert got.iloc[1].isna().all()
    pd.testing.assert_series_equal(got.iloc[0], got.iloc[2])
    assert m.build(7, [O], ace_minutes=ace([-4], [500])).isna().all().all()


def test_ace_duplicate_conflict():
    assert m.build(6, [O], ace_minutes=ace([-4, -4], [500, 500])).iloc[0, 0] == 0
    with pytest.raises(ValueError, match='Conflicting ACE'):
        m.build(6, [O], ace_minutes=ace([-4, -4], [500, 501]))


def test_cme_issue_boundary_versions_and_expiry():
    first = record(O - pd.Timedelta(hours=1))
    update = record(O + pd.Timedelta(minutes=1), 900, 'm2')
    got = m.build(8, [O, O + pd.Timedelta(minutes=1), O + pd.Timedelta(hours=74)], cme_records=[first, first, update])
    assert got.iloc[0, 0] == 500
    assert got.iloc[1, 0] == 900
    assert got.iloc[2].isna().all()
    assert m.build(8, [O], cme_records=[record(O - pd.Timedelta(hours=72))]).iloc[0, -1] == 72
    with pytest.raises(ValueError, match='Conflicting CME'):
        m.build(8, [O], cme_records=[first, dict(first, speed_kms=600)])


def test_cme_missing_and_tied_fields():
    assert m.build(8, [O], cme_records=[dict(record(O), speed_kms=None)]).isna().all().all()
    with pytest.raises(ValueError, match='Ambiguous'):
        m.build(8, [O], cme_records=[record(O), record(O, 600, 'other')])


def test_issued_parser_uses_body_seconds_and_excludes_notes():
    parser = m._module('analysis/w50_audit_20261108/notification_archive_probe.py')
    raw = {'messageID': '1', 'messageIssueTime': '2026-01-01T01:00Z',
           'messageBody': 'Message Issue Date: 2026-01-01T01:00:30Z\n## Summary:\n'
           '2026-01-01T00:00:00-CME-001 Estimated speed: 500 km/s\n'
           'Estimated opening half-angle: 30 deg Direction (lon./lat.): 4 / -5\n'
           '## Notes:\nEstimated speed: 9999 km/s'}
    parsed = parser.parse_message(raw)
    assert m.build(8, [O], cme_records=[parsed]).isna().all().all()
    assert m.build(8, [O + pd.Timedelta(seconds=30)], cme_records=[parsed]).iloc[0, 0] == 500
    assert not parser.parse_message(dict(raw, messageBody='no summary'))['single_cme_forecast_summary']


def test_readiness_missing_cache(tmp_path):
    result = m.probe_readiness(6, start=O, end=O, cache_root=tmp_path)
    assert not result['ready'] and result['missing_count'] == 1
    assert not m.probe_readiness(8, start=O, end=O, archive_root=tmp_path)['ready']
