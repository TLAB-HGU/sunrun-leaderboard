"""Protect forecast-only extraction and conservative availability semantics."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('notification_probe', ROOT / 'notification_archive_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def issued_example():
    source = json.loads((ROOT / 'cme_information_sources.json').read_text())
    return source['target44038_followup']['issued_messages'][0].copy()


def test_official_body_precision_and_forecast_values():
    parsed = PROBE.parse_message(issued_example())
    assert parsed['single_cme_forecast_summary'] is True
    assert pd.Timestamp(parsed['conservative_issue_time']) == pd.Timestamp('2026-01-18T21:39:21Z')
    assert parsed['speed_kms'] == 1431
    assert parsed['half_angle_deg'] == 41
    assert parsed['longitude_deg'] == -27
    assert parsed['latitude_deg'] == -3
    assert parsed['event_ids'] == ['2026-01-18T18:09:00-CME-001']


def test_retrospective_notes_cannot_supply_kinematics():
    row = issued_example()
    row['messageBody'] += '\nEstimated speed: 9999 km/s.\nDirection (lon./lat.): 1/2\n'
    assert PROBE.parse_message(row)['speed_kms'] == 1431
    row['messageBody'] = row['messageBody'].replace('Estimated speed: ~1431 km/s.', '')
    parsed = PROBE.parse_message(row)
    assert parsed['speed_kms'] is None
    assert parsed['single_cme_forecast_summary'] is False


def test_ambiguous_forecast_speed_is_not_silently_selected():
    row = issued_example()
    row['messageBody'] = row['messageBody'].replace('Estimated speed: ~1431 km/s.',
        'Estimated speed: ~1431 km/s.\nEstimated speed: ~1473 km/s.')
    parsed = PROBE.parse_message(row)
    assert parsed['speed_kms'] is None
    assert parsed['single_cme_forecast_summary'] is False


def test_later_api_timestamp_and_missing_summary_fail_conservatively():
    row = issued_example()
    row['messageIssueTime'] = '2026-01-18T21:40Z'
    assert pd.Timestamp(PROBE.parse_message(row)['conservative_issue_time']) == pd.Timestamp('2026-01-18T21:40Z')
    row['messageBody'] = row['messageBody'].replace('## Summary:', '## Unknown:')
    assert PROBE.parse_message(row)['single_cme_forecast_summary'] is False
