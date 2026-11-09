import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cme_controller', Path(__file__).with_name('cme_archive_controller.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('value', ['2026-13', '2026-00', '2026-1', '2026-01-01', 'no'])
def test_invalid_month(value):
    with pytest.raises(ValueError):
        m.month_bounds(value)


def test_leap_month():
    assert m.month_bounds('2024-02') == ('2024-02-01', '2024-02-29')


@pytest.mark.parametrize('value', [{}, [1], [{'messageType': 'CME'}], [{'messageType': 'FLR'}]])
def test_malformed_results(value):
    with pytest.raises(ValueError):
        m.validate_payload(json.dumps(value))


@pytest.mark.parametrize('date', ['invalid', None, '2026-01-01'])
def test_invalid_issue_dates(date):
    with pytest.raises(ValueError):
        m.validate_payload(json.dumps([dict(messageType='CME', messageID='1', messageBody='', messageIssueTime=date)]))


def test_valid_empty_and_missing_summary():
    assert m.validate_payload('[]')['messages'] == 0
    row = dict(messageType='CME', messageID='1', messageBody='No accepted summary', messageIssueTime='2026-01-01T00:00Z')
    assert m.validate_payload(json.dumps([row]))['accepted_summary_rows'] == 0
    with pytest.raises(ValueError, match='Conflicting'):
        m.validate_payload(json.dumps([row, dict(row, messageBody='changed')]))


def test_collector_is_notification_only():
    assert m.ENDPOINT.endswith('/get/notifications')
    source = Path(m.__file__).read_text()
    assert 'read_parquet' not in source and 'observations' not in source
    assert "'type': 'CME'" in source
