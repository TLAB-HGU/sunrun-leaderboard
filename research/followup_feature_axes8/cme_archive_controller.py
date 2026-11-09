"""Finite official issued-notification collection; never a provenance readiness gate."""
import argparse
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENDPOINT = 'https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications'
OLD = ROOT / 'analysis/w50_audit_20261108/notification_archive'


def month_bounds(value):
    if not re.fullmatch(r'\d{4}-\d{2}', value):
        raise ValueError('Expected YYYY-MM')
    month = pd.Period(value, freq='M')
    return str(month.start_time.date()), str(month.end_time.date())


def validate_payload(body):
    rows = json.loads(body)
    if not isinstance(rows, list):
        raise ValueError('Expected official notification list')
    spec = importlib.util.spec_from_file_location('notification_parser', OLD.parent / 'notification_archive_probe.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    seen = {}
    accepted = 0
    for row in rows:
        if not isinstance(row, dict) or row.get('messageType') != 'CME':
            raise ValueError('Expected CME notification object')
        if not row.get('messageID') or not isinstance(row.get('messageBody'), str):
            raise ValueError('Missing message identity/body')
        stamp = pd.Timestamp(row.get('messageIssueTime'))
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError('Invalid or timezone-naive issue date')
        if row['messageID'] in seen and seen[row['messageID']] != row:
            raise ValueError('Conflicting message versions in response')
        seen[row['messageID']] = row
        accepted += int(parser.parse_message(row)['single_cme_forecast_summary'])
    return {'messages': len(rows), 'accepted_summary_rows': accepted}


def collect(out, months):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive = HERE / 'input/cme_archive'
    archive.mkdir(parents=True, exist_ok=True)
    # Copy immutable source bytes and original retrieval receipts; never overwrite.
    for source in sorted(OLD.glob('????-??*.json')):
        dest = archive / source.name
        if not dest.exists():
            shutil.copy2(source, dest)
        elif dest.read_bytes() != source.read_bytes():
            raise ValueError(f'Existing archival copy differs: {dest.name}')
    failures, completed = [], []
    for number, month in enumerate(months, 1):
        start, end = month_bounds(month)
        url = ENDPOINT + '?' + urllib.parse.urlencode({'startDate': start, 'endDate': end, 'type': 'CME'})
        dest = archive / f'{month}.json'
        try:
            if dest.exists():
                body = dest.read_bytes()
                counts = validate_payload(body)
                receipt = json.loads(dest.with_suffix('.receipt.json').read_text())
            else:
                request = urllib.request.Request(url, headers={'User-Agent': 'sunrun-research-data-audit/1.0'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    body = response.read()
                    receipt = {'url': url, 'fetched_utc': datetime.now(UTC).isoformat(),
                               'status': response.status, 'response_bytes': len(body),
                               'content_type': response.headers.get('Content-Type'),
                               'archive_version': 'current official API response at fetched_utc',
                               'historical_immutability_proven': False}
                counts = validate_payload(body)
                dest.write_bytes(body)
                dest.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            event = {'month': month, **counts, **receipt, 'path': str(dest), 'unit': number, 'units': len(months)}
            completed.append(event)
        except (OSError, ValueError, KeyError, TypeError) as error:
            event = {'month': month, 'url': url, 'error': str(error), 'unit': number, 'units': len(months)}
            failures.append(event)
        with (out / 'retrieval.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(dict(event, stage='issued_notification_collection')), flush=True)
    status = {'ready': False, 'blocked_reason': 'immutable_as_issued_body_provenance_unproven',
              'archive_root': str(archive), 'requested_months': months,
              'data_gap_fixed': len(completed) == len(months), 'completed': completed, 'failures': failures,
              'target_data_read': False, 'historical_immutability_proven': False,
              'note': 'Current retrieval fixes archive coverage only. Issue timestamps and current receipt do not prove historical body versions.'}
    (HERE / 'cme-source-status.json').write_text(json.dumps(status, indent=2) + '\n')
    (out / 'complete.json').write_text(json.dumps({'publish': 'not_required', 'publish_reason': 'Data collection; axis8 provenance gate remains blocked', 'failures': len(failures)}, indent=2) + '\n')
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'runs/cme_archive_collection')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--threads', type=int, choices=[1], default=1)
    parser.add_argument('--months', nargs='+', default=[f'2026-{i:02}' for i in range(1, 10)])
    args = parser.parse_args()
    for month in args.months:
        month_bounds(month)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return collect(args.out, args.months)


if __name__ == '__main__':
    raise SystemExit(main())
