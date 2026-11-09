"""Inventory training-era, issued CME messages; no fitting or benchmark truth.

Raw API bodies and receipt metadata are preserved. Extraction intentionally
accepts only unambiguous labeled forecast summaries, never retrospective notes.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENDPOINT = 'https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications'
CUTOFF = pd.Timestamp('2025-12-29', tz='UTC')


def training_coverage(records):
    """Report support on input origins; never open any target column."""
    path = ROOT.parents[1] / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
    raw = pd.read_parquet(path, columns=['origin'], filters=[('fit_F1', '==', True), ('h', '==', 1)],
                          use_threads=False)
    origins = pd.DatetimeIndex(pd.to_datetime(raw.origin, utc=True)).sort_values()
    valid = [row for row in records if row['single_cme_forecast_summary']]
    stamps = np.sort(np.array([pd.Timestamp(row['conservative_issue_time']).value for row in valid], dtype=np.int64))
    recent = (np.searchsorted(stamps, origins.asi8, side='right')
              - np.searchsorted(stamps, origins.asi8 - pd.Timedelta(hours=72).value, side='left'))
    speeds = np.array([row['speed_kms'] for row in valid])
    return {'scope': 'F1 training input origins only; target columns not loaded',
            'origins': len(origins), 'unique_events': len({row['event_ids'][0] for row in valid}),
            'origins_with_any_unambiguous_message_issued_in_preceding_72h': int((recent > 0).sum()),
            'fraction_with_recent_message': float((recent > 0).mean()),
            'speed_quantiles_0_50_90_99_100': np.quantile(speeds, [0, .5, .9, .99, 1]).tolist() if len(speeds) else None,
            'warning': '72h is the forecast length, not a selected propagation window; exposure does not prove Earth impact.'}


def single_number(pattern, text):
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(matches[0]) if len(matches) == 1 else None


def parse_message(row):
    body = row.get('messageBody') or ''
    # The summary is a forecast statement. Notes may contain later observations.
    summary = body.split('## Summary:', 1)[-1].split('## Notes:', 1)[0]
    if '## Summary:' not in body:
        summary = ''
    stamps = re.findall(r'Message Issue Date:\s*(\S+)', body)
    issued = pd.to_datetime(row['messageIssueTime'], utc=True)
    if len(stamps) == 1:
        # Body carries seconds whereas API may round to a minute. Be conservative.
        issued = max(issued, pd.to_datetime(stamps[0], utc=True))
    speed = single_number(r'Estimated speed:\s*~?\s*([\d.]+)\s*km/s', summary)
    angle = single_number(r'Estimated opening half-angle:\s*~?\s*([\d.]+)\s*deg', summary)
    directions = re.findall(r'Direction\s*\(lon\./lat\.\):\s*([+\-\d.]+)\s*/\s*([+\-\d.]+)', summary)
    event_ids = sorted(set(re.findall(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-CME-\d+', summary)))
    if len(directions) == 1:
        lon, lat = map(float, directions[0])
    else:
        lon = lat = None
    valid = (speed is not None and speed > 0 and angle is not None and 0 < angle <= 180
             and lon is not None and -180 <= lon <= 180 and lat is not None
             and -90 <= lat <= 90 and len(event_ids) == 1)
    return {'message_id': row.get('messageID'), 'url': row.get('messageURL'),
            'api_issue_time': row['messageIssueTime'], 'conservative_issue_time': issued.isoformat(),
            'single_cme_forecast_summary': valid, 'event_ids': event_ids,
            'speed_kms': speed, 'half_angle_deg': angle, 'longitude_deg': lon, 'latitude_deg': lat,
            'notes_excluded': True, 'simulation_id_not_inferred': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'notification_archive')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, receipts, errors = {}, [], []
    for month in pd.period_range('2022-08', '2025-12', freq='M'):
        params = {'startDate': str(month.start_time.date()),
                  'endDate': str(month.end_time.date()), 'type': 'CME'}
        url = ENDPOINT + '?' + urllib.parse.urlencode(params)
        path = args.out / f'{month}.json'
        try:
            if path.exists():
                body = path.read_bytes()
                receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            else:
                request = urllib.request.Request(url, headers={'User-Agent': 'sunrun-research-data-audit/1.0'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read()
                    receipt = {'url': url, 'fetched_utc': datetime.now(UTC).isoformat(),
                               'status': response.status, 'response_bytes': len(body),
                               'content_type': response.headers.get('Content-Type')}
                rows = json.loads(body)
                if not isinstance(rows, list):
                    raise ValueError('Expected a message list')
                path.write_bytes(body)
                path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
                time.sleep(.5)
            rows = json.loads(body)
            for row in rows:
                if row.get('messageType') != 'CME':
                    raise ValueError('Unexpected non-CME message')
                parsed = parse_message(row)
                if pd.Timestamp(parsed['conservative_issue_time']) >= CUTOFF:
                    continue
                key = parsed['message_id']
                if key in records and records[key] != parsed:
                    raise ValueError('Conflicting duplicated issued message')
                records[key] = parsed
            receipts.append(receipt)
            print(json.dumps({'month': str(month), 'messages': len(rows), 'stage': 'archive_inventory'}), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'month': str(month), 'url': url, 'error': str(error)})
            print(json.dumps({'month': str(month), 'error': str(error)}), flush=True)
    values = sorted(records.values(), key=lambda row: row['conservative_issue_time'])
    result = {'scope': 'training-era issued messages only; no fitting/evaluation truth',
              'cutoff': CUTOFF.isoformat(), 'records': values, 'receipts': receipts, 'errors': errors,
              'message_count': len(values),
              'unambiguous_single_cme_kinematics': sum(row['single_cme_forecast_summary'] for row in values),
              'training_input_coverage': training_coverage(values),
              'limits': ['Issue records are not consumer receipt logs.',
                         'Current archival response does not by itself prove immutable historical bodies.',
                         'No absent-message inference about absence of eruptions.',
                         'No correspondence with stored ENLIL simulation IDs is assumed.',
                         'Extraction is an inventory, not an adopted feature or demonstrated skill gain.']}
    (args.out / 'inventory.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('message_count', 'unambiguous_single_cme_kinematics', 'errors')}))


if __name__ == '__main__':
    main()
