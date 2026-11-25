"""Publication-gated DONKI CME notification features: Earth-arrival state (T) and CME kinematics (K).

Source: CCMC DONKI notifications of type CME (https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications).
Each origin only sees messages whose body issue time is <= the origin. The parsing grammar was designed on
messages issued before 2025-11-01 and frozen; no target values are read anywhere in this module.

T (from ~/sunrun-public-event-arrival, 2026-11-13): hours to the next live Earth-arrival forecast,
hours since the most recent past one (6-96 h ago), and whether the next one is a glancing blow.
K (from ~/sunrun-event-precursor-ablation, 2026-11-14): speed, half-angle and lon/lat of the fastest
CME among the next forecast's Earth events, and the speed of the most recent past forecast's CME.
-999 means "not available" and is written at every origin, so the loader's forward fill is a no-op.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

T_FEATURES = ['pub_enlil_next_arr_h', 'pub_enlil_since_arr_h', 'pub_enlil_glancing']
K_FEATURES = ['pub_cme_next_speed_kms', 'pub_cme_next_half_angle_deg', 'pub_cme_next_lon_deg',
              'pub_cme_next_lat_deg', 'pub_cme_last_speed_kms']
FEATURES = T_FEATURES + K_FEATURES
MISSING = -999.
API = 'https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications?startDate={}&endDate={}&type=CME'
CORPUS_START = dt.date(2022, 1, 1)

TIME = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z'
CME_ID = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-CME-\d+'
MESSAGE_ID = r'\d{8}-AL-\d+'
EARTH = r'NASA missions near Earth'
ARRIVAL = re.compile(
    rf'(?P<modal>(?:will|may|might)\s+|(?:is|are)\s+estimated\s+to\s+)'
    rf'(?:possibly\s+)?(?:reach|impact)\s+{EARTH}\s+at\s+(?:about\s+)?(?P<arrival>{TIME})', re.I)
NEGATIVE = re.compile(
    rf'(?:will\s+not|is\s+not\s+expected\s+to|are\s+not\s+expected\s+to|'
    rf'is\s+no\s+longer\s+expected\s+to|are\s+no\s+longer\s+expected\s+to)'
    rf'\s+(?:reach|impact|affect)\s+{EARTH}', re.I)
KINEMATICS = re.compile(
    r'Estimated speed:\s*~(\d+) km/s\.\s*'
    r'Estimated opening half-angle:\s*(\d+) deg\.\s*'
    r'Direction \(lon\./lat\.\):\s*(-?\d+)/(-?\d+) in Heliocentric Earth Equatorial coordinates\.\s*'
    rf'Activity ID:\s*({CME_ID})')


def fetch_notifications(destination: Path, stop: dt.date | None = None) -> Path:
    """Download every CME notification since 2022-01-01 in 59-day windows; one record per message ID."""
    stop = stop or dt.datetime.now(dt.timezone.utc).date()
    start, byid = CORPUS_START, {}
    while start <= stop:
        end = min(start + dt.timedelta(days=58), stop)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(API.format(start.isoformat(), end.isoformat()), timeout=60) as response:
                    rows = json.loads(response.read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if not isinstance(rows, list):
            raise ValueError(f'unexpected DONKI response for {start}..{end}')
        for row in rows:
            byid[row.get('messageID')] = row
        start = end + dt.timedelta(days=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(byid.values()), ensure_ascii=False, indent=2) + '\n')
    return destination


def parse_message(message):
    """Return a traceable forecast/retraction or an explicit rejection record."""
    body = message.get('messageBody', '')
    record = dict(message_id=message.get('messageID'), message_url=message.get('messageURL'),
                  api_issue=message.get('messageIssueTime'), status='rejected', reason='',
                  body_issue=None, event_ids=[], previous_ids=[], arrival=None, glancing=-1,
                  earth_text='', action=None)
    issue_match = re.search(r'^##\s*Message Issue Date:\s*(\S+)', body, re.M | re.I)
    id_match = re.search(r'^##\s*Message ID:\s*(\S+)', body, re.M | re.I)
    if not issue_match or not id_match:
        record['reason'] = 'missing_body_identity_or_issue'
        return record
    api_issue = pd.to_datetime(record['api_issue'], utc=True, errors='coerce')
    issue = pd.to_datetime(issue_match.group(1), utc=True, errors='coerce')
    record['body_issue'] = None if pd.isna(issue) else issue.isoformat()
    if id_match.group(1) != record['message_id']:
        record['reason'] = 'body_message_id_mismatch'
        return record
    if pd.isna(issue) or pd.isna(api_issue) or not (api_issue <= issue < api_issue + pd.Timedelta(minutes=1)):
        record['reason'] = 'body_issue_not_within_api_issue_minute'
        return record
    section = re.split(r'##\s*Summary\s*:', body, maxsplit=1, flags=re.I)
    if len(section) != 2:
        record['reason'] = 'missing_summary'
        return record
    summary = re.split(r'##\s*Notes\s*:', section[1], maxsplit=1, flags=re.I)[0]
    if re.search(r'\bensemble\b', body[:body.find('## Summary:')] + summary, re.I):
        record['reason'] = 'ensemble_excluded'
        return record
    summary = re.split(r'Links? to (?:the )?(?:movies|simulations|modeled|modelled)', summary,
                       maxsplit=1, flags=re.I)[0]
    flat = re.sub(r'\s+', ' ', summary).strip()
    record['previous_ids'] = sorted(set(re.findall(MESSAGE_ID, flat)))
    all_ids = sorted(set(re.findall(CME_ID, flat)))
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', summary)]
    positive = [(p, m) for p in paragraphs for m in ARRIVAL.finditer(p)]
    negative = [p for p in paragraphs if NEGATIVE.search(p)]
    if positive and negative:
        record['reason'] = 'conflicting_earth_actions'
        return record
    if negative:
        earth_text = ' '.join(negative)
        action, arrival, glancing = 'retract', None, -1
    elif positive:
        arrival_values = {m.group('arrival') for _, m in positive}
        if len(arrival_values) != 1:
            record['reason'] = 'multiple_earth_arrivals_ambiguous'
            record['earth_text'] = ' '.join(p for p, _ in positive)
            return record
        earth_text = ' '.join(dict.fromkeys(p for p, _ in positive))
        action, arrival = 'forecast', pd.Timestamp(next(iter(arrival_values)))
        if arrival <= issue:
            record['reason'] = 'arrival_not_after_public_issue'
            record['earth_text'] = earth_text
            return record
        geometries = set()
        for paragraph, match in positive:
            prefix = paragraph[:match.start()]
            # The nearest geometry must directly qualify the CME reaching Earth.
            direct = re.search(r'(leading edge|flank)\s+of\s+(?:the\s+)?CMEs?\s*$', prefix, re.I)
            if direct:
                geometries.add(0 if direct.group(1).lower() == 'leading edge' else 1)
            if re.search(rf'glancing blow\s+(?:at|to)\s+{EARTH}', paragraph, re.I):
                geometries.add(1)
        glancing = next(iter(geometries)) if len(geometries) == 1 else -1
    else:
        record['reason'] = 'no_explicit_deterministic_earth_arrival_or_retraction'
        return record
    record['earth_text'] = earth_text
    earth_ids = sorted(set(re.findall(CME_ID, earth_text)))
    if earth_ids:
        event_ids = earth_ids
    elif len(all_ids) == 1:
        event_ids = all_ids
    elif len(all_ids) > 1 and re.search(r'\bCMEs\b', earth_text):
        # The publication explicitly describes the listed CMEs as a joint Earth forecast.
        event_ids = all_ids
    elif len(all_ids) > 1:
        dates = set(re.findall(TIME, earth_text))
        event_ids = [identifier for identifier in all_ids
                     if identifier.split('-CME-')[0][:16] + 'Z' in dates]
        if len(event_ids) != 1:
            record['reason'] = 'ambiguous_earth_event_identity'
            return record
    else:
        # A previous-message-only relation is resolved chronologically by parse_corpus.
        event_ids = []
        if not record['previous_ids']:
            record['reason'] = 'missing_earth_event_identity'
            return record
    record.update(status='accepted', reason='explicit_public_earth_' + action,
                  event_ids=event_ids, action=action,
                  arrival=None if arrival is None else arrival.isoformat(), glancing=glancing)
    return record


def parse_corpus(messages):
    parsed = [parse_message(m) for m in messages]
    parsed.sort(key=lambda r: (r['body_issue'] or '', r['message_id']))
    earlier = {}
    for record in parsed:
        if record['status'] != 'accepted':
            continue
        if not record['event_ids']:
            previous = [earlier[k] for k in record['previous_ids'] if k in earlier]
            groups = {tuple(r['event_ids']) for r in previous}
            if len(groups) != 1 or len(previous) != len(record['previous_ids']):
                record.update(status='rejected', reason='ambiguous_or_unavailable_previous_identity')
                continue
            record['event_ids'] = list(next(iter(groups)))
        earlier[record['message_id']] = record
    return parsed


def kinematics(body, event_ids):
    """Fastest published CME among the Earth-forecast events of one notification, or None."""
    parts = re.split(r'##\s*Summary\s*:', body, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None
    summary = re.split(r'##\s*Notes\s*:', parts[1], maxsplit=1, flags=re.I)[0]
    blocks = [dict(speed=float(a), half_angle=float(b), lon=float(c), lat=float(d), cme=e)
              for a, b, c, d, e in KINEMATICS.findall(summary)]
    if (not blocks or len(blocks) != len(re.findall(r'Estimated speed', summary))
            or len({b['cme'] for b in blocks}) != len(blocks)):
        return None
    chosen = [b for b in blocks if b['cme'] in set(event_ids)] or blocks
    return max(chosen, key=lambda b: (b['speed'], b['cme']))


def features_at_origins(messages, origins):
    """T and K at each sorted unique origin; only messages issued at or before the origin count."""
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('origins must be sorted and unique')
    bodies = {m['messageID']: m.get('messageBody', '') for m in messages}
    records = [r for r in parse_corpus(messages) if r['status'] == 'accepted']
    records.sort(key=lambda r: (pd.Timestamp(r['body_issue']), r['message_id']))
    speed = {r['message_id']: kinematics(bodies[r['message_id']], r['event_ids'])
             for r in records if r['action'] == 'forecast'}
    out = np.full((len(origins), len(FEATURES)), MISSING, np.float32)
    out[:, 2] = -1.
    active, position = {}, 0
    for i, origin in enumerate(origins):
        while position < len(records) and pd.Timestamp(records[position]['body_issue']) <= origin:
            new = records[position]
            new_ids = set(new['event_ids'])
            for key, old in list(active.items()):
                if new_ids.intersection(old['event_ids']):
                    del active[key]       # an update or retraction of the same CME replaces the old forecast
            if new['action'] == 'forecast':
                active[new['message_id']] = new
            position += 1
        for key, old in list(active.items()):
            if pd.Timestamp(old['body_issue']) < origin - pd.Timedelta(days=7):
                del active[key]
        values = [((pd.Timestamp(r['arrival']) - origin) / pd.Timedelta(hours=1), r) for r in active.values()]
        pending = sorted((x for x in values if x[0] >= -6), key=lambda x: (x[0], x[1]['message_id']))
        past = sorted((x for x in values if -96 <= x[0] < -6), key=lambda x: (-x[0], x[1]['message_id']))
        if pending:
            hours, record = pending[0]
            out[i, 0], out[i, 2] = hours, record['glancing']
            cme = speed[record['message_id']]
            if cme:
                out[i, 3:7] = cme['speed'], cme['half_angle'], cme['lon'], cme['lat']
        if past:
            hours, record = past[0]
            out[i, 1] = -hours
            cme = speed[record['message_id']]
            if cme:
                out[i, 7] = cme['speed']
    frame = pd.DataFrame(out, columns=FEATURES)
    frame.insert(0, 'origin', origins)
    return frame


def augment(source: Path, corpus: Path, destination: Path) -> dict:
    """Copy source (df.parquet) row group by row group and append the T and K columns per origin.

    Since 2026-11-25 df.parquet already carries these columns (data-v2 notebook section 6b): they are recomputed from
    the corpus, must equal the stored values exactly, and the source is then copied unchanged."""
    messages = json.loads(Path(corpus).read_text())
    parquet = pq.ParquetFile(source)
    names = parquet.schema_arrow.names
    present = set(FEATURES).intersection(names)
    if present and present != set(FEATURES):
        raise ValueError('source table has only some of the DONKI feature columns')
    origins = pd.DatetimeIndex(pq.read_table(source, columns=['origin'])['origin'].to_pandas()).unique().sort_values()
    features = features_at_origins(messages, origins)
    index = pd.DatetimeIndex(features.origin)
    if present:
        stored = pq.read_table(source, columns=['origin', *FEATURES]).to_pandas()
        position = index.get_indexer(pd.DatetimeIndex(stored.origin))
        for name in FEATURES:
            if not np.array_equal(stored[name].to_numpy(np.float32), features[name].to_numpy(np.float32)[position]):
                raise ValueError(f'stored {name} differs from the corpus replay')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return dict(corpus=str(corpus), messages=len(messages), origins=len(origins), features=FEATURES,
                    source_columns_verified=True,
                    next_forecast_active=float(features[FEATURES[0]].ne(MISSING).mean()),
                    next_kinematics_active=float(features[K_FEATURES[0]].ne(MISSING).mean()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for i in range(parquet.num_row_groups):
        table = parquet.read_row_group(i)
        position = index.get_indexer(pd.DatetimeIndex(table['origin'].to_pandas()))
        if (position < 0).any():
            raise ValueError('origin without features')
        for name in FEATURES:
            table = table.append_column(name, pa.array(features[name].to_numpy(np.float32)[position], type=pa.float32()))
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression='zstd')
        writer.write_table(table, row_group_size=table.num_rows)
    writer.close()
    active = features[FEATURES[0]].ne(MISSING)
    return dict(corpus=str(corpus), messages=len(messages), origins=len(origins), features=FEATURES,
                next_forecast_active=float(active.mean()),
                next_kinematics_active=float(features[K_FEATURES[0]].ne(MISSING).mean()))
