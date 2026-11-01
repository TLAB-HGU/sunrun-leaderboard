"""Raw-data chain for the two inference walkthroughs.

Everything here is read from the frozen raw sources and the model-input bundle, so the pages can
start at ACE/SUVI/coronal-hole/ENLIL instead of at "a forecast.npz already exists":

  raw hourly ACE snapshot   ch-breakthrough-v2/upload/raw-ace-snapshot/year=*.parquet
  SUVI hourly stage         data-v2/stage/suvi_hourly/year=*.parquet  (+ suvi-fusion patches)
  coronal-hole hourly       ch-v1/ch-hourly.parquet
  WSA-Enlil run JSON        donki-live/ + headroom-audit/donki/
  target series             ch-v1/ace.parquet  (filled_speed_kms, was_missing)
  model-input bundle        analysis/A0014/outputs/cache/{train,f1,official}.npz

The longitude-strip numbers are recomputed with the frozen arithmetic in
leaderboard/model-release-publish/model_release/strips.py and then checked against the bundle's
own strip columns, so the page can show raw pixel statistics turning into one model input.

Output: docs/explainers/data/raw_chain.json
Run:    /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_raw_chain.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
OUT = ROOT / 'docs/explainers/data'
STORE = (ROOT / 'data-v2/raw-store/v2/solar-pivot-noanalog72-20261015/pinned-v1/repo/'
         'leaderboard/store')
CACHE = ROOT / 'analysis/A0014/outputs/cache'
H = 72

_spec = importlib.util.spec_from_file_location(
    'strips', ROOT / 'leaderboard/model-release-publish/model_release/strips.py')
strips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strips)

FEATURE_JSON = ROOT / 'leaderboard/experiments/cumulative_lane12_20261024/materialized_v3/features_v3.json'
EXAMPLES = {'detpool': pd.Timestamp('2026-03-30T13:00:00Z'),
            'mixture': pd.Timestamp('2026-08-18T23:00:00Z')}


def sanitize(x):
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, (np.floating, float)) and not np.isfinite(float(x)):
        return None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return sanitize(x.tolist())
    return x


def load_sources():
    raw = []
    for p in sorted(glob.glob(str(STORE / 'ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/*.parquet'))):
        raw.append(pd.read_parquet(p))
    ace_raw = pd.concat(raw, ignore_index=True)
    ace_raw['t'] = pd.to_datetime(ace_raw['timestamp_utc'], utc=True)
    ace_raw = ace_raw.set_index('t').sort_index()

    target = pd.read_parquet(STORE / 'ch-v1/ace.parquet')
    target['t'] = pd.to_datetime(target['timestamp_utc'], utc=True)
    target = target.set_index('t').sort_index()

    ch = pd.read_parquet(STORE / 'ch-v1/ch-hourly.parquet').sort_index()
    suvi = pd.read_parquet(STORE / 'suvi-fusion/suvi-hourly-v2.parquet').sort_index()
    meta = pd.concat([pd.read_parquet(p) for p in
                      sorted(glob.glob(str(ROOT / 'data-v2/stage/suvi_hourly/year=*/*.parquet')))],
                     ignore_index=True)
    meta['slot'] = pd.to_datetime(meta['slot'], utc=True)
    meta['obs_end'] = pd.to_datetime(meta['obs_end'], utc=True)
    meta['s3_modified'] = pd.to_datetime(meta['s3_modified'], utc=True)
    meta = meta[meta['status'] == 'ok'].sort_values('slot').reset_index(drop=True)
    return ace_raw, target, ch, suvi, meta


def sources_table(ace_raw, target, ch, suvi, meta):
    def span(df):
        return str(df.index[0]), str(df.index[-1])

    enlil_hist = sorted((STORE / 'headroom-audit/donki').glob('WSAEnlilSimulations_*.json'))
    enlil_live = sorted((STORE / 'donki-live').glob('WSAEnlilSimulations_*.json'))
    a0, a1 = span(ace_raw)
    t0, t1 = span(target)
    c0, c1 = span(ch)
    s0, s1 = span(suvi)
    return [
        {'id': 'ace', 'title': 'ACE 원자료 (L1 in-situ)', 'cadence': '1시간',
         'rows': int(len(ace_raw)), 'first': a0, 'last': a1, 'cols': int(ace_raw.shape[1] - 1),
         'path': 'ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet',
         'makes': 'ace_*, v_lag*, v_m*, v_sd*, speed_d24, epam_p1_d6, rec26/27/28',
         'note': '플라스마(속도·밀도·온도)·자기장·EPAM·SIS. 79.8% 시간대에 속도 관측이 있음(나머지는 결측).'},
        {'id': 'suvi', 'title': 'SUVI EUV 영상 (GOES-18)', 'cadence': '1시간 슬롯',
         'rows': int(len(suvi)), 'first': s0, 'last': s1, 'cols': int(suvi.shape[1]),
         'path': 'stage/suvi_hourly/year=*.parquet + suvi-fusion/suvi-hourly-v2.parquet',
         'makes': 'strip_v350…750 × {logmed, dark, cover}, img_age_h, img_stale',
         'note': '디스크를 경도 12칸 × 위도 3밴드로 나눈 패치 통계. 원본 netCDF는 data-v2/suvi.'},
        {'id': 'ch', 'title': '코로나홀 지도', 'cadence': '1시간',
         'rows': int(len(ch)), 'first': c0, 'last': c1, 'cols': int(ch.shape[1]),
         'path': 'ch-v1/ch-hourly.parquet',
         'makes': 'ch_ch_t035/t045/t055_* (면적·최대·상위3 홀·자오선·적도)',
         'note': '임계 0.35/0.45/0.55 세 세트. valid_fraction 평균 0.999로 거의 전 시간 존재.'},
        {'id': 'enlil', 'title': 'WSA-Enlil 모델 실행', 'cadence': '월별 실행 파일',
         'rows': len(enlil_hist) + len(enlil_live), 'first': enlil_hist[0].stem[-7:] if enlil_hist else None,
         'last': enlil_live[-1].stem[-7:] if enlil_live else None, 'cols': 5,
         'path': 'donki-live/ + headroom-audit/donki/WSAEnlilSimulations_*.json',
         'makes': 'enlil_next_arr_h, enlil_since_arr_h, enlil_n_pending, enlil_kp, enlil_glancing, enlil_dt_target',
         'note': f'{len(enlil_hist)}개 과거 + {len(enlil_live)}개 라이브 실행 파일(DONKI JSON).'},
        {'id': 'target', 'title': '타깃 시계열 (ACE 속도)', 'cadence': '1시간',
         'rows': int(len(target)), 'first': t0, 'last': t1, 'cols': 2,
         'path': 'ch-v1/ace.parquet',
         'makes': 'y = filled_speed_kms (평가 정답)',
         'note': '관측 결측은 ffill로 채우고 was_missing 플래그로 남김.'},
        {'id': 'meta', 'title': 'SUVI 슬롯 메타데이터', 'cadence': '1시간',
         'rows': int(len(meta)), 'first': str(meta['slot'].min()), 'last': str(meta['slot'].max()),
         'cols': int(meta.shape[1]),
         'path': 'data-v2/stage/suvi_hourly/year=*.parquet (status/obs_end/s3_modified)',
         'makes': 'img_age_h, img_stale, 슬롯 선택',
         'note': 'status=ok 인 슬롯만 후보. origin−1h 이전, S3 수정시각 ≤ origin 인 최신 슬롯을 고름.'},
    ]


def families():
    feats = json.loads(FEATURE_JSON.read_text())['features']
    def n(pref):
        return len([f for f in feats if f.startswith(pref)])
    rows = [
        ('ace_*', 'ACE 플라스마·자기장·EPAM·SIS (원자료)', n('ace_')),
        ('v_lag* / v_m* / v_sd*', 'ACE 속도 이력(지연·이동평균·표준편차)', len([f for f in feats if f.startswith('v_')])),
        ('rec26/27/28', '27일 회귀: 목표 시각 −26/27/28일의 ACE 속도', n('rec')),
        ('ch_ch_*', '코로나홀 면적·위치', n('ch_')),
        ('strip_*', 'SUVI 경도 스트립(속도별 도달 경도)', n('strip_')),
        ('img_*', '이미지 신선도(age, stale)', n('img_')),
        ('enlil_*', 'WSA-Enlil 실행 상태', n('enlil_')),
        ('speed_d24 / epam_p1_d6', '속도·EPAM 변화율', 2),
        ('h', '수평선 인덱스', 1),
    ]
    return [{'prefix': a, 'meaning': b, 'n': c} for a, b, c in rows]


def strip_table():
    lam = []
    for hh in (1, 12, 24, 36, 48, 60, 72):
        lam.append({'h': hh, **{f'v{v}': round(float(strips.lam(hh, v, 0.0)), 2) for v in strips.SPEEDS}})
    return {
        'omega_deg_h': strips.OMEGA, 'distance_km': strips.D_KM,
        'speeds': list(strips.SPEEDS),
        'tau_h': {str(v): round(float(t), 2) for v, t in strips.TAU.items()},
        'strip_half_deg': strips.STRIP_HALF, 'cell_w_deg': strips.CELL_W,
        'stale_h': strips.STALE_H, 'cover_min': strips.COVER_MIN,
        'cell_centers': [round(float(c), 1) for c in strips._cell_centers()],
        'lam_by_h': lam,
    }


def pick_slot(origin, meta, suvi):
    """Newest ok slot at <= origin-1h with finite lat1 patches (the builder's rule)."""
    lat1 = [c for c in suvi.columns if c.startswith('lon') and c.split('_')[1] == 'lat1']
    j = meta.index[meta['slot'] <= origin - pd.Timedelta(hours=1)]
    for k in reversed(j):
        slot = meta.loc[k, 'slot']
        if slot not in suvi.index:
            continue
        if suvi.loc[slot, lat1].notna().all():
            return slot, meta.loc[k, 'obs_end']
    return None, None


def example(key: str, origin: pd.Timestamp, ace_raw, target, ch, suvi, meta) -> dict:
    names = json.loads((CACHE / 'names.json').read_text())
    tr = dict(np.load(CACHE / 'train.npz'))
    f1 = dict(np.load(CACHE / 'f1.npz'))
    off = dict(np.load(CACHE / 'official.npz'))
    o_f1 = pd.DatetimeIndex(f1['origins'], tz='UTC')
    o_off = pd.DatetimeIndex(off['origins'], tz='UTC')
    if origin <= o_f1[-1]:
        bundle, oo = f1, o_f1
    else:
        bundle, oo = off, o_off
    i = int(oo.searchsorted(origin))
    C = np.asarray(bundle['C'][i], dtype=float)
    E = np.asarray(bundle['E'][i], dtype=float)
    const = names['constant']
    vary = names['varying']
    cval = {n: float(v) for n, v in zip(const, C)}
    vval = {n: [float(x) for x in E[:, j]] for j, n in enumerate(vary)}

    ACE_COLS = ('ace_speed_kms', 'ace_density_cm3', 'ace_temperature_k', 'ace_bt_nt',
                'ace_bz_gsm_nt', 'ace_by_gsm_nt', 'ace_epam_p_channel3', 'ace_sis_gt10_flux')

    def at(ts):
        if ts not in ace_raw.index:
            return {c: None for c in ACE_COLS}
        r = ace_raw.loc[ts]
        return {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in ACE_COLS}

    def win(n):
        """n hourly rows ending at the origin: the production _m6/_m24 definition."""
        sl = ace_raw.loc[origin - pd.Timedelta(hours=n - 1):origin]
        return {c: (None if sl[c].dropna().empty else float(sl[c].mean())) for c in ACE_COLS} | \
               {'rows': int(len(sl)), 'obs': int(sl['ace_speed_kms'].notna().sum())}

    raw_ace = {'t0': at(origin), 'm6': win(6), 'm24': win(24)}
    raw_ace['win6_values'] = [None if v != v else round(float(v), 2) for v in
                              ace_raw.loc[origin - pd.Timedelta(hours=5):origin, 'ace_speed_kms']]
    raw_ace['win24_values'] = [None if v != v else round(float(v), 2) for v in
                               ace_raw.loc[origin - pd.Timedelta(hours=23):origin, 'ace_speed_kms']]

    # target series around the origin
    def tgt(ts):
        if ts not in target.index:
            return None
        r = target.loc[ts]
        return {'t': str(ts), 'speed': float(r['filled_speed_kms']),
                'was_missing': bool(r['was_missing'])}

    slot, obs_end = pick_slot(origin, meta, suvi)
    age = float((origin - obs_end) / pd.Timedelta(hours=1)) if obs_end is not None else None
    lat1 = {suf: [c for c in suvi.columns if c.startswith('lon') and c.split('_')[1] == 'lat1'
                  and c.endswith(suf)]
            for suf in ('log_median', 'dark_fraction', 'valid_fraction')}
    lat1 = {k: sorted(v, key=lambda c: int(c[3:5])) for k, v in lat1.items()}
    patches = {k: [round(float(v), 4) for v in suvi.loc[slot, cols]] for k, cols in lat1.items()} \
        if slot is not None else {}

    # ---- strip arithmetic, recomputed from the raw patches
    edges = -90.0 + strips.CELL_W * np.arange(13)
    calc = []
    for hh in (1, 24, 48, 72):
        row = {'h': hh}
        for v in strips.SPEEDS:
            lam_c = float(strips.lam(hh, v, age or 0.0))
            w = np.clip(np.minimum(lam_c + strips.STRIP_HALF, edges[1:])
                        - np.maximum(lam_c - strips.STRIP_HALF, edges[:-1]), 0, None) / (2 * strips.STRIP_HALF)
            ws = float(w.sum())
            lm = float((w * np.array(patches['log_median'])).sum() / max(ws, 1e-12)) if ws else None
            dk = float((w * np.array(patches['dark_fraction'])).sum() / max(ws, 1e-12)) if ws else None
            cv = float((w * np.array(patches['valid_fraction'])).sum() / max(ws, 1e-12)) if ws else None
            if cv is not None and cv < strips.COVER_MIN:   # builder keeps NaN when coverage is thin
                lm = dk = None
            row[f'v{v}'] = {'lam': round(lam_c, 2), 'logmed': lm, 'dark': dk, 'cover': cv,
                            'w': [round(float(x), 4) for x in w]}
        calc.append(row)
    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) < 1e-3
    match = {f'v{v}': [same(calc[j][f'v{v}']['logmed'], vval[f'strip_v{v}_logmed'][hh - 1])
                       for j, hh in enumerate((1, 24, 48, 72))] for v in strips.SPEEDS}
    diff = {f'v{v}': max(abs(calc[j][f'v{v}']['logmed'] - vval[f'strip_v{v}_logmed'][hh - 1])
                          for j, hh in enumerate((1, 24, 48, 72))
                          if calc[j][f'v{v}']['logmed'] is not None
                          and np.isfinite(vval[f'strip_v{v}_logmed'][hh - 1]))
            for v in strips.SPEEDS}

    rec = []
    for hh in (1, 12, 24, 48, 72):
        t = origin + pd.Timedelta(hours=hh)
        rec.append({'h': hh, 'target': str(t),
                    **{f'rec{d}': (float(target.loc[t - pd.Timedelta(days=d), 'filled_speed_kms'])
                                   if (t - pd.Timedelta(days=d)) in target.index else None)
                       for d in (26, 27, 28)}})
    return {
        'origin': str(origin), 'bundle': 'f1' if origin <= o_f1[-1] else 'official',
        'raw_ace': raw_ace,
        'map_checks': {
            'speed_last': [raw_ace['t0']['ace_speed_kms'], cval.get('ace_speed_kms_last')],
            'speed_m6': [raw_ace['m6']['ace_speed_kms'], cval.get('ace_speed_kms_m6')],
            'speed_m24': [raw_ace['m24']['ace_speed_kms'], cval.get('ace_speed_kms_m24')],
            'density_last': [raw_ace['t0']['ace_density_cm3'], cval.get('ace_density_cm3_last')],
            'temp_last': [raw_ace['t0']['ace_temperature_k'], cval.get('ace_temperature_k_last')],
            'bt_last': [raw_ace['t0']['ace_bt_nt'], cval.get('ace_bt_nt_last')],
            'by_m24': [raw_ace['m24']['ace_by_gsm_nt'], cval.get('ace_by_gsm_nt_m24')],
            'epam_last': [raw_ace['t0']['ace_epam_p_channel3'], cval.get('ace_epam_p_channel3_last')],
            'sis_last': [raw_ace['t0']['ace_sis_gt10_flux'], cval.get('ace_sis_gt10_flux_last')],
            'epam_log': [float(np.log10(max(raw_ace['t0']['ace_epam_p_channel3'] or 1e-30, 1e-30))),
                         cval.get('ace_epam_p_channel3_last')],
            'sis_log': [float(np.log10(max(raw_ace['t0']['ace_sis_gt10_flux'] or 1e-30, 1e-30))),
                        cval.get('ace_sis_gt10_flux_last')],
            'rec27_h1': [rec[0]['rec27'], vval['rec27'][0]],
        },
        'target': {'t0': tgt(origin), 't1': tgt(origin + pd.Timedelta(hours=1)),
                   't24': tgt(origin + pd.Timedelta(hours=24))},
        'recurrence': rec,
        'ch': {k: cval.get(k) for k in
               ('ch_ch_t055_area', 'ch_ch_t055_top1_area', 'ch_ch_t035_area')},
        'img': {'slot': str(slot), 'obs_end': str(obs_end), 'age_h': age,
                'stale': cval.get('img_stale')},
        'suvi_patches': patches,
        'strip_calc': calc,
        'strip_match': match, 'strip_max_diff': diff,
        'features': {
            'ace': {k: cval.get(k) for k in ('ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24',
                                             'ace_density_cm3_last', 'ace_temperature_k_last',
                                             'ace_bt_nt_last', 'ace_by_gsm_nt_m24',
                                             'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last')},
            'speed': {k: cval.get(k) for k in ('v_lag0', 'v_m6', 'v_m24', 'v_sd24')},
            'ch': {k: cval.get(k) for k in ('ch_ch_t035_area', 'ch_ch_t055_area',
                                            'ch_ch_t035_top2_area', 'ch_ch_t055_meridian30_equatorial_area')},
            'img': {k: cval.get(k) for k in ('img_age_h', 'img_stale')},
            'enlil': {k: cval.get(k) for k in ('enlil_next_arr_h', 'enlil_since_arr_h',
                                               'enlil_n_pending', 'enlil_kp')},
            'varying': {k: [vval[k][0], vval[k][23], vval[k][47], vval[k][71]] for k in
                        ('rec27', 'strip_v550_logmed', 'strip_v550_dark', 'enlil_dt_target')},
        },
        'bundle_shape': {'C': len(const), 'E': [int(bundle['E'].shape[1]), int(bundle['E'].shape[2])]},
    }


def upstream(suvi_meta_rows: int, ok_slots: int, missing_slots: int) -> dict:
    """One level above the hourly tables: instruments → files → ingest."""
    import glob as _glob

    root = ROOT / 'data-v2'
    nc = _glob.glob(str(root / 'suvi/data/**/*.nc'), recursive=True)
    nc_bytes = sum(Path(f).stat().st_size for f in nc)
    gong = _glob.glob(str(root / 'model-inputs/leaderboard-store/v2/gong-mrzqs/maps/**/*.fits.gz'),
                      recursive=True)
    gong_bytes = sum(Path(f).stat().st_size for f in gong)
    hourly_frames = _glob.glob(str(root / 'model-inputs/leaderboard-store/v2/gong-mrzqs/'
                                   'maps-hourly/**/*'), recursive=True)
    hourly_frames = [f for f in hourly_frames if Path(f).is_file()]

    enl_f, enl_runs = 0, 0
    for f in sorted(_glob.glob(str(STORE / 'headroom-audit/donki/WSAEnlilSimulations_*.json'))) + \
            sorted(_glob.glob(str(STORE / 'donki-live/WSAEnlilSimulations_*.json'))):
        try:
            d = json.loads(Path(f).read_text())
            enl_runs += len(d) if isinstance(d, list) else 1
            enl_f += 1
        except Exception:
            pass
    return {
        'instruments': [
            {'id': 'ace', 'name': 'ACE · L1', 'what': 'SWEPAM·MAG·EPAM·SIS',
             'cadence': '64 s / 16 s', 'value': 'L1 1.5×10⁶ km'},
            {'id': 'suvi', 'name': 'GOES-18 · SUVI', 'what': 'Fe195 · thin_aluminum',
             'cadence': '~1 분 영상', 'value': f'{ok_slots:,} 슬롯'},
            {'id': 'gong', 'name': 'GONG · MRZQS', 'what': '자기장 지도 (코로나홀)',
             'cadence': '시간별 지도', 'value': f'{len(gong):,} 장'},
            {'id': 'enlil', 'name': 'WSA-Enlil', 'what': '태양풍 수치모델 실행',
             'cadence': '이벤트별 실행', 'value': f'{enl_runs:,} 실행'}
        ],
        'files': [
            {'id': 'cdf', 'name': 'CDF', 'path': 'ftps://spdf.gsfc.nasa.gov/pub/data/ace/…',
             'note': 'ac_h2_mfi / ac_h2_swe Level-2', 'value': '시간당 행으로 파싱'},
            {'id': 'nc', 'name': 'netCDF', 'path': 'data-v2/suvi/data/g18/…',
             'note': f'{len(nc):,} 파일 · {nc_bytes / 1e9:.0f} GB', 'value': f'{nc_bytes / 1e9:.0f} GB'},
            {'id': 'fits', 'name': 'FITS.gz', 'path': 'gong-mrzqs/maps/…',
             'note': f'{len(gong):,} 지도 · {gong_bytes / 1e9:.1f} GB',
             'value': f'{len(hourly_frames):,} 시간프레임'},
            {'id': 'json', 'name': 'DONKI JSON', 'path': 'donki-live/WSAEnlilSimulations_*.json',
             'note': f'{enl_f} 파일 · {enl_runs:,} 실행', 'value': 'kp_18/90/135/180'}
        ],
        'ingest': [
            {'id': 'collect', 'name': 'collector', 'note': 'builders/raw_sources.py',
             'value': '다운로드 + sha256'},
            {'id': 'db', 'name': 'RawDB', 'note': 'raw-store/solar_raw.duckdb',
             'value': 'MANIFEST/FILES'},
            {'id': 'hourly', 'name': 'hourly parquet', 'note': 'stage/*/year=*/part.parquet',
             'value': '2022–2026 · 5 파티션'}
        ],
        'suvi': {'files': len(nc), 'gb': round(nc_bytes / 1e9, 1),
                 'slots': int(suvi_meta_rows), 'ok': int(ok_slots), 'missing': int(missing_slots)},
        'gong': {'maps': len(gong), 'gb': round(gong_bytes / 1e9, 2), 'hourly': len(hourly_frames)},
        'enlil': {'files': enl_f, 'runs': enl_runs},
    }


def main():
    ace_raw, target, ch, suvi, meta = load_sources()
    names_count = {}
    tr = dict(np.load(CACHE / 'train.npz'))
    n_origins = len(tr['origins'])
    c_cells = int(np.prod(tr['C'].shape))
    e_cells = int(np.prod(tr['E'].shape))
    monthly = target.copy()
    monthly['m'] = monthly.index.to_period('M').astype(str)
    miss = monthly.groupby('m')['was_missing'].agg(['mean', 'size'])
    ok_slots = int((pd.concat([pd.read_parquet(p) for p in
                               sorted(glob.glob(str(ROOT / 'data-v2/stage/suvi_hourly/year=*/*.parquet')))],
                              ignore_index=True)['status'] == 'ok').sum())
    meta_all = pd.concat([pd.read_parquet(p) for p in
                          sorted(glob.glob(str(ROOT / 'data-v2/stage/suvi_hourly/year=*/*.parquet')))],
                         ignore_index=True)
    out = {
        'upstream': upstream(len(meta_all), ok_slots, len(meta_all) - ok_slots),
        'missing_by_month': [{'month': m, 'pct': float(r['mean'] * 100), 'hours': int(r['size'])}
                             for m, r in miss.iterrows()],
        'sources': sources_table(ace_raw, target, ch, suvi, meta),
        'families': families(),
        'grid': {
            'hourly_rows': int(len(suvi)),
            'first': str(suvi.index[0]), 'last': str(suvi.index[-1]),
            'target_rows': int(len(target)),
            'was_missing_pct': float(target['was_missing'].mean() * 100),
            'missing_hours': int(target['was_missing'].sum()),
            'target_first': str(target.index[0]), 'target_last': str(target.index[-1]),
            'ace_speed_null_pct': float(ace_raw['ace_speed_kms'].isna().mean() * 100),
        },
        'bundle': {
            'train_origins': int(n_origins), 'f1_origins': int(len(np.load(CACHE / 'f1.npz')['origins'])),
            'official_origins': int(len(np.load(CACHE / 'official.npz')['origins'])),
            'C_shape': [int(tr['C'].shape[0]), int(tr['C'].shape[1])],
            'E_shape': [int(tr['E'].shape[0]), int(tr['E'].shape[1]), int(tr['E'].shape[2])],
            'past_shape': [int(tr['past'].shape[0]), int(tr['past'].shape[1])],
            'c_nan': int(np.isnan(tr['C']).sum()), 'e_nan': int(np.isnan(tr['E']).sum()),
            'c_cells': c_cells, 'e_cells': e_cells,
            'prefill': 'ffill along origin',
        },
        'strip': strip_table(),
        'examples': {k: example(k, t, ace_raw, target, ch, suvi, meta)
                     for k, t in EXAMPLES.items()},
    }
    names = json.loads((CACHE / 'names.json').read_text())
    off = dict(np.load(CACHE / 'official.npz'))
    Coff = np.asarray(off['C'], dtype=float)
    is_ = names['constant'].index('img_stale')
    ia = names['constant'].index('img_age_h')
    out['image_quality'] = {
        'official_origins': int(len(Coff)),
        'stale_pct': float(np.nanmean(Coff[:, is_]) * 100),
        'age_p50_h': float(np.nanmedian(Coff[:, ia])),
        'age_p90_h': float(np.nanpercentile(Coff[:, ia], 90)),
        'age_over_6h_pct': float(np.nanmean(Coff[:, ia] > 6) * 100),
        'age_max_h': float(np.nanmax(Coff[:, ia])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'raw_chain.json').write_text(json.dumps(sanitize(out), indent=1) + '\n')
    for k, e in out['examples'].items():
        print(k, e['origin'], 'slot', e['img']['slot'], 'age', round(e['img']['age_h'] or -1, 1),
              'strip match v550', e['strip_match']['v550'])
    print('->', OUT / 'raw_chain.json')


if __name__ == '__main__':
    main()
