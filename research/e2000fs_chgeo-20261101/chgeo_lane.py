"""e2000fs_chgeo_point4_144h — coronal-hole geometry axis, single change on the point4_144h recipe.

Round-7 literature note (research/notes/lit-review-round7-20261101.md §4.1) and the measured bottleneck
(§1.3/§1.5): the >=550 error is a discrimination problem inside the 500-600 km/s prediction band, and the
physically relevant covariates (coronal-hole latitude, shape, latitudinal extent) are already materialised
in `ch-hourly.parquet` but never entered the model — the frozen Top30 uses only 8 area columns.

Arms (identical data, masks, rounds, block layout, seed; the only difference is the added block):
  control : the e2000fs_calib_point4_144h recipe exactly (21 constant + 9 varying x 24 h = 237 columns)
  chgeo   : control + 22 coronal-hole geometry columns appended to the constant block

Geometry block (pre-registered, physical quantities only, no fitted constant):
  per threshold t in {t035, t055}: top1_area, top1_lat, top1_lon, top1_lat_width, top1_lon_width,
      share = top1_area/area, A_lat0, A_lat1, A_lat2 (lat-band area sums), A_asym=(A_lat2-A_lat0)/area
  plus d27_share = share(t) - share(t - 27 d) (one solar rotation, the same span as the rec* family)
  => 22 columns. The value for a forecast origin O is the newest ch-hourly slot with slot <= O-1h
     (the causal rule the SUVI strip builder uses), ffilled over hourly slots. Geometry summaries are
     rotation-invariant, so one value per origin is the non-redundant representation; the
     longitude-resolved, forecast-time-aligned information is already in `strip_v*`.

Data: data-v2/model-inputs/leaderboard-store/e2000fs-calib-point4-144h/df.parquet
      (flat (set, origin, h) design table on the F1/F2/F3 quarters with the 144 h maturity fit masks)
      ch-hourly.parquet sha256 96f38e2728...0870a

Rules kept: mask origin+144h <= cutoff per stage, point4 objective (y>=550 weight 4, per-horizon mean 1),
three 24-output vector trees, 900 rounds, depth 3, eta .03, subsample .8, colsample .7, seed 42,
clip [1, 3000]. F1, F2 and F3 are all produced and submitted whatever the internal result.

usage:
  python chgeo_lane.py --arm chgeo --stage F1 --block 0 --out DIR --threads 3
  python chgeo_lane.py --arm control --out DIR                 # all nine parts in this process
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
HERE = Path(__file__).resolve().parent
DF = HERE / 'inputs/input_df.parquet'          # pinned copy of leaderboard-store/input-df/df.parquet
CH = HERE / 'inputs/ch-hourly.parquet'          # pinned copy of ch-v1/ch-hourly.parquet
H = 72
BLOCKS = (0, 24, 48)
ROUNDS = 900
HIGH_SPEED, HIGH_WEIGHT = 550., 4.
STAGES = (('F1', 'fit_F1', 2089), ('F2', 'fit_F2', 2113), ('F3', 'fit_F3', 2113))
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6',
            'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last',
            'ch_ch_t035_top2_area', 'ch_ch_t035_meridian15_equatorial_area',
            'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area',
            'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian30_equatorial_area',
            'ch_ch_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed',
           'strip_v550_dark', 'strip_v650_dark', 'strip_v750_logmed']
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
              reg_lambda=5, objective='reg:squarederror', tree_method='hist', device='cpu', eval_metric='rmse',
              multi_strategy='multi_output_tree', seed=42, num_target=24)
COLUMNS = ['set', 'origin', 'h', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y'] + CONSTANT + VARYING


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry() -> pd.DataFrame:
    """Causal per-slot coronal-hole geometry block (22 columns)."""
    ch = pd.read_parquet(CH)
    ch.index = pd.DatetimeIndex(ch.index).tz_convert('UTC')
    ch = ch.sort_index()
    out = pd.DataFrame(index=ch.index)
    for t in ('t035', 't055'):
        area = ch[f'ch_{t}_area']
        share = ch[f'ch_{t}_top1_area'] / area.where(area > 0)
        bands = {j: ch[[f'ch_{t}_lon{i:02d}_lat{j}_area' for i in range(12)]].sum(axis=1) for j in range(3)}
        out[f'g_{t}_top1_area'] = ch[f'ch_{t}_top1_area']
        out[f'g_{t}_top1_lat'] = ch[f'ch_{t}_top1_lat']
        out[f'g_{t}_top1_lon'] = ch[f'ch_{t}_top1_lon']
        out[f'g_{t}_top1_lat_width'] = ch[f'ch_{t}_top1_lat_width']
        out[f'g_{t}_top1_lon_width'] = ch[f'ch_{t}_top1_lon_width']
        out[f'g_{t}_share'] = share
        for j in range(3):
            out[f'g_{t}_alat{j}'] = bands[j]
        out[f'g_{t}_asym'] = (bands[2] - bands[0]) / area.where(area > 0)
        out[f'g_{t}_d27_share'] = share - share.shift(27 * 24)
    if out.shape[1] != 22:
        raise AssertionError(f'geometry block must be 22 columns, got {out.shape[1]}')
    return out.astype(np.float32)


def at_origins(geom: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    """Newest slot <= origin - 1 h, ffilled over hourly slots (no future observation enters)."""
    hourly = geom.reindex(pd.date_range(geom.index.min(), geom.index.max(), freq='h', tz='UTC')).ffill()
    return hourly.reindex((origins - pd.Timedelta(hours=1)).floor('h')).to_numpy(dtype=np.float32)


def cube(frame: pd.DataFrame, origins: pd.DatetimeIndex):
    """(constants, varying[origin, h, feature], y[origin, h]) indexed by `origins` order."""
    pos = pd.Series(np.arange(len(origins), dtype=np.int64), index=origins)
    ri = pos.reindex(frame['origin']).to_numpy()
    hi = frame['h'].to_numpy().astype(np.int64) - 1
    if np.isnan(ri).any() or hi.min() < 0 or hi.max() >= H:
        raise ValueError('cube: origin or horizon outside the requested grid')
    ri = ri.astype(np.int64)
    const = np.empty((len(origins), len(CONSTANT)), dtype=np.float32)
    const[ri] = frame[CONSTANT].to_numpy(dtype=np.float32)
    vary = np.empty((len(origins), H, len(VARYING)), dtype=np.float32)
    vary[ri, hi] = frame[VARYING].to_numpy(dtype=np.float32)
    y = np.empty((len(origins), H), dtype=np.float32)
    y[ri, hi] = frame['y'].to_numpy(dtype=np.float32)
    return const, vary, y


def design(const, vary, extra, start):
    block = vary[:, start:start + 24, :].reshape(len(const), -1)
    parts = [const] if extra is None else [const, extra]
    return np.hstack(parts + [block]).astype(np.float32)


def weighted_l2(pred, matrix):
    target = matrix.get_label().reshape(pred.shape)
    w = np.where(target >= HIGH_SPEED, HIGH_WEIGHT, 1.)
    w = (w / w.mean(axis=0)).astype(np.float32)
    return (pred - target) * w, w


def stage_frames(df, stage, mask):
    train = df[(df['set'] == 'train') & df[mask]]
    query = df[(df['set'] == 'query') & (df['period'] == stage)]
    return (train, pd.DatetimeIndex(sorted(train['origin'].unique())),
            query, pd.DatetimeIndex(sorted(query['origin'].unique())))


def run_part(arm, stage, mask, n_query, start_h, df, geom, rounds, threads, out, tag=''):
    t0 = time.time()
    train, fit_origins, query, query_origins = stage_frames(df, stage, mask)
    if len(query_origins) != n_query:
        raise ValueError(f'{stage}: {len(query_origins)} query origins, expected {n_query}')
    c_tr, e_tr, y_tr = cube(train, fit_origins)
    c_qu, e_qu, y_qu = cube(query, query_origins)
    g_tr = at_origins(geom, fit_origins) if geom is not None else None
    g_qu = at_origins(geom, query_origins) if geom is not None else None
    x_tr = design(c_tr, e_tr, g_tr, start_h)
    x_qu = design(c_qu, e_qu, g_qu, start_h)
    block_y = y_tr[:, start_h:start_h + 24]
    dtrain = xgb.DMatrix(x_tr, label=block_y, nthread=threads)
    settings = dict(PARAMS, base_score=float(block_y.mean()), nthread=threads)
    model = xgb.train(settings, dtrain, num_boost_round=rounds, obj=weighted_l2, verbose_eval=False)
    pred = np.clip(model.inplace_predict(x_qu).astype(np.float64), 1., 3000.)
    b = start_h // 24
    model.save_model(out / f'model_{arm}_{stage}_b{b}{tag}.ubj')
    np.save(out / f'part_{arm}_{stage}_b{b}{tag}.npy', pred)
    slice_y = y_qu[:, start_h:start_h + 24]
    err = (pred - slice_y) ** 2
    tail = slice_y >= 550
    record = dict(arm=arm, stage=stage, block=b, hours=[start_h + 1, start_h + 24], rounds=rounds,
                  threads=threads, n_fit_origins=int(len(fit_origins)), n_query=int(len(query_origins)),
                  n_columns=int(x_tr.shape[1]), mse=float(err.mean()),
                  mse_ge550=float(err[tail].mean()) if tail.any() else float('nan'), n_ge550=int(tail.sum()),
                  seconds=round(time.time() - t0, 1))
    (out / f'part_{arm}_{stage}_b{b}{tag}.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('control', 'chgeo'), required=True)
    ap.add_argument('--stage', choices=[s for s, _, _ in STAGES])
    ap.add_argument('--block', type=int, choices=(0, 1, 2))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DF, columns=COLUMNS)
    if pd.DatetimeIndex(df['origin']).tz is None:
        df['origin'] = pd.DatetimeIndex(df['origin'], tz='UTC')
    geom = geometry() if a.arm == 'chgeo' else None
    rounds = 2 if a.smoke else a.rounds

    todo = [(s, m, n) for s, m, n in STAGES if a.stage in (None, s)]
    for stage, mask, n_query in todo:
        if a.smoke:
            n_query = None
        for b in ([a.block] if a.block is not None else (0, 1, 2)):
            if a.smoke:
                run_part(arm=a.arm, stage=stage, mask=mask, n_query=len(pd.DatetimeIndex(sorted(
                    df[(df['set'] == 'query') & (df['period'] == stage)]['origin'].unique()))),
                    start_h=BLOCKS[b], df=df, geom=geom, rounds=rounds, threads=a.threads, out=a.out,
                    tag=a.tag or '_smoke')
            else:
                run_part(a.arm, stage, mask, n_query, BLOCKS[b], df, geom, rounds, a.threads, a.out, tag=a.tag)
    (a.out / f'receipt_{a.arm}{a.tag}.json').write_text(json.dumps(dict(
        arm=a.arm, rounds=rounds, threads=a.threads,
        input_sha256={'df': sha256(DF), 'ch_hourly': sha256(CH) if a.arm == 'chgeo' else None},
        n_columns={'control': 237, 'chgeo': 259}), indent=1) + '\n')


if __name__ == '__main__':
    main()
