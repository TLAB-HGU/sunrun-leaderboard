"""Data for the image -> 30 features -> model walkthrough (e2000fs_calib4_detpool_gpd_flow.html).

One origin (2026-03-30 13:00 UTC, F1) is followed end to end with real files:

  SUVI Fe195 .nc  ->  12x3 patch stats  ->  longitude strips (5 of the 30)
                  ->  coronal-hole masks ->  7 of the 30
                  ->  image age           ->  1 of the 30
  ACE / recurrence / WSA-Enlil            ->  the other 17
  30 features -> 3 blocks x 237 columns -> detector / tweedie body / mixture tail / gpd arm
  -> detroute -> (1-w) detroute + w gpd

Everything is recomputed from the archived files and checked against the bundle that the model
actually consumed; a mismatch aborts the run.

Output: docs/explainers/data/flow.json, docs/explainers/data/flow/*.{jpg,png}
Run:    /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_flow.py
"""
from __future__ import annotations

import base64
import glob
import importlib.util
import io
import json
import sys
from pathlib import Path

import contourpy
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image

ROOT = Path('/home/t-lab01/sunrun')
OUT = ROOT / 'docs/explainers/data'
IMG = OUT / 'flow'
CACHE = ROOT / 'analysis/A0014/outputs/cache'
ORIGIN = pd.Timestamp('2026-04-03T19:00:00Z')
H = 72
GRID = np.round(np.arange(0.0, 0.75 + 1e-9, 0.025), 4)


def suvi_file(origin: pd.Timestamp) -> str:
    """Fe195 file of the newest slot at origin-1h (file name carries year + day-of-year + hhmmss)."""
    slot = origin - pd.Timedelta(hours=1)
    t = slot - pd.Timedelta(minutes=1)
    hits = glob.glob(str(ROOT / f'data-v2/suvi/data/g18/{slot.year}/{slot.dayofyear:03d}/'
                         f'*_s{t.year}{t.dayofyear:03d}{t.hour:02d}5*.nc'))
    assert len(hits) == 1, hits
    return hits[0]

sys.path.insert(0, str(ROOT / 'leaderboard/model-release-publish'))
sys.path.insert(0, str(ROOT / 'research/e2000fs_calib4-20261031'))
from experiments.xgboost_suvi_fusion import coronal_holes as CH  # noqa: E402
from experiments.xgboost_suvi_fusion import features as SF  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


strips = load_module('strips', ROOT / 'leaderboard/model-release-publish/model_release/strips.py')
tw = load_module('tweedie_lane', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')


def r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def png_b64(img: Image.Image, fmt='PNG', **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True, **kw)
    return 'data:image/%s;base64,%s' % (fmt.lower().replace('jpeg', 'jpeg'), base64.b64encode(buf.getvalue()).decode())


# ------------------------------------------------------------------ image side
def crop_frame(rad, dqf, hdr):
    """Same crop as coronal_holes.extract_ch_frame (so masks match the stored features)."""
    h = dict(hdr)
    rsun = float(np.asarray(h['RSUN']).item())
    mat = np.array([[float(h['PC1_1']), float(h['PC1_2'])], [float(h['PC2_1']), float(h['PC2_2'])]])
    scale = np.linalg.svd(mat, compute_uv=False).min()
    radius = rsun / scale + 2
    cx, cy = float(h['CRPIX1']) - 1, float(h['CRPIX2']) - 1
    x0, x1 = max(0, int(np.floor(cx - radius))), min(rad.shape[1], int(np.ceil(cx + radius)) + 1)
    y0, y1 = max(0, int(np.floor(cy - radius))), min(rad.shape[0], int(np.ceil(cy + radius)) + 1)
    h['CRPIX1'], h['CRPIX2'] = cx + 1 - x0, cy + 1 - y0
    return rad[y0:y1, x0:x1], dqf[y0:y1, x0:x1], h, (x0, y0, x1, y1)


def colorize(rad, disk):
    """AIA-193-like false colour of log1p radiance; north up, west (lon>0) right."""
    v = np.log1p(np.where(np.isfinite(rad) & (rad >= 0), rad, 0.0))
    lo, hi = np.percentile(v[disk], 1), np.percentile(v[disk], 99.7)
    u = np.clip((v - lo) / (hi - lo), 0, 1) ** 0.9
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[8, 6, 10], [92, 44, 12], [190, 104, 28], [247, 178, 86], [255, 247, 224]], float)
    rgb = np.stack([np.interp(u, stops, cols[:, c]) for c in range(3)], -1).astype(np.uint8)
    return np.flipud(rgb)


def overlay(mask, color, alpha=170):
    m = np.flipud(mask)
    a = np.zeros(m.shape + (4,), np.uint8)
    a[m] = (*color, alpha)
    return Image.fromarray(a, 'RGBA')


def grid_paths(lon, lat, disk, hgt):
    """SVG path strings for lon = -90..90 step 15 and lat = -30, +30, plus the disk edge."""
    ml = np.ma.masked_where(~disk, lon)
    mt = np.ma.masked_where(~disk, lat)
    yy, xx = np.indices(lon.shape)

    def lines(z, levels):
        gen = contourpy.contour_generator(xx.astype(float), yy.astype(float), z, name='serial')
        d = []
        for lev in levels:
            for seg in gen.lines(lev):
                pts = [f'{x:.1f},{hgt - 1 - y:.1f}' for x, y in seg]
                if len(pts) > 1:
                    d.append('M' + 'L'.join(pts))
        return ' '.join(d)

    return {'lon': lines(ml, np.arange(-75, 90, 15)),
            'lat': lines(mt, [-30, 30]),
            'lon0': lines(ml, [0.0]),
            'lon_q': {str(k): lines(ml, [k]) for k in (-30, -15, 15, 30)},
            'lat_eq': lines(mt, [-30, 30])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    nc = [suvi_file(ORIGIN)]
    rad, dqf, hdr = SF.read_suvi_frame(nc[0])
    full = SF.extract_frame_features(rad, dqf, hdr)
    ch_all = CH.extract_ch_frame(rad, dqf, hdr)

    radc, dqfc, hc, box = crop_frame(rad, dqf, hdr)
    H, W = radc.shape
    coords = CH.geometry(radc.shape, hc)
    disk, lat, lon, area = coords
    valid = disk & (dqfc == 0) & np.isfinite(radc) & (radc >= 0)
    ref = float(np.median(radc[valid]))

    base = Image.fromarray(colorize(radc, disk))
    base_uri = png_b64(base, 'JPEG', quality=86)

    # --- patch statistics, 12 lon x 3 lat bands
    patches = []
    lon_i = np.clip(np.floor((np.deg2rad(lon) + np.pi / 2) / np.pi * 12), 0, 11).astype(int)
    lat_i = np.clip(np.floor((np.deg2rad(lat) + np.pi / 2) / np.pi * 3), 0, 2).astype(int)
    for i in range(12):
        for j in range(3):
            reg = disk & (lon_i == i) & (lat_i == j)
            ys, xs = np.nonzero(reg)
            if len(xs):  # label anchor: the cell pixel nearest the centroid (limb cells are crescents)
                q = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
                ax, ay = float(xs[q]), float(ys[q])
            patches.append({
                'i': i, 'j': j,
                'cx': r(ax, 1) if len(xs) else None,
                'cy': r(H - 1 - ay, 1) if len(ys) else None,
                'logmed': r(full[f'lon{i:02d}_lat{j}_log_median']),
                'dark': r(full[f'lon{i:02d}_lat{j}_dark_fraction']),
                'valid': r(full[f'lon{i:02d}_lat{j}_valid_fraction']),
                'q10': r(full[f'lon{i:02d}_lat{j}_log_q10']),
            })

    # --- coronal-hole candidates: masks for the two thresholds that carry the 30 features
    def cand(th):
        mask, labels, count, v, la, lo, ar = CH._candidate_from_geometry(radc, dqfc, coords, th)
        sizes = np.bincount(labels.ravel(), weights=ar.ravel(), minlength=count + 1)
        order = [k for k in np.argsort(-sizes) if k != 0]
        return mask, labels, order, sizes

    cands = {}
    for th, key in ((0.35, 't035'), (0.55, 't055')):
        mask, labels, order, sizes = cand(th)
        cands[key] = (mask, labels, order, sizes)
        # stored-value check
        assert abs(float(area[mask].sum()) - ch_all[f'ch_{key}_area']) < 1e-9

    eq = np.abs(lat) <= 30
    m15 = np.abs(lon) <= 15
    m30 = np.abs(lon) <= 30
    ORANGE, BLUE, GREEN, PINK = (255, 255, 255), (80, 170, 255), (90, 220, 150), (255, 90, 170)
    ch_layers = {}
    ch_defs = {
        'ch_ch_t035_top2_area': ('t035', 'top2', None),
        'ch_ch_t035_equatorial_area': ('t035', 'all', eq),
        'ch_ch_t035_meridian15_equatorial_area': ('t035', 'all', eq & m15),
        'ch_ch_t035_meridian30_equatorial_area': ('t035', 'all', eq & m30),
        'ch_ch_t055_equatorial_area': ('t055', 'all', eq),
        'ch_ch_t055_meridian15_area': ('t055', 'all', m15),
        'ch_ch_t055_meridian30_equatorial_area': ('t055', 'all', eq & m30),
    }
    ch_values = {}
    for name, (key, mode, region) in ch_defs.items():
        mask, labels, order, sizes = cands[key]
        if mode == 'top2':
            sel = labels == order[1]
            val = float(area[sel].sum())
            layer = sel
        else:
            layer = mask & region
            val = float(area[layer].sum())
        stored = ch_all[name.replace('ch_ch_', 'ch_')]
        assert abs(val - stored) < 1e-9, (name, val, stored)
        ch_values[name] = r(val, 5)
        ch_layers[name] = png_b64(overlay(layer, (255, 70, 150), 200), 'PNG')
    # backdrops: all candidates at both thresholds, the largest hole highlighted
    ch_base = {}
    for key in ('t035', 't055'):
        mask, labels, order, sizes = cands[key]
        rest = overlay(mask, (70, 190, 255), 150)
        ch_base[key] = {
            'all': png_b64(rest, 'PNG'),
            'top1': png_b64(overlay(labels == order[0], (255, 255, 255), 215), 'PNG'),
            'n_holes': len(order),
            'top_areas': [r(sizes[k], 5) for k in order[:4]],
            'area': r(area[mask].sum(), 5),
        }

    paths = grid_paths(lon, lat, disk, H)
    disk_edge = {'cx': float(hc['CRPIX1']) - 1, 'cy': H - 1 - (float(hc['CRPIX2']) - 1),
                 'r': float(hc['RSUN']) * 0.98}

    # --- strips recomputed from the patch cells and compared with the bundle that the model consumed
    names = json.loads((CACHE / 'names.json').read_text())
    selected = json.loads((ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/full_v1/manifest.json'
                           ).read_text())['selected']
    f1 = dict(np.load(CACHE / 'f1.npz'))
    origins = pd.DatetimeIndex(f1['origins'], tz='UTC')
    oi = int(np.where(origins == ORIGIN)[0][0])
    C, E = f1['C'][oi], f1['E'][oi]  # (90,), (72, 20)
    cn, vn = names['constant'], names['varying']

    rc = load_module('extract_raw_chain', ROOT / 'docs/explainers/code/extract_raw_chain.py')
    raw_chain = rc.sanitize(rc.example('flow', ORIGIN, *rc.load_sources()))
    assert raw_chain['bundle'] == 'f1', raw_chain['bundle']
    age = float(raw_chain['img']['age_h'])
    lat1 = [p for p in patches if p['j'] == 1]
    stat = {k: np.array([p[k] for p in lat1]) for k in ('logmed', 'dark', 'valid')}
    strip_rows = []
    for h in range(1, 73):
        row = {'h': h}
        for v in strips.SPEEDS:
            lam = float(strips.lam(h, v, age))
            w = np.array(strips.strip_weights(lam))
            ws = w.sum()
            cover = float((w * stat['valid']).sum() / ws) if ws > 0 else None
            ok = cover is not None and cover >= strips.COVER_MIN
            row[f'v{v}'] = {
                'lam': r(lam, 2), 'w': [r(x, 4) for x in w],
                'logmed': r((w * stat['logmed']).sum() / ws, 5) if ok else None,
                'dark': r((w * stat['dark']).sum() / ws, 5) if ok else None,
                'cover': r(cover, 4),
            }
            # compare to the bundle (the model input) for every horizon
            for s in ('logmed', 'dark'):
                col = f'strip_v{v}_{s}'
                b = E[h - 1, vn.index(col)]
                mine = row[f'v{v}'][s]
                if mine is None or not np.isfinite(b):
                    assert (mine is None) == (not np.isfinite(b)), (h, col, mine, b)
                else:
                    assert abs(mine - b) < 2e-3, (h, col, mine, b)
        strip_rows.append(row)

    # --- the 30 selected features, taken from the bundle in model order
    def src(n):
        if n.startswith(('strip_', 'ch_ch_', 'img_')):
            return 'image'
        if n.startswith('enlil_'):
            return 'enlil'
        return 'ace'
    const_sel = [n for n in cn if n in selected]
    vary_sel = [n for n in vn if n in selected]
    assert len(const_sel) == 21 and len(vary_sel) == 9
    feats = []
    for n in const_sel:
        feats.append({'name': n, 'kind': 'const', 'src': src(n), 'value': r(C[cn.index(n)], 5)})
    for n in vary_sel:
        feats.append({'name': n, 'kind': 'vary', 'src': src(n),
                      'series': [r(x, 4) for x in E[:, vn.index(n)]]})

    # --- model outputs for this origin
    def row_of(path, key='predictions'):
        z = np.load(path)
        o = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
        k = int(np.where(o == ORIGIN)[0][0])
        return z[key][k].astype(float), k
    body, k = row_of(ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz')
    tail, _ = row_of(ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz')
    prob, _ = row_of(ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/exceedance_prob.npz', 'prob')
    detroute, _ = row_of(ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz')
    gpd, _ = row_of(ROOT / 'research/e2000fs_calib3-20261031/runs/gpd_v1/forecast.npz')
    pool = load_module('pool_lane', ROOT / 'research/e2000fs_calib4-20261031/pool_lane.py')
    o_all, base_all, truth_all = pool.load_inputs()
    arm_all = np.asarray(np.load(pool.ARMS['gpd'])['predictions'], dtype=np.float64)
    wz = np.load(ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/weights.npz')
    ko = int(np.where(o_all == ORIGIN)[0][0])
    assert ko == k
    w, mature = float(wz['weights'][ko]), int(wz['mature'][ko])
    truth = truth_all[ko]
    assert np.allclose(base_all[ko], detroute, atol=1e-3) and np.allclose(arm_all[ko], gpd, atol=1e-3)
    final = (1 - w) * detroute + w * gpd
    # w scan on the matured trailing window (the frozen rule, closed form)
    j0 = max(0, mature - pool.WINDOW)
    se_b = (base_all[j0:mature] - truth_all[j0:mature]) ** 2
    se_a = (arm_all[j0:mature] - truth_all[j0:mature]) ** 2
    cr = (base_all[j0:mature] - truth_all[j0:mature]) * (arm_all[j0:mature] - truth_all[j0:mature])
    hi_w = truth_all[j0:mature] >= 550.
    nh = int(hi_w.sum())
    st = {'a0': se_b.mean(), 'b0': se_a.mean(), 'x0': cr.mean(),
          'a1': (se_b * hi_w).sum() / nh, 'b1': (se_a * hi_w).sum() / nh, 'x1': (cr * hi_w).sum() / nh}
    scan = []
    for g in GRID:
        tot = (1 - g) ** 2 * st['a0'] + g ** 2 * st['b0'] + 2 * g * (1 - g) * st['x0']
        tl = (1 - g) ** 2 * st['a1'] + g ** 2 * st['b1'] + 2 * g * (1 - g) * st['x1']
        scan.append({'w': float(g), 'total': r(tot, 2), 'tail': r(tl, 2), 'ok': bool(tl <= st['a1'])})
    best = min((x for x in scan if x['ok']), key=lambda x: x['total'])
    assert abs(best['w'] - w) < 1e-9, (best['w'], w)
    dp = {'window_rows': int(mature - j0), 'window_tail_cells': nh, 'a1': r(st['a1'], 2), 'scan': scan,
          'mse_base': r(((detroute - truth) ** 2).mean(), 1), 'mse_gpd': r(((gpd - truth) ** 2).mean(), 1),
          'mse_final': r(((final - truth) ** 2).mean(), 1),
          'window_start': str(o_all[j0]), 'window_end': str(o_all[mature - 1])}
    routed = prob >= 0.234
    assert np.allclose(np.where(routed, tail, body), detroute, atol=1e-3)

    # --- one real tree path + margin accumulation of the detector (band 0..2, F1 fit)
    sel_idx = {n: i for i, n in enumerate(selected)}
    cols = []
    for b, start in enumerate((0, 24, 48)):
        blk = [n for n in const_sel] + [f'{n}@h{start + t + 1}' for t in range(24) for n in vary_sel]
        cols.append(blk)
    designs = [tw.design(f1, names, selected, s)[oi] for s in (0, 24, 48)]
    for dsg, cl in zip(designs, cols):
        assert len(dsg) == len(cl) == 237
    # pick the horizon with the highest exceedance probability
    band_models = {}
    for b_ in range(3):
        m_ = xgb.Booster()
        m_.load_model(str(ROOT / f'research/e2000fs_calib2-20261031/runs/detroute_v1/detector_f1_b{b_}.ubj'))
        band_models[b_] = m_.trees_to_dataframe()

    def image_splits(h0):
        """Image-feature splits met by this sample in the first three trees of its band."""
        b_ = h0 // 24
        xx = np.append(designs[b_], h0 + 1).astype(np.float32)
        names_b = cols[b_] + ['hz']
        n_img = 0
        for t_ in range(3):
            d_ = band_models[b_][band_models[b_].Tree == t_].set_index('ID')
            cur_ = f'{t_}-0'
            while d_.loc[cur_, 'Feature'] != 'Leaf':
                idx_ = int(d_.loc[cur_, 'Feature'][1:])
                n_img += src(names_b[idx_].split('@')[0]) == 'image'
                cur_ = d_.loc[cur_, 'Yes'] if (np.isnan(xx[idx_]) or xx[idx_] < d_.loc[cur_, 'Split']) else d_.loc[cur_, 'No']
        return n_img
    cand_h = [h0 for h0 in range(72) if 0.25 < prob[h0] < 0.85]
    hs = max(cand_h, key=lambda h0: (image_splits(h0), -abs(prob[h0] - 0.5)))
    band, off = hs // 24, hs % 24
    x = np.append(designs[band], hs + 1).astype(np.float32)[None, :]
    fn = cols[band] + ['hz']
    model = xgb.Booster()
    model.load_model(str(ROOT / f'research/e2000fs_calib2-20261031/runs/detroute_v1/detector_f1_b{band}.ubj'))
    dm = xgb.DMatrix(x, feature_names=None)
    p_full = float(model.predict(dm)[0])
    assert abs(p_full - prob[hs]) < 1e-5, (p_full, prob[hs])
    margins = [float(model.predict(dm, output_margin=True, iteration_range=(0, k))[0])
               for k in list(range(1, 41)) + list(range(50, 401, 10))]
    ks = list(range(1, 41)) + list(range(50, 401, 10))
    leaf = model.predict(dm, pred_leaf=True)[0]
    bs = float(json.loads(bytes(model.save_raw('json')))['learner']['learner_model_param']['base_score'])
    td = model.trees_to_dataframe()

    def tree_dump(t):
        d = td[td.Tree == t]
        nodes = {}
        for _, rw in d.iterrows():
            nodes[rw.ID] = rw
        path, cur = [], d.iloc[0].ID
        out = []
        for _, rw in d.iterrows():
            idx = int(rw.Feature[1:]) if rw.Feature != 'Leaf' else None
            out.append({'id': rw.ID, 'feat': fn[idx] if idx is not None else None, 'split': r(rw.Split, 4),
                        'yes': rw.Yes if isinstance(rw.Yes, str) else None,
                        'no': rw.No if isinstance(rw.No, str) else None,
                        'leaf': r(rw.Gain, 5) if rw.Feature == 'Leaf' else None,
                        'cover': r(rw.Cover, 1)})
        # actual path for this sample
        cur = f'{t}-0'
        byid = {n['id']: n for n in out}
        while byid[cur]['feat'] is not None:
            n = byid[cur]
            idx = fn.index(n['feat'])
            v = float(x[0, idx])
            path.append({'id': cur, 'feat': n['feat'], 'split': n['split'], 'value': r(v, 4),
                         'go': 'yes' if (np.isnan(v) or v < n['split']) else 'no'})
            cur = n['yes'] if (np.isnan(v) or v < n['split']) else n['no']
        return {'nodes': out, 'path': path, 'leaf_id': cur, 'leaf': byid[cur]['leaf']}
    trees = [tree_dump(t) for t in range(3)]

    # gain share by source for all detector bands (F1 fit) and the tweedie body (F1 fit)
    def total_gain(path):
        """Number of splits per feature index, read from the model JSON. get_score segfaults on
        xgboost 2.1.4 multi-output trees and those trees store no loss_changes, so split counts
        (xgboost's 'weight' importance) are the one measure available for every model."""
        m = xgb.Booster()
        m.load_model(str(path))
        trees_json = json.loads(bytes(m.save_raw('json')))['learner']['gradient_booster']['model']['trees']
        gain = {}
        for t in trees_json:
            for left, idx in zip(t['left_children'], t['split_indices']):
                if left != -1:
                    gain[idx] = gain.get(idx, 0.0) + 1.0
        return gain

    def gain_by_source(path, cols_by_band, extra_hz=False, band=None):
        sc = total_gain(path)
        tot = {'image': 0.0, 'ace': 0.0, 'enlil': 0.0, 'hz': 0.0}
        per = {}
        for i, g in sc.items():
            name = cols_by_band[i]
            base_name = name.split('@')[0]
            s = 'hz' if name == 'hz' else src(base_name)
            tot[s] += g
            per[base_name] = per.get(base_name, 0.0) + g
        z = sum(tot.values())
        return {k: v / z for k, v in tot.items()}, {k: v / z for k, v in per.items()}

    # cross-check the JSON route against xgboost's own score on a binary model
    _m = xgb.Booster()
    _m.load_model(str(ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/detector_f1_b0.ubj'))
    _a = _m.get_score(importance_type='weight')
    _b = total_gain(ROOT / 'research/e2000fs_calib2-20261031/runs/detroute_v1/detector_f1_b0.ubj')
    assert all(abs(_a[f'f{i}'] - g) / g < 1e-3 for i, g in _b.items()), 'gain route mismatch'

    det_share, det_per = [], {}
    for b in range(3):
        s, p = gain_by_source(ROOT / f'research/e2000fs_calib2-20261031/runs/detroute_v1/detector_f1_b{b}.ubj',
                              cols[b] + ['hz'])
        det_share.append(s)
        for k, v in p.items():
            det_per[k] = det_per.get(k, 0.0) + v / 3
    body_share, body_per = [], {}
    for b in range(3):
        s, p = gain_by_source(ROOT / f'research/e2000fs_calib-20261030/runs/tweedie15_v1/model_tweedie_f1_b{b}.ubj',
                              cols[b])
        body_share.append(s)
        for k, v in p.items():
            body_per[k] = body_per.get(k, 0.0) + v / 3

    # --- ACE side: the numbers already verified by extract_raw_chain.py
    ace = {'raw': raw_chain['raw_ace'], 'map_checks': raw_chain['map_checks'],
           'recurrence': raw_chain['recurrence'], 'enlil': raw_chain['features']['enlil']}

    flow = {
        'origin': str(ORIGIN),
        'slot': raw_chain['img']['slot'], 'obs_end': raw_chain['img']['obs_end'], 'age_h': age,
        'image': {
            'file': Path(nc[0]).name, 'crop': list(box), 'size': [W, H], 'rsun_px': r(hc['RSUN'], 2),
            'frame_px': [int(rad.shape[1]), int(rad.shape[0])],
            'ref_median': r(ref, 4), 'valid_pct': r(100 * valid.sum() / disk.sum(), 2),
            'rad_min': r(np.nanmin(radc), 3), 'rad_max': r(np.nanmax(radc), 3),
            'b0': r(float(hdr.get('SOLAR_B0', 0)), 3),
            'pc': [r(hc[k], 6) for k in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2')],
            # CSS rotation that puts solar north up (raw frames follow the spacecraft attitude)
            'rot_deg': r(-np.degrees(np.arctan2(float(hc['PC2_1']), float(hc['PC2_2']))), 3),
            'grid': {k: v for k, v in paths.items() if k in ('lon', 'lat', 'lon0')},
            'lon_q': paths['lon_q'], 'disk': disk_edge,
            'uri': base_uri,
        },
        'patches': patches,
        'ch': {'values': ch_values, 'layers': ch_layers, 'base': ch_base,
               'full': {k: r(v, 5) for k, v in ch_all.items() if k.endswith(('_area', 'count')) and 't045' not in k
                        and 'lon0' not in k}},
        'strip': {'speeds': list(strips.SPEEDS), 'tau': {str(v): r(strips.TAU[v], 2) for v in strips.SPEEDS},
                  'omega': strips.OMEGA, 'half': strips.STRIP_HALF, 'cell_w': strips.CELL_W,
                  'cover_min': strips.COVER_MIN, 'rows': strip_rows},
        'features': feats,
        'columns': {'const': const_sel, 'vary': vary_sel, 'block': 237},
        'ace': ace,
        'model': {
            'body': [r(v, 2) for v in body], 'tail': [r(v, 2) for v in tail], 'prob': [r(v, 4) for v in prob],
            'detroute': [r(v, 2) for v in detroute], 'gpd': [r(v, 2) for v in gpd],
            'w': w, 'final': [r(v, 2) for v in final], 'truth': [r(v, 1) for v in truth],
            'tau': 0.234,
            'scan': dp['scan'], 'detail': {k_: v_ for k_, v_ in dp.items() if k_ != 'scan'},
        },
        'detector': {'h': hs + 1, 'band': band, 'prob': r(p_full, 4), 'rounds': 400,
                     'margin_k': ks, 'margin': [r(m, 4) for m in margins],
                     'bias': r(float(np.log(bs / (1 - bs))), 4),
                     'trees': trees, 'leaf_ids': [int(v) for v in leaf[:3]]},
        'gain': {'detector': det_share, 'body': body_share,
                 'det_per': dict(sorted(det_per.items(), key=lambda t: -t[1])),
                 'body_per': dict(sorted(body_per.items(), key=lambda t: -t[1]))},
        'scores': json.loads((OUT / 'detpool_gpd.json').read_text())['scores'],
        'base_scores': json.loads((OUT / 'detpool_gpd.json').read_text())['base_scores'],
        'weights': json.loads((OUT / 'detpool_gpd.json').read_text())['weights'],
    }
    (OUT / 'flow.json').write_text(json.dumps(flow, ensure_ascii=False, separators=(',', ':')))
    size = (OUT / 'flow.json').stat().st_size
    print('flow.json', size // 1024, 'KB; origin', ORIGIN, 'h*', hs + 1, 'p', round(p_full, 3), 'w', w)
    print('gain by source detector', [{k: round(v, 3) for k, v in s.items()} for s in det_share])
    print('gain by source body', [{k: round(v, 3) for k, v in s.items()} for s in body_share])


if __name__ == '__main__':
    main()
