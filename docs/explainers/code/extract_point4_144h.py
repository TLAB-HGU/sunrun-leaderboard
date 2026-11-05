"""Extract the numbers behind docs/explainers/e2000fs_calib_point4_144h.html.

Compares the archived parent (e2000fs_dirmo_24_feat30_pointweight4, mask origin+72h < cutoff) with the
standalone 144h-mask control (mask origin+144h <= cutoff) on the official F1+F2 grid. Scores are
recomputed from the archived prediction arrays and cross-checked against the runs' results.json.

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_point4_144h.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
PARENT = ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/full_v1'
RUN144 = ROOT / 'research/e2000fs_calib-20261030/runs/control_144h_v1'
CACHE = ROOT / 'analysis/A0014/outputs/cache'
OUT = ROOT / 'docs/explainers/data/point4_144h.json'
sys.path.insert(0, str(ROOT / 'leaderboard'))
sys.path.insert(0, str(ROOT / 'research/e2000fs_calib4-20261031'))

H, U = 72, 550.0
CUTOFFS = {'f1': pd.Timestamp('2026-02-26', tz='UTC'), 'f2': pd.Timestamp('2026-05-29', tz='UTC')}
F2_CUT = CUTOFFS['f2']


def r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main() -> None:
    from pool_lane import score as lane_score          # one scorer for both arms (csi/pod/far included)

    tr = dict(np.load(CACHE / 'train.npz'))
    train_origins = pd.DatetimeIndex(tr['origins'], tz='UTC')
    f1 = np.load(PARENT / 'f1_predictions.npz')
    f2 = np.load(PARENT / 'final_predictions.npz')
    origins = pd.DatetimeIndex(np.concatenate([f1['origins'], f2['origins']]), tz='UTC')
    assert len(origins) == 4970
    parent = np.vstack([f1['pred'], f2['pred']]).astype(np.float64)
    c1 = np.load(RUN144 / '..' / 'tweedie15_v1' / 'control_f1_predictions.npz')
    c2 = np.load(RUN144 / '..' / 'tweedie15_v1' / 'control_final_predictions.npz')
    assert np.array_equal(c1['origins'], f1['origins']) and np.array_equal(c2['origins'], f2['origins'])
    m144 = np.vstack([c1['pred'], c2['pred']]).astype(np.float64)

    speed = pd.read_parquet(CACHE / 'speed.parquet')['speed'].sort_index()
    t = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    truth = speed.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    assert not np.isnan(truth).any()

    n_f1 = int((origins < F2_CUT).sum())
    assert n_f1 == 2137

    def fold_score(pred, truth, fold_name):
        err = pred - truth
        se = err ** 2
        hi, lo = truth >= U, truth < 350.
        return dict(fold=fold_name, n=len(truth), cells=int(pred.size),
                    mse=float(se.mean()), mse_ge550=float(se[hi].mean()),
                    bias=float(err.mean()), bias_lt350=float(err[lo].mean()),
                    bias_ge550=float(err[hi].mean()),
                    band={k: float(se[:, sl].mean()) for k, sl in
                          (('h1-24', slice(0, 24)), ('h25-48', slice(24, 48)), ('h49-72', slice(48, 72)))})

    sc = {'parent': {}, 'h144': {}}
    for name, arr in (('parent', parent), ('h144', m144)):
        sc[name]['f1'] = fold_score(arr[:n_f1], truth[:n_f1], 'f1')
        sc[name]['f2'] = fold_score(arr[n_f1:], truth[n_f1:], 'f2')
        sc[name]['pooled'] = lane_score(arr, truth, origins, n_f1)

    pj = json.loads((PARENT / 'results.json').read_text())
    for fold, key in (('f1', 'f1'), ('f2', 'final'), ('pooled', 'combined')):
        a, b = sc['parent'][fold], pj[key]
        assert abs(a['mse'] - b['mse']) < 1e-4, (fold, a['mse'], b['mse'])
        assert abs(a['mse_ge550'] - b['mse_ge550']) < 1e-4
    assert abs(sc['parent']['pooled']['bias'] - json.loads((PARENT / 'results.json').read_text())
               ['combined'].get('bias', sc['parent']['pooled']['bias'])) < 1.0
    qj = json.loads((RUN144 / 'results.json').read_text())['candidate_scores']
    assert abs(sc['h144']['pooled']['mse'] - qj['mse']) < 1e-6
    assert abs(sc['h144']['pooled']['mse_ge550'] - qj['mse_ge550']) < 1e-6

    # ---- masks: what the extra 72 h actually removes -----------------------------
    masks = {}
    for fold, cut in CUTOFFS.items():
        keep72 = np.flatnonzero(train_origins + pd.Timedelta(hours=72) < cut)
        keep144 = np.flatnonzero(train_origins + pd.Timedelta(hours=144) <= cut)
        dropped = sorted(set(keep72.tolist()) - set(keep144.tolist()))
        masks[fold] = dict(cutoff=str(cut.date()), n72=int(len(keep72)), n144=int(len(keep144)),
                           dropped=len(dropped),
                           last72=str(train_origins[keep72[-1]]), last144=str(train_origins[keep144[-1]]),
                           last_target72=str(train_origins[keep72[-1]] + pd.Timedelta(hours=H)),
                           last_target144=str(train_origins[keep144[-1]] + pd.Timedelta(hours=H)),
                           dropped_origins=[str(train_origins[i]) for i in dropped])
    gaps = np.diff(train_origins.asi8) / 3.6e12
    uniq, cnt = np.unique(gaps, return_counts=True)
    spacing = dict(median_h=float(np.median(gaps)), modes=[[float(u), int(c)] for u, c in zip(uniq, cnt)][:6],
                   hourly_share=float((gaps == 1).mean()), n=int(len(train_origins)),
                   span=[str(train_origins[0]), str(train_origins[-1])])

    # ---- effect: per fold, per horizon band, paired bootstrap --------------------
    bands = []
    for fold, sl in (('f1', slice(0, n_f1)), ('f2', slice(n_f1, None))):
        for lab, bsl in (('h1–24', slice(0, 24)), ('h25–48', slice(24, 48)), ('h49–72', slice(48, 72))):
            e0 = (parent[sl][:, bsl] - truth[sl][:, bsl]) ** 2
            e1 = (m144[sl][:, bsl] - truth[sl][:, bsl]) ** 2
            bands.append(dict(fold=fold, band=lab, parent=r(e0.mean(), 1), h144=r(e1.mean(), 1),
                              delta=r(e1.mean() - e0.mean(), 1)))

    per_h = dict(h=list(range(1, H + 1)),
                 parent=[r(((parent[:, i] - truth[:, i]) ** 2).mean(), 1) for i in range(H)],
                 h144=[r(((m144[:, i] - truth[:, i]) ** 2).mean(), 1) for i in range(H)])

    # label-maturity arithmetic: when does the last training label become final?
    f1_first = str(origins[0])
    maturity = {}
    fold_first = {'f1': str(origins[0]), 'f2': str(origins[n_f1])}
    for fold, cut in CUTOFFS.items():
        m = masks[fold]
        end72 = pd.Timestamp(m['last_target72'])
        end144 = pd.Timestamp(m['last_target144'])
        maturity[fold] = dict(end72=str(end72), end144=str(end144),
                              matured72=str(end72 + pd.Timedelta(hours=H)),
                              matured144=str(end144 + pd.Timedelta(hours=H)),
                              fold_first=fold_first[fold],
                              cutoff=str(cut), before_start_h=round(
                                  (pd.Timestamp(fold_first[fold]) - (end72 + pd.Timedelta(hours=H))).total_seconds() / 3600, 1))
    se0 = (parent - truth) ** 2
    se1 = (m144 - truth) ** 2
    hi = truth >= U
    per0, per1 = se0.mean(1), se1.mean(1)                       # per-origin MSE
    tail0 = np.where(hi, se0, 0.0).sum(1)
    tail1 = np.where(hi, se1, 0.0).sum(1)
    tn = hi.sum(1)
    rng = np.random.default_rng(0)
    B = 4000
    idx = rng.integers(0, len(origins), size=(B, len(origins)))
    d_mse = per1[idx].mean(1) - per0[idx].mean(1)
    d_tail = tail1[idx].sum(1) / tn[idx].sum(1) - tail0[idx].sum(1) / tn[idx].sum(1)
    boot = dict(B=B, d_mse=[r(np.quantile(d_mse, .025), 2), r(np.quantile(d_mse, .975), 2)],
                d_tail=[r(np.quantile(d_tail, .025), 2), r(np.quantile(d_tail, .975), 2)],
                share_origins_better=r(float((per1 < per0).mean()), 4),
                share_origins_tail_better=r(float(np.mean(tail1[tn > 0] / np.maximum(tn[tn > 0], 1)
                                                          < tail0[tn > 0] / np.maximum(tn[tn > 0], 1))), 4))

    per_origin = per1 - per0
    gain = int(np.argmin(np.where(tn >= 6, per_origin, np.inf)))
    loss = int(np.argmax(np.where(tn >= 6, per_origin, -np.inf)))
    def ex(i, tag):
        return dict(tag=tag, origin=str(origins[i]), fold='F1' if i < n_f1 else 'F2',
                    truth=[r(v, 1) for v in truth[i]], parent=[r(v, 1) for v in parent[i]],
                    h144=[r(v, 1) for v in m144[i]],
                    mse=dict(parent=r(per0[i], 1), h144=r(per1[i], 1)), tail_cells=int(tn[i]))
    examples = [ex(gain, 'gain'), ex(loss, 'loss')]

    d = (m144 - parent).ravel()
    edges = [-120, -40, -20, -10, -5, 5, 10, 20, 40, 120]
    cnts, _ = np.histogram(d, bins=edges)
    diff = dict(edges=edges, counts=[int(c) for c in cnts], share=[r(c / d.size, 4) for c in cnts],
                mean=r(d.mean(), 2), sd=r(d.std(), 2), share_beyond_10=r(float((np.abs(d) > 10).mean()), 4))

    payload = dict(
        run=dict(experiment='e2000fs_calib_point4_144h', dir=str(RUN144.relative_to(ROOT)),
                 parent=str(PARENT.relative_to(ROOT)),
                 parent_mask='origin + 72h < cutoff', mask='origin + 144h <= cutoff',
                 recipe='three 24-output vector trees · Top30 237 cols/block · 900 rounds · depth 3 · eta 0.03 · seed 42',
                 discipline='only the training mask differs from the archived parent'),
        masks=masks, spacing=spacing, scores=sc, bands=bands, boot=boot, per_h=per_h, maturity=maturity,
        first_origin=f1_first,
        diff=diff, examples=examples,
        n_origins=len(origins), n_f1=n_f1, cells=int(parent.size),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')
    for k in ('parent', 'h144'):
        p_ = sc[k]['pooled']
        print(k, {x: r(p_[x], 2) for x in ('mse', 'mse_ge550', 'bias', 'bias_lt350', 'bias_ge550', 'csi')})
    print('boot', boot)


if __name__ == '__main__':
    main()
