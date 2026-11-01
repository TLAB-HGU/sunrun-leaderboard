"""Extract the numbers behind docs/explainers/e2000fs_calib5_detpool_all.html.

Recomputes every score from the archived arrays (base = calib4 gpd_v1, stack = poolall_v1),
checks them against results.json / the sibling page bundle, and summarises the 81-arm NNLS
weights. Any mismatch stops the script.

Usage: /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_calib5.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
RUN = ROOT / 'research/e2000fs_calib5-20261031/runs/poolall_v1'
BASE = ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/forecast.npz'
SERIES = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
FLOW = ROOT / 'docs/explainers/data/flow.json'
OUT = ROOT / 'docs/explainers/data/calib5.json'

H, U = 72, 550.0
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
WINDOW, REFIT, RIDGE = 2160, 168, 1e-3


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    se = err ** 2
    hi, lo = truth >= U, truth < 350.
    return dict(mse=float(se.mean()), ge=float(se[hi].mean()), bias=float(err.mean()),
                lt350=float(err[lo].mean()), ge_bias=float(err[hi].mean()),
                tail_n=int(hi.sum()), cells=int(pred.size),
                band=[float(se[:, sl].mean()) for sl in (slice(0, 24), slice(24, 48), slice(48, 72))])


def main() -> None:
    res = json.loads((RUN / 'results.json').read_text())
    b = np.load(BASE)
    origins = pd.DatetimeIndex(b['origins_ns'], tz='UTC')
    base = np.asarray(b['predictions'], dtype=np.float64)
    s = np.load(RUN / 'forecast.npz')
    assert np.array_equal(s['origins_ns'], b['origins_ns'])
    stack = np.asarray(s['predictions'], dtype=np.float64)
    speed = pd.read_parquet(SERIES)['speed'].sort_index()
    t = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    truth = speed.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    assert not np.isnan(truth).any()

    n_f1 = int((origins < F2_CUT).sum())
    assert n_f1 == 2137
    folds = {'f1': slice(0, n_f1), 'f2': slice(n_f1, None)}

    out_scores = {'base': {}, 'stack': {}}
    for name, arr in (('base', base), ('stack', stack)):
        for fold, sl in folds.items():
            out_scores[name][fold] = scores(arr[sl], truth[sl])
        out_scores[name]['pooled'] = scores(arr, truth)

    # cross-checks: results.json (stack) and the sibling detpool page (base)
    off = res['official']
    assert abs(out_scores['stack']['pooled']['mse'] - off['mse']) < 1e-6
    assert abs(out_scores['stack']['pooled']['ge'] - off['mse_ge550']) < 1e-6
    assert abs(out_scores['stack']['f1']['mse'] - off['mse_f1']) < 1e-6
    assert abs(out_scores['stack']['f2']['ge'] - off['mse_f2_ge550']) < 1e-6
    flow = json.loads(FLOW.read_text())
    assert abs(out_scores['base']['pooled']['mse'] - flow['scores']['total']) < 1e-3
    assert abs(out_scores['base']['pooled']['ge'] - flow['scores']['ge550']) < 1e-3
    assert abs(out_scores['base']['pooled']['bias'] - flow['scores']['bias']) < 1e-2

    # ---- betas -----------------------------------------------------------------
    wz = np.load(RUN / 'weights.npz', allow_pickle=True)
    betas = np.asarray(wz['betas'], dtype=np.float64)
    arms = [str(a) for a in wz['arms']]
    assert betas.shape == (len(origins), len(arms)) == (4970, 81)
    bsum = betas.sum(axis=1)
    nz = (betas > 0).any(axis=1)
    assert abs(bsum.mean() - res['weights']['mean_sum']) < 1e-6
    assert abs(nz.mean() - res['weights']['share_nonzero_blocks']) < 1e-6

    # reconstruct the stack from base + sum(beta * (arm - base)) to prove the arithmetic
    deltas = []
    for name in arms:
        p = ROOT / 'research' / name
        z = np.load(p)
        deltas.append(np.asarray(z['predictions'], dtype=np.float64) - base)
    delta = np.stack(deltas)                       # (81, 4970, 72)
    recon = base + np.einsum('ij,jik->ik', betas, delta)
    recon_err = float(np.abs(recon - stack).max())

    mean_beta = betas.mean(axis=0)
    order = np.argsort(-mean_beta)
    top = []
    for i in order[:12]:
        st = scores(deltas[i] + base, truth)       # that arm alone
        top.append(dict(arm=arms[i], mean_beta=r(mean_beta[i], 4), share=r(mean_beta[i] / 0.5, 4),
                        nz_share=r(float((betas[:, i] > 0).mean()), 3),
                        mse=r(st['mse'], 1), ge=r(st['ge'], 1), alone_delta_mse=r(st['mse'] - out_scores['base']['pooled']['mse'], 1),
                        alone_delta_ge=r(st['ge'] - out_scores['base']['pooled']['ge'], 1)))

    # correlation structure of the arm increments (how redundant are 81 arms)
    rng = np.random.default_rng(0)
    cells = rng.choice(delta.shape[1] * H, size=min(120_000, delta.shape[1] * H), replace=False)
    flat = delta.reshape(len(arms), -1)[:, cells]
    flat = flat - flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True) + 1e-12
    corr = (flat / sd) @ (flat / sd).T / flat.shape[1]
    iu = np.triu_indices(len(arms), 1)
    off = corr[iu]
    corr_stats = dict(mean=r(float(off.mean()), 3), p10=r(float(np.quantile(off, .1)), 3),
                      median=r(float(np.median(off)), 3), p90=r(float(np.quantile(off, .9)), 3),
                      share_above_90=r(float((off > 0.9).mean()), 3), n_pairs=int(len(off)))
    # levels are far more correlated than the base-relative increments
    lvl = np.stack([deltas[i] + base for i in range(len(arms))]).reshape(len(arms), -1)[:, cells]
    lvl = lvl - lvl.mean(axis=1, keepdims=True)
    ls = lvl.std(axis=1, keepdims=True) + 1e-12
    lcorr = (lvl / ls) @ (lvl / ls).T / lvl.shape[1]
    loff = lcorr[iu]
    level_corr = dict(mean=r(float(loff.mean()), 4), median=r(float(np.median(loff)), 4),
                      min=r(float(loff.min()), 4), share_above_99=r(float((loff > .99).mean()), 3))

    # ---- how the weights evolve (refit every 168 origins) ----------------------
    det = arms.index('e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz')
    # post-hoc diagnostic (NOT a preregistered result): same betas with the detroute column zeroed
    b2 = betas.copy(); b2[:, det] = 0.0
    cf = base + np.einsum('ij,jik->ik', b2, delta)
    counterfactual = dict(without_detroute=scores(cf, truth),
                          detroute_alone=scores(deltas[det] + base, truth),
                          base=scores(base, truth))
    series = [dict(origin=str(origins[i])[:16], sum=r(bsum[i], 4), detroute=r(betas[i, det], 4))
              for i in range(0, len(origins), REFIT)]
    hist_edges = [0, 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    cnt, _ = np.histogram(bsum, bins=hist_edges)
    hist = dict(edges=hist_edges, counts=[int(c) for c in cnt], share=[r(c / len(bsum), 3) for c in cnt])

    # ---- prediction differences -------------------------------------------------
    d = (stack - base).ravel()
    d_edges = [-200, -60, -30, -15, -5, 5, 15, 30, 60, 200]
    dc, _ = np.histogram(d, bins=d_edges)
    diff = dict(edges=d_edges, counts=[int(c) for c in dc], share=[r(c / d.size, 4) for c in dc],
                mean=r(d.mean(), 2), sd=r(d.std(), 2), share_beyond_15=r(float((np.abs(d) > 15).mean()), 4))

    # ---- worked example origins --------------------------------------------------
    per = ((stack - truth) ** 2).mean(1) - ((base - truth) ** 2).mean(1)   # >0 → stack worse
    hi_ok = (truth >= U).sum(1) >= 6
    worst = int(np.argmax(np.where(hi_ok, per, -np.inf)))
    best = int(np.argmin(np.where(hi_ok, per, np.inf)))
    def ex(i, tag):
        return dict(tag=tag, origin=str(origins[i]), index=int(i), fold='F1' if i < n_f1 else 'F2',
                    truth=[r(v, 1) for v in truth[i]], base=[r(v, 1) for v in base[i]],
                    stack=[r(v, 1) for v in stack[i]],
                    mse=dict(base=r(((base[i] - truth[i]) ** 2).mean(), 1),
                             stack=r(((stack[i] - truth[i]) ** 2).mean(), 1)),
                    tail_cells=int((truth[i] >= U).sum()))
    examples = [ex(best, 'gain'), ex(worst, 'loss')]

    payload = dict(
        run=dict(experiment=res['experiment'], dir=str(RUN.relative_to(ROOT)),
                 base=str(BASE.relative_to(ROOT)), n_arms=res['n_arms'], window=WINDOW, refit=REFIT,
                 ridge=RIDGE, shrink=list((1.0, 0.75, 0.5, 0.25)),
                 rule=res['rule'], discipline=res['discipline'],
                 refits=int(np.ceil(len(origins) / REFIT)), origins=int(len(origins))),
        scores=out_scores,
        weights=dict(mean_sum=r(bsum.mean(), 4), median_sum=r(np.median(bsum), 4),
                     share_all_zero=r(float((bsum <= 0).mean()), 4),
                     share_nonzero=r(float(nz.mean()), 4),
                     share_sum_capped=r(float((bsum > 0.999).mean()), 4),
                     mean_nonzero_betas=r(float(betas[betas > 0].mean()), 4),
                     n_positive_cells=int((betas > 0).sum()), n_cells=int(betas.size),
                     hist=hist, series=series, top=top),
        recon=dict(max_abs_error=recon_err),
        arms=dict(names=arms, count=len(arms), corr=corr_stats, level_corr=level_corr),
        counterfactual=counterfactual,
        diff=diff, examples=examples,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')
    print('recon max abs err', recon_err)
    print('base pooled', {k: r(v, 2) for k, v in out_scores['base']['pooled'].items() if k in ('mse', 'ge')})
    print('stack pooled', {k: r(v, 2) for k, v in out_scores['stack']['pooled'].items() if k in ('mse', 'ge')})
    print('corr', corr_stats)


if __name__ == '__main__':
    main()
