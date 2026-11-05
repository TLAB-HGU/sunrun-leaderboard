"""e2000fs_blk0prev reproducible script (self-contained).

Rebuilds the t0 / t0-12h SUVI strips from the pinned cell table, trains the DirMO-24
recipe with block 0 (horizons 1-24) reading the t0-12h strip and blocks 1-2 reading t0,
and writes the 357,840-row official forecast to <out>/predictions.parquet.

    python <this file>
"""
"""Build t0 and t0-12h SUVI strips for train/f1/official splits (same-lambda convention)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/t-lab01/sunrun')
SUVI = ROOT / 'data-v2/raw-store/v2/solar-pivot-noanalog72-20261015/pinned-v1/repo/leaderboard/store/suvi-fusion/suvi-hourly-v2.parquet'
CACHE = ROOT / 'analysis/A0014/outputs/cache'
OUT = Path(__file__).resolve().parent / 'out'

OM, D = 0.55, 147858620.7
SP = (350, 450, 550, 650, 750)
TAU = {v: D / v / 3600.0 for v in SP}
CW, SH, CM, MG = 15.0, 5.0, 0.5, 1.0
HALL = np.arange(1, 73)
OFF = (0, 12)
STATS = ('log_median', 'dark_fraction', 'valid_fraction')
PICK = [('v350', 'logmed', 'log_median'), ('v450', 'logmed', 'log_median'), ('v550', 'dark', 'dark_fraction'),
        ('v650', 'dark', 'dark_fraction'), ('v750', 'logmed', 'log_median')]


def strips(cache, times, origins, off):
    edges = -90.0 + CW * np.arange(13)
    tau = np.array([TAU[v] for v in SP])
    st = {s: cache[[f'lon{i:02d}_lat1_{s}' for i in range(12)]].to_numpy(np.float32) for s in STATS}
    n = len(origins)
    out = np.full((n, 72, len(PICK)), np.nan, np.float32)
    o = pd.DatetimeIndex(origins, tz='UTC').to_numpy('datetime64[ns]').astype('int64') - off * 3_600_000_000_000
    for i, t in enumerate(o):
        j = np.searchsorted(times, t - int(MG * 3.6e12), side='right') - 1
        if j < 0:
            continue
        age = (t - times[j]) / 3.6e12
        LAM = OM * (tau[None, :] - (HALL[:, None] + age))
        W = np.clip(np.minimum(LAM[..., None] + SH, edges[1:]) - np.maximum(LAM[..., None] - SH, edges[:-1]), 0, None) / (2 * SH)
        ws = np.maximum(W.sum(-1), 1e-12)
        for vi in range(len(SP)):
            cov = (W[:, vi, :] * st['valid_fraction'][j]).sum(-1) / ws[:, vi]
            ok = cov >= CM
            for pi, (_, _, s) in enumerate(PICK):
                out[i, :, pi] = np.where(ok, (W[:, vi, :] * st[s][j]).sum(-1) / ws[:, vi], np.nan)
    return out




def _build_strips():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_parquet(SUVI)
    c.index = pd.to_datetime(c.index, utc=True)
    ti = c.index.to_numpy('datetime64[ns]').astype('int64')
    for split in ('train', 'f1', 'official'):
        o = pd.DatetimeIndex(np.load(CACHE / f'{split}.npz')['origins'], tz='UTC')
        E = np.stack([strips(c, ti, o, d) for d in OFF], axis=2)
        np.savez_compressed(OUT / f'strip2_{split}.npz', origins=o.asi8, E=E,
                            names=np.array([p[0] + '_' + p[1] for p in PICK]))
        print(split, E.shape, float(np.isnan(E).mean()), flush=True)



    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--rounds', type=int, default=900)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rec = parent_record()
    selected = rec['config']['data']['selected']
    recipe = rec['config']['hyperparameters']['recipe']['xgboost']
    params = {k: v for k, v in recipe.items() if k not in ('n_estimators', 'n_jobs', 'random_state')}
    params.update(nthread=args.jobs, seed=42)
    BASE.write_text(json.dumps({'parent_sha': PARENT_SHA, 'selected': selected, 'recipe': recipe,
                                'snapshot_per_block': SNAP, 'maturity_hours': 72}, indent=2))

    names = json.loads((CACHE / 'names.json').read_text())
    const, vary = names['constant'], names['varying']
    rest = [n for n in selected if not n.startswith('strip_')]
    ci = [i for i, n in enumerate(const) if n in rest]
    ri = [vary.index(n) for n in REC_NAMES if n in selected]

    strips = {s: np.load(OUT_STRIPS / f'strip2_{s}.npz')['E'] for s in ('train', 'f1', 'official')}

    def design(bundle, split, start, block):
        blk = bundle['E'][:, start:start + 24, :]
        rec_part = blk[:, :, ri].reshape(len(bundle['C']), -1)
        strip_part = strips[split][:, start:start + 24, SNAP[block], :].reshape(len(bundle['C']), -1)
        return np.hstack([bundle['C'][:, ci], rec_part, strip_part]).astype(np.float32)

    train = dict(np.load(CACHE / 'train.npz'))
    origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed']
    y = targets(series, origins)
    fit_int = np.flatnonzero(origins + pd.Timedelta(hours=144) < pd.Timestamp('2025-12-04', tz='UTC'))
    val = np.flatnonzero((origins >= pd.Timestamp('2025-12-04', tz='UTC')) &
                         (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))

    stages = [('internal', fit_int, train, 'train', origins[val], y[val])]
    for stage, grid in (('f1', 'f1'), ('final', 'official')):
        bundle = dict(np.load(CACHE / f'{grid}.npz'))
        oo = pd.DatetimeIndex(bundle['origins'], tz='UTC')
        cutoff = {'f1': '2026-02-26', 'final': '2026-05-29'}[stage]
        fit = np.flatnonzero(origins + pd.Timedelta(hours=144) < pd.Timestamp(cutoff, tz='UTC'))
        stages.append((stage, fit, bundle, grid, oo, targets(series, oo)))

    results, preds = {}, {}
    for stage, fit, bundle, split, oo, yy in stages:
        pred = np.empty_like(yy)
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            X = design(bundle if stage != 'internal' else train, split, b * 24, b)
            tgt = (y if stage == 'internal' else yy)[:, sl]
            if stage == 'internal':
                Xfit, Xval = X[fit], X[val]
            else:
                Xfit = design(train, 'train', b * 24, b)[fit]
                Xval = X
            tgt_fit = (y[:, sl] if stage == 'internal' else y[:, sl])[fit].astype(np.float32)
            dt = xgb.DMatrix(Xfit, label=tgt_fit, nthread=args.jobs)
            settings = dict(params, base_score=float(tgt_fit.mean()), num_target=24)
            rounds = 20 if args.smoke else args.rounds
            model = xgb.train(settings, dt, num_boost_round=rounds, obj=objective)
            pred[:, sl] = np.clip(model.inplace_predict(Xval), 1., 3000.)
        np.savez_compressed(out / f'{stage}_predictions.npz', pred=pred, origins=oo.asi8)
        preds[stage] = (oo, yy, pred)
        se = (pred - yy) ** 2
        m = yy >= HPO
        results[stage] = {'mse': float(se.mean()), 'mse_ge550': float(se[m].mean()),
                          'bias_ge550': float((pred - yy)[m].mean()), 'n_origins': int(len(oo))}
        print(stage, results[stage], flush=True)

    oo = preds['f1'][0].append(preds['final'][0])
    yy = np.vstack([preds[s][1] for s in ('f1', 'final')])
    pp = np.vstack([preds[s][2] for s in ('f1', 'final')])
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(oo.asi8, 72),
                          'horizon_hours': np.tile(np.arange(1, 73), len(oo)),
                          'pred_kms': pp.ravel()})
    frame['origin_last_input_utc'] = pd.to_datetime(frame['origin_last_input_utc'], utc=True)
    assert len(frame) == 357840 and not frame.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
    frame.to_parquet(out / 'predictions.parquet', index=False)
    se = (pp.ravel() - yy.ravel()) ** 2
    m = yy.ravel() >= HPO
    results['combined'] = {'mse': float(se.mean()), 'mse_ge550': float(se[m].mean())}
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print('combined', results['combined'], flush=True)




OUT_STRIPS = OUT


if __name__ == '__main__':
    import sys
    target = Path('out')
    target.mkdir(parents=True, exist_ok=True)
    _build_strips()
    sys.argv = ['run.py', '--out', str(target)]
    main()
