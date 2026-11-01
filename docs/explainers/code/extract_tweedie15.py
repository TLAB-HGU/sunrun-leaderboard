"""Extract the numbers behind docs/explainers/e2000fs_calib_tweedie15.html.

Everything the page shows is recomputed from the archived run
`research/e2000fs_calib-20261030/runs/tweedie15_v1/` and checked against the run log,
so no constant is typed by hand. Any mismatch stops the script.

Usage:  /home/t-lab01/sunrun/.venv/bin/python docs/explainers/code/extract_tweedie15.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
RUN = ROOT / 'research/e2000fs_calib-20261030/runs/tweedie15_v1'
CACHE = ROOT / 'analysis/A0014/outputs/cache'
MANIFEST = ROOT / 'research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/full_v1/manifest.json'
OUT = ROOT / 'docs/explainers/data/tweedie15.json'

H, U, S, P = 72, 550.0, 500.0, 1.5
BLOCKS = (0, 24, 48)
F1_END = pd.Timestamp('2026-02-26', tz='UTC')
F2_END = pd.Timestamp('2026-05-29', tz='UTC')
VAL = (pd.Timestamp('2025-12-04', tz='UTC'), pd.Timestamp('2026-02-22T23:00:00Z'))
LOG = {
    'control': {
        'internal': dict(n=648, mse=7660.435071399218, ge=11373.621130441355, bias=7.3589030017251,
                         lt350=79.44780937499999, ge_bias=-57.9627040132484,
                         band=(4111.36154170984, 9142.325681747554, 9727.617990740262)),
        'f1': dict(n=2137, mse=4049.4522988229915, ge=5245.454930568053, bias=16.60685329715925,
                   lt350=71.67928519953664, ge_bias=-31.729103526445773,
                   band=(2023.4296764947399, 4606.105468962694, 5518.821751011541)),
        'f2': dict(n=2833, mse=3069.738348977694, ge=6292.477687546873, bias=11.143383121247355,
                   lt350=59.79039462237157, ge_bias=-48.62391130347469,
                   band=(1843.7153188821578, 3270.9279635785124, 4094.571764472413)),
    },
    'tweedie': {
        'internal': dict(n=648, mse=8115.4713889013165, ge=15664.189801288367, bias=-12.317156615865574,
                         lt350=70.41959584350585, ge_bias=-78.4195744224338,
                         band=(4256.490257757704, 9252.229880923791, 10837.694028022452)),
        'f1': dict(n=2137, mse=3512.7149222078965, ge=8655.53847950938, bias=-1.5258306642233952,
                   lt350=54.72403643152235, ge_bias=-57.5280444256153,
                   band=(1936.9244898709699, 4181.1811950952415, 4420.03908165748)),
        'f2': dict(n=2833, mse=3012.5248351553114, ge=10213.018384756455, bias=-1.235119181830973,
                   lt350=56.3628486300455, ge_bias=-76.3280756474655,
                   band=(1750.2426876718414, 3360.6344733548103, 3926.6973444392797)),
    },
}


def r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    hi, lo = truth >= U, truth < 350.
    se = err ** 2
    return dict(n=int(len(truth)), cells=int(pred.size), mse=float(se.mean()),
                ge=float(se[hi].mean()) if hi.any() else None,
                bias=float(err.mean()),
                lt350=float(err[lo].mean()) if lo.any() else None,
                ge_bias=float(err[hi].mean()) if hi.any() else None,
                tail_share=float(hi.mean()),
                band=[float(se[:, sl].mean()) for sl in (slice(0, 24), slice(24, 48), slice(48, 72))])


def deviance(pred: np.ndarray, truth: np.ndarray, power: float = P) -> np.ndarray:
    """Tweedie unit deviance for 1 < p < 2, per cell."""
    y, mu = truth, np.clip(pred, 1., 3000.)
    a = y ** (2. - power) / ((1. - power) * (2. - power))
    b = y * mu ** (1. - power) / (1. - power)
    c = mu ** (2. - power) / (2. - power)
    return 2. * (a - b + c)


def loss_curve(name: str, step: int = 6) -> dict:
    frame = pd.read_csv(RUN / name)
    keep = sorted(set(list(range(0, len(frame), step)) + [len(frame) - 1]))
    sub = frame.iloc[keep]
    return dict(rounds=[int(v) for v in sub['round']],
                train=[r(v, 1) for v in sub['train']], valid=[r(v, 1) for v in sub['valid']],
                final=dict(train=r(frame['train'].iloc[-1], 1), valid=r(frame['valid'].iloc[-1], 1),
                           best_valid=r(frame['valid'].min(), 1),
                           best_round=int(frame['round'].iloc[int(frame['valid'].idxmin())])))


def model_base_scores() -> dict:
    import xgboost as xgb
    out = {}
    for path in sorted(RUN.glob('model_*.ubj')):
        b = xgb.Booster()
        b.load_model(path)
        cfg = json.loads(b.save_config())
        lp = cfg['learner']['learner_model_param']
        out[path.name] = dict(base_score=float(lp['base_score']), num_feature=int(lp['num_feature']),
                              num_target=int(lp['num_target']), rounds=int(b.num_boosted_rounds()))
    return out


def main() -> None:
    names = json.loads((CACHE / 'names.json').read_text())
    selected = json.loads(MANIFEST.read_text())['selected']
    ci = [i for i, n in enumerate(names['constant']) if n in selected]
    ei = [i for i, n in enumerate(names['varying']) if n in selected]
    assert len(ci) == 21 and len(ei) == 9, (len(ci), len(ei))
    assert len(names['constant']) == 90 and len(names['varying']) == 20

    f1_bundle = dict(np.load(CACHE / 'f1.npz'))
    f2_bundle = dict(np.load(CACHE / 'official.npz'))
    series = pd.read_parquet(CACHE / 'speed.parquet')['speed'].sort_index()

    def truth_of(origins: pd.DatetimeIndex) -> np.ndarray:
        t = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
        return series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, H)

    z = np.load(RUN / 'forecast.npz')
    origins = pd.DatetimeIndex(z['origins_ns'], tz='UTC')
    tweedie = np.asarray(z['predictions'], dtype=np.float64)
    control = np.asarray(np.load(RUN / 'forecast_control.npz')['predictions'], dtype=np.float64)
    truth = truth_of(origins)
    assert not np.isnan(truth).any()
    n_f1 = int((origins < F2_END).sum())
    assert n_f1 == 2137 and len(origins) == 4970
    assert np.array_equal(origins[:n_f1].asi8, pd.DatetimeIndex(f1_bundle['origins'], tz='UTC').asi8)
    assert np.array_equal(origins[n_f1:].asi8, pd.DatetimeIndex(f2_bundle['origins'], tz='UTC').asi8)

    folds = {'f1': slice(0, n_f1), 'f2': slice(n_f1, None)}
    scores = {'control': {}, 'tweedie': {}}
    preds = {'control': control, 'tweedie': tweedie}
    for arm in scores:
        for fold, sl in folds.items():
            s = score(preds[arm][sl], truth[sl])
            ref = LOG[arm][fold]
            assert s['n'] == ref['n']
            assert abs(s['mse'] - ref['mse']) < 1e-6, (arm, fold, s['mse'], ref['mse'])
            assert abs(s['ge'] - ref['ge']) < 1e-6
            assert abs(s['bias'] - ref['bias']) < 1e-6
            assert abs(s['lt350'] - ref['lt350']) < 1e-6
            for a, b in zip(s['band'], ref['band']):
                assert abs(a - b) < 1e-6, (arm, fold, a, b)
            scores[arm][fold] = s
        scores[arm]['pooled'] = score(preds[arm], truth)

    # internal (validation-window) fold: re-read the archived predictions and re-score
    internal = {}
    for arm in ('control', 'tweedie'):
        npz = np.load(RUN / f'{arm}_internal_predictions.npz')
        o = pd.DatetimeIndex(npz['origins'], tz='UTC')
        assert ((o >= VAL[0]) & (o <= VAL[1])).all() and len(o) == 648
        s = score(npz['pred'], truth_of(o))
        ref = LOG[arm]['internal']
        assert abs(s['mse'] - ref['mse']) < 1e-6, (arm, s['mse'], ref['mse'])
        assert abs(s['ge'] - ref['ge']) < 1e-6
        internal[arm] = s
    scores['control']['internal'] = internal['control']
    scores['tweedie']['internal'] = internal['tweedie']

    # Tweedie deviance vs MSE: which prediction wins which metric
    obj = {}
    for fold, sl in folds.items():
        obj[fold] = {arm: dict(mse=float(((preds[arm][sl] - truth[sl]) ** 2).mean()),
                               dev=float(deviance(preds[arm][sl], truth[sl]).mean()))
                     for arm in preds}

    # per-band objective: MSE and Tweedie deviance by truth band
    band_rows = []
    for fold, sl in folds.items():
        for lo, hi in ((0, 350), (350, 550), (550, 3000)):
            m = (truth[sl] >= lo) & (truth[sl] < hi)
            row = dict(fold=fold, lo=lo, hi=hi, n=int(m.sum()))
            for arm in preds:
                row[arm] = dict(
                    mse=float(((preds[arm][sl][m] - truth[sl][m]) ** 2).mean()) if m.any() else None,
                    dev=float(deviance(preds[arm][sl], truth[sl])[m].mean()) if m.any() else None,
                    bias=float((preds[arm][sl][m] - truth[sl][m]).mean()) if m.any() else None)
            band_rows.append(row)

    # gradient magnitude by prediction level: same data, two objectives
    bins = [(0, 350), (350, 450), (450, 550), (550, 650), (650, 800), (800, 3000)]
    grad = []
    for lo, hi in bins:
        row = dict(lo=lo, hi=hi)
        for arm in preds:
            mu = np.clip(preds[arm], 1., 3000.)
            m = (mu >= lo) & (mu < hi)
            row[arm] = dict(
                n=int(m.sum()),
                mse=float(((mu[m] - truth[m]) ** 2).mean()) if m.any() else None,
                abs_err=float(np.abs(mu[m] - truth[m]).mean()) if m.any() else None,
                g=float(np.mean(np.abs((S / mu[m]) ** P * (mu[m] - truth[m])))) if m.any() else None,
            )
        # control gradient with its own 4x point weight, normalised per horizon like the code
        w = np.where(truth >= U, 4., 1.)
        wn = w / w.mean(axis=0)
        mu = np.clip(control, 1., 3000.)
        m = (mu >= lo) & (mu < hi)
        row['control_weighted'] = dict(
            g=float(np.mean(np.abs(wn[m] * (mu[m] - truth[m])))) if m.any() else None,
            w_mean=float(wn[m].mean()) if m.any() else None)
        grad.append(row)

    # per-horizon tail share and the point4 normaliser, computed on the TRAINING labels
    train = dict(np.load(CACHE / 'train.npz'))
    train_origins = pd.DatetimeIndex(train['origins'], tz='UTC')
    y_train = truth_of(train_origins)
    tail_share = (y_train >= U).mean(axis=0)
    norm = 1. + 3. * tail_share
    weights = dict(h=list(range(1, H + 1)),
                   n_train=int(len(train_origins)),
                   tail=[r(float(v), 4) for v in tail_share],
                   norm=[r(float(v), 4) for v in norm],
                   w_hi=[r(float(v), 4) for v in 4. / norm],
                   w_lo=[r(float(v), 4) for v in 1. / norm],
                   grid_tail=r(float((truth >= U).mean()), 5))

    # two worked examples: best improvement and worst tail regression
    err_c = ((control - truth) ** 2)
    err_t = ((tweedie - truth) ** 2)
    mse_o = np.stack([err_c.mean(1), err_t.mean(1)], 1)
    hi_cells = (truth >= U).sum(1)
    gain_idx = int(np.argmax(np.where(hi_cells >= 6, mse_o[:, 0] - mse_o[:, 1], -np.inf)))
    tail_idx = int(np.argmin(np.where(hi_cells >= 6, mse_o[:, 0] - mse_o[:, 1], np.inf)))

    def example(idx: int, tag: str) -> dict:
        return dict(
            tag=tag, origin=str(origins[idx]), index=int(idx), fold='F1' if idx < n_f1 else 'F2',
            truth=[r(v, 1) for v in truth[idx]], tweedie=[r(v, 1) for v in tweedie[idx]],
            control=[r(v, 1) for v in control[idx]],
            mse=dict(control=r(err_c[idx].mean(), 1), tweedie=r(err_t[idx].mean(), 1)),
            tail_cells=int(hi_cells[idx]),
        )

    examples = [example(gain_idx, 'gain'), example(tail_idx, 'tail')]

    # how one prediction is assembled from 900 rounds (block 0 model of the F1 stage)
    import xgboost as xgb
    pick = gain_idx if gain_idx < n_f1 else tail_idx
    j = pick if pick < n_f1 else None
    bundle = f1_bundle if j is not None else f2_bundle
    k = pick if j is not None else pick - n_f1
    x = np.hstack([bundle['C'][k:k + 1, ci],
                   bundle['E'][k:k + 1, 0:24, :][:, :, ei].reshape(1, -1)]).astype(np.float32)
    dm = xgb.DMatrix(x)
    stage = 'f1' if j is not None else 'final'
    booster = xgb.Booster()
    booster.load_model(RUN / f'model_tweedie_{stage}_b0.ubj')
    base = float(json.loads(booster.save_config())['learner']['learner_model_param']['base_score'])
    margin = np.asarray(booster.predict(dm, output_margin=True))[0]
    cum = []                                   # cumulative margin after each round, h = 1 and 24
    for k in range(1, 901):
        m = np.asarray(booster.predict(dm, output_margin=True, iteration_range=(0, k)))[0]
        cum.append((float(m[0]), float(m[23])))
    deltas = [(cum[0][0] - base, 1)]           # first tree = prediction after 1 round − base score
    deltas += [(cum[k][0] - cum[k - 1][0], k + 1) for k in range(1, len(cum))]
    ranked = sorted(deltas, key=lambda q: -abs(q[0]))[:5]
    rounds = dict(stage=stage, origin=str(origins[pick]), base_score=r(base, 3),
                  cumulative=dict(h1=r(cum[-1][0], 2), h24=r(cum[-1][1], 2)),
                  h1_after_1=r(cum[0][0], 3),
                  trajectory=[dict(round=k + 1, h1=r(cum[k][0], 2), h24=r(cum[k][1], 2))
                              for k in range(0, 900, 25)] + [dict(round=900, h1=r(cum[-1][0], 2),
                                                                 h24=r(cum[-1][1], 2))],
                  top=[dict(round=int(k), delta=r(d, 3)) for d, k in ranked],
                  largest=r(max(deltas, key=lambda q: abs(q[0]))[0], 3),
                  tail_sum=r(sum(d for d, _ in deltas[450:]), 2))

    # distribution of the two predictions' disagreement
    diff = (tweedie - control).ravel()
    edges = [-120, -60, -40, -20, -10, -5, 5, 10, 20, 40, 60, 120]
    counts, _ = np.histogram(diff, bins=edges)
    hist = dict(edges=edges, counts=[int(c) for c in counts],
                share=[r(float(c) / diff.size, 5) for c in counts],
                mean=r(diff.mean(), 2), sd=r(diff.std(), 2),
                share_beyond_20=r(float((np.abs(diff) > 20).mean()), 4))

    # a few real cells: how the two objectives score the same (y, mu) pair
    cells = []
    for fold, sl in folds.items():
        ec = np.abs(control[sl] - truth[sl]).ravel()
        order = np.argsort(-ec)
        tt, cc, ww = truth[sl].ravel(), control[sl].ravel(), tweedie[sl].ravel()
        for label, pos in (('상위 0.1%', int(len(order) * 0.001)), ('상위 1%', int(len(order) * 0.01)),
                           ('상위 10%', int(len(order) * 0.10)), ('중앙값', int(len(order) * 0.50))):
            k = int(order[pos])
            y, mc, mw = float(tt[k]), float(cc[k]), float(ww[k])
            cells.append(dict(fold=fold, label=label, h=int(k % H) + 1, y=r(y, 1),
                              control=r(mc, 1), tweedie=r(mw, 1),
                              se_c=r((mc - y) ** 2, 0), se_t=r((mw - y) ** 2, 0),
                              dev_c=r(float(deviance(np.array([mc]), np.array([y]))[0]), 3),
                              dev_t=r(float(deviance(np.array([mw]), np.array([y]))[0]), 3)))

    # the detpool example window (A/B/C) comes from the sibling page's bundle, not from hand typing
    flow = json.loads((ROOT / 'docs/explainers/data/flow.json').read_text())
    sc = flow['model']['scan']
    at = lambda wq: next(q for q in sc if abs(q['w'] - wq) < 1e-9)
    A = at(0.0)['total']
    b1, k1, m1 = (1 - .25) ** 2, .25 ** 2, 2 * .25 * (1 - .25)
    b2, k2, m2 = (1 - .5) ** 2, .5 ** 2, 2 * .5 * (1 - .5)
    r1 = at(0.25)['total'] - b1 * A
    r2 = at(0.5)['total'] - b2 * A
    det = k1 * m2 - m1 * k2
    B, C = (r1 * m2 - m1 * r2) / det, (k1 * r2 - r1 * k2) / det
    pool = dict(origin=flow['origin'], A=r(A, 2), B=r(B, 2), C=r(C, 2), w=flow['model']['w'],
                a1=flow['model']['detail']['a1'], rows=flow['model']['detail']['window_rows'])

    payload = dict(
        run=dict(lane='tweedie', power=P, scale=S, rounds=900, blocks=list(BLOCKS), h=H, u=U,
                 dir=str(RUN.relative_to(ROOT)), fits=str(MANIFEST.relative_to(ROOT)),
                 stages=dict(internal=dict(fit='origin < 2025-12-04 − 144h', apply='2025-12-04 ~ 2026-02-22 23:00',
                                           n=648),
                             f1=dict(fit='origin + 144h ≤ 2026-02-26', apply='F1 격자', n=2137),
                             final=dict(fit='origin + 144h ≤ 2026-05-29', apply='F2 격자', n=2833)),
                 discipline='internal window only for any selection; F1/F2 scored once after fitting'),
        recipe=dict(params=dict(json.loads(MANIFEST.read_text())['recipe']['xgboost'])),
        models=model_base_scores(),
        pool=pool,
        cells=cells,
        scores=scores,
        objective=obj,
        bands=band_rows,
        gradient=grad,
        weights=weights,
        examples=examples,
        rounds=rounds,
        hist=hist,
        features=dict(n_const_all=len(names['constant']), n_vary_all=len(names['varying']),
                      n_selected=len(selected), const=[names['constant'][i] for i in ci],
                      vary=[names['varying'][i] for i in ei], wide=int(len(ci) + 24 * len(ei))),
        loss=dict(tweedie=[loss_curve(f'losses_tweedie_internal_b{b}.csv') for b in range(3)],
                  control=[loss_curve(f'losses_control_internal_b{b}.csv') for b in range(3)],
                  tweedie_f1=loss_curve('losses_tweedie_f1_b0.csv'),
                  control_f1=loss_curve('losses_control_f1_b0.csv')),
        provenance=dict(inputs=json.loads((RUN / 'results.json').read_text())['inputs'],
                        manifest_sha=json.loads((RUN / 'results.json').read_text())['frozen']['manifest_sha256']),
    )
    payload['recipe']['params'] = {k: payload['recipe']['params'][k] for k in
                                   sorted(payload['recipe']['params'])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')
    print('example gain', examples[0]['origin'], examples[0]['mse'], '| tail', examples[1]['origin'],
          examples[1]['mse'])
    print('pooled', {a: {k: r(v, 2) for k, v in scores[a]['pooled'].items() if k in ('mse', 'ge', 'bias')}
                     for a in ('control', 'tweedie')})
    print('objective', json.dumps(obj))


if __name__ == '__main__':
    main()
