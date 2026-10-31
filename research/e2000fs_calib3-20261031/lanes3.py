"""e2000fs calib3 lanes (2026-10-31, round 5).

All lanes reuse the frozen parent recipe (3x24 vector-leaf XGBoost, point4 objective,
origin+144h<=cutoff maturity mask) and change exactly one mechanism each. Every lane fits
F1 and F2 unconditionally and writes the 357,840-row official forecast; no internal gate.

Lanes
  extrap  : LTBoost / PILOT style hybrid -- weighted ridge trend + tree on the residual
            (CIKM 2024 10.1145/3627673.3679527; RaFFLE 2502.10185). Trees cannot
            extrapolate, which is the measured >=550 shortfall (-40 km/s bias).
  restack : two-stage stack -- point4 level head, then an UNWEIGHTED-MSE residual model
            given the out-of-fold level as a feature. Corrects the dispersion that the
            4x high-target weights inject (bias +63 below 350, -40 above 550).
  affine  : MOS/spread recalibration -- per-horizon affine map fitted on out-of-fold
            predictions (Sansom et al. recalibration family; Veenhuis 2013 spread-skill).
  bag3    : 3-seed equal-weight averaging of the unchanged level head (variance reduction;
            static equal weights beat adaptive weighting at long leads, ECML AALTD 2025).
  era     : era-wise adversarial reweighting (Group DRO / era splitting, 2309.14496) to
            make the fit less dependent on one sub-period -> aims at internal->official transfer.
  gpd     : peaks-over-threshold tail replacement -- where P(y>=550) >= 0.5 the level head is
            replaced by 550 + exp(excess head), otherwise untouched (gbex, Extremes 2024).
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1' if _k != 'OMP_NUM_THREADS' else '3')
import argparse, hashlib, importlib.util, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/home/t-lab01/sunrun')
H = 72
HPO = 550.
F1_CUT = pd.Timestamp('2026-02-26', tz='UTC')
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')
U = 550.


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


tw = imported('calib_tweedie', ROOT / 'research/e2000fs_calib-20261030/tweedie_lane.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wj(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=1, default=str) + '\n')


# ---------------------------------------------------------------- fitting primitives
def point_w(t):
    return np.where(t >= HPO, 4., 1.)


def fit_multi(x_tr, y_tr, params, rounds, objective, log_path=None, w_row=None):
    """Multi-output fit with a caller-supplied objective factory. Returns the model."""
    dt = xgb.DMatrix(x_tr, label=y_tr.astype(np.float32), nthread=3)
    xv = x_tr[-240:] if len(x_tr) > 240 else x_tr
    yv = y_tr[-240:] if len(y_tr) > 240 else y_tr
    dv = xgb.DMatrix(xv, label=yv.astype(np.float32), nthread=3)
    hist = {'round': [], 'train': [], 'valid': []}

    class Rec(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            hist['round'].append(epoch + 1)
            hist['train'].append(evals_log['train']['rmse'][-1] ** 2)
            hist['valid'].append(evals_log['validation']['rmse'][-1] ** 2)
            return False

    model = xgb.train(params, dt, num_boost_round=rounds, obj=objective(dt, w_row),
                      evals=[(dt, 'train'), (dv, 'validation')],
                      custom_metric=None, callbacks=[Rec()], verbose_eval=False)
    if log_path:
        pd.DataFrame(hist).to_csv(log_path, index=False)
    return model, hist


def obj_point4(dt, w_row=None):
    def f(pred, matrix):
        t = matrix.get_label().reshape(pred.shape).astype(np.float64)
        w = point_w(t)
        return ((pred - t) * w).astype(np.float32), w.astype(np.float32)
    return f


def obj_point4_w(dt, w_row=None):
    """point4 objective (4x above 550) multiplied by a per-row group weight."""
    assert w_row is not None, 'obj_point4_w needs per-row weights'
    w_row = np.asarray(w_row, dtype=np.float64)
    def f(pred, matrix):
        t = matrix.get_label().reshape(pred.shape).astype(np.float64)
        w = point_w(t) * w_row.reshape(-1, 1)
        return ((pred - t) * w).astype(np.float32), w.astype(np.float32)
    return f


def obj_mse(dt, w_row=None):
    def f(pred, matrix):
        t = matrix.get_label().reshape(pred.shape).astype(np.float64)
        if w_row is None:
            return (pred - t).astype(np.float32), np.ones_like(pred, dtype=np.float32)
        w = np.asarray(w_row, dtype=np.float64).reshape(-1, 1) * np.ones_like(pred)
        return ((pred - t) * w).astype(np.float32), w.astype(np.float32)
    return f


def params_of(over=None):
    recipe = tw.read_json(tw.MANIFEST)['recipe']['xgboost']
    p = {k: v for k, v in recipe.items()
         if k not in ('n_estimators', 'n_jobs', 'random_state', 'base_score', 'objective')}
    p.update(objective='reg:squarederror', tree_method='hist', device='cpu',
             multi_strategy='multi_output_tree', nthread=3, seed=42, eval_metric='rmse', verbosity=0)
    if over:
        p.update(over)
    return p


def level_fit(x_tr, y_tr, rounds, params, out, tag, objective=obj_point4, w_row=None):
    m, _ = tw.fit_block('control', x_tr, y_tr, x_tr[-240:], y_tr[-240:],
                        dict(params, base_score=float(np.nanmean(y_tr))), rounds, 1.5,
                        out / f'losses_{tag}.csv')
    return m


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, smoke=False):
        names, selected, origins, series, y, bundle = tw.load_data()
        blocks = [tw.design(bundle, names, selected, h) for h in (0, 24, 48)]
        f1b = dict(np.load(tw.CACHE / 'f1.npz')); f2b = dict(np.load(tw.CACHE / 'official.npz'))
        off = np.load(tw.OFFICIAL)
        self.official = pd.DatetimeIndex(off['origins_ns'], tz='UTC')
        self.n_f1 = int((self.official < F2_CUT).sum())
        self.o_f1 = pd.DatetimeIndex(f1b['origins'], tz='UTC')
        self.o_f2 = pd.DatetimeIndex(f2b['origins'], tz='UTC')
        assert np.array_equal(self.o_f1.asi8, self.official[:self.n_f1].asi8)
        assert np.array_equal(self.o_f2.asi8, self.official[self.n_f1:].asi8)
        self.off = {'f1': [tw.design(f1b, names, selected, h) for h in (0, 24, 48)],
                    'final': [tw.design(f2b, names, selected, h) for h in (0, 24, 48)]}
        self.blocks = blocks
        self.origins = origins
        self.y = y
        self.series = series
        self.train_bundle = bundle
        self.fit = {'f1': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F1_CUT),
                    'final': np.flatnonzero(origins + pd.Timedelta(hours=144) <= F2_CUT)}
        self.smoke = smoke
        if smoke:
            self.fit = {k: v[-150:] for k, v in self.fit.items()}
            self.o_f1 = self.o_f1[:48]; self.o_f2 = self.o_f2[:48]
            self.off = {'f1': [x[:48] for x in self.off['f1']], 'final': [x[:48] for x in self.off['final']]}
            self.n_f1 = 48

    def truth(self, o):
        t = o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
        return self.series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, H)


def score(pred, y, n_f1):
    e = pred - y; se = e ** 2
    hi = y >= HPO
    f1, f2 = slice(0, n_f1), slice(n_f1, len(y))
    out = {'total': float(se.mean()), 'ge550': float(se[hi].mean()),
           'f1_ge550': float(se[f1][hi[f1]].mean()), 'f2_ge550': float(se[f2][hi[f2]].mean()),
           'f1': float(se[f1].mean()), 'f2': float(se[f2].mean()),
           'bias': float(e.mean()), 'bias_lt350': float(e[y < 350].mean()),
           'bias_ge550': float(e[hi].mean())}
    out['meets_target'] = bool(out['total'] < 3200. and
                               max(out['ge550'], out['f1_ge550'], out['f2_ge550']) < 6000.)
    return out


# ---------------------------------------------------------------- lanes
def standardize(x_tr, x_ap):
    mu = np.nanmean(x_tr, axis=0); sd = np.nanstd(x_tr, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    a = np.nan_to_num((x_tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num((x_ap - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return a, b


def ridge_fit(x, y, lam=30.0):
    """Per-column weighted ridge with intercept; x already standardized."""
    n, d = x.shape
    xa = np.hstack([np.ones((n, 1)), x])
    reg = lam * np.eye(d + 1); reg[0, 0] = 0.0
    coef = np.empty((d + 1, y.shape[1]))
    for h in range(y.shape[1]):
        w = point_w(y[:, h])[:, None]
        a = xa * np.sqrt(w); b = y[:, [h]] * np.sqrt(w)
        coef[:, h] = np.linalg.solve(a.T @ a + reg, a.T @ b).ravel()
    return coef


def lane_extrap(ctx, out, rounds):
    params = params_of()
    p_f1, p_f2 = [], []
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        fit = ctx.fit[stage]
        pred = np.empty((len(o), H))
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[fit, sl]
            x_tr, x_ap = standardize(ctx.blocks[b][fit], ctx.off[stage][b])
            coef = ridge_fit(x_tr, y_tr)
            lin_tr = np.hstack([np.ones((len(x_tr), 1)), x_tr]) @ coef
            lin_ap = np.hstack([np.ones((len(x_ap), 1)), x_ap]) @ coef
            resid = y_tr - lin_tr
            m, _ = fit_multi(x_tr, resid, dict(params, base_score=0.0), rounds,
                             obj_point4, out / f'losses_extrap_resid_{stage}_b{b}.csv')
            pred[:, sl] = lin_ap + m.inplace_predict(x_ap)
        pred = np.clip(pred, 1., 3000.)
        (p_f1 if stage == 'f1' else p_f2).append(pred)
        print(stage, json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p_f1[0], p_f2[0]])


def oof_two_fold(ctx, stage, b, rounds, params, out, tag):
    """Two contiguous folds; returns (oof predictions per fit row, full-apply model preds)."""
    fit = ctx.fit[stage]
    y_tr = ctx.y[fit, b * 24:(b + 1) * 24]
    x_tr = ctx.blocks[b][fit]
    x_ap = ctx.off[stage][b]
    n = len(fit); mid = n // 2
    oof = np.empty_like(y_tr)
    for (a, c, flip) in ((slice(0, mid), slice(mid, n), False), (slice(mid, n), slice(0, mid), True)):
        m, _ = fit_multi(x_tr[a], y_tr[a], dict(params, base_score=float(np.nanmean(y_tr[a]))),
                         rounds, obj_point4, out / f'losses_{tag}_oof{int(mid if not flip else 0)}_{stage}_b{b}.csv')
        oof[c] = m.inplace_predict(x_tr[c])
    full, _ = fit_multi(x_tr, y_tr, dict(params, base_score=float(np.nanmean(y_tr))),
                        rounds, obj_point4, out / f'losses_{tag}_full_{stage}_b{b}.csv')
    return oof, full.inplace_predict(x_ap)


def affine_map(oof, y):
    coef = []
    for h in range(y.shape[1]):
        a = np.vstack([np.ones(len(oof)), oof[:, h]]).T
        beta, *_ = np.linalg.lstsq(a, y[:, h], rcond=None)
        coef.append(beta)
    return np.array(coef)          # (24, 2)


def lane_affine(ctx, out, rounds):
    params = params_of()
    p = {}
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        pred = np.empty((len(o), H)); coefs = []
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[ctx.fit[stage], sl]
            oof, ap = oof_two_fold(ctx, stage, b, rounds, params, out, 'affine')
            c = affine_map(oof, y_tr); coefs.append(c)
            pred[:, sl] = ap * c[:, 1] + c[:, 0]
        pred = np.clip(pred, 1., 3000.)
        p[stage] = pred
        wj(out / f'affine_coef_{stage}.json',
           {f'h{i + 1}': [float(r[1]), float(r[0])] for i, r in enumerate(np.vstack(coefs))})
        print(stage, json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p['f1'], p['final']])


def lane_restack(ctx, out, rounds):
    params = params_of()
    p = {}
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        pred = np.empty((len(o), H))
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[ctx.fit[stage], sl]
            oof, ap = oof_two_fold(ctx, stage, b, rounds, params, out, 'restack')
            x2 = np.hstack([ctx.blocks[b][ctx.fit[stage]], oof.astype(np.float32)])
            x2ap = np.hstack([ctx.off[stage][b], ap.astype(np.float32)])
            m, _ = fit_multi(x2, y_tr - oof, dict(params, base_score=0.0), max(200, rounds // 2),
                             obj_mse, out / f'losses_restack_resid_{stage}_b{b}.csv')
            pred[:, sl] = ap + m.inplace_predict(x2ap)
        pred = np.clip(pred, 1., 3000.)
        p[stage] = pred
        print(stage, json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p['f1'], p['final']])


def lane_bag3(ctx, out, rounds):
    params = params_of()
    p = {}
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        fit = ctx.fit[stage]; pred = np.empty((len(o), H))
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[fit, sl]; x_tr = ctx.blocks[b][fit]; x_ap = ctx.off[stage][b]
            acc = np.zeros((len(o), 24))
            for seed in (42, 43, 44):
                m, _ = fit_multi(x_tr, y_tr, dict(params, base_score=float(np.nanmean(y_tr)), seed=seed),
                                 rounds, obj_point4, out / f'losses_bag3_{stage}_b{b}_s{seed}.csv')
                acc += m.inplace_predict(x_ap)
            pred[:, sl] = acc / 3.
        pred = np.clip(pred, 1., 3000.)
        p[stage] = pred
        print(stage, json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p['f1'], p['final']])


def lane_era(ctx, out, rounds, inner=3, eta=0.5):
    """Group-DRO-style adversarial reweighting over calendar months of the origin."""
    params = params_of()
    p = {}
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        fit = ctx.fit[stage]; pred = np.empty((len(o), H))
        months = ctx.origins[fit].month
        groups = np.unique(months)
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[fit, sl]; x_tr = ctx.blocks[b][fit]; x_ap = ctx.off[stage][b]
            gw = np.ones(len(groups))
            model = None
            for it in range(inner):
                row_w = gw[np.searchsorted(groups, months)]
                row_w = row_w / row_w.mean()
                model, _ = fit_multi(x_tr, y_tr, dict(params, base_score=float(np.nanmean(y_tr))),
                                     rounds, obj_point4_w,
                                     out / f'losses_era_{stage}_b{b}_it{it}.csv', w_row=row_w)
                inpred = model.inplace_predict(x_tr)
                loss = np.array([float(((inpred[months == g] - y_tr[months == g]) ** 2).mean()) for g in groups])
                gw = gw * np.exp(eta * (loss / loss.mean() - 1.0))
                gw = gw / gw.mean()
            pred[:, sl] = model.inplace_predict(x_ap)
            wj(out / f'era_weights_{stage}_b{b}.json', {str(int(g)): float(w) for g, w in zip(groups, gw)})
        pred = np.clip(pred, 1., 3000.)
        p[stage] = pred
        print(stage, json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p['f1'], p['final']])


def lane_gpd(ctx, out, rounds, tau=0.5):
    """Peaks-over-threshold replacement: level everywhere, 550+exp(excess) where p>=tau."""
    params = params_of()
    p = {}
    for stage, o in (('f1', ctx.o_f1), ('final', ctx.o_f2)):
        fit = ctx.fit[stage]; pred = np.empty((len(o), H))
        routed = 0
        for b in range(3):
            sl = slice(b * 24, (b + 1) * 24)
            y_tr = ctx.y[fit, sl]; x_tr = ctx.blocks[b][fit]; x_ap = ctx.off[stage][b]
            lvl, _ = tw.fit_block('control', x_tr, y_tr, x_tr[-240:], y_tr[-240:],
                                  dict(params, base_score=float(np.nanmean(y_tr))), rounds, 1.5,
                                  out / f'losses_gpd_level_{stage}_b{b}.csv')
            ind = (y_tr >= U).astype(np.float32)
            pr, _ = fit_multi(x_tr, ind, dict(params, base_score=float(np.mean(ind))), 400,
                              obj_mse, out / f'losses_gpd_prob_{stage}_b{b}.csv')
            ev = (y_tr >= U).any(axis=1)
            z_tr = np.log(np.maximum(y_tr[ev] - U, 1.0))
            ez, _ = fit_multi(x_tr[ev], z_tr, dict(params, base_score=float(np.mean(z_tr))), 400,
                              obj_mse, out / f'losses_gpd_excess_{stage}_b{b}.csv')
            L = lvl.inplace_predict(x_ap).astype(np.float64)
            P = np.clip(pr.inplace_predict(x_ap), 0., 1.)
            Z = np.clip(ez.inplace_predict(x_ap), 0., np.log(3000. - U))
            route = P >= tau
            pred[:, sl] = np.where(route, U + np.exp(Z), L)
            routed += int(route.sum())
        pred = np.clip(pred, 1., 3000.)
        p[stage] = pred
        print(stage, 'routed_share_pct', float(100 * routed / pred.size),
              json.dumps(score(pred, ctx.truth(o), len(o))), flush=True)
    return np.vstack([p['f1'], p['final']])


LANES = {'extrap': lane_extrap, 'restack': lane_restack, 'affine': lane_affine,
         'bag3': lane_bag3, 'era': lane_era, 'gpd': lane_gpd}
DISCIPLINE = {
    'extrap': 'weighted ridge trend + tree on residual (LTBoost/PILOT); point4 objective unchanged on the residual',
    'restack': 'second-stage residual model with unweighted MSE over out-of-fold level predictions',
    'affine': 'per-horizon affine recalibration fitted on out-of-fold predictions only',
    'bag3': 'three-seed equal-weight average of the unchanged level head',
    'era': 'three inner rounds of month-wise adversarial reweighting on the point4 objective',
    'gpd': 'where P(y>=550)>=0.5 replace the level with 550+exp(log-excess head); no uplift elsewhere',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True, choices=sorted(LANES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=900)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = Ctx(smoke=a.smoke)
    rounds = 40 if a.smoke else a.rounds
    pred = LANES[a.lane](ctx, out, rounds)
    o_eval = ctx.o_f1.append(ctx.o_f2)
    y = ctx.truth(o_eval)
    res = {'lane': a.lane, 'rounds': rounds, 'smoke': a.smoke,
           'discipline': DISCIPLINE[a.lane],
           'results': {'official': score(pred, y, ctx.n_f1)},
           'inputs': {str(tw.OFFICIAL): sha(tw.OFFICIAL)}}
    wj(out / 'results.json', res)
    np.savez_compressed(out / 'forecast.npz', origins_ns=o_eval.asi8, predictions=pred.astype(np.float32))
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(o_eval.strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                          'horizon_hours': np.tile(np.arange(1, H + 1), len(o_eval)),
                          'pred_kms': pred.ravel()})
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(json.dumps(res['results'], indent=1))
    print(f'elapsed {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
