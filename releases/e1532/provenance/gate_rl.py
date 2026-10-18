"""HUXT-LHX head (expos): champion-anchored residual head with two propagation channels.

  r_h = y_h - champ_h ~ b1_h (HUXt_h - champ_h) + b2_h bias + b3_h (mean72 LHX - mean72 champ) + c_h
HUXt_h  : E1466 causal TD-HUXt (GONG SW-012 -> PFSS/WSA) Earth trajectory issued at the origin (channels.Huxt).
bias    : mean(ACE - HUXt nowcast) over the last 72 available ACE hours (obs + 2h <= origin).
LHX     : E1506 OMNI2 long-history expert path (same seed).
Ridge per horizon on +-3h pooled rows (standardised, lambda 1e-2 * n), fit cross-fold on roll3 (control E1508, same seed);
official: fit on all folds with targets < 2026-05-29, champion = E1149 official same seed.
"""
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
from expos import evaluate
from expos.roll3 import FOLDS, fold_bounds
from experiments.roll3_common import write
from experiments.solar_lhx72_omni2_20261018 import channels as C
from experiments.solar_lhx72_omni2_20261018.gate_level import paths, frame, RUNS, CUTOFF, OFFICIAL

CONTROL, CHAMP, LHX = 'E1508', 'E1149', 'E1506'
LAM = 1e-2


def design(champ, ch, lhx):
    raw = ch['huxt_raw']; ok = np.isfinite(raw)
    f = [np.where(ok, raw - champ, 0), np.broadcast_to(np.nan_to_num(ch['huxt_bias'])[:, None], champ.shape),
         np.broadcast_to((lhx.mean(1) - champ.mean(1))[:, None], champ.shape), np.ones_like(champ)]
    return np.stack(f, -1)


def fit(X, r):
    beta = []
    for h in range(72):
        sl = slice(max(0, h - 3), min(72, h + 4))
        A = X[:, sl].reshape(-1, X.shape[-1]); b = r[:, sl].ravel(); k = np.isfinite(b)
        A, b = A[k], b[k]
        sd = A.std(0) + 1e-6; sd[-1] = 1
        pen = LAM * len(b) * np.diag(np.r_[np.ones(A.shape[1] - 1), 0])
        beta.append(np.linalg.solve((A / sd).T @ (A / sd) + pen, (A / sd).T @ b) / sd)
    return np.array(beta)


def apply(X, beta):
    return np.einsum('nhk,hk->nh', X, beta)


def main():
    out = Path(os.environ['EXPOS_OUT']); out.mkdir(parents=True, exist_ok=True)
    filled, observed = evaluate._ace()
    y_obs = filled.where(observed)
    hux = C.Huxt()
    chans = {}
    for k in sorted(FOLDS):
        a, b = fold_bounds(k)
        org = pd.date_range(a, b, freq='h')
        chans[k] = (org, hux.channels(org, y_obs))
    chans['off'] = (OFFICIAL, hux.channels(OFFICIAL, y_obs))
    for key, (org, ch) in chans.items():
        assert np.isfinite(ch['huxt_raw']).all(), key
    for seed in (0, 1, 2):
        roll = {}
        for k in sorted(FOLDS):
            org, ch = chans[k]
            c = paths(RUNS / CONTROL / 'roll3' / f'f{k}_s{seed}.parquet', org)
            l = paths(RUNS / LHX / 'lhx' / 'roll3' / f'f{k}_s{seed}.parquet', org)
            y = np.stack([y_obs.reindex(org + pd.Timedelta(hours=h)).to_numpy() for h in range(1, 73)], 1)
            roll[k] = (org, c, design(c, ch, l), y)
        coef = {}
        for k in roll:
            o = [j for j in roll if j != k]
            beta = fit(np.concatenate([roll[j][2] for j in o]), np.concatenate([roll[j][3] - roll[j][1] for j in o]))
            coef[k] = beta.tolist()
            org, c, X, _ = roll[k]
            write(out, k, seed, org, c + apply(X, beta))
        ys = []
        for k in sorted(roll):
            org, c, _, y = roll[k]
            tt = org.values[:, None] + (np.arange(1, 73) * np.timedelta64(1, 'h'))[None, :]
            ys.append(np.where(tt < CUTOFF.tz_convert(None).to_datetime64(), y - c, np.nan))
        beta = fit(np.concatenate([roll[k][2] for k in roll]), np.concatenate(ys))
        org, ch = chans['off']
        c = paths(RUNS / CHAMP / f'predictions_official_s{seed}.parquet', org)
        l = paths(RUNS / LHX / 'lhx' / f'official_s{seed}.parquet', org)
        fin = c + apply(design(c, ch, l), beta)
        assert np.isfinite(fin).all()
        frame(org, fin, out / f'predictions_official_s{seed}.parquet')
        (out / f'receipt_s{seed}.json').write_text(json.dumps(dict(seed=seed, control=CONTROL, champion=CHAMP, lhx=LHX,
            huxt='E1466 lead_fields.h5 -> cache/huxt_earth.npz', coef_roll3=coef, coef_official=beta.tolist(),
            code={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, C.__file__)}), indent=1))
        print('RL SEED DONE', seed, 'b_huxt@24,48,72', [round(beta[h, 0], 3) for h in (23, 47, 71)], flush=True)


if __name__ == '__main__':
    main()
