"""HUXT-LHX head on roll6 (pre-registered 2026-10-18): same head as gate_rl.py (E1527), cross-fit over six folds.

roll3 folds 1-3: control E1508, LHX E1506; extra folds 4-6: control E1529, LHX E1530 (roll6.py layout).
Fold k coefficients are fit on the other five folds (same seed). Official: fit on all six folds with targets < 2026-05-29,
champion = E1149 official same seed, LHX = E1506 official same seed. Writes roll3/ (k=1..3) and roll3x/ (k=4..6).
"""
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
from expos import evaluate
from experiments.roll3_common import write as write3
from experiments.solar_lhx72_omni2_20261018 import channels as C, roll6
from experiments.solar_lhx72_omni2_20261018.gate_level import paths, frame, RUNS, CUTOFF, OFFICIAL
from experiments.solar_lhx72_omni2_20261018.gate_rl import design, fit, apply

CTRL = {1: 'E1508', 2: 'E1508', 3: 'E1508', 4: 'E1529', 5: 'E1529', 6: 'E1529'}
LHX = {1: 'E1506', 2: 'E1506', 3: 'E1506', 4: 'E1530', 5: 'E1530', 6: 'E1530'}
CHAMP = 'E1149'


def main():
    out = Path(os.environ['EXPOS_OUT']); out.mkdir(parents=True, exist_ok=True)
    filled, observed = evaluate._ace()
    y_obs = filled.where(observed)
    hux = C.Huxt()
    chans = {}
    for k in range(1, 7):
        a, b = roll6.bounds(k)
        org = pd.date_range(a, b, freq='h')
        chans[k] = (org, hux.channels(org, y_obs))
        assert np.isfinite(chans[k][1]['huxt_raw']).all(), k
    chans['off'] = (OFFICIAL, hux.channels(OFFICIAL, y_obs))
    for seed in (0, 1, 2):
        roll = {}
        for k in range(1, 7):
            org, ch = chans[k]
            sub = 'roll3' if k <= 3 else 'roll3x'
            c = paths(RUNS / CTRL[k] / sub / f'f{k}_s{seed}.parquet', org)
            l = paths(RUNS / LHX[k] / 'lhx' / sub / f'f{k}_s{seed}.parquet', org)
            y = np.stack([y_obs.reindex(org + pd.Timedelta(hours=h)).to_numpy() for h in range(1, 73)], 1)
            roll[k] = (org, c, design(c, ch, l), y)
        coef = {}
        for k in roll:
            o = [j for j in roll if j != k]
            beta = fit(np.concatenate([roll[j][2] for j in o]), np.concatenate([roll[j][3] - roll[j][1] for j in o]))
            coef[k] = beta.tolist()
            org, c, X, _ = roll[k]
            pred = c + apply(X, beta)
            (write3(out, k, seed, org, pred) if k <= 3 else roll6.write(out, k, seed, org, pred))
        ys = []
        for k in sorted(roll):
            org, c, _, y = roll[k]
            tt = org.values[:, None] + (np.arange(1, 73) * np.timedelta64(1, 'h'))[None, :]
            ys.append(np.where(tt < CUTOFF.tz_convert(None).to_datetime64(), y - c, np.nan))
        beta = fit(np.concatenate([roll[k][2] for k in sorted(roll)]), np.concatenate(ys))
        org, ch = chans['off']
        c = paths(RUNS / CHAMP / f'predictions_official_s{seed}.parquet', org)
        l = paths(RUNS / 'E1506' / 'lhx' / f'official_s{seed}.parquet', org)
        fin = c + apply(design(c, ch, l), beta)
        assert np.isfinite(fin).all()
        frame(org, fin, out / f'predictions_official_s{seed}.parquet')
        (out / f'receipt_s{seed}.json').write_text(json.dumps(dict(seed=seed, ctrl=CTRL, lhx=LHX, champion=CHAMP, coef_roll6=coef,
            coef_official=beta.tolist(), code={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, C.__file__, roll6.__file__)}), indent=1))
        print('RL6 SEED DONE', seed, 'b_huxt@24,48,72', [round(beta[h, 0], 3) for h in (23, 47, 71)], flush=True)


if __name__ == '__main__':
    main()
