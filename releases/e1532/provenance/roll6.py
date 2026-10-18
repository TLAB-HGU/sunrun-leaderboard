"""roll6 (user decision 2026-10-18, option 1 after E1527): roll3 + three earlier folds of the same layout.

PRE-REGISTERED before any extra-fold result was seen:
  extra folds F4 2024-06-01..09-30, F5 2024-10-01..2025-01-31, F6 2025-02-01..2025-05-31 23:00 (hourly origins x 72h),
  cutoff = fold start - 72h. Files: $EXPOS_OUT/roll3x/f{k}_s{seed}.parquet (k = 4, 5, 6); roll3 folds stay in roll3/.
  Pass (same statistics as protected expos/roll3.py, applied to 6 folds): pooled relative MSE of seed-mean predictions
  <= -1%, 72h-block bootstrap (2000 draws, seed 0) 95% upper bound < 0, >= 4/6 folds improved, pooled event F1 (mean of
  per-seed F1) >= control, >= 3 seeds.
  Control = E1508 rule (official 79-day alpha calibration) on all six folds.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from expos import core, evaluate, roll3

EXTRA = {4: ('2024-06-01', '2024-09-30 23:00'), 5: ('2024-10-01', '2025-01-31 23:00'), 6: ('2025-02-01', '2025-05-31 23:00')}
FILE = re.compile(r'f(\d+)_s(\d+)\.parquet$')


def bounds(k):
    if k in roll3.FOLDS:
        return roll3.fold_bounds(k)
    a, b = EXTRA[k]
    return pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')


def cutoff(k):
    return bounds(k)[0] - pd.Timedelta(hours=72)


def folds_extra():
    for k in sorted(EXTRA):
        a, b = bounds(k)
        yield k, cutoff(k), pd.date_range(a, b, freq='h')


def write(out, k, seed, origins, paths):
    paths = np.asarray(paths, dtype=float)
    assert paths.shape == (len(origins), 72) and np.isfinite(paths).all()
    d = Path(out) / 'roll3x'; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'origin_last_input_utc': np.repeat(pd.DatetimeIndex(origins).strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                  'horizon_hours': np.tile(np.arange(1, 73), len(origins)), 'pred_kms': paths.ravel()}).to_parquet(d / f'f{k}_s{seed}.parquet', index=False)


def load(eid):
    """eid may be 'EA+EB': roll3/ from EA, roll3x/ from EB (a control split across two registered runs)."""
    ea, eb = (eid.split('+') + [eid])[:2]
    files = {}
    for sub, e in (('roll3', ea), ('roll3x', eb)):
        for p in sorted((core.RUNS / e / sub).glob('f*_s*.parquet')):
            m = FILE.search(p.name); files[(int(m.group(1)), int(m.group(2)))] = p
    ks = sorted({k for k, _ in files}); ss = sorted({s for _, s in files})
    assert ks == [1, 2, 3, 4, 5, 6], (eid, ks)
    assert all((k, s) in files for k in ks for s in ss), eid
    filled, observed = evaluate._ace()
    frames = []
    for k in ks:
        a, b = bounds(k); base = None
        for s in ss:
            df = evaluate._load(files[(k, s)]); df = df[df.origin.between(a, b)]
            assert df.origin.nunique() == len(pd.date_range(a, b, freq='h')), (eid, k, s)
            df = df.sort_values(['origin', 'h']).rename(columns={'y_pred': f's{s}'}).reset_index(drop=True)
            base = df if base is None else base.merge(df, on=['origin', 'h'], validate='one_to_one')
        t = pd.DatetimeIndex(base.origin + pd.to_timedelta(base.h, unit='h'))
        base['y'] = filled.reindex(t).to_numpy(); base['obs'] = observed.reindex(t).fillna(False).to_numpy(); base['fold'] = k
        frames.append(base)
    return pd.concat(frames, ignore_index=True), ss


def compare(eid, control):
    cand, cs = load(eid); ref, rs = load(control)
    m = cand.merge(ref, on=['origin', 'h', 'fold'], suffixes=('', '_ref'), validate='one_to_one')
    q = m[m.obs]
    cm = q[[f's{s}' for s in cs]].mean(1); rm = q[[f's{s}_ref' for s in rs]].mean(1)
    se_c, se_r = ((cm - q.y) ** 2).to_numpy(), ((rm - q.y) ** 2).to_numpy()
    rel, draws = roll3.bootstrap_rel(se_c, se_r, roll3._blocks(q))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    by = {int(k): float(se_c[(q.fold == k).to_numpy()].sum() / se_r[(q.fold == k).to_numpy()].sum() - 1) for k in range(1, 7)}
    f1c, f1r = np.mean(list(roll3._f1(cand, cs).values())), np.mean(list(roll3._f1(ref, rs).values()))
    nimp = sum(v < 0 for v in by.values())
    ok = rel <= -0.01 and hi < 0 and nimp >= 4 and f1c >= f1r and len(cs) >= 3
    line = (f'R6| E={eid} base={control} folds=1..6 seeds={",".join(map(str, cs))} d_pool={rel:+.4%} ci=[{lo:+.4%},{hi:+.4%}] '
            f'folds_better={nimp}/6 f1={f1c:.3f}/{f1r:.3f} n_blocks={len(np.unique(roll3._blocks(q)))} pass={"Y" if ok else "n"}')
    return dict(line=line, pooled_rel=rel, ci95=[lo, hi], rel_by_fold=by, f1=[f1c, f1r], passed=ok)


if __name__ == '__main__':
    import sys, json
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1, default=float))
