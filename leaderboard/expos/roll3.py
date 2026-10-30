"""Rolling 3-fold x 3-seed evaluation (user decision 2026-10-17, ROLL3:20261017): the selection standard.

Three chronological 4-month folds over the year before the official window (all preofficial). For fold k the runner
trains with cutoff = fold_start - 72h (targets <= cutoff) and predicts every hourly fold origin x 72h, once per seed,
writing $EXPOS_OUT/roll3/f{k}_s{seed}.parquet (columns origin_last_input_utc, horizon_hours, pred_kms), k = 1, 2, 3.
Helpers for runners: experiments/roll3_common.py.

A candidate is compared with a control run of the champion recipe on the same folds and seeds (paired). It passes when
  - pooled relative MSE of seed-mean predictions over all fold rows <= -1%,
  - the 95% upper bound of a 72h-block bootstrap of that pooled relative difference is < 0,
  - seed-mean MSE improves in >= 2 of 3 folds,
  - pooled unique-event F1 (seed mean) >= the control's,
  - >= 3 seeds and the seeds are not identical.
The official window is not scored here; lb-eval is a one-time confirmation after a roll3 pass.
"""
import re

import numpy as np
import pandas as pd

from . import core, evaluate

FOLDS = {1: ("2025-06-01", "2025-09-30 23:00"), 2: ("2025-10-01", "2026-01-31 23:00"), 3: ("2026-02-01", "2026-05-28 23:00")}
GAP_HOURS = 72
POOLED_MAX, MIN_FOLDS, MIN_SEEDS, BOOT, BLOCK = -0.01, 2, 3, 2000, 72
FILE = re.compile(r"f(\d+)_s(\d+)\.parquet$")


def fold_bounds(k):
    a, b = FOLDS[k]
    return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")


def fold_cutoff(k):
    return fold_bounds(k)[0] - pd.Timedelta(hours=GAP_HOURS)


def _files(eid):
    d = core.RUNS / eid / "roll3"
    out = {}
    for p in sorted(d.glob("f*_s*.parquet")) if d.exists() else []:
        m = FILE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def load(eid):
    """Observed-target rows per fold with one prediction column per seed."""
    files = _files(eid)
    if not files:
        raise core.ExposError(f"{eid}: no roll3/f{{k}}_s{{seed}}.parquet files")
    folds, seeds = sorted({k for k, _ in files}), sorted({s for _, s in files})
    if folds != sorted(FOLDS):
        raise core.ExposError(f"{eid}: roll3 needs folds {sorted(FOLDS)}, got {folds}")
    missing = [(k, s) for k in folds for s in seeds if (k, s) not in files]
    if missing:
        raise core.ExposError(f"{eid}: every fold needs every seed; missing {missing[:5]}")
    filled, observed = evaluate._ace()
    frames = []
    for k in folds:
        a, b = fold_bounds(k)
        base = None
        for s in seeds:
            df = evaluate._load(files[(k, s)])
            df = df[df["origin"].between(a, b)]
            want = pd.date_range(a, b, freq="h")
            if df["origin"].nunique() != len(want):
                raise core.ExposError(f"{eid} f{k}_s{s}: needs all {len(want)} hourly fold origins, got {df['origin'].nunique()}")
            df = df.sort_values(["origin", "h"]).rename(columns={"y_pred": f"s{s}"}).reset_index(drop=True)
            base = df if base is None else base.merge(df, on=["origin", "h"], how="inner", validate="one_to_one")
        t = pd.DatetimeIndex(base["origin"] + pd.to_timedelta(base["h"], unit="h"))
        base["y"] = filled.reindex(t).to_numpy()
        base["obs"] = observed.reindex(t).fillna(False).to_numpy()
        base["fold"] = k
        frames.append(base)
    return pd.concat(frames, ignore_index=True), seeds


def _f1(df, seeds):
    filled, observed = evaluate._ace()
    out = {}
    for s in seeds:
        tp = fp = fn = 0
        for k in sorted(df["fold"].unique()):
            part = df[df["fold"] == k][["origin", "h", f"s{s}"]].rename(columns={f"s{s}": "y_pred"})
            ev = evaluate._events(part, filled, observed)
            tp, fp, fn = tp + (ev["tp"] or 0), fp + (ev["fp"] or 0), fn + (ev["fn"] or 0)
        out[s] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return out


def summarise(df, seeds):
    q = df[df["obs"]]
    cols = [f"s{s}" for s in seeds]
    mean = q[cols].mean(axis=1)
    return {"seeds": seeds, "rows": int(len(q)),
            "mse_seedmean_pred": float(((mean - q["y"]) ** 2).mean()),
            "mse_by_seed": {s: float(((q[f"s{s}"] - q["y"]) ** 2).mean()) for s in seeds},
            "fold_mse_seedmean": {int(k): float(np.mean([((g[c] - g["y"]) ** 2).mean() for c in cols]))
                                  for k, g in q.groupby("fold")},
            "bands": {f"{lo}-{hi}": float(((mean - q["y"]) ** 2)[q["h"].between(lo, hi)].mean()) for lo, hi in evaluate.BANDS}}


def _blocks(q):
    """72h blocks of origins, numbered within each fold (folds never share a block)."""
    start = q.groupby("fold")["origin"].transform("min")
    return (q["fold"].astype(str) + "-" + ((q["origin"] - start) // pd.Timedelta(hours=BLOCK)).astype(str)).to_numpy()


def bootstrap_rel(se_cand, se_ref, blocks, n=BOOT, seed=0):
    """Pooled relative MSE difference and its 72h-block bootstrap distribution."""
    keys, inv = np.unique(blocks, return_inverse=True)
    c, r = np.bincount(inv, se_cand), np.bincount(inv, se_ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    draws = c[idx].sum(1) / r[idx].sum(1) - 1
    return float(c.sum() / r.sum() - 1), draws


def noise_floor(df, seeds):
    """MDE proxy: block-bootstrap SD of the relative difference between two seeds of the same recipe."""
    q = df[df["obs"]]
    if len(seeds) < 2:
        return None
    a, b = q[f"s{seeds[0]}"], q[f"s{seeds[1]}"]
    rel, draws = bootstrap_rel((a - q["y"]) ** 2, (b - q["y"]) ** 2, _blocks(q))
    sd = float(np.std(draws, ddof=1))
    return {"seed_pair_rel": rel, "boot_sd": sd, "mde80": 2.8 * sd}


def compare(eid, control):
    cand, cs = load(eid)
    ref, rs = load(control)
    m = cand.merge(ref, on=["origin", "h", "fold"], suffixes=("", "_ref"), validate="one_to_one")
    if len(m) != len(cand):
        raise core.ExposError(f"{eid}: fold rows differ from control {control}")
    q = m[m["obs"]]
    cm = q[[f"s{s}" for s in cs]].mean(axis=1)
    rm = q[[f"s{s}_ref" for s in rs]].mean(axis=1)
    se_c, se_r = (cm - q["y"]) ** 2, (rm - q["y"]) ** 2
    rel, draws = bootstrap_rel(se_c.to_numpy(), se_r.to_numpy(), _blocks(q))
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    rel_fold = {int(k): float(se_c[q["fold"] == k].sum() / se_r[q["fold"] == k].sum() - 1) for k in sorted(FOLDS)}
    return cand, cs, ref, rs, {"pooled_rel": rel, "ci95": [lo, hi], "boot_sd": float(np.std(draws, ddof=1)),
                               "rel_by_fold": rel_fold, "n_folds_improved": sum(v < 0 for v in rel_fold.values()),
                               "n_blocks": int(len(np.unique(_blocks(q))))}


def evaluate_roll3(eid, control, baseline=False):
    s = core.state()
    if eid not in s["experiments"]:
        raise core.ExposError(f"unknown experiment {eid}")
    changed = core.check_protected(s)
    if changed:
        raise core.ExposError(f"protected files changed: {changed}")
    if baseline:
        df, seeds = load(eid)
        summ = summarise(df, seeds)
        summ["f1_by_seed"] = _f1(df, seeds)
        summ["f1_seedmean"] = float(np.mean(list(summ["f1_by_seed"].values())))
        gate = {"pass": False, "reasons": ["control run (no gate)"], "noise": noise_floor(df, seeds)}
        return core.append("roll3", id=eid, baseline=True, control=None, gate=gate, candidate=summ)
    if control not in s["experiments"]:
        raise core.ExposError(f"unknown control {control}")
    cand, cs, ref, rs, cmp = compare(eid, control)
    cs_sum, rs_sum = summarise(cand, cs), summarise(ref, rs)
    cs_sum["f1_by_seed"], rs_sum["f1_by_seed"] = _f1(cand, cs), _f1(ref, rs)
    f1c, f1r = (float(np.mean(list(x["f1_by_seed"].values()))) for x in (cs_sum, rs_sum))
    reasons = []
    if len(cs) < MIN_SEEDS:
        reasons.append(f"needs >= {MIN_SEEDS} seeds, got {len(cs)}")
    if len(cs) > 1 and len({round(v, 9) for v in cs_sum["mse_by_seed"].values()}) == 1:
        reasons.append("seeds produced identical results: seed variation was not exercised")
    if cmp["pooled_rel"] > POOLED_MAX:
        reasons.append(f"pooled relative {cmp['pooled_rel']:+.4f} > {POOLED_MAX}")
    if not cmp["ci95"][1] < 0:
        reasons.append(f"block-bootstrap 95% upper bound {cmp['ci95'][1]:+.4f} >= 0")
    if cmp["n_folds_improved"] < MIN_FOLDS:
        reasons.append(f"improves in {cmp['n_folds_improved']}/3 folds (< {MIN_FOLDS})")
    if f1c < f1r:
        reasons.append(f"pooled F1 seed-mean {f1c:.4f} < control {f1r:.4f}")
    gate = {"pass": not reasons, "reasons": reasons, **cmp, "f1_seedmean": f1c, "control_f1_seedmean": f1r,
            "noise": noise_floor(cand, cs)}
    cs_sum["f1_seedmean"], rs_sum["f1_seedmean"] = f1c, f1r
    return core.append("roll3", id=eid, baseline=False, control=control, gate=gate, candidate=cs_sum, control_summary=rs_sum)


def line(rec):
    """`R3|` result line for the sheet (first line of the result cell)."""
    g, c = rec["gate"], rec["candidate"]
    if rec.get("baseline"):
        n = g.get("noise") or {}
        return (f"R3| E={rec['id']} base=control folds=1,2,3 seeds={','.join(map(str, c['seeds']))} "
                f"mse={c['mse_seedmean_pred']:.1f} folds_mse={'/'.join(f'{v:.1f}' for v in c['fold_mse_seedmean'].values())} "
                f"mde80={n.get('mde80', float('nan')):+.4f} type=control")
    return (f"R3| E={rec['id']} base={rec['control']} folds=1,2,3 seeds={','.join(map(str, c['seeds']))} "
            f"d_pool={g['pooled_rel']:+.4%} ci=[{g['ci95'][0]:+.4%},{g['ci95'][1]:+.4%}] "
            f"folds_better={g['n_folds_improved']}/3 f1={g['f1_seedmean']:.3f}/{g['control_f1_seedmean']:.3f} "
            f"mde80={(g.get('noise') or {}).get('mde80', float('nan')):+.4f} pass={'Y' if g['pass'] else 'n'}")
