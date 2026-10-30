"""Arrival-timing pre-test (user decision 2026-10-17, ROLL3:20261017).

Hindsight L1 labels (store/arrival-labels-v1, built by experiments/arrival_labels/build_labels.py) give t_onset per
speed-ramp event: the first time the 3h-smoothed L1 speed stays >= base + 20% of the base->peak rise for 3 hours.
For every eligible event inside the roll3 folds and every lead L in LEADS, the origin t_onset - L is scored on the run's
roll3 trajectories (all seeds):
  - predicted ramp: in the 72h trajectory, base = min before the trajectory max, rise = max - base;
    the ramp counts only if rise >= max(MIN_RISE, RISE_FRAC * label dV), else it is a miss;
  - predicted onset: first time the trajectory stays >= base + 20% of rise for 3 hours (same rule as the label);
  - error = predicted onset - t_onset (hours); capped error = min(|error|, CAP) with misses scored CAP.
Primary metric: capped MAE (mean over events of the seed/lead mean). Also: hit rate (|error| <= HIT and not a miss),
MAE on hits. Paired against the control run with an event bootstrap. A run passes when the capped-MAE difference has a
95% upper bound < 0, or the hit-rate difference has a 95% lower bound > 0.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from . import core, roll3

LABELS = core.STORE_ROOT / "arrival-labels-v1"
LEADS = (12, 24, 36, 48)
FRAC, SUSTAIN, MIN_RISE, RISE_FRAC, CAP, HIT, BOOT = 0.20, 3, 50.0, 0.25, 24.0, 12.0, 4000


def labels():
    sums = dict(reversed(l.split("  ", 1)) for l in (LABELS / "SHA256SUMS").read_text().splitlines() if l.strip())
    for name in ("events.parquet", "definition.json"):
        if hashlib.sha256((LABELS / name).read_bytes()).hexdigest() != sums.get(name):
            raise core.ExposError(f"arrival labels changed: {name} does not match SHA256SUMS")
    ev = pd.read_parquet(LABELS / "events.parquet")
    definition = json.loads((LABELS / "definition.json").read_text())
    ev = ev[(ev["quality"] != "X") & ev["weak"].eq(False) & ev["t_onset"].notna()].copy()
    return ev, definition


def predicted_onset(path, label_dv):
    """Hour index (1..72) of the predicted onset in one 72h trajectory, or None for a miss."""
    peak = int(np.argmax(path))
    base = float(np.min(path[:peak + 1]))
    rise = float(path[peak]) - base
    if rise < max(MIN_RISE, RISE_FRAC * label_dv):
        return None
    above = path >= base + FRAC * rise
    run = np.convolve(above.astype(int), np.ones(SUSTAIN, dtype=int), "full")[SUSTAIN - 1:len(above) + SUSTAIN - 1]
    idx = np.flatnonzero(run[:len(above) - SUSTAIN + 1] >= SUSTAIN)
    return int(idx[0]) + 1 if len(idx) else None


def _trajectories(eid):
    files = roll3._files(eid)
    if not files:
        raise core.ExposError(f"{eid}: no roll3 files")
    seeds = sorted({s for _, s in files})
    out = {}
    for (k, s), p in files.items():
        df = pd.read_parquet(p)
        o = pd.to_datetime(df["origin_last_input_utc"], utc=True)
        df = df.assign(o=o).sort_values(["o", "horizon_hours"])
        arr = df["pred_kms"].to_numpy().reshape(-1, 72)
        out.setdefault(s, {}).update(dict(zip(df["o"].drop_duplicates(), arr)))
    return out, seeds


def score(eid, ev):
    traj, seeds = _trajectories(eid)
    rows = []
    for _, e in ev.iterrows():
        onset = pd.Timestamp(e["t_onset"]).tz_convert("UTC") if pd.Timestamp(e["t_onset"]).tzinfo else pd.Timestamp(e["t_onset"], tz="UTC")
        for lead in LEADS:
            origin = onset - pd.Timedelta(hours=lead)
            if not all(origin in traj[s] for s in seeds):
                continue
            for s in seeds:
                h = predicted_onset(traj[s][origin], float(e["dV_kms"]))
                err = None if h is None else float(h - lead)
                rows.append(dict(event_id=e["event_id"], kind=e["kind"], lead=lead, seed=s, err=err,
                                 capped=CAP if err is None else min(abs(err), CAP),
                                 hit=err is not None and abs(err) <= HIT))
    r = pd.DataFrame(rows)
    if r.empty:
        raise core.ExposError(f"{eid}: no labelled event has origins inside the roll3 trajectories")
    return r, seeds


def per_event(r):
    g = r.groupby("event_id")
    hits = r[r["hit"]]
    return pd.DataFrame({"capped": g["capped"].mean(), "hit": g["hit"].mean(), "kind": g["kind"].first()}), \
        (float(hits["err"].abs().mean()) if len(hits) else None)


def summary(r):
    pe, mae_hits = per_event(r)
    return {"events": int(len(pe)), "capped_mae": float(pe["capped"].mean()), "hit_rate": float(pe["hit"].mean()),
            "mae_hits": mae_hits, "miss_rate": float(r["err"].isna().mean()),
            "by_lead": {int(l): {"capped_mae": float(g["capped"].mean()), "hit_rate": float(g["hit"].mean())} for l, g in r.groupby("lead")},
            "by_kind": {k: {"events": int(len(g)), "capped_mae": float(g["capped"].mean()), "hit_rate": float(g["hit"].mean())} for k, g in pe.groupby("kind")}}


def evaluate_arrival(eid, control=None, baseline=False):
    s = core.state()
    if eid not in s["experiments"]:
        raise core.ExposError(f"unknown experiment {eid}")
    changed = core.check_protected(s)
    if changed:
        raise core.ExposError(f"protected files changed: {changed}")
    ev, definition = labels()
    r, seeds = score(eid, ev)
    summ = summary(r)
    label_info = {"version": definition.get("version"), "status": definition.get("status", "draft"),
                  "sha256": hashlib.sha256((LABELS / "events.parquet").read_bytes()).hexdigest()}
    if baseline:
        return core.append("arrival", id=eid, baseline=True, control=None, labels=label_info, candidate=summ,
                           gate={"pass": False, "reasons": ["control run (no gate)"]})
    if control not in s["experiments"]:
        raise core.ExposError(f"unknown control {control}")
    rc, _ = score(control, ev)
    a, b = per_event(r)[0], per_event(rc)[0]
    m = a.join(b, rsuffix="_ref", how="inner")
    d_cap, d_hit = (m["capped"] - m["capped_ref"]).to_numpy(), (m["hit"] - m["hit_ref"]).to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(m), (BOOT, len(m)))
    cap_ci = [float(v) for v in np.percentile(d_cap[idx].mean(1), [2.5, 97.5])]
    hit_ci = [float(v) for v in np.percentile(d_hit[idx].mean(1), [2.5, 97.5])]
    reasons = []
    if not (cap_ci[1] < 0 or hit_ci[0] > 0):
        reasons.append(f"no timing gain: capped-MAE diff CI {cap_ci[0]:+.2f}..{cap_ci[1]:+.2f}h, hit-rate diff CI {hit_ci[0]:+.3f}..{hit_ci[1]:+.3f}")
    if len(m) < 50:
        reasons.append(f"only {len(m)} events (< 50)")
    gate = {"pass": not reasons, "reasons": reasons, "events": int(len(m)), "d_capped_mae": float(d_cap.mean()),
            "capped_ci95": cap_ci, "d_hit_rate": float(d_hit.mean()), "hit_ci95": hit_ci}
    return core.append("arrival", id=eid, baseline=False, control=control, labels=label_info, candidate=summ,
                       control_summary=summary(rc), gate=gate)


def line(rec):
    c, g = rec["candidate"], rec["gate"]
    lab = f"labels=v{rec['labels']['version']}{'' if rec['labels']['status'] == 'frozen' else '-draft'}"
    if rec.get("baseline"):
        return f"ARR| E={rec['id']} base=control events={c['events']} mae={c['capped_mae']:.2f}h hit={c['hit_rate']:.3f} {lab}"
    return (f"ARR| E={rec['id']} base={rec['control']} events={g['events']} mae={c['capped_mae']:.2f}h hit={c['hit_rate']:.3f} "
            f"d_mae={g['d_capped_mae']:+.2f}h ci=[{g['capped_ci95'][0]:+.2f},{g['capped_ci95'][1]:+.2f}] "
            f"d_hit={g['d_hit_rate']:+.3f} ci=[{g['hit_ci95'][0]:+.3f},{g['hit_ci95'][1]:+.3f}] pass={'Y' if g['pass'] else 'n'} {lab}")
