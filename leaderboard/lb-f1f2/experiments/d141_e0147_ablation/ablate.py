"""D141 E0147 ablation runner: one-factor changes to the E0147 recipe (D76 per-lead mixture) on the official window.

E0147 = 7 XGBoost fits (short 1-6 / 7-24 at level seeds 0,1,2 + one long head at event seed 1) -> 3 stitched members
(e0003_line, d33_hsplit, d45_lvlmean) -> fixed per-lead convex weights, train targets < 2026-05-29, 3h grid, 2833 official origins.
This runner reuses that code path (d76_moe runner helpers, read-only) and changes exactly one factor per `--arm`.

Seed cells (the lb-eval gate needs >= 3 genuinely different cells): cell s fits level seeds (3s, 3s+1, 3s+2), d33 uses the middle one,
the long head uses event seed 1 for s=0 and 100+s otherwise, so cell 0 is the E0147 seed set (control sanity check) and cells 1/2 are fresh.
Writes $EXPOS_OUT/predictions_official_s{0,1,2}.parquet + poison.json + selection.json. CPU only (XGBoost device=cpu, n_jobs 6).
No stored prediction file is read; every arm is a fresh retrain. EXPOS_SMOKE=1: seed 0, 24h train grid, 20 trees, first 8 origins.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d76_moe"))
import runner as R  # noqa: E402  (E0145 code, read-only)
import official_members as OM  # noqa: E402

ia, ST, HS = R.ia, R.ST, R.HS
OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
SCRATCH = LB / "store/scratch/worker-d141"
CUTOFF = OM.CUTOFF
ORIGINS = OM.ORIGINS
SEEDS = (0, 1, 2)
MEMBERS = ("e0003_line", "d33_hsplit", "d45_lvlmean")
# E0145 selection.json per-lead weights (dev-period OOF only), identical to d76_official.WEIGHTS
WEIGHTS = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272, "d45_lvlmean": 0.3946530834632206},
           "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209, "d45_lvlmean": 0.3411415011482895},
           "25-48": {"e0003_line": 1 / 3, "d33_hsplit": 1 / 3, "d45_lvlmean": 1 / 3},
           "49-72": {"e0003_line": 1 / 3, "d33_hsplit": 1 / 3, "d45_lvlmean": 1 / 3}}
BANDS = (("1-6", 1, 6), ("7-24", 7, 24), ("25-48", 25, 48), ("49-72", 49, 72))


# ---- arm registry -------------------------------------------------------------------------------------------------
def _speed(c):
    return c.startswith(("v_", "rec"))


def _plasma(c):
    return c.startswith("ace_") or c in ("speed_d24", "epam_p1_d6")


DROP = {  # arm -> predicate over a column name (columns removed from the 105-col existing_all + strips set)
    "no_strips": lambda c: c in ST.strip_columns(),
    "no_ch": lambda c: c.startswith("ch_"),
    "no_speed": lambda c: _speed(c),
    "no_plasma": _plasma,
    "no_rec": lambda c: c.startswith("rec"),
    "no_vlag": lambda c: c.startswith("v_lag"),
    "no_vroll": lambda c: c.startswith(("v_m", "v_sd")),
    "no_epam": lambda c: "epam" in c,
    "no_imf": lambda c: c.startswith(("ace_bt", "ace_bz", "ace_by")),
    "no_density_temp": lambda c: c.startswith(("ace_density", "ace_temperature")),
    "no_acespeed": lambda c: c.startswith("ace_speed") or c == "speed_d24",
    "no_strip_logmed": lambda c: c.startswith("strip_") and c.endswith("_logmed"),
    "no_strip_dark": lambda c: c.startswith("strip_") and c.endswith("_dark"),
    "no_strip_cover": lambda c: c.startswith("strip_") and c.endswith("_cover"),
    "no_img_meta": lambda c: c in ("img_age_h", "img_stale"),
}
for _v in ST.SPEEDS:
    DROP[f"no_strip_v{_v}"] = (lambda v: lambda c: c.startswith(f"strip_v{v}_"))(_v)

ARMS = {"control": {}}
ARMS.update({a: {"drop": a} for a in DROP})
ARMS.update({  # B4: members (retrain only what the kept members need; weights renormalised per lead)
    "drop_e0003_line": {"members": ("d33_hsplit", "d45_lvlmean")},
    "drop_d33_hsplit": {"members": ("e0003_line", "d45_lvlmean")},
    "drop_d45_lvlmean": {"members": ("e0003_line", "d33_hsplit")},
    "only_e0003_line": {"members": ("e0003_line",)},
    "only_d33_hsplit": {"members": ("d33_hsplit",)},
    "only_d45_lvlmean": {"members": ("d45_lvlmean",)},
})
ARMS.update({  # B5: blend / split
    "w_equal": {"weights": "equal"},
    "w_global": {"weights": "global"},
    "split12": {"split": 12},
    "split48": {"split": 48},
})
ARMS.update({  # B3: strip knobs (features recomputed under the patched constants)
    "omega045": {"strip": {"OMEGA": 0.45}}, "omega065": {"strip": {"OMEGA": 0.65}},
    "half25": {"strip": {"STRIP_HALF": 2.5}}, "half10": {"strip": {"STRIP_HALF": 10.0}},
    "cover03": {"strip": {"COVER_MIN": 0.3}}, "cover07": {"strip": {"COVER_MIN": 0.7}},
    "stale3": {"strip": {"STALE_H": 3.0}}, "stale12": {"strip": {"STALE_H": 12.0}},
    "margin0": {"strip": {"MARGIN_H": 0.0}}, "margin3": {"strip": {"MARGIN_H": 3.0}},
})
ARMS.update({  # B7: data / model
    "grid1h": {"freq": "1h"},
    "start2023": {"start": "2023-10-01"},
    "start2024": {"start": "2024-10-01"},
    "depth4": {"params": {"max_depth": 4}}, "depth8": {"params": {"max_depth": 8}},
    "trees300": {"params": {"n_estimators": 300}}, "trees1200": {"params": {"n_estimators": 1200}},
    "mcw5": {"params": {"min_child_weight": 5}}, "mcw50": {"params": {"min_child_weight": 50}},
    "lambda1": {"params": {"reg_lambda": 1}}, "lambda20": {"params": {"reg_lambda": 20}},
    "colsample04": {"params": {"colsample_bytree": 0.4}}, "colsample10": {"params": {"colsample_bytree": 1.0}},
    "subsample06": {"params": {"subsample": 0.6}}, "subsample10": {"params": {"subsample": 1.0}},
})
ARMS.update({"enlil": {"enlil": True}})  # B8: existing+enlil (DONKI WSA-Enlil, modelCompletionTime <= origin)


def global_weights():
    """E0146 (D71) global convex weights, same for every lead (E0141 dev-OOF pre-rule, expos_runners/d71_official.py:23-24)."""
    w = {"e0003_line": 0.3282447610385238, "d33_hsplit": 0.33603162776340306, "d45_lvlmean": 0.33572361119807315}
    return {b: dict(w) for b, _, _ in BANDS}


# ---- feature frame --------------------------------------------------------------------------------------------------
def _fkey(kind, payload):
    code = ["experiments/d141_e0147_ablation/ablate.py", "experiments/d1_longitude_strips/strips.py",
            "experiments/headroom_audit/info_ablation.py", "experiments/headroom_audit/headroom.py",
            "experiments/expos_runners/existing_all.py"]
    h = hashlib.sha256()
    for f in code:
        h.update((LB / f).read_bytes())
    for f in R.FCACHE_DATA:
        st = (LB / f).stat()
        h.update(f"{f}{st.st_size}{st.st_mtime_ns}".encode())
    h.update(json.dumps({"kind": kind, **payload}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def build_train(freq, start, strip_cfg):
    key = _fkey("train", {"freq": freq, "start": start, "strip": strip_cfg})
    pq = SCRATCH / "featcache" / f"{key}.parquet"
    if pq.is_file():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    df = ia.build(freq=freq, start=start)
    meta, block = ST.load_inputs()
    df, _ = R.add_strips(df, meta, block)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(pq)
    return df


def feature_columns(df, cfg):
    cols = ia.arms(df.columns)["existing+enlil" if cfg.get("enlil") else "existing_all"] + ST.strip_columns()
    if cfg.get("drop"):
        keep = [c for c in cols if not DROP[cfg["drop"]](c)]
        if len(keep) == len(cols):
            raise SystemExit(f"arm {cfg['drop']} dropped no columns")
        cols = keep
    return cols


# ---- fit / members / blend ------------------------------------------------------------------------------------------
def fit_params(cfg, seed):
    p = dict(ia.PARAMS)
    p.update(device=os.environ.get("D141_DEVICE", "cpu"), n_jobs=6 if os.environ.get("D141_DEVICE", "cpu") == "cpu" else 4, random_state=int(seed))
    p.update(cfg.get("params", {}))
    if SMOKE:
        p["n_estimators"] = 20
    return p


def cell_seeds(s):
    lvl = (3 * s, 3 * s + 1, 3 * s + 2)
    return lvl, (1 if s == 0 else 100 + s)


def fit_cell(tr, cols, cfg, s):
    members = cfg.get("members", MEMBERS)
    split = cfg.get("split", HS.SPLIT_H)
    lvl, ev = cell_seeds(s)
    need = set()
    if "d33_hsplit" in members:
        need.add(lvl[1])
    if "d45_lvlmean" in members:
        need.update(lvl)
    tr1 = tr[tr["h"] <= 6]
    tr2 = tr[(tr["h"] >= 7) & (tr["h"] <= split)]
    short = {sd: (xgb.XGBRegressor(**fit_params(cfg, sd)).fit(tr1[cols], tr1["y"]),
                  xgb.XGBRegressor(**fit_params(cfg, sd)).fit(tr2[cols], tr2["y"])) for sd in sorted(need)}
    long_ = xgb.XGBRegressor(**fit_params(cfg, ev)).fit(tr[cols], tr["y"])
    return short, long_, lvl


def member_preds(models, te, cols, cfg):
    short, long_, lvl = models
    members = cfg.get("members", MEMBERS)
    split = cfg.get("split", HS.SPLIT_H)
    h = te["h"].to_numpy()
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= split), h > split
    long_all = np.asarray(long_.predict(te[cols]), dtype=np.float64)
    out = {}
    if "e0003_line" in members:
        out["e0003_line"] = long_all

    def stitched(sds):
        p = np.empty(len(te), dtype=np.float64)
        p[m1] = np.mean([short[sd][0].predict(te.loc[m1, cols]) for sd in sds], axis=0)
        p[m2] = np.mean([short[sd][1].predict(te.loc[m2, cols]) for sd in sds], axis=0)
        p[m3] = long_all[m3]
        return p

    if "d33_hsplit" in members:
        out["d33_hsplit"] = stitched([lvl[1]])
    if "d45_lvlmean" in members:
        out["d45_lvlmean"] = stitched(list(lvl))
    return out


def blend(mem, h, cfg):
    members = [m for m in MEMBERS if m in mem]
    mode = cfg.get("weights", "default")
    W = global_weights() if mode == "global" else WEIGHTS
    pred = np.empty(len(h), dtype=np.float64)
    for b, lo, hi in BANDS:
        mask = (h >= lo) & (h <= hi)
        w = {m: (1.0 if mode == "equal" else W[b][m]) for m in members}
        z = sum(w.values())
        pred[mask] = sum(w[m] / z * mem[m][mask] for m in members)
    return pred


# ---- main -----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    for k, v in cfg.get("strip", {}).items():  # module constants read at call time by strip_features / select_slots
        setattr(ST, k, v)
    if "split" in cfg:
        HS.SPLIT_H = cfg["split"]
    seeds = (0,) if SMOKE else SEEDS
    origins = ORIGINS[:8] if SMOKE else ORIGINS
    freq = "24h" if SMOKE else cfg.get("freq", "3h")
    print(f"d141 arm={a.arm} cfg={cfg} smoke={SMOKE} freq={freq} seeds={seeds} origins={len(origins)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    df = build_train(freq, cfg.get("start"), cfg.get("strip", {}))
    cols = feature_columns(df, cfg)
    if a.arm == "control":
        assert len(cols) == 105, len(cols)
    tr = df[df["target_t"] < CUTOFF]
    meta, block = ST.load_inputs()
    te, sel = R.add_strips(R.base_grid(origins), meta, block)
    print(f"train rows={len(tr)} cols={len(cols)} test rows={len(te)}", flush=True)

    t0 = time.perf_counter()
    cells = []
    for s in seeds:
        models = fit_cell(tr, cols, cfg, s)
        pred = blend(member_preds(models, te, cols, cfg), te["h"].to_numpy(), cfg)
        if not np.isfinite(pred).all() or len(pred) != len(origins) * 72:
            raise SystemExit(f"seed {s}: non-finite or incomplete predictions; refusing to write")
        pd.DataFrame({"origin_last_input_utc": te["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "horizon_hours": te["h"].astype(int), "pred_kms": pred.astype(float)}
                     ).to_parquet(OUT / f"predictions_official_s{s}.parquet", index=False)
        cells.append({"seed": s, "level_seeds": list(models[2]), "event_seed": cell_seeds(s)[1], "n_train": int(len(tr)),
                      "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"seed {s} done {cells[-1]}", flush=True)

    cut = pd.to_datetime(sel["slot"]).dropna().median().isoformat()
    probe = pd.DatetimeIndex(ORIGINS[::24]) if not SMOKE else pd.DatetimeIndex(origins)
    poison = ST.poison_test(pd.DatetimeIndex(np.concatenate([pd.DatetimeIndex(df["origin"].unique())[::50], probe])).unique(),
                            cut, meta=meta, block=block)
    (OUT / "poison.json").write_text(json.dumps(poison, indent=1, default=ST.manifest_json_default))
    print("poison:", poison["passed"], flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed; refusing to present predictions as causal")
    (OUT / "selection.json").write_text(json.dumps(
        {"rule": "E0147 recipe with exactly one factor changed (see arm); fixed weights, no fitted choice, no official truth consumed",
         "arm": a.arm, "cfg": cfg, "n_cols": len(cols), "cols_dropped": sorted(set(ia.arms(df.columns)["existing_all"] + ST.strip_columns()) - set(cols)),
         "smoke": SMOKE, "freq": freq, "cells": cells, "fitted_choice": None}, indent=1, default=str))
    print(f"done arm={a.arm}", flush=True)


if __name__ == "__main__":
    main()
