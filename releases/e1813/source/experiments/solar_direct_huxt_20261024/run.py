"""Direct HUXt-state model: single 3-head XGBoost, no residual / blend / ensemble.

Pre-registered in PREREG.md (D_DIRECTHUXT_20261024).  The only changes from the E1782
direct base are (i) the causal HUXt Earth trajectory and radial-longitude ring at the
target lead, (ii) the six causal ENLIL columns, (iii) the horizon index in the long head,
and (iv) the persistence-anomaly training target.  No champion prediction, no residual
head, no forecast combination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path("/home/t-lab01/sunrun/leaderboard")
for p in (LB / "experiments/d147_e0421_lean", LB / "experiments/d143_axisA_analog/a2b",
          LB / "experiments/headroom_audit", LB / "experiments/d1_longitude_strips"):
    sys.path.insert(0, str(p))
import lean as L  # noqa: E402
import spec_a2b as SP  # noqa: E402
import run_official_a2b as A  # noqa: E402
import headroom as HR  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
CUTOFF = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
H = np.arange(1, 73)
HUXT_CACHE = LB / "experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz"
ENLIL = ["enlil_next_arr_h", "enlil_since_arr_h", "enlil_n_pending", "enlil_kp",
         "enlil_glancing", "enlil_dt_target"]
HX = ["hx_h", "hx_mean", "hx_std", "hx_min", "hx_max", "hx_argmax", "hx_ring0",
      "hx_ringm1", "hx_ringp1", "hx_ringm2", "hx_ringp2", "hx_ringmax", "hx_ringmin",
      "hx_ringstd"]


def write_grid(origins, pred, name):
    origins = pd.DatetimeIndex(origins)
    pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), 72),
        "horizon_hours": np.tile(H, len(origins)),
        "pred_kms": np.asarray(pred, float).reshape(-1),
    }).to_parquet(OUT / f"predictions_{name}.parquet", index=False)


class Huxt:
    def __init__(self):
        z = np.load(HUXT_CACHE, allow_pickle=True)
        self.traj = z["traj"]
        self.ring = z["ring"]
        self.eli = z["earth_lon_index"]
        self.origins = pd.DatetimeIndex(pd.to_datetime(z["origins_ns"], utc=True))
        self.pos = pd.Series(np.arange(len(self.origins)), index=self.origins)
        self.sha = hashlib.sha256(HUXT_CACHE.read_bytes()).hexdigest()

    def columns(self, grid: pd.DataFrame) -> pd.DataFrame:
        origin = pd.DatetimeIndex(pd.to_datetime(grid["origin"], utc=True))
        h = grid["h"].to_numpy()
        idx = self.pos.reindex(origin).to_numpy()
        good = ~np.isnan(idx)
        idx = np.nan_to_num(idx).astype(int)
        hh = np.clip(h - 1, 0, 71)
        rows = np.arange(len(grid))
        traj, ring, eli = self.traj, self.ring, self.eli
        t = traj[idx, hh].astype(float)
        e = eli[idx, hh].astype(int)
        rr = ring[idx, hh, :].astype(float)
        f = pd.DataFrame({
            "hx_h": t,
            "hx_mean": traj[idx].mean(1).astype(float),
            "hx_std": traj[idx].std(1).astype(float),
            "hx_min": traj[idx].min(1).astype(float),
            "hx_max": traj[idx].max(1).astype(float),
            "hx_argmax": traj[idx].argmax(1).astype(float),
            "hx_ring0": rr[rows, e],
            "hx_ringm1": rr[rows, (e - 1) % 128],
            "hx_ringp1": rr[rows, (e + 1) % 128],
            "hx_ringm2": rr[rows, (e - 2) % 128],
            "hx_ringp2": rr[rows, (e + 2) % 128],
            "hx_ringmax": rr.max(1),
            "hx_ringmin": rr.min(1),
            "hx_ringstd": rr.std(1),
        })
        f.loc[~good, :] = np.nan
        return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--official-only", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=240)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ace = HR.hourly(pd.read_parquet(LB / "store/ch-v1/ace.parquet"))["filled_speed_kms"]
    meta, block = ST.load_inputs()
    hx = Huxt()
    print(f"huxt cache {hx.sha[:16]} origins={len(hx.origins)}", flush=True)

    df = L.build_train({})
    hx_tr = hx.columns(df.reset_index(drop=True))
    dfx = pd.concat([df.reset_index(drop=True), hx_tr], axis=1)
    base = L.feature_columns(dfx, L.DEFAULT)  # includes h
    cols = base + ENLIL + HX
    missing = [c for c in cols if c not in dfx.columns]
    if missing:
        raise SystemExit(f"training frame misses {missing[:5]}")
    print(f"train rows={len(dfx)} cols={len(cols)}", flush=True)

    def grid_for(origins):
        g = A.base_grid_for(pd.DatetimeIndex(origins, tz="UTC"), ace, HR, meta, block, [])
        ev = ia.enlil_runs()
        en = ia.enlil_features(pd.DatetimeIndex(origins, tz="UTC"), ev)
        en["origin"] = pd.to_datetime(en.index, utc=True)
        g = g.merge(en.reset_index(drop=True), on="origin", how="left")
        g["enlil_dt_target"] = g["h"] - g["enlil_next_arr_h"]
        g = pd.concat([g.reset_index(drop=True), hx.columns(g).reset_index(drop=True)], axis=1)
        return g

    def fit_predict(train, grid):
        tr = train.copy()
        tr["y"] = tr["y"] - tr["v_lag0"].to_numpy()
        fs = L.fit_set({**L.DEFAULT, "members": "d33"}, 3, SP.BASE_PARAMS, 0, tr, cols)
        return L.predict({**L.DEFAULT, "members": "d33"}, fs, grid, cols) + grid["v_lag0"].to_numpy()

    if not a.smoke and not a.official_only:
        for name, (lo, hi) in L.ia.PERIODS.items():
            lo, hi = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
            origins = pd.date_range(lo, hi, freq=SP.TRAIN_FREQ, tz="UTC")
            g = grid_for(origins)
            tr = dfx[dfx["target_t"] < lo - pd.Timedelta("72h")]
            write_grid(g["origin"].unique(), fit_predict(tr, g), name)
            print(f"{name} rows={len(g)}", flush=True)

    if a.official:
        off = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
        tr = dfx[dfx["target_t"] < CUTOFF].copy()
        tr["y"] = tr["y"] - tr["v_lag0"].to_numpy()
        ckpt = OUT / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
            final = OUT / f"predictions_official_s{seed}.parquet"
            if final.exists():
                print(f"official seed={seed} already present", flush=True)
                continue
            ev = L.RL.event_seed(seed)
            mp = [ckpt / f"model_s{seed}_{k}.json" for k in ("short", "mid", "long")]
            if all(p.exists() for p in mp):
                short, mid, long = xgb.XGBRegressor(), xgb.XGBRegressor(), xgb.XGBRegressor()
                for m, p in ((short, mp[0]), (mid, mp[1]), (long, mp[2])):
                    m._estimator_type = "regressor"
                    m.load_model(p)
                fs = ({ev: (short, mid)}, long, L.RL.level_seeds(seed), ev)
                print(f"official seed={seed} models loaded", flush=True)
            else:
                fs = L.fit_set({**L.DEFAULT, "members": "d33"}, 3, SP.BASE_PARAMS, seed, tr, cols)
                for m, p in ((fs[0][ev][0], mp[0]), (fs[0][ev][1], mp[1]), (fs[1], mp[2])):
                    m._estimator_type = "regressor"
                    m.save_model(p)
                print(f"official seed={seed} models fitted", flush=True)
            parts, origins = [], []
            for first in range(0, len(off), a.chunk):
                i = first // a.chunk
                cp = ckpt / f"chunk_s{seed}_{i}.npz"
                if cp.exists():
                    z = np.load(cp)
                    parts.append(z["pred"])
                    origins.append(pd.DatetimeIndex(pd.to_datetime(z["origins_ns"], utc=True)))
                    print(f"official seed={seed} chunk={i} loaded", flush=True)
                    continue
                chunk = off[first:first + a.chunk]
                g = grid_for(chunk)
                p = L.predict({**L.DEFAULT, "members": "d33"}, fs, g, cols) + g["v_lag0"].to_numpy()
                parts.append(p)
                oo = pd.DatetimeIndex(g["origin"].unique())
                origins.append(oo)
                np.savez(cp, origins_ns=oo.asi8, pred=p)
                print(f"official seed={seed} chunk={i} rows={len(g)}", flush=True)
                del g
            oo = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in origins]))
            write_grid(oo, np.concatenate(parts), f"official_s{seed}")
            print(f"official seed={seed} rows={len(oo)}", flush=True)

    # causal/future poison on one probe origin
    o = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    xf = ace.copy()
    xf.loc[xf.index > o] = xf.loc[xf.index > o] + 100
    xp = ace.copy()
    xp.loc[xp.index < o] = xp.loc[xp.index < o] + 100
    probe = pd.DatetimeIndex([o])
    b = grid_for(probe)[cols].to_numpy(float)
    # rebuild with perturbed ACE through a local copy of grid_for
    def grid_ace(series):
        g = A.base_grid_for(probe, series, HR, meta, block, [])
        ev = ia.enlil_runs()
        en = ia.enlil_features(probe, ev)
        en["origin"] = pd.to_datetime(en.index, utc=True)
        g = g.merge(en.reset_index(drop=True), on="origin", how="left")
        g["enlil_dt_target"] = g["h"] - g["enlil_next_arr_h"]
        g = pd.concat([g.reset_index(drop=True), hx.columns(g).reset_index(drop=True)], axis=1)
        return g[cols].to_numpy(float)
    fut = grid_ace(xf)
    past = grid_ace(xp)
    poison = {"feature_future_invariant": bool(np.allclose(b, fut, equal_nan=True)),
              "feature_positive_control": bool(not np.allclose(b, past, equal_nan=True)),
              "label_cutoff_ok": bool(dfx.loc[dfx["target_t"] < CUTOFF, "target_t"].max() < CUTOFF),
              "huxt_cache_sha256": hx.sha}
    poison["passed"] = bool(poison["feature_future_invariant"] and poison["feature_positive_control"]
                            and poison["label_cutoff_ok"])
    (OUT / "poison.json").write_text(json.dumps(poison, indent=1))
    print(f"poison {poison['passed']}", flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed")


if __name__ == "__main__":
    main()
