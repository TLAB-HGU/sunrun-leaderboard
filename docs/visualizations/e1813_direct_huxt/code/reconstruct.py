"""Post-hoc per-round train/validation MSE for E1813 (D_DIRECTHUXT_20261024), reconstructed from its saved models.

E1813 did not log per-round losses. This script reloads the frozen official-fit boosters
(research/expos/runs/E1813/checkpoints/model_s{seed}_{short,mid,long}.json) and computes, for every
boosting round k = 0..900, the prediction base_score + sum of the first k trees via leaf indices.

- train: the official fit rows (targets < 2026-05-29). Each head is scored on its own fit rows, and the
  combined curve is scored on the rows each head serves (h<=6 short, 7..24 mid, >24 long).
- validation: the leaderboard window (2,833 origins x 72h, snapshot truth). This window was already
  scored and exposed, so these curves are descriptive only and are never used for selection.

Run inside a private mount namespace whose leaderboard/store is the hash-verified pinned copy
(see README.md). No model is fitted, nothing is written outside this directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
LB = Path("/home/t-lab01/sunrun/leaderboard")
RUN = Path("/home/t-lab01/sunrun/research/expos/runs/E1813")
SNAP = Path("/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab--sunrun-lb-store/snapshots/"
            "bcf5d5417d99eaa331ddca40581d48953e09a58e")
sys.path.insert(0, str(LB / "experiments/solar_direct_huxt_20261024"))
os.environ["EXPOS_OUT"] = str(HERE / "unused-out")
import run as R  # noqa: E402  (E1813 runner, read-only)

L, A, HR, ST, ia, SP = R.L, R.A, R.HR, R.ST, R.ia, R.SP
HEADS = ("short", "mid", "long")
SEEDS = (0, 1, 2)
NT = int(os.environ.get("RECON_THREADS", "4"))
CHUNK = 20000


def head_masks(h):
    return {"short": h <= 6, "mid": (h >= 7) & (h <= SP.SPLIT_H), "long": h > SP.SPLIT_H}


class Staged:
    """Per-round margins from leaf indices: margin_k = base_score + sum_{t<k} leaf_t."""

    def __init__(self, path):
        self.b = xgb.Booster(model_file=str(path))
        self.b.set_param({"nthread": NT})
        cfg = json.loads(self.b.save_config())
        self.base = float(cfg["learner"]["learner_model_param"]["base_score"])
        t = self.b.trees_to_dataframe()
        self.T = int(t["Tree"].max()) + 1
        nmax = int(t["Node"].max()) + 1
        self.leaf = np.zeros((self.T, nmax))
        lv = t[t["Feature"] == "Leaf"]
        self.leaf[lv["Tree"].to_numpy(), lv["Node"].to_numpy()] = lv["Gain"].to_numpy()

    def cum(self, X):
        """(n, T+1) staged margins, column k uses the first k trees."""
        idx = self.b.predict(xgb.DMatrix(X, nthread=NT), pred_leaf=True).astype(np.int64)
        v = self.leaf[np.arange(self.T)[None, :], idx]
        out = np.empty((len(X), self.T + 1))
        out[:, 0] = self.base
        np.cumsum(v, axis=1, out=out[:, 1:])
        out[:, 1:] += self.base
        return out

    def final(self, X):
        return self.b.predict(xgb.DMatrix(X, nthread=NT))


def main():
    print("building training frame", flush=True)
    ace = HR.hourly(pd.read_parquet(LB / "store/ch-v1/ace.parquet"))["filled_speed_kms"]
    meta, block = ST.load_inputs()
    hx = R.Huxt()
    df = L.build_train({})
    dfx = pd.concat([df.reset_index(drop=True), hx.columns(df.reset_index(drop=True))], axis=1)
    cols = L.feature_columns(dfx, L.DEFAULT) + R.ENLIL + R.HX
    assert len(cols) == len(set(cols)), "duplicate feature names"
    tr = dfx[dfx["target_t"] < R.CUTOFF].copy()
    tr["y"] = tr["y"] - tr["v_lag0"].to_numpy()
    print(f"train rows={len(tr)} cols={len(cols)}", flush=True)

    def grid_for(origins):  # identical to run.py main().grid_for
        g = A.base_grid_for(pd.DatetimeIndex(origins, tz="UTC"), ace, HR, meta, block, [])
        ev = ia.enlil_runs()
        en = ia.enlil_features(pd.DatetimeIndex(origins, tz="UTC"), ev)
        en["origin"] = pd.to_datetime(en.index, utc=True)
        g = g.merge(en.reset_index(drop=True), on="origin", how="left")
        g["enlil_dt_target"] = g["h"] - g["enlil_next_arr_h"]
        return pd.concat([g.reset_index(drop=True), hx.columns(g).reset_index(drop=True)], axis=1)

    keep = list(dict.fromkeys(["origin", "h", "v_lag0"] + cols))
    cache = HERE / "official-grid-cache.parquet"
    if cache.exists():
        va = pd.read_parquet(cache)
        print(f"official grid cache rows={len(va)}", flush=True)
    else:
        off = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
        parts = []
        for first in range(0, len(off), 240):
            g = grid_for(off[first:first + 240])
            parts.append(g[keep])
            print(f"official grid {first // 240} rows={len(g)}", flush=True)
        va = pd.concat(parts, ignore_index=True)
        va.to_parquet(cache, index=False)
    truth = pd.read_parquet(SNAP / "truth.parquet")
    key = pd.DataFrame({"origin_last_input_utc": va["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "horizon_hours": va["h"].astype(int)})
    yv = key.merge(truth[["origin_last_input_utc", "horizon_hours", "target_kms"]], how="left",
                   on=["origin_last_input_utc", "horizon_hours"])["target_kms"].to_numpy()
    va_ok = ~np.isnan(yv)
    print(f"validation rows={len(va)} scored={int(va_ok.sum())}", flush=True)

    models = {(s, k): Staged(RUN / "checkpoints" / f"model_s{s}_{k}.json") for s in SEEDS for k in HEADS}
    T = models[(0, "long")].T
    R1 = T + 1
    h_tr = tr["h"].to_numpy()
    fit_rows = {"short": h_tr <= 6, "mid": (h_tr >= 7) & (h_tr <= SP.SPLIT_H), "long": np.ones(len(tr), bool)}
    serve_tr = head_masks(h_tr)
    serve_va = head_masks(va["h"].to_numpy())

    sse = {}  # (split, seed|'mean', head|'all') -> (R1,)
    cnt = {}

    def add(name, e2, n):
        sse[name] = sse.get(name, 0) + e2
        cnt[name] = cnt.get(name, 0) + n

    finals = {s: np.full(len(va), np.nan) for s in SEEDS}
    for k in HEADS:
        # train: each head on its own fit rows, plus served rows for the combined curve
        Xk = tr.loc[fit_rows[k], cols]
        yk = tr.loc[fit_rows[k], "y"].to_numpy()
        sk = serve_tr[k][fit_rows[k]]
        for a in range(0, len(Xk), CHUNK):
            X, y, sv = Xk.iloc[a:a + CHUNK], yk[a:a + CHUNK], sk[a:a + CHUNK]
            cs = {s: models[(s, k)].cum(X) for s in SEEDS}
            for s in SEEDS:
                e = (cs[s] - y[:, None]) ** 2
                add(("train", s, k), e.sum(0), len(y))
                add(("train", s, "all"), e[sv].sum(0), int(sv.sum()))
            m = sum(cs.values()) / 3
            add(("train", "mean", "all"), ((m[sv] - y[sv, None]) ** 2).sum(0), int(sv.sum()))
        # validation: rows this head serves in the leaderboard window
        rows = np.flatnonzero(serve_va[k] & va_ok)
        for a in range(0, len(rows), CHUNK):
            r = rows[a:a + CHUNK]
            X = va.iloc[r][cols]
            y = yv[r] - va["v_lag0"].to_numpy()[r]
            cs = {s: models[(s, k)].cum(X) for s in SEEDS}
            for s in SEEDS:
                e = (cs[s] - y[:, None]) ** 2
                add(("validation", s, k), e.sum(0), len(r))
                add(("validation", s, "all"), e.sum(0), len(r))
                finals[s][r] = cs[s][:, -1] + va["v_lag0"].to_numpy()[r]
                ref = models[(s, k)].final(X)
                gap = float(np.max(np.abs(ref - cs[s][:, -1])))
                if gap > 1e-3:
                    raise SystemExit(f"staged sum != booster predict ({s},{k}): {gap}")
            m = sum(cs.values()) / 3
            add(("validation", "mean", "all"), ((m - y[:, None]) ** 2).sum(0), len(r))
        print(f"head {k} done", flush=True)

    # reproduction check against the saved official predictions and the lb-eval scores
    check = {}
    for s in SEEDS:
        p = pd.read_parquet(RUN / f"predictions_official_s{s}.parquet")
        p = key.assign(rec=finals[s]).merge(p, on=["origin_last_input_utc", "horizon_hours"])
        ok = ~np.isnan(p["rec"].to_numpy())
        check[f"s{s}_max_abs_diff_vs_saved_kms"] = float(np.max(np.abs(p["rec"][ok] - p["pred_kms"][ok])))
        check[f"s{s}_validation_mse_final_round"] = float(sse[("validation", s, "all")][-1] / cnt[("validation", s, "all")])
    check["seedmean_validation_mse_final_round"] = float(sse[("validation", "mean", "all")][-1] / cnt[("validation", "mean", "all")])
    check["lbeval_reference"] = {"seedmean": 3277.1166, "s0": 3285.89, "s1": 3239.88, "s2": 3305.58}
    print(json.dumps(check, indent=1), flush=True)

    out = []
    for (split, s, k), v in sse.items():
        out.append(pd.DataFrame({"round": np.arange(R1), "split": split, "seed": str(s), "head": k,
                                 "mse": v / cnt[(split, s, k)], "rows": cnt[(split, s, k)]}))
    pd.concat(out).to_csv(HERE / "e1813_loss_history_long.csv", index=False)
    for s in SEEDS:
        pd.DataFrame({"step": np.arange(R1),
                      "train_mse": sse[("train", s, "all")] / cnt[("train", s, "all")],
                      "validation_mse": sse[("validation", s, "all")] / cnt[("validation", s, "all")]}
                     ).to_csv(HERE / f"loss_history_s{s}.csv", index=False)
        for k in HEADS:
            pd.DataFrame({"step": np.arange(R1),
                          "train_mse": sse[("train", s, k)] / cnt[("train", s, k)],
                          "validation_mse": sse[("validation", s, k)] / cnt[("validation", s, k)]}
                         ).to_csv(HERE / f"loss_history_s{s}_{k}.csv", index=False)
    (HERE / "reproduction-check.json").write_text(json.dumps(check, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
