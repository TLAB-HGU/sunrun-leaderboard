"""e2000fs_calib11g_rawdf: the board-leading calib11g recipe, re-fitted on data-v2/input/df.parquet.

This is a port, not a new hypothesis. The parent (`research/e2000fs_calib11g_serarounds2/experiment.py`)
is frozen except for the three changes that the new input table forces:

1. `ch_ch_t035_*` / `ch_ch_t055_*` -> `suvi_t035_*` / `suvi_t055_*` (the new table computes the SUVI
   statistics from the raw NetCDF itself, so the values differ from the old frozen bundle).
2. Masked squared error: the new table's `y` exists only where ACE SWEPAM status == 0 (32.18 % NaN),
   which is the official scoring rule. Missing targets get zero grad/hess weight; an origin whose whole
   24 h block is unobserved is dropped from that block's training rows. No filling of missing truth.
3. Per-period booster rounds are re-selected on the quarter that ends immediately before each period's
   training cutoff (rolling origin, same rule the parent used on the old table).

Everything else - the 30-column design (21 constant + 9 varying x 24 h = 237 columns per block), the
level target, the SERA relevance weights with the median anchor, the three 24-h vector-leaf boosters per
period, depth 3, lr 0.03, subsample 0.8, colsample 0.7, min_child_weight 20, reg_lambda 5, seed 42 and the
144 h maturity mask - is unchanged.

    python experiment.py --phase observations        # raw ACE -> observations.parquet (+ y agreement check)
    python experiment.py --phase select --jobs 5     # per-period rounds on the pre-cutoff quarter
    python experiment.py --all --jobs 5              # B0 fit at the selected rounds, then merge
    python experiment.py --phase score               # official scorer: total / F1 / F2 / F3
    python experiment.py --smoke                    # 2-round pipeline check in /tmp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/home/t-lab01/sunrun")
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data-v2/input/df.parquet"
RAW_ACE = ROOT / "data-v2/raw/ACE/monthly-txt"
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT / "leaderboard/scorer"))
import competition as C  # noqa: E402

H, BLOCK, MATURITY_H = 72, 24, 144
PERIODS = {"F1": 2089, "F2": 2113, "F3": 2113}  # period -> query origins (hourly)
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
              reg_lambda=5, objective="reg:squarederror", tree_method="hist", device="cpu",
              multi_strategy="multi_output_tree", eval_metric="rmse", seed=42, verbosity=0, nthread=2)
# parent's 30-column design, with the two SUVI dark-area prefixes renamed
CONSTANT = ["v_lag0", "ace_speed_kms_last", "ace_density_cm3_last", "ace_density_cm3_m6", "ace_temperature_k_last",
            "ace_temperature_k_m24", "ace_bt_nt_last", "ace_bt_nt_m6", "ace_by_gsm_nt_m24", "ace_epam_p_channel3_last",
            "ace_sis_gt10_flux_last", "suvi_t035_top2_area", "suvi_t035_meridian15_equatorial_area",
            "suvi_t035_meridian30_equatorial_area", "suvi_t035_equatorial_area", "suvi_t055_meridian15_area",
            "suvi_t055_meridian30_equatorial_area", "suvi_t055_equatorial_area", "img_age_h", "enlil_n_pending",
            "enlil_kp"]
VARYING = ["rec26", "rec27", "rec28", "rec_mean", "strip_v350_logmed", "strip_v450_logmed", "strip_v550_dark",
           "strip_v650_dark", "strip_v750_logmed"]
COLS = ["set", "origin", "h", "period", "fit_F1", "fit_F2", "fit_F3", "y", *CONSTANT, *VARYING]
# rolling-origin selection windows: (start, end, training cutoff of the window) - the parent's D6/E1/E2
WINDOWS = {"F1": ("2025-10-01T00:00Z", "2026-01-01T00:00Z", "2025-09-29T00:00Z"),
           "F2": ("2026-01-01T00:00Z", "2026-04-01T00:00Z", "2025-12-29T00:00Z"),
           "F3": ("2026-04-01T00:00Z", "2026-07-01T00:00Z", "2026-03-29T00:00Z")}
CANDIDATES = (200, 300, 450, 600, 900, 1200)

_CACHE: dict[str, pd.DataFrame] = {}


# --------------------------------------------------------------------------------------- data

def load(period: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not _CACHE:
        _CACHE["df"] = pd.read_parquet(DATA, columns=COLS)
    df = _CACHE["df"]
    train = df[df[f"fit_{period}"].astype(bool)]
    query = df[(df["set"] == "query") & (df["period"] == period)]
    return df, train, query


def table(rows: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Long rows -> (origins, constants[origin, f], varying[origin, h, f], y[origin, h]). Target may be NaN."""
    rows = rows.sort_values(["origin", "h"], kind="stable")
    n = len(rows) // H
    assert len(rows) == n * H and (rows["h"].to_numpy().reshape(n, H) == np.arange(1, H + 1)).all(), "not a full origin x 72 grid"
    origins = pd.DatetimeIndex(rows["origin"].to_numpy()[::H], tz="UTC")
    const = rows[CONSTANT].to_numpy(np.float32).reshape(n, H, len(CONSTANT))[:, 0, :]
    vary = rows[VARYING].to_numpy(np.float32).reshape(n, H, len(VARYING))
    return origins, const, vary, rows["y"].to_numpy(np.float64).reshape(n, H)


def design(const: np.ndarray, vary: np.ndarray, start: int) -> np.ndarray:
    return np.hstack([const, vary[:, start:start + BLOCK, :].reshape(len(const), -1)]).astype(np.float32)


def build_observations(out: Path) -> dict:
    """Canonical truth from the NASA monthly files: status == 0, finite, positive, no fill."""
    rows = []
    for f in sorted(RAW_ACE.glob("*_ace_swepam_1h.txt")):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) != 10 or not (p[0].isdigit() and len(p[0]) == 4):
                continue
            rows.append((pd.Timestamp(int(p[0]), int(p[1]), int(p[2]), int(p[3]) // 100, tz="UTC"),
                         int(p[6]), float(p[8])))
    raw = (pd.DataFrame(rows, columns=["t", "status", "spd"]).sort_values("t").drop_duplicates("t", keep="last")
           .set_index("t"))
    valid = raw["status"].eq(0) & np.isfinite(raw["spd"]) & raw["spd"].gt(0)
    obs = pd.DataFrame({"timestamp_utc": raw.index, "speed_kms": raw["spd"].where(valid).to_numpy()})
    out.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(out / "observations.parquet", index=False)
    # agreement check against the input table's y (must be exact: same rule, same source)
    yy = pd.read_parquet(DATA, columns=["target_time", "y"]).groupby("target_time")["y"].first()
    j = obs.set_index("timestamp_utc")["speed_kms"].align(yy, join="inner")
    a, b = j[0], j[1]
    both = a.notna() & b.notna()
    agree = int(np.isclose(a[both], b[both]).sum())
    nan_agree = int((a.isna() == b.isna()).sum())
    report = {"phase": "observations", "raw_hours": int(len(obs)), "status0_hours": int(obs["speed_kms"].notna().sum()),
              "overlap_with_y": int(len(a)), "both_present": int(both.sum()), "identical": agree,
              "nan_pattern_equal": nan_agree, "missing_on_one_side": int((both.sum()) - agree)}
    print(json.dumps(report), flush=True)
    assert agree == both.sum() and nan_agree == len(a), "observations disagree with the input table's y"
    (out / "observations_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def context(periods: dict | None, out: Path) -> dict:
    obs = C.load_observations(out / "observations.parquet")
    return C.prepare_context(obs, periods)


# ------------------------------------------------------------------------------------ objective

def masked_weighted_l2(weights: np.ndarray, mask: np.ndarray):
    """SERA weights (median-anchored) with unobserved targets masked out of grad and hess.

    `mask` is 1 where the target was observed. Both the anchor (median) and the mask are computed from
    observed entries only, so unobserved truth never enters the fit in any form. xgboost rejects NaN
    labels, so the DMatrix carries a sanitised label (0 where missing) and the mask is applied here.
    """
    med = np.nanmedian(np.where(mask > 0, weights, np.nan), axis=0)
    # `weights` is NaN where the target is missing; mask first so 0 * NaN cannot leak into grad/hess
    w = (np.where(mask > 0, weights, 0.0) / med[None, :]).astype(np.float32)
    keep = mask.astype(np.float32)

    def obj(pred, matrix):
        target = matrix.get_label().reshape(pred.shape)  # sanitised: 0 where unobserved
        grad = (pred - target) * keep * w
        return grad.astype(np.float32), (keep * w).astype(np.float32)

    return obj


def fit(c: np.ndarray, e: np.ndarray, y: np.ndarray, start: int, rounds: int, weights: np.ndarray,
        seed: int = 42, nthread: int = 2) -> xgb.Booster:
    label = y[:, start:start + BLOCK]
    mask = np.isfinite(label)
    obj = masked_weighted_l2(weights, mask)
    matrix = xgb.DMatrix(design(c, e, start), label=np.nan_to_num(label, nan=0.0).astype(np.float32),
                         nthread=nthread)
    base = float(np.nanmean(label))
    model = xgb.train(dict(PARAMS, seed=seed, base_score=base, nthread=nthread), matrix,
                      num_boost_round=rounds, obj=obj, verbose_eval=False)
    return model


def sera_weights(tr_y: np.ndarray) -> np.ndarray:
    """w = (y - q0)/(q100 - q0) per lead, from observed labels only (parent's SERA ramp)."""
    q0, q1 = np.nanmin(tr_y, axis=0), np.nanmax(tr_y, axis=0)
    return np.clip((tr_y - q0[None, :]) / (q1 - q0)[None, :], 1e-3, None)


# --------------------------------------------------------------------------------------- phases

def phase_weights(period: str, out: Path) -> dict:
    t0 = time.perf_counter()
    _, train, _ = load(period)
    tr_o, _, _, tr_y = table(train)
    w = sera_weights(tr_y)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{period}_w.npz", w=w.astype(np.float32), origins=tr_o.asi8)
    report = {"phase": "weights", "scheme": "sera", "period": period, "train_origins": len(tr_y),
              "observed_fraction": float(np.isfinite(tr_y).mean()),
              "weight_pct": {q: float(np.nanpercentile(w, q)) for q in (50, 90, 99, 99.9, 100)},
              "seconds": round(time.perf_counter() - t0)}
    print(json.dumps(report), flush=True)
    return report


def fit_from_arrays(period: str, block: int, rounds: int, out: Path, tr_arr: tuple, qu_arr: tuple,
                    seed: int = 42) -> dict:
    """One (period, block) fit from precomputed float32 arrays (keeps the sweep's resident set small)."""
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    tr_o, tr_c, tr_e, tr_y = tr_arr
    qu_o, qu_c, qu_e, _ = qu_arr
    w = sera_weights(tr_y)
    start = block * BLOCK
    keep_row = np.isfinite(tr_y[:, start:start + BLOCK]).any(axis=1)
    model = fit(tr_c[keep_row], tr_e[keep_row], tr_y[keep_row], start, rounds,
                w[keep_row][:, start:start + BLOCK], seed=seed)
    t1 = time.perf_counter()
    pred = model.inplace_predict(design(qu_c, qu_e, start))
    per_fold = (time.perf_counter() - t1) / len(qu_o)
    np.savez_compressed(out / f"{period}_b{block}.npz", pred=pred, origins=qu_o.asi8)
    report = {"phase": "fit", "period": period, "block": block, "rounds": rounds, "seed": seed,
              "train_origins": int(len(tr_y)), "kept_rows": int(keep_row.sum()),
              "dropped_unobserved_block": int((~keep_row).sum()),
              "observed_fraction": float(np.isfinite(tr_y).mean()), "queries": int(len(qu_o)),
              "inference_seconds_per_fold": per_fold, "seconds": round(time.perf_counter() - t0)}
    print(json.dumps(report), flush=True)
    return report


def prepare_arrays(rows: pd.DataFrame) -> tuple:
    """Long rows -> (origins, const float32, varying float32, y float32) without keeping the frame alive."""
    o, c, e, y = table(rows)
    return o, c, e, y.astype(np.float32)


def _fit_block(period: str, block: int, rounds: int, out: Path, tr: pd.DataFrame | None = None,
               query: pd.DataFrame | None = None, seed: int = 42) -> dict:
    if tr is None:
        _, tr, query = load(period)
    return fit_from_arrays(period, block, rounds, out, prepare_arrays(tr), prepare_arrays(query), seed=seed)


def assemble(period: str, out: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    parts = [np.load(out / f"{period}_b{b}.npz") for b in range(3)]
    pred = np.hstack([np.asarray(z["pred"], np.float64) for z in parts])
    origins = pd.DatetimeIndex(parts[0]["origins"], tz="UTC")
    assert pred.shape == (len(origins), H) and np.isfinite(pred).all(), (period, pred.shape)
    return pred, origins


def frame(pred: np.ndarray, origins: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                         "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)),
                         "pred_kms": pred.ravel()})


def phase_select(out: Path, jobs: int, candidates=CANDIDATES) -> dict:
    """Per-period rounds on the quarter immediately before that period's training cutoff.

    All (period, candidate rounds, block) fits go to one pool so the sweep is limited by total rounds, not
    by the 3 blocks of a single candidate. Each candidate writes to its own directory.
    """
    sel_ctx = C.prepare_context(C.load_observations(out / "observations.parquet"),
                                {f"F{i+1}": WINDOWS[p][:2] for i, p in enumerate(PERIODS)})
    slot = {"F1": "f1", "F2": "f2", "F3": "f3"}
    df = pd.read_parquet(DATA, columns=COLS)
    sets = {}
    for period, (w_start, w_end, w_cut) in WINDOWS.items():
        first, end, cut = (pd.Timestamp(t) for t in (w_start, w_end, w_cut))
        tr = df[(df["set"] == "train") & (df["origin"] + pd.Timedelta(hours=MATURITY_H) <= cut)]
        qu = df[(df["set"] == "train") & (df["origin"] >= first) & (df["origin"] < end)]
        origins = pd.DatetimeIndex(sorted(qu["origin"].unique()), tz="UTC")
        origins = origins[(origins.hour % 6 == 0) & (origins <= end - pd.Timedelta(hours=73))]
        qu = qu[qu["origin"].isin(origins)]
        period_start = pd.Timestamp(C.PERIODS[period][0], tz="UTC")
        assert (origins < period_start).all(), f"{period}: selection window is not strictly before the forecast period"
        assert cut <= pd.Timestamp(WINDOWS[period][2]), f"{period}: window cutoff is later than the window rule"
        # float32 arrays only: the sweep must not hold three long-format frames alive at once
        sets[period] = (prepare_arrays(tr), prepare_arrays(qu))
        print(json.dumps({"phase": "select", "period": period, "window": [str(first), str(end)], "cutoff": str(cut),
                          "train_origins": int(len(sets[period][0][0])), "window_origins": int(len(origins)),
                          "candidates": list(candidates)}), flush=True)
    del df

    tasks = []
    for p in PERIODS:
        for r in candidates:
            done = all((out / "select" / p / str(r) / f"{p}_b{b}.npz").exists() for b in range(3))
            if not done:
                tasks += [(p, r, b) for b in range(3)]
    t0 = time.perf_counter()
    print(json.dumps({"phase": "select", "pending_fits": len(tasks), "jobs": jobs}), flush=True)

    def run(t):
        period, rounds, block = t
        tr_arr, qu_arr = sets[period]
        return fit_from_arrays(period, block, rounds, out / "select" / period / str(rounds), tr_arr, qu_arr)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, tasks))
    print(json.dumps({"phase": "select", "fits": len(tasks), "seconds": round(time.perf_counter() - t0)}), flush=True)

    chosen, report = {}, {}
    for period in PERIODS:
        scores = {}
        for rounds in candidates:
            pred, og = assemble(period, out / "select" / period / str(rounds))
            res = C.score_competition(frame(pred, og), sel_ctx)
            scores[rounds] = {"s": res[slot[period]]["s"], "rmse": res[slot[period]]["rmse"],
                              "timse": res[slot[period]]["timse"], "bias": res[slot[period]]["bias"]}
            print(json.dumps({"phase": "select", "period": period, "rounds": rounds, **scores[rounds]}), flush=True)
        best = min(scores, key=lambda r: (round(scores[r]["s"], 6), r))
        chosen[period] = best
        report[period] = {"window": list(WINDOWS[period]), "scores": scores,
                          "chosen_rounds": best, "s_at_chosen": scores[best]["s"]}
    (out / "select.json").write_text(json.dumps({"chosen": chosen, "detail": report}, indent=1) + "\n")
    print(json.dumps({"phase": "select", "chosen": chosen}), flush=True)
    return chosen


def rounds_by_period(out: Path, override: int | None = None) -> dict:
    if override:
        return {p: override for p in PERIODS}
    path = out / "select.json"
    assert path.exists(), f"no {path}; run --phase select first (rounds must be re-selected on the new table)"
    chosen = json.loads(path.read_text())["chosen"]
    return {p: int(chosen[p]) for p in PERIODS}


def phase_score(out: Path, periods: dict | None = None) -> dict:
    ctx = context(periods, out)
    preds = pd.concat([frame(*assemble(p, out)) for p in PERIODS], ignore_index=True)
    res = C.score_competition(preds, ctx)
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_score.json").write_text(json.dumps(res, indent=1) + "\n")
    flat = {k: {m: res[k][m] for m in ("s", "timse", "rmse", "trmse", "bias", "n_pairs", "n_replaced")}
            for k in ("total", "f1", "f2", "f3")}
    print(json.dumps({"phase": "score", **flat}, indent=1), flush=True)
    rep = res["total"].get("replacement_fraction")
    assert rep == 0.0, f"persistence replacement fraction must be 0, got {rep}"
    return res


def phase_diagnose(out: Path) -> dict:
    """Where the remaining S lives: per-period error share, lead bands, and loss concentration."""
    res = json.loads((out / "local_score.json").read_text())
    obs = C.load_observations(out / "observations.parquet").set_index("timestamp_utc")["speed_kms"]
    diag = {"periods": {}, "lead_bands": {}, "concentration": {}}
    total_sq, total_n = 0.0, 0
    for p in PERIODS:
        pred, og = assemble(p, out)
        truth = np.column_stack([obs.reindex(og + pd.Timedelta(hours=int(h))).to_numpy() for h in np.arange(1, H + 1)])
        err = pred - truth
        sq = np.where(np.isfinite(err), err ** 2, 0.0)
        n = int(np.isfinite(err).sum())
        total_sq += float(sq.sum())
        total_n += n
        bands = {}
        for lo, hi in ((1, 24), (25, 48), (49, 72)):
            e = err[:, lo - 1:hi]
            b = np.isfinite(e)
            bands[f"h{lo}-{hi}"] = {"rmse": float(np.sqrt((e[b] ** 2).mean())), "bias": float(e[b].mean()),
                                    "n": int(b.sum())}
        per_origin = np.where(np.isfinite(err), err ** 2, 0.0).sum(axis=1)
        order = np.argsort(per_origin)[::-1]
        slot = {"F1": "f1", "F2": "f2", "F3": "f3"}[p]
        diag["periods"][p] = {"s": res[slot]["s"], "rmse": res[slot]["rmse"], "timse": res[slot]["timse"],
                              "bias": res[slot]["bias"], "n_pairs": n, "sum_sq": float(sq.sum()),
                              "origins": int(len(og))}
        diag["lead_bands"][p] = bands
        diag["concentration"][p] = {
            "top20_share": float(per_origin[order[:20]].sum() / max(per_origin.sum(), 1e-9)),
            "origins_for_50pct": int(np.searchsorted(np.cumsum(per_origin[order]) / per_origin.sum(), 0.5) + 1),
            "share_of_sq": None}
    for p in PERIODS:
        diag["periods"][p]["share_of_sq"] = diag["periods"][p]["sum_sq"] / total_sq
        diag["concentration"][p]["share_of_sq"] = diag["periods"][p]["share_of_sq"]
    diag["total_sq_share_note"] = "share_of_sq = period share of pooled squared error over the same valid pairs"
    (out / "diagnosis.json").write_text(json.dumps(diag, indent=1) + "\n")
    print(json.dumps(diag, indent=1), flush=True)
    return diag


def phase_provenance(out: Path, rounds: dict, jobs: int) -> dict:
    import xgboost, pandas, numpy
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {"experiment": HERE.name, "parent": "e2000fs_calib11g_serarounds2",
            "data": str(DATA), "git_commit": commit,
            "versions": {"xgboost": xgboost.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__},
            "seed": 42, "rounds_by_period": rounds, "jobs": jobs,
            "features": {"constant": CONSTANT, "varying": VARYING,
                         "columns_per_block": len(CONSTANT) + len(VARYING) * BLOCK,
                         "count": len(CONSTANT) + len(VARYING)},
            "blocks": [[b * BLOCK + 1, (b + 1) * BLOCK] for b in range(3)],
            "params": {k: v for k, v in PARAMS.items() if k != "objective"},
            "objective": "masked SERA-weighted squared error (median anchor, unobserved target weight 0)",
            "training_cutoffs": {p: str(C.PERIODS[p][0]) for p in PERIODS},
            "selection_windows": {p: {"window": list(WINDOWS[p][:2]), "window_training_cutoff": WINDOWS[p][2]}
                                  for p in PERIODS},
            "score_version": C.VERSION}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["observations", "select", "weights", "fit", "merge", "score",
                                        "diagnose"])
    ap.add_argument("--period", choices=sorted(PERIODS))
    ap.add_argument("--block", type=int, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--origin-stride", type=int, default=1,
                    help="keep only training origins whose UTC hour is a multiple of this value (ablation arm)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        smoke = Path("/tmp/calib11g_rawdf_smoke")
        build_observations(RESULTS)  # canonical truth is shared, cheap, and asserted
        _, train, query = load("F1")
        tr = train[train["origin"] < pd.Timestamp("2024-01-01", tz="UTC")]
        qu = query[query["origin"] < query["origin"].min() + pd.Timedelta(hours=24)]
        for b in range(3):
            _fit_block("F1", b, 2, smoke, tr=tr, query=qu)
        pred, og = assemble("F1", smoke)
        res = C.score_competition(frame(pred, og), context(None, RESULTS))
        print(json.dumps({"phase": "smoke", "origins": len(og), "s": res["f1"]["s"],
                          "replaced": res["f1"]["n_replaced"]}), flush=True)
        return
    if a.phase == "observations":
        build_observations(a.out)
        return
    if a.phase == "select":
        phase_select(a.out, a.jobs)
        return
    if a.all:
        a.out.mkdir(parents=True, exist_ok=True)
        rounds = rounds_by_period(a.out, a.rounds)
        print(json.dumps({"phase": "rounds", "rounds": rounds}), flush=True)
        pending = [(p, b) for p in PERIODS for b in range(3)
                   if not (a.out / f"{p}_b{b}.npz").exists()]
        print(json.dumps({"phase": "fits", "pending": len(pending), "jobs": a.jobs}), flush=True)
        if pending:
            # float32 arrays for the pending periods, then one pool over all (period, block) fits
            arrs = {}
            for p in sorted({x[0] for x in pending}):
                _, tr, qu = load(p)
                if a.origin_stride > 1:
                    tr = tr[tr["origin"].dt.hour % a.origin_stride == 0]
                    print(json.dumps({"phase": "stride", "period": p, "stride": a.origin_stride,
                                      "train_origins": int(tr["origin"].nunique())}), flush=True)
                arrs[p] = (prepare_arrays(tr), prepare_arrays(qu))
                del tr, qu
            with ThreadPoolExecutor(max_workers=a.jobs) as pool:
                list(pool.map(lambda pb: fit_from_arrays(pb[0], pb[1], rounds[pb[0]], a.out,
                                                        arrs[pb[0]][0], arrs[pb[0]][1], seed=a.seed),
                              pending))
            del arrs
        for p in PERIODS:
            pred, og = assemble(p, a.out)
            np.savez_compressed(a.out / f"{p}_predictions.npz", pred=pred, origins=og.asi8)
        allp = pd.concat([frame(*assemble(p, a.out)) for p in PERIODS], ignore_index=True)
        allp.to_parquet(a.out / "predictions.parquet", index=False)
        merge_report = {p: {"origins": int(PERIODS[p])} for p in PERIODS}
        merge_report["rows"] = len(allp)
        (a.out / "merge.json").write_text(json.dumps(merge_report, indent=1) + "\n")
        phase_provenance(a.out, rounds, a.jobs)
        phase_score(a.out)
        return
    if a.phase == "weights":
        phase_weights(a.period, a.out)
    elif a.phase == "fit":
        _fit_block(a.period, a.block or 0, rounds_by_period(a.out, a.rounds)[a.period], a.out, seed=a.seed)
    elif a.phase == "merge":
        allp = pd.concat([frame(*assemble(p, a.out)) for p in PERIODS], ignore_index=True)
        allp.to_parquet(a.out / "predictions.parquet", index=False)
        print(f"{len(allp)} rows", flush=True)
    elif a.phase == "score":
        phase_score(a.out)
    elif a.phase == "diagnose":
        phase_diagnose(a.out)


if __name__ == "__main__":
    main()
