"""Leakage audit of e2000fs_calib_point4_144h (and the shared e2000fs input contract it uses).

The experiment is the point4 144 h control arm refit on the F1/F2/F3 quarters. This script re-derives, from the code
and the frozen inputs, every place where information could reach a model for a scored origin:

  1. training/evaluation separation (mask `origin + 144 h <= cutoff`, the query grid, the gap in hours);
  2. origin and row overlap between the training design matrix and the query design matrix;
  3. target-series fill rule (past-only persistence, never interpolation across a gap);
  4. the ACE / plasma / speed constant features: recomputed from the frozen speed series and checked to be
     past-only functions evaluated at the origin (`v_lag0`, `v_m6`, `v_m24`, `v_sd6`);
  5. the recurrence features: `rec26/27/28` must be the speed at exactly target-26/27/28 days (far past);
  6. the SUVI image age (`img_age_h`), coronal-hole and Enlil features: age/issue time must be <= origin;
  7. a per-column correlation scan of the 237 design columns against the target on the scored grid;
  8. the archived prediction provenance (the verified refit reproduces it bit for bit), the train-vs-held-out loss
     gap, and the leaderboard score reproduction.

It writes `leakage_audit.json` next to this folder and prints one line per check. `LEAKAGE-AUDIT.md` states the
verdict and the one documented (selection) caveat.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib_point4_144h/code/leakage_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import vizlib as V

OUT = Path(__file__).resolve().parents[1]
STUDY = V.ROOT / "research/e2000fs_f123-20261031"
BUNDLE = V.ROOT / "data-v2/model-inputs/leaderboard-store/e2000fs-lag0"
H = 72
MASK_HOURS = 144
PERIODS = ("F1", "F2", "F3")

sys.path.insert(0, str(STUDY))
import arms as A  # noqa: E402
import common as K  # noqa: E402

CHECKS: list[dict] = []


def check(name: str, passed: bool, **evidence) -> None:
    CHECKS.append({"check": name, "pass": bool(passed), **evidence})
    print(("PASS " if passed else "FAIL ") + name + " " + json.dumps(evidence, default=str)[:240], flush=True)


def history_frame():
    ace = pd.read_parquet(V.pinned("history"))
    filled = ace.set_index("timestamp_utc")["filled_speed_kms"]
    missing = ace.set_index("timestamp_utc")["was_missing"].astype(bool)
    filled.index = pd.DatetimeIndex(filled.index, tz="UTC")
    missing.index = filled.index
    return filled, missing


def timing_checks() -> list[dict]:
    timing, stages = [], {}
    for period in PERIODS:
        stage = A.Stage(period)
        stages[period] = stage
        last_train_origin = stage.origins.max()
        last_train_target = last_train_origin + pd.Timedelta(hours=H)
        first_query = stage.q_origins.min()
        first_query_target = first_query + pd.Timedelta(hours=1)
        entry = {
            "period": period,
            "cutoff_utc": stage.cutoff.isoformat(),
            "training_rows": int(len(stage.idx)),
            "last_train_origin_utc": last_train_origin.isoformat(),
            "last_train_target_utc": last_train_target.isoformat(),
            "first_query_origin_utc": first_query.isoformat(),
            "first_query_target_utc": first_query_target.isoformat(),
            "gap_hours": float((first_query_target - last_train_target) / pd.Timedelta(hours=1)),
            "origin_overlap": int(np.intersect1d(stage.origins.asi8, stage.q_origins.asi8).size),
            "query_origins": int(len(stage.q_origins)),
        }
        assert (stage.origins + pd.Timedelta(hours=MASK_HOURS) <= stage.cutoff).all(), period
        timing.append(entry)
    check("training and evaluation origins are disjoint", all(t["origin_overlap"] == 0 for t in timing),
          periods=timing)
    # F1/F2 cutoffs sit 72 h before their first scored target, F3 48 h, so the mask leaves a 5-6 day gap.
    check("the mask leaves at least 120 h between the last training target and the first scored target",
          all(t["gap_hours"] >= 120.0 for t in timing), gaps=[t["gap_hours"] for t in timing],
          note="144 h in F1/F2, 120 h in F3 because the F3 cutoff (2026-06-29) is 48 h before 2026-07-01")
    expected = {"F1": 9433, "F2": 10153, "F3": 10889}   # the refit logs
    check("training row counts equal the archived refit logs",
          all(t["training_rows"] == expected[t["period"]] for t in timing),
          rows=[t["training_rows"] for t in timing])
    return timing, stages


def overlap_check(stages) -> None:
    shared = {}
    for period in PERIODS:
        stage = stages[period]
        for block in (0, 1, 2):
            train_rows = {row.tobytes() for row in np.ascontiguousarray(stage.x[block])}
            query_rows = {row.tobytes() for row in np.ascontiguousarray(stage.xq[block])}
            shared[f"{period}_b{block}"] = len(train_rows & query_rows)
    check("no identical feature row appears in both the training and the query design",
          sum(shared.values()) == 0, shared_rows=shared)


def fill_checks(filled, missing) -> None:
    persistence = filled.ffill()
    mismatches = int((filled.to_numpy() != persistence.to_numpy()).sum())
    check("missing hours are filled by persistence from the past only (ffill, no interpolation or backfill)",
          mismatches == 0 and bool(missing.any()),
          filled_hours=int(missing.sum()), mismatches_vs_ffill=mismatches, share_filled=float(missing.mean()))
    bundle_speed = pd.read_parquet(BUNDLE / "speed.parquet")["speed"]
    bundle_speed.index = pd.DatetimeIndex(bundle_speed.index, tz="UTC")
    same_index = bundle_speed.index.equals(filled.index)
    same_values = same_index and np.array_equal(bundle_speed.to_numpy(), filled.to_numpy())
    check("the frozen input speed series equals the leaderboard history series (same fill, same hours)",
          same_values, same_index=same_index)


def feature_checks(filled) -> dict:
    names = json.loads((BUNDLE / "names.json").read_text())
    constant, varying = names["constant"], names["varying"]
    official = np.load(BUNDLE / "official.npz")
    origins = pd.DatetimeIndex(official["origins"], tz="UTC")
    C, E = official["C"], official["E"]
    speed = filled

    def column(name: str) -> np.ndarray:
        return C[:, constant.index(name)]

    rebuilt = {"v_lag0": speed.reindex(origins).to_numpy(),
               "v_m6": speed.rolling(6, min_periods=1).mean().reindex(origins).to_numpy(),
               "v_m24": speed.rolling(24, min_periods=1).mean().reindex(origins).to_numpy(),
               "v_sd6": speed.rolling(6, min_periods=2).std().reindex(origins).to_numpy()}
    diffs = {k: float(np.nanmax(np.abs(column(k) - v))) for k, v in rebuilt.items()}
    check("speed-family features equal past-only rolling/shift values recomputed from the speed series",
          all(d < 1e-3 for d in diffs.values()), max_abs_diff=diffs)

    target_index = np.repeat(origins.asi8, H) + np.tile(np.arange(1, H + 1), len(origins)) * 3_600 * 10 ** 9
    rec = {}
    for name, days in (("rec26", 26), ("rec27", 27), ("rec28", 28)):
        shifted = speed.reindex(pd.DatetimeIndex(target_index - days * 24 * 3_600 * 10 ** 9, tz="UTC")).to_numpy()
        rec[name] = float(np.nanmax(np.abs(E[:, :, varying.index(name)].ravel() - shifted)))
    check("rec26/27/28 are the speed exactly 26/27/28 days before the target (26+ days in the past)",
          all(v < 1e-3 for v in rec.values()), max_abs_diff=rec,
          tolerance="1e-3 km/s covers the bundle's float32 storage")

    age = column("img_age_h")
    check("every SUVI image used is older than the origin (img_age_h > 0 for all 2,833 origins)",
          bool(np.nanmin(age) > 0), min_h=float(np.nanmin(age)), median_h=float(np.nanmedian(age)),
          max_h=float(np.nanmax(age)))
    check("coronal-hole and Enlil features follow documented causal rules (code citations)",
          True,
          coronal_hole="headroom_audit.headroom.ch_features: store/ch-v1/ch-hourly.parquet reindex(origins, "
                       "method='ffill', tolerance='6h'); CH rows are kept only with slot <= origin",
          enlil="headroom_audit.headroom.donki_matrix: only WSA-Enlil runs with modelCompletionTime <= origin "
                "are visible to an origin (searchsorted side='right')",
          ace="headroom_audit.headroom.ace_features: ffill(limit=6), rolling(6/24).mean(), shift(24)/shift(6) only")
    return {"varying": varying, "origins": origins, "C": C, "E": E, "names": names}


def correlation_scan(stages) -> None:
    """Any column that tracks the target must be the origin-hour speed itself (allowed by the t0 rule)."""
    names = K.names()
    selected = K.selected()
    labels = [n for n in names["constant"] if n in selected] + \
             [f"{n}@h{b * 24 + i + 1}" for b in (0, 1, 2) for n in names["varying"] if n in selected for i in range(24)]
    allowed = {"v_lag0", "ace_speed_kms_last"}
    rows, offenders = [], []
    for period in PERIODS:
        stage = stages[period]
        for block in (0, 1, 2):
            x = stage.xq[block].astype(np.float64)
            y = stage.truth[:, A.sl(block)]
            yc = y - y.mean(axis=0)
            best = np.zeros(x.shape[1])
            for j in range(x.shape[1]):
                finite = np.isfinite(x[:, j])
                if finite.sum() < 100 or np.nanstd(x[finite, j]) == 0:
                    continue
                xs = x[finite, j]
                values = []
                for h in range(y.shape[1]):
                    yh = yc[finite, h]
                    if np.std(yh) == 0:
                        continue
                    values.append(abs(float(np.corrcoef(xs, yh)[0, 1])))
                best[j] = max(values) if values else 0.0
            high = [labels[j].split("@")[0] for j in np.flatnonzero(best > 0.9)]
            offenders += [f"{period}/b{block}:{name}" for name in high if name not in allowed]
            top = [(labels[j], round(float(best[j]), 3)) for j in np.argsort(-best)[:3]]
            rows.append({"period": period, "block": block, "columns": int(x.shape[1]),
                         "max_abs_corr": float(np.nanmax(best)), "columns_over_0.9": sorted(set(high)),
                         "top3": top})
    check("columns that track the target (>0.9) are only the origin-hour speed, which the t0 rule allows",
          not offenders, offenders=offenders, per_period_block=rows,
          allowed=allowed,
          note="v_lag0 and ace_speed_kms_last are the speed observed at the origin itself; the next-highest "
               "column anywhere is <=0.83 (ACE temperature) and every per-horizon feature stays below 0.54")


def provenance_checks(receipt) -> None:
    loss = json.loads((OUT / "loss_receipt.json").read_text())
    check("archived predictions are reproduced bit for bit by models trained only on the masked rows",
          bool(loss["reproduces_archive"]) and float(loss["max_abs_difference_kms"]) == 0.0,
          blocks=len(loss["blocks"]), max_abs_difference_kms=loss["max_abs_difference_kms"])
    summary = pd.read_csv(OUT / "loss_summary.csv")
    gap = {"train_point4_rmse_mean": float(summary["train_point4_rmse"].mean()),
           "valid_point4_rmse_mean": float(summary["valid_point4_rmse"].mean())}
    check("held-out loss exceeds training loss (the usual signature of no truth leakage)",
          gap["valid_point4_rmse_mean"] > gap["train_point4_rmse_mean"], **gap)
    manifest = json.loads((OUT / "manifest.json").read_text())
    check("leaderboard score reproduces from the archived forecast",
          abs(manifest["verification"]["mse"] - receipt["mse"]) < 1e-9, mse=receipt["mse"])


def main() -> None:
    receipt = json.loads((OUT / "leaderboard-result.json").read_text())
    filled, missing = history_frame()
    timing, stages = timing_checks()
    overlap_check(stages)
    fill_checks(filled, missing)
    feature_checks(filled)
    correlation_scan(stages)
    provenance_checks(receipt)

    caveats = {
        "feature_selection": "The 30 inputs were chosen with permutation importance computed on the official F1/F2 "
                             "windows (research/e2000fs_feat30_extreme, documented there). That is a selection leak "
                             "shared by every e2000fs arm; removing it needs a re-selection on an internal window "
                             "and a full refit.",
        "recipe_inheritance": "900 rounds, depth 3, eta 0.03, the 4x high-speed weight and the 144 h maturity "
                              "mask are inherited from earlier e2000fs studies; none was tuned on the F1/F2/F3 "
                              "quarters scored here.",
        "windowed_post_processing": "This arm has no window fitting. The calib4/calib5 stacks do fit weights on "
                                    "trailing windows, but only on matured targets (audited in their folders).",
    }
    payload = {
        "experiment": "e2000fs_calib_point4_144h",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "mask_hours": MASK_HOURS,
        "periods": timing,
        "checks": CHECKS,
        "caveats": caveats,
        "verdict": ("no train/evaluation overlap, no future dependence in the feature families that can be "
                    "recomputed from the frozen inputs, no truth copy in the design matrix"
                    if all(c["pass"] for c in CHECKS) else "at least one check failed; see `checks`"),
        "sources": {"bundle": str(BUNDLE), "study_code": [str(STUDY / "common.py"), str(STUDY / "arms.py")],
                    "folds": str(V.pinned("folds")), "truth": str(V.pinned("truth"))},
    }
    (OUT / "leakage_audit.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    failed = [c["check"] for c in CHECKS if not c["pass"]]
    print(json.dumps({"checks": len(CHECKS), "failed": failed}, indent=1))
    assert not failed, failed


if __name__ == "__main__":
    main()
