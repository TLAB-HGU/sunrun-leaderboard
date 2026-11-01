"""Hourly e2000fs-lag0 input bundles for origins the frozen bundle does not cover (2026-10-31 F1/F2/F3 re-split).

    python data-v2/builders/e2000fs_lag0_extend.py --out data-v2/model-inputs/leaderboard-store/e2000fs-lag0-ext

The frozen `e2000fs-lag0` bundle has hourly queries only for 2026-02-28T23Z..05-28T23Z (f1.npz) and 2026-05-31T23Z.. (official.npz);
its training matrix is every third hour. The new F1 (targets 2026-01-01..03-31) also needs the hourly origins
2025-12-31T23Z..2026-02-28T22Z ("head"), and the new F2 the origins 2026-05-29T00Z..05-31T22Z ("gap").

The features come from the same historical builder the frozen bundle used (`LE.base_grid_for` -> `A.base_grid_for`, ACE evaluated at the
origin, i.e. lag 0), read from the pinned store snapshot. Each segment is built contiguously with a warm-up (speed_d24 is a 24-row
difference), ffilled along the origin axis like the frozen bundle, and checked against the frozen f1.npz / official.npz on the origins both
cover: columns that differ are reported in receipt.json with their mismatch share. Nothing in the frozen bundle is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/t-lab01/sunrun")
DATA = ROOT / "data-v2"
FROZEN = DATA / "model-inputs/leaderboard-store/e2000fs-lag0"
PINNED_STORE = DATA / "raw-store/v2/solar-pivot-noanalog72-20261015/pinned-v1/repo/leaderboard/store"
SUNIMG = DATA / "sun-img"
EXPERIMENTS = ROOT / "leaderboard/model-release-publish/experiments"
LANE = ROOT / "leaderboard/experiments/cumulative_lane12_20261024"
MODULES = ("d147_e0421_lean", "d148_enlil_live", "d143_axisA_analog", "d1_longitude_strips", "headroom_audit")
H = 72
UTC = "UTC"

# (name, build_start, build_end, emit_start, emit_end)  -- all hourly origins, inclusive
SEGMENTS = [
    ("head", "2025-12-20T00:00Z", "2026-03-05T23:00Z", "2025-12-31T23:00Z", "2026-02-28T22:00Z"),
    ("gap", "2026-05-20T00:00Z", "2026-05-31T23:00Z", "2026-05-29T00:00Z", "2026-05-31T22:00Z"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_builder(tmp: Path, enlil_dir: Path | None = None):
    """Import the historical builder modules from a temp tree whose `store` is the pinned snapshot."""
    (tmp / "experiments").mkdir()
    for m in MODULES:
        shutil.copytree(EXPERIMENTS / m, tmp / "experiments" / m, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "store").symlink_to(PINNED_STORE)
    os.environ["EXPOS_OUT"] = str(tmp / "store/headroom-audit")
    for p in [tmp / "experiments" / m for m in ("d147_e0421_lean", "d148_enlil_live", "d143_axisA_analog/a2b",
                                                  "headroom_audit", "d1_longitude_strips")] + [LANE, ROOT / "leaderboard"]:
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(ROOT / "leaderboard/experiments/cumulative_20261024_lane9"))
    import headroom as HR
    import info_ablation as ia
    import lean as L
    import lean_enlil as LE
    import run_official_a2b as A
    import strips as ST
    ia.enlil_runs = LE.enlil_runs
    A.base_grid_for = LE.base_grid_for
    L.feature_columns = LE.feature_columns
    HR.LB = tmp
    HR.OUT = ia.OUT = tmp / "store/headroom-audit"
    if enlil_dir is not None:  # local WSA-Enlil copy for months before 2026-06 (read as OUT/"donki")
        (tmp / "enlil_local").mkdir()
        (tmp / "enlil_local/donki").symlink_to(enlil_dir)
        HR.OUT = ia.OUT = tmp / "enlil_local"
    LE.LIVE = tmp / "store/donki-live"
    LE.RUNS = LE.enlil_runs()
    LE._TABLE.clear()
    ST.SUNIMG = SUNIMG
    import prepare_base111_v2 as P2  # read-only CH selection (SW-008 causal rule), as used for the frozen bundle
    P2.CH = tmp / "store/ch-v1/ch-hourly.parquet"
    return HR, ia, LE, A, ST, P2


def frozen_state(before: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """C and E rows of the frozen (already prefilled) bundle at its last origin before `before`: the ffill state."""
    best, state = None, None
    for n in ("train", "f1", "official"):
        z = np.load(FROZEN / f"{n}.npz")
        o = pd.DatetimeIndex(z["origins"], tz=UTC)
        k = o.searchsorted(before, side="left") - 1
        if k >= 0 and (best is None or o[k] > best):
            best, state = o[k], (z["C"][k].astype(float), z["E"][k].astype(float))
    return state


def build_segment(mods, names: dict, features: list[str], seg: tuple, speed: pd.Series) -> dict:
    HR, ia, LE, A, ST, P2 = mods
    name, b0, b1, e0, e1 = seg
    origins = pd.date_range(b0, b1, freq="h")
    meta, block = ST.load_inputs()
    ace = A.filled_hourly()
    t = time.time()
    g = LE.base_grid_for(origins, ace, HR, meta, block, features)
    g["origin"] = pd.to_datetime(g["origin"], utc=True)
    ch_cols = [c for c in features if c.startswith("ch_")]
    vals = P2.filtered_table(origins, ch_cols)[0]
    for c in ch_cols:
        g[c] = g["origin"].map(vals[c]).astype("float32")
    g = g.sort_values(["origin", "h"]).reset_index(drop=True)
    print(f"[{name}] built {len(origins)} origins x {H} h in {time.time() - t:.0f}s", flush=True)
    n = len(origins)
    assert len(g) == n * H and (g["h"].to_numpy().reshape(n, H) == np.arange(1, H + 1)).all()
    const, var = names["constant"], names["varying"]
    C = np.stack([g[c].to_numpy(float).reshape(n, H)[:, 0] for c in const], axis=1)
    for c in const:  # constant columns really are constant across the 72 horizons
        v = g[c].to_numpy(float).reshape(n, H)
        assert np.array_equal(v, np.repeat(v[:, :1], H, axis=1), equal_nan=True), f"{c} varies with horizon"
    E = np.stack([g[c].to_numpy(float).reshape(n, H) for c in var], axis=2)
    # prefill like the frozen bundle: ffill along the origin axis (history comes from the warm-up rows)
    # the frozen bundle was prefilled along ONE timeline (train + f1 + official), so a value that is empty here
    # (e.g. no pending ENLIL arrival) continues from the frozen bundle's row just before this segment
    sC, sE = frozen_state(origins[0])
    C = pd.DataFrame(np.vstack([sC[None], C])).ffill().to_numpy()[1:]
    E = pd.DataFrame(np.vstack([sE.reshape(1, -1), E.reshape(n, -1)])).ffill().to_numpy()[1:].reshape(n, H, len(var))
    lags = np.arange(-23, 1)
    past = np.stack([speed.reindex(origins + pd.Timedelta(hours=int(k))).to_numpy(float) for k in lags], axis=1)
    return {"origins": origins, "C": C, "E": E, "past": past, "const": const, "var": var}


def pick(seg_out: dict, start, end) -> dict:
    m = (seg_out["origins"] >= pd.Timestamp(start)) & (seg_out["origins"] <= pd.Timestamp(end))
    return {k: (v[m] if isinstance(v, np.ndarray) or isinstance(v, pd.DatetimeIndex) else v) for k, v in seg_out.items()}


def compare(built: dict, frozen_path: Path, label: str, names: dict) -> dict:
    """Mismatch share per column on the origins both bundles cover."""
    z = np.load(frozen_path)
    fo = pd.DatetimeIndex(z["origins"], tz=UTC)
    common = built["origins"].intersection(fo)
    if not len(common):
        return {"label": label, "n_origins": 0}
    bi = built["origins"].get_indexer(common)
    fi = fo.get_indexer(common)
    bad = {}
    for k, c in enumerate(names["constant"]):
        a, b = built["C"][bi, k], z["C"][fi, k].astype(float)
        share = float((~np.isclose(a, b, rtol=1e-4, atol=1e-3, equal_nan=True)).mean())
        if share:
            bad[c] = share
    for k, c in enumerate(names["varying"]):
        a, b = built["E"][bi, :, k], z["E"][fi, :, k].astype(float)
        share = float((~np.isclose(a, b, rtol=1e-4, atol=1e-3, equal_nan=True)).any(axis=1).mean())
        if share:
            bad[c] = share
    pshare = float((~np.isclose(built["past"][bi], z["past"][fi].astype(float), rtol=1e-4, atol=1e-3, equal_nan=True)).any(axis=1).mean())
    if pshare:
        bad["past"] = pshare
    return {"label": label, "n_origins": int(len(common)), "first": str(common.min()), "last": str(common.max()),
            "columns_with_any_mismatch": len(bad), "columns_total": len(names["constant"]) + len(names["varying"]) + 1,
            "mismatch_share_by_column": dict(sorted(bad.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=[s[0] for s in SEGMENTS])
    ap.add_argument("--enlil-dir", type=Path, help="directory with WSAEnlilSimulations_YYYY-MM.json for months before 2026-06")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((FROZEN / "names.json").read_text())
    features = json.loads((LANE / "materialized_v2/features_v2.json").read_text())["features"]
    speed = pd.read_parquet(FROZEN / "speed.parquet")["speed"].sort_index()
    speed.index = pd.to_datetime(speed.index, utc=True)
    receipt = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "frozen_bundle": str(FROZEN),
               "pinned_store": str(PINNED_STORE), "enlil_dir": str(a.enlil_dir), "script_sha256": sha256(Path(__file__)), "segments": {}}
    with tempfile.TemporaryDirectory(prefix="e2000fs_ext_") as td:
        mods = load_builder(Path(td), a.enlil_dir)
        for seg in SEGMENTS:
            if a.only and seg[0] != a.only:
                continue
            built = build_segment(mods, names, features, seg, speed)
            checks = [compare(built, FROZEN / "f1.npz", f"{seg[0]} vs frozen f1.npz", names),
                      compare(built, FROZEN / "official.npz", f"{seg[0]} vs frozen official.npz", names),
                      # train rows are every 3rd hour; speed_d24 there is a 24-ROW (= 72 h) difference, so it is expected to differ
                      compare(built, FROZEN / "train.npz", f"{seg[0]} vs frozen train.npz (3-hourly)", names)]
            emit = pick(built, seg[3], seg[4])
            assert np.isfinite(emit["C"]).all() and np.isfinite(emit["E"]).all() and np.isfinite(emit["past"]).all(), \
                f"{seg[0]}: NaN left after prefill"
            path = out / f"{seg[0]}.npz"
            np.savez_compressed(path, origins=emit["origins"].asi8, C=emit["C"].astype(np.float32),
                                E=emit["E"].astype(np.float32), past=emit["past"].astype(np.float32))
            receipt["segments"][seg[0]] = {"emit": [seg[3], seg[4]], "n_origins": int(len(emit["origins"])),
                                           "sha256": sha256(path), "checks": checks}
            for c in checks:
                print(f"[{seg[0]}] {c['label']}: {c.get('n_origins')} common origins, "
                      f"{c.get('columns_with_any_mismatch')} of {c.get('columns_total')} columns differ", flush=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
