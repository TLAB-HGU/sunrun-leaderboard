"""Build the persistence-image feature cache for every admissible SUVI Fe195 frame.

Resumable: frames are split into a deterministic list of parts; a part that is
already on disk is skipped. Run under the lbstatus wrapper:

    python build_cache.py [--workers N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/t-lab01/sunrun")
DB = ROOT / "data-v2/suvi/solar_g18.duckdb"
NCDIR = ROOT / "data-v2/suvi"
OUT = Path(__file__).resolve().parent / "features"
PART_SIZE = 1000

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tda  # noqa: E402


def resolve(nc_path: str) -> str:
    p = str(nc_path)
    if p.startswith("/"):
        p = "data/" + p.split("/data/", 1)[1]
    return p if p.startswith("/") else str(NCDIR / p)


def frame_list(start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = pd.read_sql(
        "select slot, nc_path from suvi_hourly "
        "where status='ok' and nc_path is not null and coalesce(eclipse,0)<>2 "
        f"and slot >= TIMESTAMP '{start}' and slot < TIMESTAMP '{end}' order by slot",
        con,
    )
    con.close()
    df["slot"] = pd.to_datetime(df["slot"], utc=True)
    df["path"] = df["nc_path"].map(resolve)
    return df[["slot", "path"]]


def _work(item):
    slot, path = item
    try:
        vec, meta = tda.features(path)
        return slot, vec, meta, None
    except Exception as exc:  # noqa: BLE001
        return slot, None, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TDA_WORKERS", "20")))
    ap.add_argument("--start", default="2022-08-20")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--parts-only", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frames = frame_list(args.start, args.end)
    parts = [frames.iloc[i:i + PART_SIZE] for i in range(0, len(frames), PART_SIZE)]
    print(f"frames={len(frames)} parts={len(parts)} workers={args.workers}", flush=True)

    wanted = {int(x) for x in args.parts_only.split(",") if x.strip()} if args.parts_only else None
    todo = [k for k in range(len(parts)) if (wanted is None or k in wanted)
            and not (OUT / f"part-{k:04d}.parquet").exists()]
    print(f"todo parts={len(todo)} (done={len(parts) - len(todo) - (len(parts) - len(todo))})", flush=True)

    cols = [f"f{i:03d}" for i in range(tda.NFEAT)]
    t0 = time.time()
    done_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
        for k in todo:
            chunk = parts[k]
            res = list(pool.map(_work, list(zip(chunk["slot"], chunk["path"])), chunksize=8))
            ok = [r for r in res if r[1] is not None]
            bad = [r for r in res if r[1] is None]
            if not ok:
                raise SystemExit(f"part {k}: every frame failed, first error {bad[0][3] if bad else 'n/a'}")
            df = pd.DataFrame(np.vstack([r[1] for r in ok]), columns=cols)
            df.insert(0, "slot", pd.DatetimeIndex([r[0] for r in ok], tz="UTC"))
            for key in ("n_h0", "n_h1", "max_h1_persistence", "shape"):
                df[key] = [r[2][key] for r in ok]
            tmp = OUT / f".part-{k:04d}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, OUT / f"part-{k:04d}.parquet")
            done_rows += len(ok)
            rate = done_rows / max(time.time() - t0, 1e-9)
            eta = (len(frames) - sum(len(p) for p in parts[:k + 1])) / max(rate, 1e-9) / 60
            print(f"part {k:04d}: +{len(ok)} ok, {len(bad)} failed, {rate:.1f} frames/s, eta {eta:.1f} min",
                  flush=True)
            if bad:
                print("  first failures:", bad[:2], flush=True)

    have = sorted(OUT.glob("part-*.parquet"))
    manifest = {
        "frames": int(len(frames)),
        "parts": len(have),
        "params": {"A": tda.A, "GRID": tda.GRID, "SIGMA": tda.SIGMA, "REBIN": tda.REBIN,
                   "weight": "w(p)=p", "filtration": "sublevel on inverted asinh-normalized radiance"},
        "code_sha256": hashlib.sha256(Path(tda.__file__).read_bytes()).hexdigest(),
        "source_db": str(DB),
        "window": [args.start, args.end],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", OUT / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
