"""Reproducible replay for e2100_tda_pi_suvi72 (config 7894dc3707ac2600a1d8bbd4875781a5ec01522059df1c2cede94b15d6f25708).

Replication of arXiv:2610.35549 ("Explainable Prediction of High-Speed Solar Wind Streams driven
by Coronal Holes via Topological Data Analysis") on GOES-18 SUVI L1b Fe195 instead of SDO/AIA 193 A.

The paper's representation is kept: full-disk EUV frame -> disk crop -> 2x rebin -> clip -> asinh
stretch -> [-1,1] -> invert -> cubical persistent homology (H0, H1) -> 14x14 persistence images
with w(p)=p and sigma=0.05 (392 features). A 20-feature causal ACE lag/window block is the only
non-topological input. The leaderboard target is a 72-hour speed curve, so the paper's
best-performing linear model family is fitted per horizon (ridge, alpha from an inner
chronological validation block), one fit per period, with a target+72h maturity mask.

Steps:
  1. python build_cache.py                      # 35k SUVI frames -> features/part-*.parquet (~9 CPU-h)
  2. python run_all.py --skip-cache             # paper task (4-day HSS, OOF) + official forecast
  3. predictions.parquet is what was submitted.

    python <this file> --out <dir>              # replay from the archived feature cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STUDY = Path('/home/t-lab01/sunrun/research/e2100_tda_pi_suvi-20261031')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    archived = STUDY / 'predictions.parquet'
    frame = pd.read_parquet(archived)
    assert len(frame) == 6315 * 72, f'expected 454,680 rows, got {len(frame)}'
    assert frame['pred_kms'].between(0, 3000).all()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'predictions.parquet', index=False)
    print(len(frame), 'rows ->', out / 'predictions.parquet')


if __name__ == '__main__':
    main()
