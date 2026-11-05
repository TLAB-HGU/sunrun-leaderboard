"""Shared data layer for the e2000fs re-runs on the F1/F2/F3 quarters (user instruction 2026-10-31).

Periods (targets, [start, end)): F1 2026-01..03, F2 2026-04..06, F3 2026-07..09-29. A period's model is trained and tuned only on
rows whose targets are matured before its cutoff = period start - 72 h (origin + 72 h < cutoff, the 72 h maturity rule):
F1 < 2025-12-29, F2 < 2026-03-29, F3 < 2026-06-29. The 30 selected features and the XGBoost recipe are the frozen
e2000fs_dirmo_24_feat30_pointweight4 ones; the selection itself was made earlier (a known, documented limitation).

Inputs: the frozen e2000fs-lag0 bundle (train.npz is every third hour up to 2026-05-28T21Z; f1/official are hourly queries) plus the
hourly extension bundles from data-v2/builders/e2000fs_lag0_extend.py (head, gap). The F3 fit needs training rows after the frozen
train.npz ends (2026-05-29..06-26); they are taken from the hourly bundles at the same every-third-hour stride.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/t-lab01/sunrun")
STORE = ROOT / "data-v2/model-inputs/leaderboard-store"
FROZEN = STORE / "e2000fs-lag0"
EXT = STORE / "e2000fs-lag0-ext"
SOURCE_MANIFEST = ROOT / "research/e2000fs_dirmo_24_feat30_pointweight4/results/runs/full_v1/manifest.json"
H = 72
MATURITY = pd.Timedelta(hours=H)
UTC = "UTC"

PERIODS = {  # name -> (first query origin, last query origin, training cutoff)
    "F1": ("2025-12-31T23:00Z", "2026-03-28T23:00Z", "2025-12-29T00:00Z"),
    "F2": ("2026-03-31T23:00Z", "2026-06-27T23:00Z", "2026-03-29T00:00Z"),
    "F3": ("2026-06-30T23:00Z", "2026-09-26T23:00Z", "2026-06-29T00:00Z"),
}
N_FOLDS = {"F1": 2089, "F2": 2113, "F3": 2113}


def names() -> dict:
    return json.loads((FROZEN / "names.json").read_text())


def selected() -> list[str]:
    sel = json.loads(SOURCE_MANIFEST.read_text())["selected"]
    assert len(set(sel)) == 30
    return sel


def speed() -> pd.Series:
    s = pd.read_parquet(FROZEN / "speed.parquet")["speed"].sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def targets(series: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    t = origins.values[:, None] + np.arange(1, H + 1).astype("timedelta64[h]")[None, :]
    return series.reindex(pd.DatetimeIndex(t.ravel(), tz=UTC)).to_numpy(float).reshape(-1, H)


def design(bundle: dict, nm: dict, sel: list[str], start: int) -> np.ndarray:
    """Selected constant columns + selected varying columns of horizons start..start+23 (same as the frozen trainers)."""
    ci = [i for i, n in enumerate(nm["constant"]) if n in sel]
    ei = [i for i, n in enumerate(nm["varying"]) if n in sel]
    return np.hstack([bundle["C"][:, ci], bundle["E"][:, start:start + 24, :][:, :, ei].reshape(len(bundle["C"]), -1)]).astype(np.float32)


def _load(path: Path) -> dict:
    z = np.load(path)
    return {"origins": pd.DatetimeIndex(z["origins"], tz=UTC), "C": z["C"], "E": z["E"], "past": z["past"]}


def _cat(parts: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("C", "E", "past")}
    out["origins"] = pd.DatetimeIndex(np.concatenate([p["origins"].asi8 for p in parts]), tz=UTC)
    order = np.argsort(out["origins"].asi8, kind="stable")
    return {k: (v[order] if k != "origins" else v[order]) for k, v in out.items()}


def hourly_queries() -> dict:
    """Hourly bundle 2025-12-31T23Z..2026-09-26T23Z: head + frozen f1 + gap + frozen official (contiguous, unique)."""
    q = _cat([_load(EXT / "head.npz"), _load(FROZEN / "f1.npz"), _load(EXT / "gap.npz"), _load(FROZEN / "official.npz")])
    o = q["origins"]
    assert o.is_unique and (np.diff(o.asi8) == 3_600_000_000_000).all(), "hourly query timeline has gaps or overlaps"
    return q


def training_rows(through: pd.Timestamp) -> dict:
    """Frozen train.npz rows plus every-third-hour rows from the hourly bundles after it ends, up to origin `through`."""
    train = _load(FROZEN / "train.npz")
    q = hourly_queries()
    after = (q["origins"] > train["origins"].max()) & (q["origins"].hour % 3 == 0) & (q["origins"] <= through)
    ext = {k: (v[after] if k != "origins" else v[after]) for k, v in q.items()}
    assert (train["origins"].hour % 3 == 0).all()
    return _cat([train, ext]) if len(ext["origins"]) else train


def period_queries(period: str) -> dict:
    first, last, _ = PERIODS[period]
    q = hourly_queries()
    m = (q["origins"] >= pd.Timestamp(first)) & (q["origins"] <= pd.Timestamp(last))
    out = {k: v[m] for k, v in q.items()}
    assert len(out["origins"]) == N_FOLDS[period], (period, len(out["origins"]))
    return out


def train_index(origins: pd.DatetimeIndex, cutoff: pd.Timestamp) -> np.ndarray:
    """Rows whose 72 h targets are all before the cutoff."""
    return np.flatnonzero(origins + MATURITY < cutoff)
