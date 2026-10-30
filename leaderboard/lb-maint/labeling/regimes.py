"""QuitoBench-style 8-regime labels (arXiv 2603.26017) for 72h forecast folds.

Each fold is one origin t (last input timestamp); its window is the targets
t+1..t+72.
  T = STL trend strength (period 24, 168h left context, scored on the window)
  S = 27-day solar-rotation recurrence: max(0, Pearson r)^2 between the window
      and the same window 654h earlier. Solar wind has no physical diurnal
      seasonality; its real period is the ~27.25d corotation (ACF peak 652-654h).
      (STL with period 654 was tried and rejected: only 2 cycles of context, so
      the seasonal part memorises the data and S collapses to ~1.0.)
  F = 1 - normalised Welch spectral entropy of the window
Each is binarised at the frozen per-axis THRESHOLDS (config.py) into 2^3 = 8 cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from statsmodels.tsa.seasonal import STL

from config import EVAL_END, EVAL_START, THRESHOLDS

HORIZON = 72
SHORT_PERIOD, SHORT_CTX = 24, 168
ROT_LAG = 654
HERE = Path(__file__).resolve().parent
DEFAULT_SERIES = HERE.parent / "data" / "hourly_series.csv"  # written by refresh_data.py


def _strength(full: np.ndarray, n_window: int, period: int, component: str) -> float:
    """1 - Var(resid)/Var(component+resid), on the last n_window points of an STL fit of `full`."""
    res = STL(full, period=period, robust=True).fit()
    comp = getattr(res, component)[-n_window:]
    resid = res.resid[-n_window:]
    return float(max(0.0, 1.0 - np.var(resid) / np.var(comp + resid)))


def forecastability(x: np.ndarray) -> float:
    """1 - normalised Welch spectral entropy (1 = one dominant frequency)."""
    _, p = welch(x - x.mean(), nperseg=min(len(x), 48))
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1 / len(p))
    p = p[p > 0]
    return float(1.0 + (p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def recurrence(window: np.ndarray, prev: np.ndarray) -> float:
    if np.std(window) == 0 or np.std(prev) == 0:
        return 0.0
    return float(max(0.0, np.corrcoef(window, prev)[0, 1]) ** 2)


def label_window(window: np.ndarray, ctx_short: np.ndarray, prev_window: np.ndarray) -> dict:
    """`ctx_short` = SHORT_CTX hours right before `window`; `prev_window` = the window ROT_LAG hours earlier."""
    if len(ctx_short) != SHORT_CTX or len(prev_window) != len(window):
        raise ValueError(f"insufficient context: short {len(ctx_short)}/{SHORT_CTX}, prev window {len(prev_window)}/{len(window)}")
    n = len(window)
    t = _strength(np.concatenate([ctx_short, window]), n, SHORT_PERIOD, "trend")
    s = recurrence(window, prev_window)
    f = forecastability(window)
    name = "_".join("high" if v > THRESHOLDS[k] else "low" for k, v in (("T", t), ("S", s), ("F", f)))
    return {"T": t, "S": s, "F": f, "regime": name}


def build_folds(series: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Folds whose 72 targets all lie in [start, end). One row per origin."""
    s = series.set_index(pd.to_datetime(series["timestamp_utc"], utc=True))["filled_speed_kms"]
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    h = pd.Timedelta(hours=1)
    rows = []
    for o in pd.date_range(t0 - h, t1 - (HORIZON + 1) * h, freq="h"):
        w = s.loc[o + h: o + HORIZON * h].to_numpy()
        if len(w) != HORIZON:
            raise ValueError(f"incomplete window for origin {o}: {len(w)} points (data not yet available?)")
        ctx_s = s.loc[o - (SHORT_CTX - 1) * h: o].to_numpy()
        prev = s.loc[o + h - ROT_LAG * h: o + HORIZON * h - ROT_LAG * h].to_numpy()
        rows.append({"origin_last_input_utc": o.strftime("%Y-%m-%dT%H:%M:%SZ"), **label_window(w, ctx_s, prev)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(DEFAULT_SERIES))
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--out", default=str(HERE / "folds.parquet"))
    a = ap.parse_args()
    df = build_folds(pd.read_csv(a.series), a.start, a.end)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} folds -> {a.out}")
    print(df["regime"].value_counts().reindex(
        [f"{t}_{s}_{f}" for t in ("low", "high") for s in ("low", "high") for f in ("low", "high")],
        fill_value=0).to_string())
    print(df[["T", "S", "F"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
