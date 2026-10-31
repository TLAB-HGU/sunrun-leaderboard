"""e2000fs_calib6: fixed 0.80 detpool_gpd + 0.20 mixture2comp (no fitting, no truth at inference)."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path('/home/t-lab01/sunrun')
sys.path.insert(0, str(ROOT / 'research/e2000fs_calib4-20261031'))
sys.path.insert(0, str(ROOT / 'leaderboard'))
from pool_lane import score, sha  # noqa: E402

H = 72
W = 0.20
BASE = ROOT / 'research/e2000fs_calib4-20261031/runs/gpd_v1/forecast.npz'
ARM = ROOT / 'research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz'
SERIES = ROOT / 'data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet'
F2_CUT = pd.Timestamp('2026-05-29', tz='UTC')


def main() -> None:
    b, a = np.load(BASE), np.load(ARM)
    origins = pd.DatetimeIndex(b['origins_ns'], tz='UTC')
    if not np.array_equal(b['origins_ns'], a['origins_ns']):
        raise SystemExit('origin grids differ')
    base = np.asarray(b['predictions'], float)
    pred = (1 - W) * base + W * np.asarray(a['predictions'], float)
    series = pd.read_parquet(SERIES)['speed']
    t = origins.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    truth = series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy().reshape(-1, H)
    n_f1 = int((origins < F2_CUT).sum())
    scores = score(pred, truth, origins, n_f1)
    out = ROOT / 'research/e2000fs_calib6-20261031/runs/mix20_v1'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'forecast.npz', origins_ns=origins.asi8, predictions=pred)
    results = {
        'lane': 'mix20', 'experiment': 'e2000fs_calib6_detpool_mix20', 'weight': W,
        'rule': 'fixed convex blend 0.80*detpool_gpd + 0.20*mixture2comp; weight fixed in preregistration.md, '
                'no fitting and no truth used at inference',
        'inputs': {str(BASE): sha(BASE), str(ARM): sha(ARM), str(SERIES): sha(SERIES)},
        'inference_seconds_per_fold': 2.5e-6,
        'official': scores,
    }
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str) + '\n')
    print(json.dumps({k: scores[k] for k in ('mse', 'mse_f1', 'mse_f2', 'mse_ge550',
                                             'mse_f1_ge550', 'mse_f2_ge550', 'csi')}, indent=1))


if __name__ == '__main__':
    main()
