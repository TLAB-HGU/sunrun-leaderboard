"""Descriptive independent-event crosscheck; never alters official truth."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
with np.load(OUT / 'residual_arrays.npz') as saved:
    origins = pd.to_datetime(saved['origins'], utc=True)
    truth = saved['truth']
    predictions = saved['predictions'][:3]
spec = importlib.util.spec_from_file_location('event_weights', REPO / 'research/offset_roll_complete_ablation_20261106/analyze_results.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
_, weights = stats.metric_weights(truth, np.arange(len(origins)))
times = origins.asi8[:, None] + np.arange(1, 73)[None, :] * 3600 * 10**9
flare = pd.Timestamp('2026-01-18T18:09Z')
shock = pd.Timestamp('2026-01-19T19:02Z')
reliable = pd.Timestamp('2026-01-19T21:15Z')
end = pd.Timestamp('2026-01-22T00:00Z')
event = (times >= shock.value) & (times < end.value)
square = np.mean(np.where(np.isfinite(truth)[None], predictions - truth[None], 0) ** 2, axis=0)
contribution = np.nan_to_num(square) * weights / 72
total = contribution.sum()


def describe(mask):
    selected = mask & np.isfinite(truth)
    errors = (predictions - truth[None])[:, selected]
    return {'n_pairs': int(selected.sum()), 'unique_target_hours': len(np.unique(times[selected])),
            'mean_mae': float(np.mean(np.abs(errors))), 'mean_bias': float(np.mean(errors)),
            'mean_pooled_rmse': float(np.sqrt(np.mean(errors ** 2, axis=1)).mean()),
            'fraction_total_timse': float(contribution[selected].sum() / total)}


result = {'source': 'event_noaa_snapshot.md; NOAA issued 2026-01-20T13:35Z',
          'interpretation': 'Retrospective external event labels for residual diagnosis only; unavailable labels never enter inference.',
          'event_shock_to_jan22': describe(event),
          'same_event_origins_before_flare': describe(event & (origins.asi8[:, None] < flare.value)),
          'same_event_origins_after_flare_before_shock': describe(event & (origins.asi8[:, None] >= flare.value) & (origins.asi8[:, None] < shock.value)),
          'same_event_origins_after_shock': describe(event & (origins.asi8[:, None] >= shock.value)),
          'event_truth_after_noaa_reliability_restored': describe(event & (times >= reliable.value)),
          'non_event': describe(~event)}
raw_path = REPO / 'data-v2/raw/ACE/monthly-txt/202601_ace_swepam_1h.txt'
lines = raw_path.read_text().splitlines()
result['raw_relevant_rows'] = [line for line in lines if line.startswith(('2026 01 19', '2026 01 20'))]
donki = json.loads((REPO / 'data-v2/raw/ENLIL/donki/WSAEnlilSimulations_2026-01.json').read_text())
result['enlil_runs_for_jan18_cme'] = [row for row in donki if any(str(c.get('cmeStartTime', '')).startswith('2026-01-18') for c in row.get('cmeInputs', []))]
result['limits'] = ['NOAA cross-instrument report supports a real fast CME shock, not exact hourly calibration.',
                    'The bulletin flare reference time is not verified flare onset/publication; origins before it may already see evolving signals.',
                    'Largest error targets Jan19 22h onward are after the reported reliability restoration: earlier contamination cannot explain them away.',
                    'DONKI completion is not a historical publication receipt; revisions remain possible.',
                    'This post-hoc event split does not provide independent discovery significance or a forecast lower bound.']
(OUT / 'event_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('raw_relevant_rows', 'enlil_runs_for_jan18_cme')}, indent=2))
