"""Official metric algebra on existing residual energies; no forecast outputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('energy_metric',
    REPO / 'research/offset_roll_complete_ablation_20261106/analyze_results.py')
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)
with np.load(ROOT / 'residual_arrays.npz') as saved:
    truth = saved['truth']
    forecasts = saved['predictions'][:3]
    names = saved['names'][:3].tolist()
    origins = pd.to_datetime(saved['origins'], utc=True)
weights = metric.metric_weights(truth, np.arange(len(origins)))
energies = np.where(np.isfinite(truth)[None], (forecasts - truth[None])**2, 0.)
baseline = metric.scores_from_weights(energies, *weights)
official = json.loads((ROOT / 'residual_official_scores.json').read_text())
for i, name in enumerate(names):
    assert np.isclose(baseline['s'][i], official[name]['total']['s'], rtol=0, atol=1e-8)
assert np.isclose(baseline['s'][0], 97.49244208059969, rtol=0, atol=1e-8)
target_times = origins.asi8[:, None] + np.arange(1, 73)[None, :] * 3600 * 10**9
masks = {
    'previously_diagnosed_jan19_event': ((target_times >= pd.Timestamp('2026-01-19T19:02Z').value)
                                      & (target_times < pd.Timestamp('2026-01-22T00:00Z').value)),
    'previously_defined_speed_ge700': truth >= 700.,
    'previously_defined_lead49_72': np.broadcast_to(np.arange(1, 73)[None, :] >= 49, truth.shape),
}
result = {'classification': 'restricted residual-energy sensitivity, not model performance',
          'reference_names': names, 'official_baseline_s': baseline['s'].tolist(),
          'half_best_single_target': 48.746221040299844, 'cases': {},
          'forecast_files_generated': False, 'fit_performed': False,
          'new_treatment_outcomes_read': False,
          'limits': ['Post-hoc descriptive masks, not operational selection rules.',
                     'All errors outside each mask remain fixed by assumption.',
                     'These floors are not achievable scores or physical information bounds.',
                     'No prediction array is recomposed or exported.']}
for name, mask in masks.items():
    remaining = np.where(mask[None], 0., energies)
    values = metric.scores_from_weights(remaining, *weights)
    result['cases'][name] = {
        'valid_pairs_in_mask': int((mask & np.isfinite(truth)).sum()),
        'restricted_s_floor_per_seed': values['s'].tolist(),
        'conditional_relative_reduction_per_seed': (1 - values['s'] / baseline['s']).tolist(),
        'all_restricted_floors_still_above50pct_target': bool((values['s'] > result['half_best_single_target']).all())}
(ROOT / 'residual_energy_budget.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
