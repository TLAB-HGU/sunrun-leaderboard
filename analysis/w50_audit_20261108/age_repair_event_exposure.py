"""Post-hoc input exposure diagnosis only; no targets, fitting or scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'research/w50_understanding_20261108'))
from age_repair import DATA, repaired_age

fixed, _ = repaired_age()
data = pd.read_parquet(DATA, columns=['origin', 'img_age_h'],
    filters=[('set', '==', 'query'), ('period', '==', 'F1'), ('h', '==', 1)],
    use_threads=False).drop_duplicates('origin').sort_values('origin')
origins = pd.to_datetime(data.origin, utc=True)
ages = fixed.reindex(origins).to_numpy()
changed = ~np.isclose(ages, data.img_age_h.to_numpy(), equal_nan=True, atol=1e-5, rtol=0)
event = ((origins >= pd.Timestamp('2026-01-18', tz='UTC'))
         & (origins < pd.Timestamp('2026-01-22', tz='UTC')))
result = {'scope': 'F1 query input ages only, no target columns or official observations',
    'posthoc_event_window': '2026-01-18 inclusive to2026-01-22 exclusive; chosen by prior residual diagnosis, not an independent test',
    'all_F1_origins': len(origins), 'all_F1_changed_origins': int(changed.sum()),
    'event_window_origins': int(event.sum()), 'event_window_changed_origins': int(changed[event].sum()),
    'event_window_max_repaired_age_h': float(np.max(ages[event])),
    'limits': ['No inference of forecast skill from exposure counts.',
               'Fresh retraining may alter predictions globally even at unchanged query inputs.',
               'No feature/parameter fitting or ongoing configuration change.']}
(ROOT / 'age_repair_event_exposure.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
