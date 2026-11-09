"""Read-only data audit; outputs data_probes.json beside this script."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
raw = ROOT / 'data-v2/input/df.parquet'
filled = ROOT / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
cols = ['set', 'origin', 'h', 'target_time', 'period', 'fit_F1', 'fit_F2', 'fit_F3', 'y']
a = pd.read_parquet(raw, columns=cols)
b = pd.read_parquet(filled, columns=cols)
changed = ~np.isclose(a.y, b.y, equal_nan=True)
r = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), rows=len(a),
         identical_non_target_metadata=a.drop(columns='y').equals(b.drop(columns='y')),
         target_time_mismatch=int(((a.target_time-a.origin) != pd.to_timedelta(a.h, unit='h')).sum()),
         query_y_changed=int((changed & a['set'].eq('query')).sum()),
         observed_y_changed=int((changed & a.y.notna()).sum()),
         train_imputed_target_rows=int((a.y.isna() & b.y.notna() & a['set'].eq('train')).sum()), periods={})
for p, cutoff in {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}.items():
    tr = b[b['fit_'+p]]
    q = a[a['set'].eq('query') & a.period.eq(p)]
    r['periods'][p] = dict(train_origins=int(tr.origin.nunique()), train_origin_min=str(tr.origin.min()),
        train_origin_max=str(tr.origin.max()), train_target_max=str(tr.target_time.max()), cutoff=cutoff,
        maturity_violations=int((tr.origin+pd.Timedelta(hours=144) > pd.Timestamp(cutoff, tz='UTC')).sum()),
        query_origins=int(q.origin.nunique()), query_origin_min=str(q.origin.min()), query_origin_max=str(q.origin.max()),
        missing_truth_fraction=float(q.y.isna().mean()))
spec = importlib.util.spec_from_file_location('cd', ROOT/'leaderboard/scorer/competition_data.py')
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
sw = pd.concat([cd.parse_month(p.read_bytes(), p.name) for p in sorted((ROOT/'data-v2/raw/ACE/monthly-txt').glob('*_ace_swepam_1h.txt'))]).set_index('timestamp_utc').speed_kms.sort_index()
r['raw_vs_df_y_mismatch'] = int((~np.isclose(sw.reindex(pd.DatetimeIndex(a.target_time)), a.y, equal_nan=True)).sum())
obs_path = ROOT/'research/offset_roll_complete_ablation_20261106/competition-observations.parquet'
obs = pd.read_parquet(obs_path).set_index('timestamp_utc').speed_kms
r['raw_vs_frozen_official_obs'] = dict(n=len(obs), mismatch=int((~np.isclose(sw.reindex(obs.index), obs, equal_nan=True)).sum()))
cols = ['origin', 'img_age_h']
ra = pd.read_parquet(raw, columns=cols, filters=[('h', '==', 1)]).drop_duplicates('origin').sort_values('origin').set_index('origin')
fb = pd.read_parquet(filled, columns=cols, filters=[('h', '==', 1)]).drop_duplicates('origin').sort_values('origin').set_index('origin')
source = pd.Series(ra.index-pd.to_timedelta(ra.img_age_h, unit='h'), index=ra.index).ffill()
age = (pd.Series(ra.index, index=ra.index)-source).dt.total_seconds()/3600
stale = ra.img_age_h.isna() & fb.img_age_h.notna()
qo = pd.DatetimeIndex(a.loc[a['set'].eq('query') & a.period.notna(), 'origin'].unique())
r['stale_image_age'] = dict(all_origins=len(ra), filled_missing_frame_origins=int(stale.sum()),
    maximum_actual_age_hours=float(age[stale].max()), maximum_reported_age_hours=float(fb.img_age_h[stale].max()),
    official_query_origins_with_stale_frame=int(stale.reindex(qo).sum()),
    query_maximum_actual_age_hours=float(age.reindex(qo)[stale.reindex(qo)].max()))
r['file_identity'] = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                      for p in [raw, filled, ROOT/'data-v2/notebooks/raw_to_input_df.ipynb', obs_path]}
(OUT/'data_probes.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r, indent=2))
