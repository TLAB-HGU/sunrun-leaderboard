"""Verify generated hashes and historical feature equality; no truth or scores."""
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT/'data-v2/model-inputs/leaderboard-store'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def check(name,new,old):
 manifest=json.loads((STORE/'v2'/name/'MANIFEST.json').read_text())
 assert sha(new)==manifest['sha256'],name
 a,b=pd.read_parquet(new),pd.read_parquet(old)
 for frame in (a,b):
  if 'origin_last_input_utc' in frame:frame.set_index('origin_last_input_utc',inplace=True)
 common=a.index.intersection(b.index)
 pd.testing.assert_frame_equal(a.loc[common],b.loc[common],check_freq=False)
 return {'name':name,'output_sha256_verified':True,'historical_rows_identical':len(common),'official_window_rows_identical':int(((common>=pd.Timestamp('2026-05-31T23:00:00Z'))&(common<=pd.Timestamp('2026-09-26T23:00:00Z'))).sum()),'new_rows':len(a)-len(common)}
rows=[]
rows.append(check('euvwave',STORE/'v2/euvwave/euvwave_origin_features.parquet',STORE/'solar-inc-euvwave72-20261014/euvwave_origin_features.parquet'))
rows.append(check('fe171',STORE/'v2/fe171/fe171-hourly-v2.parquet',STORE/'fe171-cache/fe171-hourly-v1.parquet'))
dim=STORE/'v2/suvi-dim/suvi_dim72_features.parquet'
if dim.exists():
 pointer=json.loads((STORE/'suvi-dim72-cache/solar-inc-suvidim72-20261014-pointer.json').read_text())
 rows.append(check('suvi-dim',dim,Path(pointer['feature_dir'])/'suvi_dim72_features.parquet'))
maskpath=STORE/'v2/chmask/chmask-hourly.parquet'
if maskpath.exists():
 import sys
 sys.path.insert(0,str(ROOT/'leaderboard'))
 from experiments.d145_axisC_chmask import chmask
 old=pd.read_parquet(STORE/'suvi-fusion/suvi-hourly-v2.parquet')
 expected=chmask.build_mask(old[[f'lon{i:02d}_lat1_dark_fraction' for i in range(12)]],old[[f'lon{i:02d}_lat1_valid_fraction' for i in range(12)]])
 actual=pd.read_parquet(maskpath)
 common=actual.index.intersection(expected.index)
 pd.testing.assert_frame_equal(actual.loc[common],expected.loc[common],check_freq=False)
 rows.append({'name':'chmask','historical_rows_identical':len(common),'official_window_rows_identical':int(((common>=pd.Timestamp('2026-05-31T23:00:00Z'))&(common<=pd.Timestamp('2026-09-26T23:00:00Z'))).sum())})
output=Path(__file__).with_name('verification.json');output.write_text(json.dumps({'verified_at':pd.Timestamp.now(tz='UTC').isoformat(),'results':rows},indent=2)+'\n')
print(output.read_text())
