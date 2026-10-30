"""Generate label-free MambaTS S1/S2 ACE input arrays using registered preprocessing."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from build import V2, HERE, sha, atomic_json

CHANNELS=(('speed','ace_speed_kms'),('density','ace_density_cm3'),('temperature','ace_temperature_k'),('Bt','ace_bt_nt'),('By','ace_by_gsm_nt'),('Bz','ace_bz_gsm_nt'),('e1','ace_epam_e_channel1'),('e2','ace_epam_e_channel2'),('p1','ace_epam_p_channel1'),('p3','ace_epam_p_channel3'),('p5','ace_epam_p_channel5'))

def prepare(raw,end):
    raw['timestamp_utc']=pd.to_datetime(raw.timestamp_utc,utc=True).dt.floor('h')
    raw=raw[raw.timestamp_utc<=end].sort_values('timestamp_utc').drop_duplicates('timestamp_utc',keep='last').set_index('timestamp_utc')
    raw=raw[[c for _,c in CHANNELS]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    full_index=pd.date_range(raw.index.min(),raw.index.max(),freq='h',tz='UTC')
    raw=raw.reindex(full_index);missing=raw.isna().to_numpy(dtype=np.uint8);values=raw.ffill(limit=6).to_numpy(dtype=np.float32)
    # Each 720h sequence ends at origin-2h; retain indices instead of expanding overlapping windows.
    origins=pd.date_range(max(full_index[0]+pd.Timedelta(hours=721),pd.Timestamp('2022-10-01',tz='UTC')),min(end,full_index[-1]+pd.Timedelta(hours=2)),freq='h')
    indices=((origins.asi8-full_index[0].value)//pd.Timedelta(hours=1).value).astype(np.int64)
    return {'values':values,'missing':missing,'source_start_ns':np.asarray(full_index[0].value),'source_end_ns':np.asarray(full_index[-1].value),'origin_ns':origins.asi8,'origin_idx':indices},full_index

def main():
    p=argparse.ArgumentParser();p.add_argument('--end',default='now');p.add_argument('--output',type=Path,default=V2/'mambats-inputs');a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    a.output.mkdir(parents=True,exist_ok=True);files=sorted((V2/'ace-snapshot').glob('year=*/part.parquet'))
    raw=pd.concat([pd.read_parquet(f,columns=['timestamp_utc',*[c for _,c in CHANNELS]]) for f in files],ignore_index=True)
    arrays,index=prepare(raw,end);np.savez_compressed(a.output/'numeric-inputs.npz',**arrays)
    manifest={'dataset':'mambats-inputs','status':'completed','returncode':0,'as_of':str(pd.Timestamp.now(tz='UTC')),'requested_end':str(end),'source_observation_first':str(index[0]),'source_observation_last':str(index[-1]),'source_hashes':{str(f):sha(f) for f in files},'builder_sha256':sha(__file__),'rows':len(index),'origins':len(arrays['origin_ns']),'channels':[n for n,_ in CHANNELS],'sequence_length':720,'availability_lag_hours':2,'fill_limit_hours':6,'files':{'numeric-inputs.npz':sha(a.output/'numeric-inputs.npz')},'scope':'Label-free shared S1/S2 numeric inputs. No labels, model fitting, or S2 image encoder embeddings.'}
    atomic_json(a.output/'MANIFEST.json',manifest);print(json.dumps(manifest))
if __name__=='__main__':main()
