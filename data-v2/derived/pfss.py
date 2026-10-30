"""Run isolated registered GONG topology kernels against the v2 source receipts."""
import argparse, importlib.util, json, os, sys, traceback
from pathlib import Path
import pandas as pd
from build import HERE,V2,sha,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['gong-ocb',*[f'gong-ocb-r{i}' for i in range(2,8)],'pfssconn','cme-hcs','pfss-wsa']);p.add_argument('--end',default='now');p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tz is None:end=end.tz_localize('UTC')
    if a.workers!=1:p.error('maintenance topology kernels currently reserve one CPU')
    out=a.output or V2/a.job;out.mkdir(parents=True,exist_ok=True)
    filename='gong_ocb.py' if a.job.startswith('gong-ocb') else ('compute_pfss.py' if a.job=='pfssconn' else 'compute_hcs_fixed.py')
    source=HERE/'vendor'/a.job/filename if a.job!='pfss-wsa' else HERE/'vendor/pfss_wsa.py'
    spec=importlib.util.spec_from_file_location('v2topology',source);m=importlib.util.module_from_spec(spec);sys.modules['v2topology']=m;spec.loader.exec_module(m)
    result={'dataset':a.job,'as_of':str(pd.Timestamp.now(tz='UTC')),'requested_end':str(end),'builder_sha256':sha(__file__),'kernel_sha256':sha(source)}
    try:
        receipt_paths=sorted((V2/'gong-derived/receipts').glob('*.json'))
        maps=[json.loads(f.read_text()) for f in receipt_paths]
        maps=[dict(r,body_sha256=r['sha256'],date_obs_utc=r['observation_time_utc']) for r in maps if pd.Timestamp(r['observation_time_utc'])<=end]
        maps.sort(key=lambda r: r['observation_time_utc'])
        if not maps:raise FileNotFoundError('No v2 GONG metadata receipts ready; run gong.py first')
        if a.job=='pfss-wsa':
            from astropy.io import fits
            for r in maps:
                r['crval1']=float(fits.getheader(r['path'])['CRVAL1'])
                r['date_obs_utc']=pd.Timestamp(r['observation_time_utc']).strftime('%Y-%m-%dT%H:%M:%S')
        manifest=out/'source.json';atomic_json(manifest,{'source':'GONG mrzqs v2','maps':maps})
        m.MANIFEST_PATH=manifest;m.CACHE_ROOT=out/'cache'
        if a.job.startswith('gong-ocb'):
            effective_end=min(end,max(pd.Timestamp(r['observation_time_utc']) for r in maps)+pd.Timedelta(hours=2))
            cfg=dict(m.OCB_CONFIG,cache_dir=str(out/'cache'),origin_feature_end_utc=str(effective_end))
            receipt=m.precompute_origin_table(manifest_path=manifest,config=cfg,n_jobs=1)
        elif a.job=='pfss-wsa':
            m.TRACE_TIMEOUT_SECONDS=120
            cache=out/'cache';cache.mkdir(exist_ok=True);vendor_sha=m.vendor_hash();rows=[]
            for i,r in enumerate(maps):
                rows.append(m._task((r,str(cache),sha(source),vendor_sha)))
                if i%20==0:print(f'pfss-wsa maps {i+1}/{len(maps)}',flush=True)
            atomic_json(out/'profiles.json',rows)
            receipt={'maps':len(rows),'trace_timeout_seconds':m.TRACE_TIMEOUT_SECONDS,'timeouts':sum(bool(r['timed_out']) for r in rows),'cache':str(cache)}
        else:
            cfg=json.loads(m.CONFIG_PATH.read_text());code_hash=sha(source);config_hash=sha(m.CONFIG_PATH)
            cache=out/'cache'/__import__('hashlib').sha256((code_hash+config_hash).encode()).hexdigest();cache.mkdir(parents=True,exist_ok=True)
            for i,r in enumerate(maps):
                if a.job=='pfssconn':
                    dest=cache/f'{r["body_sha256"]}.json'
                    if not m.cache_valid(dest,r,code_hash,config_hash):m.features_for_map(r,cfg,code_hash,config_hash,cache)
                elif not m._cache_valid(r,cache,code_hash,config_hash):m._compute_map(r,cfg,code_hash,config_hash,cache)
                if i%20==0:print(f'{a.job} maps {i+1}/{len(maps)}',flush=True)
            receipt={'cache':str(cache),'maps':len(maps)}
            if a.job=='cme-hcs':
                ephemeris=m._write_ephemeris(cache,cache.name,min(pd.Timestamp(r['date_obs_utc']) for r in maps)-pd.Timedelta(days=7),end+pd.Timedelta(days=7))
                valid=sum(bool(json.loads((cache/f'{r["body_sha256"]}.json').read_text()).get('valid')) for r in maps)
                receipt.update(ephemeris=ephemeris,valid_count=valid,invalid_count=len(maps)-valid)
                atomic_json(cache/'cache_manifest.json',dict(receipt,namespace=cache.name,compute_sha256=code_hash,config_sha256=config_hash,source_manifest_sha256=sha(manifest),config=cfg,source_availability_rule='max(observation+2h, HTTP Last-Modified)'))
        last=max(pd.Timestamp(r['observation_time_utc']) for r in maps)
        raw_files=list((V2/'gong-mrzqs/maps').rglob('*.fits.gz'))
        source_complete=(V2/'gong-mrzqs/MANIFEST.json').exists() and len(maps)>=len(raw_files)
        result.update(status='completed' if source_complete and not receipt.get('timeouts',0) and not receipt.get('invalid_count',0) and last>=end-pd.Timedelta(days=1) else 'partial',returncode=0,source_observation_last=str(last),source_maps=len(maps),source_manifest_sha256=sha(manifest),receipt=receipt)
    except Exception as e:result.update(status='failed',returncode=1,error=f'{type(e).__name__}: {e}');traceback.print_exc()
    atomic_json(out/'MANIFEST.json',result);print(json.dumps(result,default=str));return result['returncode']
if __name__=='__main__':raise SystemExit(main())
