"""Isolated, source-bound SUVI v2 derivation; never trains or reads targets."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, importlib, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
BUILDER_CONTENT = Path(__file__).read_bytes()
BUILDER_HASH = hashlib.sha256(BUILDER_CONTENT).hexdigest()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LB = ROOT / 'leaderboard'
STORE = LB / 'store/v2'
sys.path.insert(0, str(LB))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def write(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,default=str)+'\n'); tmp.replace(p)

def receipt(name, output, sources, extra):
    version=HERE/'versions'/(BUILDER_HASH+'.py')
    version.parent.mkdir(exist_ok=True)
    if not version.exists(): version.write_bytes(BUILDER_CONTENT)
    df=pd.read_parquet(output)
    index=df['origin_last_input_utc'] if 'origin_last_input_utc' in df else df.index
    info={'dataset':name,'as_of':pd.Timestamp.now(tz='UTC').isoformat(),'output':str(output),'sha256':sha(output),
          'rows':len(df),'range_start':str(min(index)),'range_end':str(max(index)),
          'builder_path':str(version),'builder_sha256':BUILDER_HASH,'inputs':{str(p):sha(p) for p in sources},
          'training_cutoff':'2026-05-29T00:00:00Z','fitting':'none; fixed transforms; no imputation added',**extra}
    write(STORE/name/'MANIFEST.json', info)
    print(json.dumps(info,default=str),flush=True)

def catalog(channel='Fe195'):
    import prepare_catalog as c
    out=STORE/('euvwave' if channel=='Fe195' else 'fe171'); out.mkdir(parents=True,exist_ok=True)
    source=out/'source_catalog.jsonl'
    old=source if source.exists() else LB/'store/solar-inc-euvwave72-20261014/source_catalog.jsonl'
    rows={}
    if old.exists() and (channel=='Fe195' or old==source):
        rows={r['filename']:r for r in map(json.loads,old.read_text().splitlines())}
    paths=sorted((ROOT/'sun-img/data').glob('**/*'+channel+'*.nc'))
    if channel!='Fe195':
        # Same immutable NOAA NetCDF contract, with channel-specific S3 key.
        import types
        text=(HERE/'prepare_catalog.py').read_text().replace('SUVI-L1b-Fe195','SUVI-L1b-Fe171')
        m=types.ModuleType('fe171_catalog'); m.__file__=str(HERE/'prepare_catalog.py'); exec(compile(text,m.__file__,'exec'),m.__dict__); c=m
    remote={}
    if channel=='Fe171':
        import dim_prepare as d
        namespace=dict(d.__dict__)
        import inspect
        exec(inspect.getsource(d.list_day).replace('SUVI-L1b-Fe195','SUVI-L1b-Fe171'),namespace)
        dates=sorted({c.stamp(c.NAME.search(path.name).groups()[:6]).date() for path in paths if path.name not in rows or rows[path.name].get('local_size')!=path.stat().st_size})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,items in enumerate(pool.map(namespace['list_day'],dates),1):
                remote.update(items)
                if i%100==0: print(f'Fe171 S3 days {i}/{len(dates)}',flush=True)
    def check(path):
        oldrow=rows.get(path.name)
        if oldrow and oldrow.get('byte_identity_passed') and oldrow.get('local_size')==path.stat().st_size:
            md5,digest=c.file_hashes(path)
            if md5==oldrow['s3_etag'] and digest==oldrow['local_sha256']:
                return {**oldrow,'local_path':str(path),'local_mtime_ns':path.stat().st_mtime_ns}
        if channel=='Fe171':
            parts=c.NAME.search(path.name).groups(); start=c.stamp(parts[:6]); end=c.stamp(parts[6:])
            key=f"SUVI-L1b-Fe171/{start:%Y}/{start.timetuple().tm_yday:03d}/{start:%H}/{path.name}"
            obj=remote.get(key)
            if obj:
                md5,digest=c.file_hashes(path); stat=path.stat()
                return {'filename':path.name,'local_path':str(path),'local_size':stat.st_size,'local_mtime_ns':stat.st_mtime_ns,'local_md5':md5,'local_sha256':digest,'s3_key':key,'s3_url':f'https://{c.HOST}/{key}','s3_etag':obj['etag'],'s3_size':obj['size'],'s3_last_modified_utc':obj['last_modified'].isoformat(),'obs_start_utc':start.isoformat(),'obs_end_utc':end.isoformat(),'byte_identity_passed':bool(md5==obj['etag'] and stat.st_size==obj['size']),'http_status':200}
        return c.inspect(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=[]
        for i,r in enumerate(pool.map(check,paths),1):
            results.append(r)
            if i%1000==0: print(f'{channel} provenance {i}/{len(paths)}',flush=True)
    if source.exists():
        history=out/'catalog-history'; history.mkdir(exist_ok=True)
        frozen=history/(sha(source)+'.jsonl')
        if not frozen.exists(): os.link(source,frozen)
    tmp=source.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results)); tmp.replace(source)
    good=[r for r in results if r.get('byte_identity_passed')]
    info={'catalog_path':str(source),'catalog_sha256':sha(source),'local_file_count':len(results),'byte_identity_count':len(good),
          'byte_identity_mismatch_count':len(results)-len(good),'last_obs_end_utc':max(r['obs_end_utc'] for r in good),
          'last_s3_modified_utc':max(r['s3_last_modified_utc'] for r in good),'retrieved_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
          'reuse':'previous public S3 identity reused only after fresh local SHA256 and MD5 match; new files HEAD checked'}
    write(out/'source_manifest.json',info); return results,info

def wave(end):
    from experiments.solar_inc_euvwave72_20261014_keyfix import wave_features as w
    out=STORE/'euvwave'; w.STORE=out; w.CATALOG=out/'source_catalog.jsonl'; w.SOURCE_MANIFEST=out/'source_manifest.json'; w.FRAME_CACHE=out/'euvwave_origin_features.parquet'
    w._origin_list=lambda:pd.date_range(pd.Timestamp('2022-10-01',tz='UTC'),end,freq='h')
    legacy=LB/'store/solar-inc-euvwave72-20261014'
    oldinfo=json.loads((legacy/'euvwave_origin_features.manifest.json').read_text())
    oldfile=legacy/'euvwave_origin_features.parquet'; oldcatalog=legacy/'source_catalog.jsonl'
    if w.FRAME_CACHE.exists() and w.FRAME_CACHE.with_suffix('.manifest.json').exists():
        currentinfo=json.loads(w.FRAME_CACHE.with_suffix('.manifest.json').read_text())
        frozen=out/'catalog-history'/(currentinfo['catalog_sha256']+'.jsonl')
        if frozen.exists(): oldinfo=currentinfo; oldfile=w.FRAME_CACHE; oldcatalog=frozen
    if not w.FRAME_CACHE.with_suffix('.partial.parquet').exists() and oldinfo['code_sha256']==sha(w.__file__) and oldinfo['frame_cache_sha256']==sha(oldfile):
        old=pd.read_parquet(oldfile)
        oldrows=[json.loads(x) for x in oldcatalog.read_text().splitlines()]
        newrows=[json.loads(x) for x in w.CATALOG.read_text().splitlines()]
        def signature(rows):
            return sorted((r['filename'],r.get('local_sha256'),r.get('s3_last_modified_utc'),r.get('byte_identity_passed')) for r in rows if pd.Timestamp(r['obs_end_utc'])<=old.index.max())
        assert oldinfo['catalog_sha256']==sha(oldcatalog)
        if signature(oldrows)==signature(newrows):
            manifest=json.loads(w.SOURCE_MANIFEST.read_text()); key,_=w._feature_cache_key(manifest,w._origin_list())
            old.to_parquet(w.FRAME_CACHE.with_suffix('.partial.parquet'))
            write(w.FRAME_CACHE.with_suffix('.partial.json'),{'cache_key':key,'completed_origins':len(old),'totals':oldinfo['diagnostics']})
            write(out/'historical-reuse.json',{'rows':len(old),'sha256':sha(oldfile),'verified_identical_source_membership':True,'verified_identical_transform':True})
    frame,source,info,_=w._ensure_cache()
    from types import SimpleNamespace
    rows,_,_=w.read_registered_frames()
    origin=frame.index[frame.wave_present==1][0] if (frame.wave_present==1).any() else frame.index[-10]
    selected=w.select_frames(origin,rows)
    injected=rows+[{'filename':'future','s3_key':'future','obs_end':origin+pd.Timedelta(hours=1),'last_modified':origin,'byte_identity_passed':True},
                   {'filename':'late','s3_key':'late','obs_end':origin-pd.Timedelta(hours=2),'last_modified':origin+pd.Timedelta(seconds=1),'byte_identity_passed':True}]
    unchanged=[r['s3_key'] for r in selected]==[r['s3_key'] for r in w.select_frames(origin,injected)]
    positive=w.Features.positive_control_features(SimpleNamespace(source_records=rows),origin)
    changed=not np.allclose(frame.loc[[origin]].to_numpy(),positive.to_numpy(),equal_nan=True)
    poison={'passed':bool(unchanged and changed),'future_and_late_publication_selection_invariant':unchanged,'admitted_image_positive_control_changed':changed,'origin':str(origin)}
    assert poison['passed'],poison
    write(out/'poison.json',poison)
    receipt('euvwave',w.FRAME_CACHE,[w.CATALOG,Path(w.__file__)],{'source_policy':'obs_end+1h<=origin; S3 LastModified<=origin; MD5/size bound; SHA256 on decode','coverage':info['coverage'],'source_latest_obs':source['last_obs_end_utc'],'poison':poison})


def snapshot_meta(out):
    source=STORE/'suvi-meta'
    if not source.exists(): source=Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly')
    paths=sorted(source.glob('year=*/part.parquet'))
    if not paths: raise FileNotFoundError(source)
    frame=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    out.mkdir(parents=True,exist_ok=True); path=out/'metadata.parquet'; frame.to_parquet(path,index=False)
    return path

def morphology(end):
    from experiments.solar_inc_chmorph72_20261014 import morph_features as m
    s=m.spec; s.STORE=STORE/'chmorph'; s.SOURCE_MANIFEST=STORE/'euvwave/source_manifest.json'
    s.CH_PATH=STORE/'ch/ch-hourly.parquet'; s.CH_FEATURE_MANIFEST=STORE/'ch/ch-feature-manifest.json'; s.CH_INPUT_IDENTITY=STORE/'ch/input-identity.json'; s.SUVI_CACHE=STORE/'suvi-fusion/suvi-hourly-v2.parquet'
    s.SUVI_META_GLOB=str(snapshot_meta(s.STORE))
    s.CH_INPUT_IDENTITY=s.STORE/'input-identity.json'
    write(s.CH_INPUT_IDENTITY,{'ch_sha256':sha(s.CH_PATH),'suvi_sha256':sha(s.SUVI_CACHE),'metadata_sha256':sha(s.SUVI_META_GLOB),'source_catalog_sha256':sha(s.SOURCE_MANIFEST)})
    ctx=m.MorphologyContext(end); origins=pd.date_range(pd.Timestamp('2022-10-01',tz='UTC'),end,freq='h')
    values,selected=ctx.features_for(origins); ctx._verify_source_files(selected)
    frame=pd.DataFrame(values,index=origins,columns=s.MORPH_COLUMNS); frame.index.name='origin_last_input_utc'
    path=s.STORE/'chmorph.parquet'; frame.to_parquet(path)
    poison=ctx.poison_report(pd.Timestamp('2026-06-02T00:00:00Z')); assert poison['passed']
    receipt('chmorph',path,[s.CH_PATH,s.SOURCE_MANIFEST,Path(m.__file__)],{'poison':poison,'coverage':ctx.coverage(origins)[1],'source_latest_obs':ctx.source_manifest['last_obs_end_utc']})

def mask(end):
    from experiments.d145_axisC_chmask import chmask as m
    m.SUVI_CACHE=STORE/'suvi-fusion/suvi-hourly-v2.parquet'; m.MASK_CACHE=STORE/'chmask'; m.MASK_CACHE.mkdir(parents=True,exist_ok=True)
    m.META_GLOB=str(snapshot_meta(m.MASK_CACHE))
    meta,dark,valid=m.load_inputs()
    records={r['filename']:r for r in map(json.loads,(STORE/'euvwave/source_catalog.jsonl').read_text().splitlines()) if r.get('byte_identity_passed')}
    # Bind metadata to verified NetCDF publication times (stage may describe FITS).
    for i,row in meta.iterrows():
        r=records.get(Path(str(row.nc_path)).name)
        if r and abs((pd.Timestamp(r['obs_end_utc'])-row.obs_end).total_seconds())<=1:
            meta.at[i,'s3_modified']=pd.Timestamp(r['s3_last_modified_utc'])
        else: meta.at[i,'status']='unverified'
    masks=m.build_mask(dark,valid)
    path=m.MASK_CACHE/'chmask-hourly.parquet'; masks.to_parquet(path)
    origins=pd.date_range(pd.Timestamp('2022-10-01',tz='UTC'),end,freq='h'); selection=m.select_slots(origins,meta,masks)
    admitted=selection.slot_idx>=0
    assert (selection.loc[admitted,'obs_end']+pd.Timedelta(hours=1)<=selection.loc[admitted,'origin']).all()
    selection.to_parquet(m.MASK_CACHE/'selection.parquet',index=False)
    probe=pd.date_range('2026-06-01','2026-06-03',freq='3h',tz='UTC'); poison=m.poison_test(probe,pd.Timestamp('2026-06-02',tz='UTC'),meta=meta,dark=dark,valid=valid); assert poison['passed'],poison
    receipt('chmask',path,[m.SUVI_CACHE,STORE/'euvwave/source_catalog.jsonl',Path(m.__file__)],{'poison':poison,'coverage_selected':int(admitted.sum()),'origin_count':len(origins),'source_policy':'NetCDF byte bound; obs_end+1h and S3 LastModified <= origin'})

def fe171(end):
    from experiments.d146_fe171_channel import fe171_hourly as f
    out=STORE/'fe171'; out.mkdir(parents=True,exist_ok=True)
    # Collector has a writer lock. Consistent database+WAL copy permits read-only snapshot.
    source=ROOT/'sun-img/solar_g18_fe171.duckdb'; db=out/'metadata.duckdb'
    for attempt in range(5):
        before=(source.stat().st_size,source.stat().st_mtime_ns)
        shutil.copy2(source,db)
        wal=Path(str(source)+'.wal')
        if wal.exists(): shutil.copy2(wal,Path(str(db)+'.wal'))
        if before==(source.stat().st_size,source.stat().st_mtime_ns): break
    else: raise RuntimeError('Fe171 database changed during every snapshot attempt')
    # Replay copied WAL locally, then open snapshot read-only.
    import subprocess
    export="import duckdb,sys; c=duckdb.connect(sys.argv[1]); c.execute('CHECKPOINT'); c.execute(\"COPY (SELECT * FROM suvi_hourly ORDER BY slot) TO '\"+sys.argv[2]+\"' (FORMAT PARQUET)\"); c.close()"
    subprocess.run([str(ROOT/'sun-img/.venv/bin/python'),'-c',export,str(db),str(out/'metadata.parquet')],check=True)
    meta=pd.read_parquet(out/'metadata.parquet')
    for col in ('slot','obs_start','obs_end','s3_modified'): meta[col]=pd.to_datetime(meta[col],utc=True)
    meta=meta.loc[meta.slot<=end].copy(); meta.to_parquet(out/'metadata.parquet',index=False)
    path=out/'fe171-hourly-v2.parquet'
    oldpath=path if path.exists() and path.with_suffix('.parquet.json').exists() else LB/'store/fe171-cache/fe171-hourly-v1.parquet'
    oldinfo=json.loads(oldpath.with_suffix('.parquet.json').read_text())
    historical=meta.loc[meta.slot<=pd.Timestamp(oldinfo['slot_max'])]
    partial=out/'scratch'/('.'+path.name+'.partial.parquet'); partial.parent.mkdir(exist_ok=True)
    identity=f.source_identity(meta)['source_sha256']
    partial_identity=out/'scratch/source-identity.json'
    if partial.exists() and (not partial_identity.exists() or json.loads(partial_identity.read_text()).get('source_sha256')!=identity): partial.unlink()
    write(partial_identity,{'source_sha256':identity})
    if f.source_identity(historical)['source_sha256']==oldinfo['source_sha256']:
        previous=pd.read_parquet(oldpath)
        assert len(previous)==len(historical) and list(previous.columns)==f._columns()
        partial=out/'scratch'/('.'+path.name+'.partial.parquet'); partial.parent.mkdir(exist_ok=True)
        if not partial.exists(): previous.reset_index(drop=True).to_parquet(partial)
        write(out/'historical-reuse.json',{'rows':len(previous),'sha256':sha(oldpath),'historical_source_metadata_sha256':oldinfo['source_sha256'],'identity_matches_current_metadata':True,'extractor_version':oldinfo['extractor_version']})
    frame,info=f.build_hourly(meta,path,workers=2,scratch_dir=out/'scratch')
    receipt('fe171',path,[out/'metadata.parquet',Path(f.__file__)],{'coverage':info,'source_policy':'raw per-slot features; causal downstream requires obs_end+1h<=origin and S3 LastModified<=origin and byte-bound source catalog','admitted_for_experiments':False,'limitation':'full Fe171 S3 byte binding is a separate catalog stage'})


def fe171_verify(end):
    from experiments.d146_fe171_channel import strips171 as f
    out=STORE/'fe171'; meta=pd.read_parquet(out/'metadata.parquet')
    for col in ('slot','obs_start','obs_end','s3_modified'): meta[col]=pd.to_datetime(meta[col],utc=True)
    records={r['filename']:r for r in map(json.loads,(out/'source_catalog.jsonl').read_text().splitlines()) if r.get('byte_identity_passed')}
    for i,row in meta.iterrows():
        r=records.get(Path(str(row.nc_path)).name)
        if r and row.obs_end<=row.slot and abs((pd.Timestamp(r['obs_end_utc'])-row.obs_end).total_seconds())<=1:
            meta.at[i,'s3_modified']=pd.Timestamp(r['s3_last_modified_utc'])
        else: meta.at[i,'status']='unverified'
    meta.to_parquet(out/'metadata-bound.parquet',index=False)
    raw=pd.read_parquet(out/'fe171-hourly-v2.parquet'); block=raw[sorted(c for c in raw if '_lat1_' in c)]
    origins=pd.date_range(pd.Timestamp('2022-10-01',tz='UTC'),end,freq='h')
    selection=f.select_slots(origins,meta,block); admitted=selection.slot_idx>=0
    assert (selection.loc[admitted,'obs_end']+pd.Timedelta(hours=1)<=selection.loc[admitted,'origin']).all()
    features,_=f.strip_features(origins,meta=meta,block=block,selection=selection)
    features.to_parquet(out/'fe171-strips.parquet',index=False); selection.to_parquet(out/'selection.parquet',index=False)
    probe=pd.date_range('2026-06-01','2026-06-03',freq='3h',tz='UTC')
    poison=f.poison_test(probe,pd.Timestamp('2026-06-02',tz='UTC'),meta=meta,block=block); assert poison['passed'],poison
    previous=json.loads((out/'MANIFEST.json').read_text())
    previous.update({'source_manifest':str(out/'source_manifest.json'),'source_catalog_sha256':sha(out/'source_catalog.jsonl'),'admitted_for_experiments':True,'limitation':None,'poison':poison,'causal_strip_output':str(out/'fe171-strips.parquet'),'causal_strip_sha256':sha(out/'fe171-strips.parquet'),'source_latest_obs':max(r['obs_end_utc'] for r in records.values()),'source_policy':'metadata-bound.parquet and Fe171 strips enforce obs_end+1h<=origin, NetCDF S3 LastModified<=origin, and byte binding; raw hourly features require this selector'})
    write(out/'MANIFEST.json',previous)
    print(json.dumps({'fe171_verified':True,'poison':poison,'strip_rows':len(features)}),flush=True)

def dim(end):
    import dim_prepare as d
    d.MAP_CACHE_ROOT=STORE/'suvi-dim/frame-maps'; d.PRODUCT_CACHE_ROOT=STORE/'suvi-dim/frame-products'
    original=LB/'experiments/lbre_suvidim72_20261014'
    assert sha(HERE/'dim_features.py')==sha(original/'dim_features.py')
    fingerprint=hashlib.sha256()
    for name in ('config.json','dim_features.py','prepare.py'):
        fingerprint.update(name.encode()); fingerprint.update((original/name).read_bytes())
    d.LEGACY_FRAME_MAP_CODE_HASHES=(*d.LEGACY_FRAME_MAP_CODE_HASHES,fingerprint.hexdigest())
    # Reuse unchanged transform keys; v2 orchestration identity remains in MANIFEST.
    d.code_hash=lambda:fingerprint.hexdigest()
    d.N_THREADS=1
    legacy=Path('/home/t-lab01/.local/state/research-dispatch/runs/solar/solar-inc-suvidim72-20261014-20261014-031438/cache/frame-maps')
    d.MAP_CACHE_ROOT.mkdir(parents=True,exist_ok=True)
    for src in legacy.glob('*.npz'):
        target=d.MAP_CACHE_ROOT/src.name
        if not target.exists(): os.link(src,target)
    d.PRODUCT_CACHE_ROOT.mkdir(parents=True,exist_ok=True)
    for src in legacy.parent.joinpath('frame-products').glob('*.npz'):
        target=d.PRODUCT_CACHE_ROOT/src.name
        if not target.exists(): os.link(src,target)

    pointer=json.loads((LB/'store/suvi-dim72-cache/solar-inc-suvidim72-20261014-pointer.json').read_text())
    historical=pd.read_parquet(Path(pointer['feature_dir'])/'suvi_dim72_features.parquet')
    original_origins=pd.DatetimeIndex(historical['origin_last_input_utc'])
    extension=pd.date_range(original_origins.max()+pd.Timedelta(hours=1),end,freq='h')
    origins=original_origins.append(extension).unique().sort_values()
    train=original_origins[original_origins<pd.Timestamp('2026-05-29',tz='UTC')]; official=origins[origins>=pd.Timestamp('2026-05-31T23:00:00Z')]
    d.load_origins=lambda:(train,official,origins)
    legacy_receipt=json.loads((Path(pointer['feature_dir'])/'suvi_dim72_preparation_receipt.json').read_text())
    assert legacy_receipt['feature_parquet_sha256']==sha(Path(pointer['feature_dir'])/'suvi_dim72_features.parquet')
    assert legacy_receipt['code_config_sha256']==fingerprint.hexdigest()
    oldvalues=historical.set_index('origin_last_input_utc')
    original_arrays=d._origin_arrays; reused_selection=[]
    def arrays(origin, catalog, get_map, product_cache=None):
        if product_cache is None or origin not in oldvalues.index:
            return original_arrays(origin,catalog,get_map,product_cache)
        # Reuse only after proving the identical per-origin input graph and
        # cached product bytes through their input keys; no image statistics
        # or source availability are inferred from the old feature values.
        selected=set()
        for index,refs in d.current_reference_groups(origin,catalog):
            product=product_cache.get(index,refs)
            if product['valid'].sum()<100: continue
            selected.add(catalog[index]['frame_id'])
            selected.update(catalog[j]['frame_id'] for j in refs)
        ids=sorted(selected)
        reused_selection.append({'origin':origin.isoformat(),'selected_ids_sha256':d._digest_ids(ids),'selected_count':len(ids)})
        if len(reused_selection)==len(historical):
            payload='\n'.join(d._json_bytes(row).decode().rstrip('\n') for row in reused_selection).encode()
            assert d.sha_bytes(payload)==legacy_receipt['origin_selection_sha256'],'Historical dimming source selection changed; cannot reuse'
            assert product_cache.receipt()['input_key_manifest_sha256']==legacy_receipt['frame_product_cache']['input_key_manifest_sha256'],'Historical dimming products changed; cannot reuse'
            write(STORE/'suvi-dim/historical-reuse.json',{'rows':len(historical),'source_selection_sha256':d.sha_bytes(payload),'products_input_key_manifest_sha256':product_cache.receipt()['input_key_manifest_sha256'],'transform_hash':fingerprint.hexdigest(),'feature_sha256':legacy_receipt['feature_parquet_sha256']})
        return oldvalues.loc[origin].to_dict(),ids,{'reused_identical_source_graph':True}
    d._origin_arrays=arrays

    def remote(frame):
        byname={r['filename']:r for r in map(json.loads,(STORE/'euvwave/source_catalog.jsonl').read_text().splitlines())}
        rows=[]
        for r in frame.to_dict('records'):
            src=byname.get(Path(str(r['local_file'])).name)
            if not src or not src.get('byte_identity_passed'): continue
            mod=pd.Timestamp(src['s3_last_modified_utc']); obs=pd.Timestamp(src['obs_end_utc'])
            rows.append({'frame_id':r['frame_id'],'source_url':src['s3_url'],'s3_key':src['s3_key'],'obs_start':src['obs_start_utc'],'obs_end':obs.isoformat(),'obs_end_ns':obs.value,'last_modified':mod.isoformat(),'last_modified_ns':mod.value,'remote_etag':src['s3_etag'],'remote_size':src['s3_size'],'local_path':src['local_path'],'byte_equal':False,'source_usable':False,'source_unusable_reason':None,'map_reused_from_previous_code':False,'map_error':None})
        return rows
    d.remote_catalog=remote
    try:
        result=d.prepare()
    except AssertionError as exc:
        if not str(exc).startswith('Historical dimming'): raise
        print(str(exc)+'; rebuilding all feature values',flush=True)
        d._origin_arrays=original_arrays
        result=d.prepare()
    source=Path(result['features_path']); output=STORE/'suvi-dim/suvi_dim72_features.parquet'; shutil.copy2(source,output)
    receipt('suvi-dim',output,[STORE/'euvwave/source_catalog.jsonl',HERE/'dim_features.py',HERE/'dim_prepare.py'],{'preparation':result,'poison':json.loads(Path(result['probe_path']).read_text())})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['catalog','fe171-catalog','euvwave','chmorph','chmask','fe171','fe171-verify','suvi-dim']); p.add_argument('--end',default='now'); a=p.parse_args()
    end=pd.Timestamp.now(tz='UTC').floor('h') if a.end=='now' else pd.Timestamp(a.end)
    if end.tzinfo is None: end=end.tz_localize('UTC')
    {'catalog':lambda e:catalog(),'fe171-catalog':lambda e:catalog('Fe171'),'euvwave':wave,'chmorph':morphology,'chmask':mask,'fe171':fe171,'fe171-verify':fe171_verify,'suvi-dim':dim}[a.dataset](end)
