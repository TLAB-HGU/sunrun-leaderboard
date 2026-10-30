"""Restartable real-GPU extraction with immutable source and clip identities."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.ch_breakthrough_v2.common import acquire_gpu_lease
from experiments.mse20_four_regimes.protocol import write_json
from .encoder import encode, load_encoder, sha256
from .images import IntensityTransform, preprocess_frame

NAT = np.iinfo(np.int64).min
QUALITY = ['missing_fraction','filled_fraction','mean_age_hours','last_age_hours','valid_disk_fraction']


def plan_sources(inventory, origins):
    """Vectorized selected hourly source lookup, bounded by actual frame age."""
    rows = inventory.loc[inventory.available].sort_values('slot').reset_index(drop=True)
    slots = pd.DatetimeIndex(rows.slot).as_unit('ns').asi8
    ends = pd.DatetimeIndex(rows.obs_end).as_unit('ns').asi8
    requested = origins[:, None] - np.arange(45, -1, -3, dtype=np.int64)[None, :] * 3600_000_000_000
    pos = np.searchsorted(slots, requested, side='right')-1
    safe = np.maximum(pos, 0)
    good = (pos >= 0) & (ends[safe] <= requested) & (requested-ends[safe] <= 6*3600_000_000_000)
    ids = np.where(good, pos, -1)
    source_ends = np.where(good, ends[safe], NAT)
    age = np.where(good, (requested-ends[safe])/3600_000_000_000, 7.)
    filled = good & (slots[safe] < requested)
    if np.any(source_ends > requested):
        raise ValueError('future frame in planned clip')
    return rows, ids, source_ends, age, filled


def source_binding(run):
    root = Path(__file__).resolve().parent
    files = [root/'images.py', root/'encoder.py', root/'extract.py', root.parents[0]/'xgboost_suvi_fusion/features.py']
    hashes = {str(p): sha256(p) for p in files}
    hashes.update({str(p.relative_to(run)): sha256(p) for p in sorted((run/'upstream').rglob('*.py')) if '.git' not in p.parts})
    return hashes


def prepare_frames(run, rows, ids, workers):
    transform_data = json.loads((run/'intensity.json').read_text())
    transform = IntensityTransform(**transform_data)
    frame_dir = run/'cache/frames'; frame_dir.mkdir(parents=True,exist_ok=True)
    needed = np.unique(ids[ids >= 0])
    transform_hash = hashlib.sha256(json.dumps(transform_data,sort_keys=True).encode()).hexdigest()
    binding = source_binding(run)
    preprocessing_hash = hashlib.sha256(json.dumps({k:v for k,v in binding.items() if k.endswith(('images.py','xgboost_suvi_fusion/features.py'))},sort_keys=True).encode()).hexdigest()
    def one(i):
        row = rows.iloc[i]
        key = hashlib.sha256(f'{row.path}:{row.size_bytes}:{row.mtime_ns}:{transform_hash}:{preprocessing_hash}'.encode()).hexdigest()
        out, meta = frame_dir/(key+'.npy'), frame_dir/(key+'.json')
        if out.exists() and meta.exists():
            record=json.loads(meta.read_text())
            stat=Path(row.path).stat()
            if stat.st_size!=row.size_bytes or stat.st_mtime_ns!=row.mtime_ns or sha256(row.path)!=record['raw_sha256']:
                raise ValueError('raw source identity changed on cache resume')
            if sha256(out) != record['preprocessed_sha256']:
                raise ValueError('frame cache hash mismatch')
            return i, key, record
        before=Path(row.path).stat()
        image, _, geom = preprocess_frame(row.path, transform)
        raw_hash=sha256(row.path)
        after=Path(row.path).stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or after.st_size!=row.size_bytes or after.st_mtime_ns!=row.mtime_ns:
            raise ValueError('raw source changed during preparation')
        temp=out.with_suffix(f'.{os.getpid()}.tmp')
        with temp.open('wb') as f: np.save(f,image[0],allow_pickle=False)
        temp.replace(out)
        record={'raw_path':row.path,'raw_sha256':raw_hash,'bytes':int(row.size_bytes),'mtime_ns':int(row.mtime_ns),
                'preprocessed_sha256':sha256(out),'geometry':geom,'source_slot':row.slot.isoformat(),'obs_end':row.obs_end.isoformat()}
        write_json(meta,record)
        return i,key,record
    mapping={}; start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,(i,key,record) in enumerate(pool.map(one,needed),1):
            mapping[int(i)]={'key':key,**record}
            if n%256==0 or n==len(needed):
                write_json(run/'extraction_progress.json',{'stage':'preprocessing','done':n,'total':len(needed),'seconds':time.perf_counter()-start})
                print(f'preprocess {n}/{len(needed)} {time.perf_counter()-start:.1f}s',flush=True)
    write_json(run/'frame-cache-manifest.json',{'transform_sha256':transform_hash,'frames':mapping,'seconds':time.perf_counter()-start})
    return mapping


def extract(run_dir, batch_size=4, workers=6, pilot=16):
    run=Path(run_dir)
    origins=np.load(run/'planned_origins.npz',allow_pickle=False)['origins']
    inventory=pd.read_parquet(run/'inventory.parquet')
    rows,ids,ends,age,filled=plan_sources(inventory,origins)
    np.savez(run/'clip-source-plan.npz',origins=origins,source_ids=ids,source_end_ns=ends,age_hours=age,filled=filled)
    binding = source_binding(run)
    write_json(run/'execution-source-hashes.json',binding)
    snapshot = run/'source-snapshot'; snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    mapping=prepare_frames(run,rows,ids,workers)
    zero=np.zeros((384,384),np.float32)
    @lru_cache(maxsize=256)
    def frame(i):
        return np.load(run/'cache/frames'/ (mapping[i]['key']+'.npy'),allow_pickle=False) if i>=0 else zero
    def clips_for(positions):
        clips=[];qs=[];iqs=[]
        for at in positions:
            channel=np.stack([frame(int(i)) for i in ids[at]],axis=0)
            clips.append(np.repeat(channel[None],3,axis=0))
            valid=np.array([mapping[int(i)]['geometry']['valid_disk_fraction'] if i>=0 else 0. for i in ids[at]])
            qs.append([float((ids[at]<0).mean()),float(filled[at].mean()),float(age[at].mean()),float(age[at,-1]),float(valid.mean())])
            iqs.append([float(ids[at,-1]<0),float(filled[at,-1]),float(age[at,-1]),float(age[at,-1]),float(valid[-1])])
        return np.stack(clips),np.asarray(qs,np.float32),np.asarray(iqs,np.float32)
    chunk_dir=run/'cache/embeddings';chunk_dir.mkdir(exist_ok=True)
    signature={'origins_sha256':hashlib.sha256(origins.tobytes()).hexdigest(),'clip_plan_sha256':sha256(run/'clip-source-plan.npz'),
               'checkpoint_sha256':sha256(run/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt'),
               'frame_manifest_sha256':sha256(run/'frame-cache-manifest.json'),'execution_source_hashes':binding}
    # Frame-manifest wall time may vary on resume: bind content rather than duration.
    signature['frame_manifest_sha256']=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    bind=run/'extraction_identity.json'
    if bind.exists() and json.loads(bind.read_text())!=signature: raise ValueError('extraction identity drift')
    write_json(bind,signature)
    timings=[];start=time.perf_counter()
    with acquire_gpu_lease(Path(__file__).resolve().parents[2] / 'store/ch-breakthrough-v2/gpu-leases') as gpu:
        torch.set_num_threads(4);device=f'cuda:{gpu}'
        model=load_encoder(run/'upstream',run/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt',device)
        torch.cuda.reset_peak_memory_stats(gpu)
        # Distributed temporal pilot is shape/throughput only, never a partial skill score.
        pilot_ids=np.unique(np.linspace(0,len(origins)-1,min(pilot,len(origins)),dtype=int))
        pclips,_,_=clips_for(pilot_ids[:batch_size]);encode(model,pclips,device);encode(model,pclips[:,:,-1:],device)
        torch.cuda.synchronize(gpu)
        pilot_start=time.perf_counter();encode(model,pclips,device);torch.cuda.synchronize(gpu)
        pilot_seconds=time.perf_counter()-pilot_start
        write_json(run/'extraction-pilot.json',{'n_planned':len(origins),'pilot_batch':len(pclips),'video_seconds':pilot_seconds,
                   'estimated_gpu_video_seconds':pilot_seconds/len(pclips)*len(origins), 'gpu':gpu,'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
                   'note':'throughput pilot only; no partial model performance claim'})
        for offset in range(0,len(origins),256):
            chunk=chunk_dir/f'{offset:06d}.npz'; meta=chunk.with_suffix('.json')
            if chunk.exists() and meta.exists():
                if sha256(chunk)!=json.loads(meta.read_text())['sha256']:raise ValueError('embedding checkpoint corrupt')
                continue
            finish=min(offset+256,len(origins));vals={k:[] for k in ('image','video','image_quality','video_quality')};seconds={'image':0.,'video':0.}
            for at in range(offset,finish,batch_size):
                positions=np.arange(at,min(at+batch_size,finish));clips,quality,iq=clips_for(positions)
                for arm,x in [('image',clips[:,:,-1:]),('video',clips)]:
                    torch.cuda.synchronize(gpu);begin=time.perf_counter();pooled,_=encode(model,x,device);torch.cuda.synchronize(gpu)
                    seconds[arm]+=time.perf_counter()-begin
                    if arm=='image':pooled[ids[positions,-1]<0]=np.nan
                    else:pooled[(ids[positions]<0).all(axis=1)]=np.nan
                    vals[arm].append(pooled)
                vals['image_quality'].append(iq);vals['video_quality'].append(quality)
            temp=chunk.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f,origins=origins[offset:finish],image_source_end_ns=ends[offset:finish,-1],
                         video_source_end_ns=ends[offset:finish].max(axis=1),**{k:np.concatenate(v) for k,v in vals.items()})
            temp.replace(chunk);write_json(meta,{'sha256':sha256(chunk),'rows':finish-offset,'gpu_seconds':seconds});timings.append(seconds)
            write_json(run/'extraction_progress.json',{'stage':'encoding','done':finish,'total':len(origins),'seconds':time.perf_counter()-start,'gpu':gpu})
            print(f'encode {finish}/{len(origins)} {time.perf_counter()-start:.1f}s',flush=True)
        memory=torch.cuda.max_memory_allocated(gpu)
    chunks=[np.load(p,allow_pickle=False) for p in sorted(chunk_dir.glob('*.npz'))]
    merged={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0].files}
    if not np.array_equal(merged['origins'],origins):raise ValueError('incomplete embedding coverage')
    if source_binding(run)!=binding:
        raise ValueError('execution source code changed during extraction')
    np.savez(run/'embeddings.npz',**merged)
    stats={'stage':'complete','n_origins':len(origins),'wall_seconds':time.perf_counter()-start,
           'gpu':gpu,'peak_allocated_bytes':memory,'batch_size':batch_size,'embeddings_sha256':sha256(run/'embeddings.npz'),
           'gpu_seconds':{arm:sum(json.loads(p.read_text())['gpu_seconds'][arm] for p in chunk_dir.glob('*.json')) for arm in ['image','video']},
           'image_quality_columns':QUALITY,'video_quality_columns':QUALITY,'source_identity':signature,
           'coverage':{arm:{'any_image_fraction':float(np.isfinite(merged[arm]).any(1).mean()),'quality_mean':merged[arm+'_quality'].mean(0).tolist()} for arm in ['image','video']}}
    write_json(run/'extraction.json',stats);write_json(run/'extraction_progress.json',stats);print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();extract(a.run_dir,a.batch_size,a.workers)
