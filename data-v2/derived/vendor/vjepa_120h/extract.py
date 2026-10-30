"""Immutable, two-GPU, restartable extraction of all 120 hourly slots."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from vjepa_frozen.encoder import encode, load_encoder, sha256
from .temporal import (CLIP_OFFSETS, QUALITY_COLUMNS, all_quality, coverage_manifest,
                       deduplicate_clips, plan_sources)

CHUNK_SIZE = 256
LEASE_DIR = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/ch-breakthrough-v2/gpu-leases')


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **arrays)
    temp.replace(path)


def binding(run):
    old = Path(__file__).resolve().parent.parent / 'vjepa_frozen'
    files = [Path(__file__), Path(__file__).with_name('temporal.py'), Path(__file__).with_name('cache.py'), old/'encoder.py', old/'images.py',
             Path('/home/t-lab01/sunrun/leaderboard/experiments/xgboost_suvi_fusion/features.py'), run/'inventory.parquet', run/'intensity.json',
             run/'planned_origins.npz', run/'frame-cache-manifest.json', run/'protocol.json', run/'extraction-runtime.json',
             run/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt']
    files += sorted((run/'upstream').rglob('*.py'))
    return {str(p): sha256(p) for p in files if '.git' not in p.parts}


def prepare(run):
    run = Path(run)
    origins = np.load(run/'planned_origins.npz', allow_pickle=False)['origins']
    rows, ids, ends, age, filled = plan_sources(pd.read_parquet(run/'inventory.parquet'), origins)
    plan = deduplicate_clips(origins, ids, ends, age, filled)
    mapping = {int(k): v for k, v in json.loads((run/'frame-cache-manifest.json').read_text())['frames'].items()}
    required = np.unique(ids[ids >= 0])
    if any(int(i) not in mapping for i in required):
        raise ValueError('frame cache does not cover all 120-hour sources')
    lookup = np.zeros(len(rows)+1, np.float32)
    for i in required:
        lookup[int(i)+1] = mapping[int(i)]['geometry']['valid_disk_fraction']
    validity = lookup[ids+1]
    quality = all_quality(ids, age, filled, validity)
    signature = binding(run)
    signature['plan_content_sha256'] = hashlib.sha256(b''.join(plan[k].tobytes() for k in sorted(plan))).hexdigest()
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    previous = run/'extraction_identity.json'
    if previous.exists():
        if json.loads(previous.read_text()) != {'identity': identity, 'sources': signature}:
            raise ValueError('extraction source identity drift')
    # Recover interrupted preparation, validating every existing artifact before
    # committing the identity. An identity alone must never skip preparation.
    ensure_npz(run/'clip-source-plan.npz', origins=origins, **plan)
    source_max = np.stack([ends[:, c].max(1) for c in CLIP_OFFSETS], axis=1)
    ensure_npz(run/'embedding_metadata.npz', origins=origins, source_end_ns=source_max, quality=quality)
    coverage = {**coverage_manifest(), 'origins': len(origins),
        'requested_clips': len(origins)*8, 'unique_clips': len(plan['clip_ends']),
        'quality_columns': QUALITY_COLUMNS, 'quality_mean': quality.mean(0).tolist(),
        'all_source_times_causal': True}
    coverage_path = run/'coverage-manifest.json'
    if coverage_path.exists() and json.loads(coverage_path.read_text()) != coverage:
        raise ValueError('coverage manifest drift')
    if not coverage_path.exists():
        write_json(coverage_path, coverage)
    if not previous.exists():
        write_json(previous, {'identity': identity, 'sources': signature})
    return plan, mapping, identity


def ensure_npz(path, **arrays):
    path = Path(path)
    if not path.exists():
        atomic_npz(path, **arrays)
        return
    with np.load(path, allow_pickle=False) as old:
        if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
            raise ValueError(f'prepared artifact drift: {path.name}')


@contextmanager
def gpu_lease(gpu):
    import fcntl
    if gpu not in (0, 1):
        raise ValueError('absolute GPU must be 0 or 1')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0,1'):
        raise ValueError('CUDA_VISIBLE_DEVICES changes absolute GPU identity')
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    with (LEASE_DIR/f'gpu{gpu}.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if memory > 1024:
                raise RuntimeError(f'GPU {gpu} has {memory} MiB allocated by another process')
            yield memory
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def chunk_bounds(total, chunk_id):
    start = chunk_id*CHUNK_SIZE
    return start, min(start+CHUNK_SIZE, total)


def read_chunk(path, identity, clip_ends, chunk_id):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['identity'] != identity or metadata['sha256'] != sha256(path):
        raise ValueError('chunk hash/source identity mismatch')
    if metadata['chunk_id'] != chunk_id or metadata['gpu'] != chunk_id % 2:
        raise ValueError('chunk deterministic owner mismatch')
    lo, hi = chunk_bounds(len(clip_ends), chunk_id)
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['clip_ends'], clip_ends[lo:hi]):
            raise ValueError('chunk time identity mismatch')
        values = data['embedding'].copy()
    if values.shape != (hi-lo, 3840) or np.isinf(values).any():
        raise ValueError('chunk embedding shape/finite mismatch')
    missing = np.isnan(values)
    if np.any(missing.any(1) != missing.all(1)):
        raise ValueError('partially nonfinite clip embedding')
    return values, metadata


def extract(run_dir, gpu, batch_size=4, pilot_only=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    run = Path(run_dir)
    runtime = {'batch_size': batch_size, 'seed': 42, 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'deterministic_algorithms': True,
               'cudnn_benchmark': False, 'tf32': False, 'precision': 'BF16 autocast; FP32 pooling',
               'cublas_workspace_config': ':4096:8', 'chunk_size': CHUNK_SIZE,
               'physical_owner': 'chunk_id % 2'}
    runtime_path = run/'extraction-runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError('extraction runtime drift across restart')
    if not runtime_path.exists():
        write_json(runtime_path, runtime)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan, mapping, identity = prepare(run)
    chunk_dir = run/'cache/hourly-clip-embeddings'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    zero = np.zeros((384,384), np.float32)
    @lru_cache(maxsize=256)
    def frame(i):
        if i < 0:
            return zero
        entry = mapping[i]
        path = run/'cache/frames'/(entry['key']+'.npy')
        if sha256(path) != entry['preprocessed_sha256']:
            raise ValueError('immutable frame hash mismatch')
        value = np.load(path, allow_pickle=False)
        if value.shape != (384,384) or not np.isfinite(value).all():
            raise ValueError('invalid cached frame')
        return value
    def clips(positions):
        return np.stack([np.repeat(np.stack([frame(int(i)) for i in plan['source_ids'][p]])[None], 3, axis=0)
                         for p in positions])
    total = len(plan['clip_ends'])
    started = time.perf_counter()
    with gpu_lease(gpu) as pre_memory:
        torch.set_num_threads(4)
        device = f'cuda:{gpu}'
        model = load_encoder(run/'upstream', run/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt', device)
        torch.cuda.reset_peak_memory_stats(gpu)
        available = np.flatnonzero((plan['source_ids'] >= 0).all(1))
        if not len(available):
            raise ValueError('no complete clip for GPU pilot')
        pilot_ids = available[np.linspace(0,len(available)-1,min(batch_size,len(available)),dtype=int)]
        batch = clips(pilot_ids)
        encode(model, batch, device)
        torch.cuda.synchronize(gpu)
        before = time.perf_counter()
        pooled, shape = encode(model, batch, device)
        torch.cuda.synchronize(gpu)
        seconds = time.perf_counter()-before
        nonconstant = bool(np.ptp(pooled, axis=0).max() > 0) if len(pooled)>1 else bool(np.ptp(pooled)>0)
        if not nonconstant:
            raise ValueError('constant GPU pilot embedding')
        write_json(run/f'extraction-pilot-gpu{gpu}.json', {'gpu':gpu,'batch_size':len(batch),
            'seconds':seconds,'clips_per_second':len(batch)/seconds,'unique_clips':total,
            'estimated_single_gpu_seconds':total*seconds/len(batch),'token_shape':list(shape),
            'embedding_shape':list(pooled.shape),'finite':bool(np.isfinite(pooled).all()),
            'nonconstant':nonconstant,'frozen':not model.training and not any(p.requires_grad for p in model.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),'pre_use_memory_mib':pre_memory,
            'device_name':torch.cuda.get_device_name(gpu),'identity':identity})
        if pilot_only:
            return
        for chunk_id in range((total+CHUNK_SIZE-1)//CHUNK_SIZE):
            if chunk_id % 2 != gpu:
                continue
            path = chunk_dir/f'{chunk_id:06d}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                read_chunk(path,identity,plan['clip_ends'],chunk_id)
                continue
            lo, hi = chunk_bounds(total,chunk_id)
            values = np.full((hi-lo,3840),np.nan,np.float32)
            gpu_seconds = 0.
            for at in range(lo,hi,batch_size):
                positions = np.arange(at,min(at+batch_size,hi))
                present = (plan['source_ids'][positions] >= 0).any(1)
                selected = positions[present]
                if not len(selected):
                    continue
                x = clips(selected)
                torch.cuda.synchronize(gpu)
                before = time.perf_counter()
                out,_ = encode(model,x,device)
                torch.cuda.synchronize(gpu)
                gpu_seconds += time.perf_counter()-before
                values[selected-lo] = out
            atomic_npz(path, clip_ends=plan['clip_ends'][lo:hi], embedding=values)
            write_json(path.with_suffix('.json'), {'sha256':sha256(path),'identity':identity,
                'chunk_id':chunk_id,'gpu':gpu,'rows':hi-lo,'gpu_seconds':gpu_seconds})
            write_json(run/f'extraction-progress-gpu{gpu}.json', {'stage':'encoding',
                'last_chunk':chunk_id,'total_unique_clips':total,'wall_seconds':time.perf_counter()-started})
            print(f'gpu{gpu} chunk {chunk_id} end {hi}/{total}',flush=True)
        write_json(run/f'extraction-worker-gpu{gpu}.json', {'stage':'complete','gpu':gpu,
            'wall_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
            'batch_size':batch_size,'identity':identity})


def merge(run_dir):
    run = Path(run_dir)
    plan,mapping,identity = prepare(run)
    total = len(plan['clip_ends'])
    unique = np.full((total,3840),np.nan,np.float32)
    timings = []
    for chunk_id in range((total+CHUNK_SIZE-1)//CHUNK_SIZE):
        path = run/'cache/hourly-clip-embeddings'/f'{chunk_id:06d}.npz'
        values,metadata = read_chunk(path,identity,plan['clip_ends'],chunk_id)
        lo,hi = chunk_bounds(total,chunk_id)
        missing = (plan['source_ids'][lo:hi] < 0).all(1)
        if not np.array_equal(np.isnan(values).all(1),missing):
            raise ValueError('embedding presence inconsistent with sources')
        unique[lo:hi] = values
        timings.append(metadata['gpu_seconds'])
    destination = run/'embeddings.npy'
    temp = run/f'embeddings.{os.getpid()}.tmp.npy'
    output = np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(plan['origin_to_clip']),30720))
    for lo in range(0,len(output),256):
        hi = min(lo+256,len(output))
        output[lo:hi] = unique[plan['origin_to_clip'][lo:hi]].reshape(hi-lo,30720)
    output.flush()
    del output
    temp.replace(destination)
    write_json(run/'extraction.json', {'stage':'complete','n_origins':len(plan['origin_to_clip']),
        'unique_clips':total,'gpu_seconds':sum(timings),'embedding_shape':[len(plan['origin_to_clip']),30720],
        'embeddings_sha256':sha256(destination),'metadata_sha256':sha256(run/'embedding_metadata.npz'),
        'identity':identity,'coverage':json.loads((run/'coverage-manifest.json').read_text()),
        'workers':[json.loads((run/f'extraction-worker-gpu{gpu}.json').read_text()) for gpu in (0,1)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1))
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args = parser.parse_args()
    if args.merge:
        merge(args.run_dir)
    elif args.gpu is not None and args.batch_size > 0:
        extract(args.run_dir,args.gpu,args.batch_size,args.pilot_only)
    else:
        parser.error('specify --merge or --gpu, and a positive batch size')
