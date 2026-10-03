"""Pinned frozen V-JEPA2.1 encoder and deterministic positional pooling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

UPSTREAM_SHA = '204698b45b3712590f06245fbfba32d3be539812'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(upstream, checkpoint, device):
    """Use official 2.1 encoder factory arguments, never initialize a predictor."""
    sys.path.insert(0, str(Path(upstream).resolve()))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    encoder = vit_base(patch_size=16, img_size=(384, 384), num_frames=64,
                       tubelet_size=2, use_sdpa=True, use_SiLU=False,
                       wide_SiLU=True, uniform_power=False, use_rope=True,
                       img_temporal_dim_size=1, interpolate_rope=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = {key.replace('module.', '').replace('backbone.', ''): val
               for key, val in state['ema_encoder'].items()}
    encoder.load_state_dict(weights, strict=True)
    del state, weights
    encoder.requires_grad_(False).eval().to(device)
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('encoder freeze failed')
    return encoder


def pool_tokens(tokens):
    """Time mean plus disk/meridian/equator/left/right in fixed solar coords."""
    if tokens.ndim != 3 or tokens.shape[1] not in (576, 4608) or tokens.shape[-1] != 768:
        raise ValueError(f'unexpected encoder tokens {tokens.shape}')
    grid = tokens.float().reshape(tokens.shape[0], -1, 24, 24, 768).mean(dim=1)
    coord = (torch.arange(24, device=grid.device) + .5 - 12) / 9.6
    yy, xx = torch.meshgrid(coord, coord, indexing='ij')
    disk = xx.square() + yy.square() <= 1
    regions = [disk, disk & (xx.abs() <= .25), disk & (yy.abs() <= .25),
               disk & (xx < 0), disk & (xx >= 0)]
    pooled = torch.cat([grid[:, mask, :].mean(dim=1) for mask in regions], dim=-1)
    if not torch.isfinite(pooled).all():
        raise ValueError('nonfinite frozen embedding')
    return pooled


@torch.inference_mode()
def encode(encoder, clips, device):
    x = torch.as_tensor(clips, dtype=torch.float32, device=device)
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] not in (1, 16) or x.shape[3:] != (384, 384):
        raise ValueError('expected B,3,(1 or 16),384,384')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tokens = encoder(x)
    return pool_tokens(tokens).cpu().numpy().astype(np.float32), tuple(tokens.shape)


def smoke(run_dir):
    from experiments.ch_breakthrough_v2.common import acquire_gpu_lease
    from .images import inventory_sources, fit_intensity, select_clip, build_clip
    p = Path(run_dir)
    inventory = inventory_sources()
    inventory.to_parquet(p/'inventory.parquet', index=False)
    transform = fit_intensity(inventory, '2025-07-31T23:00:00Z')
    (p/'intensity.json').write_text(json.dumps(transform.to_dict(), indent=2))
    origin = inventory.loc[inventory.available & (inventory.slot <= '2025-07-31T23:00:00Z'), 'slot'].iloc[-1]
    clip, stats = build_clip(select_clip(inventory, origin), transform)
    np.save(p/'cache/smoke-clip.npy', clip)
    with acquire_gpu_lease(Path(__file__).resolve().parents[2] / 'store/ch-breakthrough-v2/gpu-leases') as gpu:
        device = f'cuda:{gpu}'
        torch.set_num_threads(4)
        model = load_encoder(p/'upstream', p/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt', device)
        torch.cuda.reset_peak_memory_stats(gpu)
        results = {}
        for name, x in [('image',clip[None,:,-1:]),('video',clip[None])]:
            encode(model, x, device)
            torch.cuda.synchronize(gpu)
            started = time.perf_counter()
            pooled, shape = encode(model, x, device)
            torch.cuda.synchronize(gpu)
            results[name] = {'input_shape':list(x.shape),'token_shape':shape,
                'pooled_shape':list(pooled.shape),'seconds':time.perf_counter()-started,
                'finite':bool(np.isfinite(pooled).all()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu)}
        result = {'gpu':gpu,'device_name':torch.cuda.get_device_name(gpu),
                  'parameters':sum(x.numel() for x in model.parameters()),
                  'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
                  'eval_mode':not model.training,'precision':'BF16 autocast, FP32 pooling',
                  'origin':origin.isoformat(),'frame_stats':stats,'arms':results,
                  'checkpoint_sha256':sha256(p/'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt')}
    (p/'gpu-smoke.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    smoke(parser.parse_args().run_dir)
