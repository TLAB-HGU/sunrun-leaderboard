"""D23: frozen DINOv3 vitb16 patch tokens pooled in solar-geometry longitude strips.

Hypothesis (X0007: generic frozen embeddings + PCA lose solar structure, prefer
physics-aligned longitude features): keep the encoder frozen, but replace the
generic full-disk pooling with per-strip pooling. Each patch token is mapped to
observer-relative longitude/latitude through the SUVI disk geometry
(center/RSUN/north-up orientation + SOLAR_B0). For every (origin, horizon,
speed) row only patches inside the [lambda-5, lambda+5] west-positive strip
are averaged, where lambda is the D1 strip formula (shared OMEGA/TAU,
imported from strips, never redefined here). Per-strip means (768-d) are
compressed with PCA fit on train rows only, 4 dims per strip, 20 total, and
appended to the E0003 design matrix. No fine-tuning anywhere.

Proxy rule (same slots as D1, read-only import): newest slot S<=O-1h with
S3 LastModified<=O, local bytes present, finite cache cells; else the newest
earlier qualifying slot plus age; none -> NaN + stale. Byte identity is
evidenced by the availability manifest (local MD5 vs S3 ETag HEAD).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

LB = Path('/home/t-lab01/sunrun/leaderboard')

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
# sha256 of model.safetensors content (== HF blob name); weights live only in
# the HF cache / local cache, never committed or distributed.
MODEL_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
MODEL_BYTES = 342662192
TRANSFORMERS_PIN = "4.56.2"
TORCH_PIN = "2.13.0+cu126"

CANVAS = 224
RADIUS_FRAC = 0.4
PATCH = 16
GRID_N = 14
N_PREFIX = 5  # 1 CLS + 4 register tokens precede the 196 patch tokens
DIM = 768
STRIP_HALF = 5.0
VALID_MIN = 0.5  # patch needs >=50% valid disk pixels to join a strip mean

SPEEDS = (350, 450, 550, 650, 750)
PCA_PER_STRIP = 4
N_STRIPS = len(SPEEDS)

# ImageNet normalisation from the model's preprocessor_config.json.
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE_VERSION = 1
CACHE_DIR = LB / "store" / "dinov3-cache" / f"v{CACHE_VERSION}"
SCRATCH = LB / "store" / "scratch" / "worker-d23b"


def dino_columns() -> list:
    return [f"dino_v{v}_pca{k}" for v in SPEEDS for k in range(PCA_PER_STRIP)]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_sha256() -> str:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    return sha256_file(Path(snap) / "model.safetensors")


def _header_numbers(header: dict) -> dict:
    h = {str(k).upper(): v for k, v in header.items()}

    def num(key, default=None):
        value = h.get(key, default)
        if value is None:
            raise ValueError(f"missing SUVI geometry: {key}")
        return float(np.asarray(value).item())

    angle = np.deg2rad(num("CROTA", 0))
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    if all(k in h for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        matrix = np.array([[num("PC1_1"), num("PC1_2")],
                           [num("PC2_1"), num("PC2_2")]])
    rsun = num("RSUN")
    if not np.isfinite(rsun) or rsun <= 0:
        raise ValueError("RSUN must be finite and positive")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("invalid orientation matrix")
    return {"matrix": matrix, "crpix1": num("CRPIX1"), "crpix2": num("CRPIX2"),
            "rsun": rsun, "b0": np.deg2rad(num("SOLAR_B0", 0))}


def patch_lonlat(header: dict, size: int = CANVAS) -> tuple:
    """West-positive longitude / latitude (deg) of every patch centre.

    North-up canvas, disk radius RADIUS_FRAC*size, same convention as the
    vjepa canonical frame: canvas right is solar west, so lon > 0 is west.
    """
    g = _header_numbers(header)
    c = PATCH / 2.0 + PATCH * np.arange(GRID_N)
    xx, yy = np.meshgrid(c, c)  # (14, 14) pixel centres, row-major
    x = (xx - (size - 1) / 2) / (RADIUS_FRAC * size)
    y = ((size - 1) / 2 - yy) / (RADIUS_FRAC * size)
    r2 = x * x + y * y
    ondisk = r2 <= 1.0
    z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    cb, sb = np.cos(g["b0"]), np.sin(g["b0"])
    lat = np.arcsin(np.clip(y * cb + z * sb, -1.0, 1.0))
    lon = np.arctan2(x, z * cb - y * sb)
    return np.rad2deg(lon), np.rad2deg(lat), ondisk


def canvas_image(path: str, low: float, high: float,
                 size: int = CANVAS) -> tuple:
    """North-up disk canvas (3, size, size) in processor normalisation.

    Bilinear resampling of physical radiance with a validity weight, mirroring
    the vjepa canonical frame; invalid disk maps to 0 after normalisation.
    Returns (image, valid_patch(14,14), geometry dict).
    """
    import sys

    sys.path.insert(0, str(LB / "experiments" / "xgboost_suvi_fusion"))
    from features import read_suvi_frame  # noqa: E402
    from scipy.ndimage import map_coordinates  # noqa: E402

    rad, dqf, header = read_suvi_frame(path)
    g = _header_numbers(header)
    rad = np.asarray(rad, dtype=float)
    valid_src = ((np.asarray(dqf) == 0) & np.isfinite(rad) & (rad >= 0)).astype(float)
    vals_src = np.where(valid_src > 0, rad, 0.0).astype(float)
    yy, xx = np.indices((size, size), dtype=np.float64)
    x = (xx - (size - 1) / 2) / (RADIUS_FRAC * size)
    y = ((size - 1) / 2 - yy) / (RADIUS_FRAC * size)
    src = np.linalg.solve(g["matrix"], np.stack((x.ravel(), y.ravel())) * g["rsun"])
    rows = (src[1].reshape(size, size) + g["crpix2"] - 1).ravel()
    cols = (src[0].reshape(size, size) + g["crpix1"] - 1).ravel()
    weight = map_coordinates(valid_src, [rows, cols], order=1,
                             mode="constant", cval=0.0).reshape(size, size)
    values = map_coordinates(vals_src, [rows, cols], order=1,
                             mode="constant", cval=0.0).reshape(size, size)
    good = (weight >= 0.999) & ((x * x + y * y) <= 1.0)
    values = np.where(good, values / np.maximum(weight, 1e-12), 0.0)
    img01 = np.clip((np.log1p(np.maximum(values, 0.0)) - low) / max(high - low, 1e-6),
                    0.0, 1.0)
    img01 = np.where(good, img01, 0.0).astype(np.float32)
    valid_patch = good.reshape(GRID_N, PATCH, GRID_N, PATCH).mean(axis=(1, 3))
    rgb = np.repeat(img01[None], 3, axis=0)
    rgb = ((rgb - IM_MEAN[:, None, None]) / IM_STD[:, None, None]).astype(np.float32)
    geom = {"rsun": g["rsun"], "crpix1": g["crpix1"], "crpix2": g["crpix2"],
            "solar_b0_deg": float(np.rad2deg(g["b0"])),
            "valid_disk_fraction": float(good.sum() / max(1, good.size))}
    return rgb, valid_patch.astype(np.float32), geom


def fit_intensity(paths: tuple, max_frames: int = 64) -> tuple:
    """Fixed log-radiance q1/q99 window from train-cutoff images only."""
    import sys

    sys.path.insert(0, str(LB / "experiments" / "xgboost_suvi_fusion"))
    from features import read_suvi_frame  # noqa: E402

    take = np.linspace(0, len(paths) - 1, min(max_frames, len(paths))).astype(int)
    samples = []
    for i in take:
        rad, dqf, _ = read_suvi_frame(paths[int(i)])
        rad = np.asarray(rad, dtype=float)
        ok = (np.asarray(dqf) == 0) & np.isfinite(rad) & (rad >= 0)
        samples.append(np.log1p(rad[ok]))
    values = np.concatenate(samples)
    if not len(values):
        raise ValueError("intensity fit found no valid radiance pixels")
    return float(np.quantile(values, 0.01)), float(np.quantile(values, 0.99))


def load_encoder(device: str = "cuda"):
    """Load the frozen vitb16 encoder: eval mode, no gradients, no training."""
    import torch
    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def encode_batch(model, batch: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Patch tokens (B, 196, 768) float32 on CPU; inference only."""
    import torch

    x = torch.as_tensor(np.asarray(batch, dtype=np.float32), device=device)
    with torch.inference_mode():
        out = model(pixel_values=x)
    toks = out.last_hidden_state[:, N_PREFIX:, :].float().cpu().numpy()
    assert toks.shape[1] == GRID_N * GRID_N and toks.shape[2] == DIM
    return toks.astype(np.float32)


def strip_means(tokens: np.ndarray, lon_deg: np.ndarray, use: np.ndarray,
                lam: np.ndarray) -> tuple:
    """Mean token per (h, v) strip; empty strips -> NaN.

    tokens (P,768), lon_deg/use (P,), lam (H,V) west-positive deg.
    """
    d = np.abs(lon_deg[None, None, :] - np.asarray(lam)[:, :, None])
    m = (d <= STRIP_HALF) & np.asarray(use, dtype=bool)[None, None, :]
    t = np.where(np.asarray(use, dtype=bool)[:, None],
                 np.asarray(tokens, dtype=np.float32), 0.0)
    cnt = m.sum(axis=-1).astype(np.float32)  # (H,V)
    s = np.einsum("hvp,pd->hvd", m.astype(np.float32), t)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (s / cnt[..., None]).astype(np.float32)
    means[cnt == 0] = np.nan
    return means, cnt


def fit_pcas(sample: np.ndarray) -> list:
    """One PCA(4) per strip on finite train rows; sample (N,5,768)."""
    from sklearn.decomposition import PCA

    pcas = []
    for vi in range(N_STRIPS):
        v = np.asarray(sample[:, vi, :], dtype=np.float64)
        v = v[np.isfinite(v).all(axis=1)]
        if len(v) < PCA_PER_STRIP + 1:
            raise ValueError(f"strip {SPEEDS[vi]}: only {len(v)} finite train rows")
        p = PCA(n_components=PCA_PER_STRIP, random_state=0, svd_solver="full")
        p.fit(v)
        pcas.append(p)
    return pcas


def transform_pcas(pcas: list, vecs: np.ndarray) -> np.ndarray:
    """PCA features (N,20); strips with any non-finite input stay NaN."""
    n = len(vecs)
    out = np.full((n, N_STRIPS * PCA_PER_STRIP), np.nan, dtype=np.float32)
    for vi, p in enumerate(pcas):
        v = np.asarray(vecs[:, vi, :], dtype=np.float64)
        ok = np.isfinite(v).all(axis=1)
        if ok.any():
            out[np.ix_(ok, range(vi * PCA_PER_STRIP, (vi + 1) * PCA_PER_STRIP))] = (
                p.transform(v[ok]).astype(np.float32))
    return out


def cache_file(slot: str) -> Path:
    safe = str(slot).replace(":", "").replace("+0000", "Z")
    return CACHE_DIR / f"{safe}.npz"


def save_cache(slot: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = str(slot).replace(":", "").replace("+0000", "Z")
    tmp = CACHE_DIR / (safe + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, cache_file(slot))


def load_cache(slot: str, low: float, high: float):
    """Cached (tokens, lon, lat, ondisk, valid_patch, geom) or None on miss."""
    p = cache_file(slot)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
        if (int(z["cache_version"]) != CACHE_VERSION
                or str(z["model_sha"]) != MODEL_SHA256
                or not np.isclose(float(z["low"]), low)
                or not np.isclose(float(z["high"]), high)):
            return None
        return z
    except Exception:
        return None


def poison_check(base: np.ndarray, mod: np.ndarray, pre: np.ndarray,
                 post: np.ndarray, cols: list) -> dict:
    """Exact pre-cutoff invariance (NaN-aware) plus positive-control movement."""
    pre_inv = all(np.array_equal(np.asarray(base[:, i])[pre], np.asarray(mod[:, i])[pre],
                                 equal_nan=True) for i in range(len(cols)))
    moved = [c for i, c in enumerate(cols)
             if not np.array_equal(np.asarray(base[:, i])[post],
                                   np.asarray(mod[:, i])[post], equal_nan=True)]
    return {"pre_invariant": bool(pre_inv), "moved_columns": moved,
            "n_pre_rows": int(np.asarray(pre).sum()),
            "n_post_rows": int(np.asarray(post).sum()),
            "passed": bool(pre_inv and len(moved) > 0 and bool(np.asarray(post).any()))}


def default_json(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    try:
        import pandas as pd

        if isinstance(o, pd.Timestamp):
            return o.isoformat()
    except ImportError:
        pass
    raise TypeError(repr(o))


def geometry_record() -> dict:
    return {"canvas": CANVAS, "radius_frac": RADIUS_FRAC, "patch": PATCH,
            "grid": GRID_N, "n_prefix_tokens": N_PREFIX, "dim": DIM,
            "strip_half_width_deg": STRIP_HALF, "patch_valid_min": VALID_MIN,
            "speeds": list(SPEEDS), "lambda": "OMEGA*(TAU[v]-(h+age_h)) west-positive (D1 strips)",
            "encoder": "frozen, no fine-tuning", "pca_per_strip": PCA_PER_STRIP,
            "pca_total": N_STRIPS * PCA_PER_STRIP}
