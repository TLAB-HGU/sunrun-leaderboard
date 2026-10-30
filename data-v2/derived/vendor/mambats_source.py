"""Causal SUVI/GONG frame cache and full-history temporal attention for S2.

The raw image stream is never reduced to selected lags, strips, thresholds, or
scalar morphology. A shared small spatial autoencoder is fitted without labels
on source frames available before the corresponding training boundary. Its
frame embeddings are cached once and reused by all supervised seeds. Each
origin receives a normalized learned-attention pool over every frame whose
documented availability time is at or before that origin.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
LB = HERE.parents[2]
STORE = LB / "store"
SOURCE_ROOT = STORE / "mambats-s2-source-cache"
SUVI_META = HERE / "fe171_meta.parquet"
GONG_MANIFEST = HERE / "gong_manifest.json"

IMAGE_H = 64
IMAGE_W = 128
LATENT_DIM = 32
PRETRAIN_EPOCHS = 8
PRETRAIN_PATIENCE = 2
PRETRAIN_BATCH = 128
SOURCE_WORKERS = 4
HOUR_NS = 3_600_000_000_000
SUVI_SOURCE = "GOES-18 SUVI Fe171"
GONG_SOURCE = "GONG mrzqs magnetic maps"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _source_code_hash() -> str:
    # The shared source encoder/cache has its own explicit identity. Runner
    # changes that do not alter image decoding or encoder semantics do not
    # invalidate the expensive common frame cache.
    return _sha256_file(Path(__file__))


def _utc_ns(value) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.value)


def _iso_utc(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def _load_records(official_end: pd.Timestamp):
    if not SUVI_META.is_file() or not GONG_MANIFEST.is_file():
        raise FileNotFoundError("S2 source metadata or GONG receipt manifest is missing")
    suvi_meta_hash = _sha256_file(SUVI_META)
    gong_meta_hash = _sha256_file(GONG_MANIFEST)
    meta_identity = _json_hash({"suvi_meta_sha256": suvi_meta_hash, "gong_manifest_sha256": gong_meta_hash})

    suvi = pd.read_parquet(SUVI_META)
    suvi["obs_end"] = pd.to_datetime(suvi["obs_end"], utc=True, errors="coerce")
    suvi["s3_modified"] = pd.to_datetime(suvi["s3_modified"], utc=True, errors="coerce")
    # Parquet stores some datetime columns at microsecond resolution. Use
    # Timestamp.value explicitly so every source timestamp is normalized to
    # nanoseconds before it is compared with origin_ns.
    nat = np.iinfo(np.int64).min
    obs_ns = np.asarray([
        pd.Timestamp(value).value if pd.notna(value) else nat
        for value in suvi["obs_end"]
    ], dtype=np.int64)
    lm_ns = np.asarray([
        pd.Timestamp(value).value if pd.notna(value) else nat
        for value in suvi["s3_modified"]
    ], dtype=np.int64)
    suvi["available_at_ns"] = np.maximum(obs_ns + HOUR_NS, lm_ns)
    end_ns = _utc_ns(official_end)
    suvi = suvi.loc[
        (suvi["status"] == "ok")
        & suvi["nc_path"].notna()
        & (suvi["available_at_ns"] <= end_ns)
    ].copy()
    suvi.sort_values(["available_at_ns", "obs_end", "slot"], inplace=True)
    suvi.drop_duplicates("nc_path", keep="first", inplace=True)

    suvi_records = []
    suvi_missing = 0
    for row in suvi.itertuples(index=False):
        path = Path(str(row.nc_path))
        expected = int(row.nc_size) if pd.notna(row.nc_size) else -1
        if not path.is_file() or path.stat().st_size != expected or expected <= 0:
            suvi_missing += 1
            continue
        suvi_records.append({
            "source": "suvi",
            "path": str(path),
            "bytes": expected,
            "slot_utc": _iso_utc(row.slot),
            "obs_start_utc": _iso_utc(row.obs_start),
            "obs_end_utc": _iso_utc(row.obs_end),
            "obs_ns": int(pd.Timestamp(row.obs_end).value),
            "available_ns": int(row.available_at_ns),
            "s3_modified_utc": pd.Timestamp(row.s3_modified).isoformat(),
            "crpix1": float(row.crpix1),
            "crpix2": float(row.crpix2),
        })

    gong_data = json.loads(GONG_MANIFEST.read_text())
    gong_records = []
    gong_missing = 0
    for row in gong_data["maps"]:
        obs = _utc_ns(row["observation_time_utc"])
        last_modified = _utc_ns(row["last_modified"])
        available = max(obs + 2 * HOUR_NS, last_modified)
        if int(pd.Timestamp(row["available_at_utc"]).value) != available:
            raise ValueError(f"GONG availability receipt mismatch: {row['file']}")
        if available > end_ns:
            continue
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            gong_missing += 1
            continue
        gong_records.append({
            "source": "gong",
            "path": str(path),
            "file": row["file"],
            "bytes": int(row["bytes"]),
            "source_sha256": row["sha256"],
            "obs_ns": obs,
            "available_ns": available,
            "observation_time_utc": row["observation_time_utc"],
            "available_at_utc": pd.Timestamp(available, tz="UTC").isoformat(),
            "last_modified": row["last_modified"],
            "url": row.get("url", ""),
        })

    suvi_records.sort(key=lambda r: (r["available_ns"], r["obs_ns"], r["path"]))
    gong_records.sort(key=lambda r: (r["available_ns"], r["obs_ns"], r["path"]))
    counts = {
        "suvi_metadata_rows": int(len(suvi)),
        "suvi_files_missing_or_size_mismatch": int(suvi_missing),
        "gong_manifest_rows": int(len(gong_data["maps"])),
        "gong_files_missing_or_size_mismatch": int(gong_missing),
    }
    return suvi_records, gong_records, meta_identity, counts


def _resize_suvi(rad, good):
    """Area-resize the complete square SUVI detector without cropping its disk."""
    good = np.asarray(good, dtype=bool) & np.isfinite(rad)
    if good.mean() < 0.5:
        raise ValueError(f"SUVI DQF-valid fraction too low: {good.mean():.4f}")
    fill = float(np.median(rad[good]))
    phys = np.where(good, rad, fill).astype(np.float32)
    phys = np.log10(1.0 + np.maximum(phys, 0.0))
    if phys.ndim != 2 or phys.shape[0] != phys.shape[1] or phys.shape[0] % IMAGE_H:
        raise ValueError(f"SUVI detector must be square and evenly area-resizable: {phys.shape}")
    side_scale = phys.shape[0] // IMAGE_H
    # Standard area downsampling covers the entire detector. In particular,
    # no center crop removes the solar limb or other source-side context.
    coarse = phys.reshape(IMAGE_H, side_scale, IMAGE_H, side_scale).mean((1, 3))
    coarse_mask = good.astype(np.float32).reshape(
        IMAGE_H, side_scale, IMAGE_H, side_scale,
    ).mean((1, 3))
    frame = np.zeros((2, IMAGE_H, IMAGE_W), dtype=np.float16)
    frame[0, :, IMAGE_W // 2 - IMAGE_H // 2:IMAGE_W // 2 + IMAGE_H // 2] = coarse.astype(np.float16)
    frame[1, :, IMAGE_W // 2 - IMAGE_H // 2:IMAGE_W // 2 + IMAGE_H // 2] = coarse_mask.astype(np.float16)
    return frame


def _decode_suvi(record):
    path = Path(record["path"])
    raw_bytes = path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if len(raw_bytes) != record["bytes"]:
        raise ValueError(f"SUVI local size changed: {path.name}")
    import h5py

    with h5py.File(io.BytesIO(raw_bytes), "r") as frame:
        raw = frame["RAD"][:].astype(np.float32)
        dqf = frame["DQF"][:]
        scale = float(np.asarray(frame["RAD"].attrs["scale_factor"]).reshape(-1)[0])
        offset = float(np.asarray(frame["RAD"].attrs["add_offset"]).reshape(-1)[0])
    radiance = raw * scale + offset
    image = _resize_suvi(radiance, dqf == 0)
    return digest, image, None


def _fits_header_and_image(blob: bytes):
    decompressed = gzip.decompress(blob)
    cards = []
    pos = 0
    found_end = False
    while pos + 2880 <= len(decompressed):
        block = decompressed[pos:pos + 2880]
        pos += 2880
        for start in range(0, 2880, 80):
            card = block[start:start + 80].decode("ascii", "replace")
            cards.append(card)
            if card[:8].strip() == "END":
                found_end = True
                break
        if found_end:
            break
    if not found_end:
        raise ValueError("GONG FITS has no END card")
    header = {}
    for card in cards:
        key = card[:8].strip()
        if not key or key == "END" or card[8:10] != "= ":
            continue
        value = card[10:80].split("/", 1)[0].strip()
        if value.startswith("'"):
            value = value[1:value.find("'", 1)] if "'" in value[1:] else value[1:].strip()
        elif value in {"T", "F"}:
            value = value == "T"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value.replace("D", "E"))
                except ValueError:
                    pass
        header[key] = value
    if int(header.get("NAXIS", 0)) != 2:
        raise ValueError(f"GONG FITS primary data is not 2-D: NAXIS={header.get('NAXIS')}")
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    bitpix = int(header["BITPIX"])
    dtype_by_bitpix = {
        8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8",
        -32: ">f4", -64: ">f8",
    }
    if bitpix not in dtype_by_bitpix:
        raise ValueError(f"unsupported GONG FITS BITPIX={bitpix}")
    expected = width * height * abs(bitpix) // 8
    if pos + expected > len(decompressed):
        raise ValueError("truncated GONG FITS primary data")
    image = np.frombuffer(decompressed, dtype=np.dtype(dtype_by_bitpix[bitpix]), count=width * height, offset=pos)
    image = image.reshape(height, width).astype(np.float32)
    if bitpix > 0 and "BLANK" in header:
        image[image == float(header["BLANK"])] = np.nan
    image = image * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    return image


def _decode_gong(record):
    from scipy.ndimage import zoom

    path = Path(record["path"])
    raw_bytes = path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if digest != record["source_sha256"]:
        raise ValueError(f"GONG source SHA256 mismatch: {path.name}")
    if len(raw_bytes) != record["bytes"]:
        raise ValueError(f"GONG local size changed: {path.name}")
    image = _fits_header_and_image(raw_bytes)
    valid = np.isfinite(image)
    if valid.mean() < 0.5:
        raise ValueError(f"GONG finite-pixel fraction too low: {path.name}")
    # Bilinear resampling preserves the complete longitude/latitude map. It
    # does not threshold, select strips, or form hand-designed summaries.
    sy, sx = IMAGE_H / image.shape[0], IMAGE_W / image.shape[1]
    values = zoom(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), (sy, sx), order=1, mode="nearest", prefilter=False)
    mask = zoom(valid.astype(np.float32), (sy, sx), order=1, mode="nearest", prefilter=False)
    frame = np.stack([values, mask], axis=0).astype(np.float16)
    if frame.shape != (2, IMAGE_H, IMAGE_W):
        raise ValueError(f"unexpected GONG resized map shape: {frame.shape}")
    return digest, frame, None


def _decode_job(job):
    index, record = job
    try:
        result = _decode_suvi(record) if record["source"] == "suvi" else _decode_gong(record)
        return index, result[0], result[1], None
    except Exception as exc:
        return index, None, None, f"{type(exc).__name__}: {exc}"


class SpatialAutoencoder(nn.Module):
    """Small shared encoder; decoder is used only in unlabeled pretraining."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, LATENT_DIM, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.project = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.decoder_seed = nn.Linear(LATENT_DIM, LATENT_DIM * 8 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 24, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1),
        )

    def encode(self, image):
        return self.project(self.encoder(image))

    def forward(self, image):
        z = self.encode(image)
        decoded = self.decoder(self.decoder_seed(z).reshape(-1, LATENT_DIM, 8, 16))
        return decoded, z


class _MemmapFrames(Dataset):
    def __init__(self, source_arrays, indices, mean, scale):
        self.source_arrays = source_arrays
        self.indices = indices
        self.mean = float(mean)
        self.scale = float(scale)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        source_index = int(self.indices[local_index])
        image = np.asarray(self.source_arrays[source_index], dtype=np.float32).copy()
        mask = np.clip(image[1], 0.0, 1.0)
        image[0] = (image[0] - self.mean) / self.scale
        image[0, mask <= 0] = 0.0
        image[1] = mask
        return torch.from_numpy(image)


def _pixel_stats(images, records, end_ns):
    n = len(records)
    valid_indices = np.asarray([i for i, row in enumerate(records) if row["available_ns"] < end_ns], dtype=np.int64)
    if len(valid_indices) == 0:
        raise ValueError("no source frames are available before the requested encoder boundary")
    total = 0.0
    total_sq = 0.0
    total_weight = 0.0
    for start in range(0, len(valid_indices), 128):
        batch = np.asarray(images[valid_indices[start:start + 128]], dtype=np.float32)
        vals = batch[:, 0]
        weights = np.clip(batch[:, 1], 0.0, 1.0)
        total += float((vals * weights).sum(dtype=np.float64))
        total_sq += float((vals.astype(np.float64) ** 2 * weights).sum(dtype=np.float64))
        total_weight += float(weights.sum(dtype=np.float64))
    mean = total / max(total_weight, 1.0)
    var = max(total_sq / max(total_weight, 1.0) - mean * mean, 1e-8)
    return float(mean), float(math.sqrt(var)), valid_indices


def _train_spatial_encoder(
    suvi_images, suvi_records, gong_images, gong_records, train_boundary_ns,
    out_path: Path, device: torch.device, *, seed: int,
):
    """Unsupervised, chronological pretraining using only source frames before a boundary."""
    suvi_mean, suvi_scale, suvi_train = _pixel_stats(suvi_images, suvi_records, train_boundary_ns)
    gong_mean, gong_scale, gong_train = _pixel_stats(gong_images, gong_records, train_boundary_ns)

    def make_indices(images, records, source_indices):
        validation_start = int(np.quantile([records[i]["available_ns"] for i in source_indices], 0.9))
        train_idx = np.asarray([i for i in source_indices if records[i]["available_ns"] < validation_start], dtype=np.int64)
        valid_idx = np.asarray([i for i in source_indices if records[i]["available_ns"] >= validation_start], dtype=np.int64)
        return train_idx, valid_idx

    st, sv = make_indices(suvi_images, suvi_records, suvi_train)
    gt, gv = make_indices(gong_images, gong_records, gong_train)
    if min(len(st), len(gt), len(sv), len(gv)) == 0:
        raise ValueError("source encoder chronological training/validation split is empty")
    train_dataset = torch.utils.data.ConcatDataset([
        _MemmapFrames(suvi_images, st, suvi_mean, suvi_scale),
        _MemmapFrames(gong_images, gt, gong_mean, gong_scale),
    ])
    valid_dataset = torch.utils.data.ConcatDataset([
        _MemmapFrames(suvi_images, sv, suvi_mean, suvi_scale),
        _MemmapFrames(gong_images, gv, gong_mean, gong_scale),
    ])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=PRETRAIN_BATCH, shuffle=True, generator=generator,
        num_workers=3, pin_memory=True, persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=PRETRAIN_BATCH, shuffle=False,
        num_workers=3, pin_memory=True, persistent_workers=True,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SpatialAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = []
    best_state = None
    best_valid = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        start = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = batch.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            mask = batch[:, 1:2].clamp(0, 1)
            image_loss = (((recon[:, :1] - batch[:, :1]) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
            mask_loss = torch.mean((recon[:, 1:2] - mask) ** 2)
            loss = image_loss + 0.1 * mask_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite shared source encoder pretraining loss")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().item())
            train_batches += 1

        model.eval()
        valid_loss_sum = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device=device, non_blocking=True)
                recon, _ = model(batch)
                mask = batch[:, 1:2].clamp(0, 1)
                image_loss = (((recon[:, :1] - batch[:, :1]) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
                mask_loss = torch.mean((recon[:, 1:2] - mask) ** 2)
                valid_loss_sum += float((image_loss + 0.1 * mask_loss).item())
                valid_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)
        valid_loss = valid_loss_sum / max(valid_batches, 1)
        history.append({
            "epoch": epoch,
            "train_reconstruction_loss": train_loss,
            "valid_reconstruction_loss": valid_loss,
            "seconds": time.monotonic() - start,
        })
        print(f"source_encoder epoch={epoch} train={train_loss:.6f} valid={valid_loss:.6f}", flush=True)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PRETRAIN_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("source encoder pretraining did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, out_path)
    model.cpu()
    del model, train_loader, valid_loader
    return {
        "seed": seed,
        "train_boundary_utc": pd.Timestamp(train_boundary_ns, tz="UTC").isoformat(),
        "selected_epoch": best_epoch,
        "best_validation_reconstruction_loss": best_valid,
        "train_curve": history,
        "suvi_pixel_mean": suvi_mean,
        "suvi_pixel_scale": suvi_scale,
        "gong_pixel_mean": gong_mean,
        "gong_pixel_scale": gong_scale,
        "train_frames": {"suvi": int(len(st)), "gong": int(len(gt))},
        "validation_frames": {"suvi": int(len(sv)), "gong": int(len(gv))},
        "normalization_fit": "pixel-weighted mean/std using valid pixels from source frames available before this encoder boundary",
    }


def _embed_all(images, records, stats, checkpoint, device, output_path: Path):
    model = SpatialAutoencoder()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)
    mean, scale = float(stats["mean"]), float(stats["scale"])
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(records), LATENT_DIM))
    with torch.no_grad():
        for start in range(0, len(records), 256):
            end = min(len(records), start + 256)
            batch = np.asarray(images[start:end], dtype=np.float32)
            mask = np.clip(batch[:, 1], 0, 1)
            batch[:, 0] = (batch[:, 0] - mean) / scale
            batch[:, 0][mask <= 0] = 0.0
            encoded = model.encode(torch.from_numpy(batch).to(device=device)).float().cpu().numpy()
            if not np.isfinite(encoded).all():
                raise FloatingPointError("non-finite cached spatial frame embedding")
            out[start:end] = encoded
    out.flush()
    del model


def _time_stats(records, boundary_ns):
    train_obs = np.asarray([r["obs_ns"] for r in records if r["available_ns"] < boundary_ns], dtype=np.float64)
    if len(train_obs) == 0:
        raise ValueError("no source observation timestamps available before encoder boundary")
    mean = float(train_obs.mean())
    scale = float(train_obs.std())
    return mean, max(scale, float(HOUR_NS))


def _artifact_hashes(folder: Path):
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(folder.iterdir()) if path.is_file() and path.name not in {"cache_manifest.json"}
    }


def _encode_records(records):
    return [dict(row) for row in records]


def _build_cache(
    suvi_records, gong_records, meta_identity, meta_counts, val_start_ns,
    cutoff_ns, official_end_ns, device: torch.device, output_dir: Path,
):
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "image_size": [IMAGE_H, IMAGE_W],
        "latent_dim": LATENT_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretrain_patience": PRETRAIN_PATIENCE,
        "pretrain_batch": PRETRAIN_BATCH,
        "availability": {
            "suvi": "max(S3 LastModified, observation_end + 1 h); local bytes must equal the recorded S3 object size",
            "gong": "max(issue/last-contributor time + 2 h, HTTP Last-Modified), from the hash-pinned GONG receipt",
        },
        "boundaries": {"validation_start_ns": int(val_start_ns), "cutoff_ns": int(cutoff_ns), "official_end_ns": int(official_end_ns)},
    }
    code_hash = _source_code_hash()
    index_key = _json_hash({"metadata": meta_identity, "code": code_hash, "config": config})
    index_path = SOURCE_ROOT / f"index-{index_key}.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        cache_dir = SOURCE_ROOT / index["cache_key"]
        manifest_path = cache_dir / "cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("S2 source cache index points to a missing cache manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("metadata_identity") != meta_identity or manifest.get("source_code_sha256") != code_hash:
            raise AssertionError("S2 source cache identity differs from the registered source metadata/code")
        for name, evidence in manifest["artifacts"].items():
            path = cache_dir / name
            if not path.is_file() or path.stat().st_size != evidence["bytes"] or _sha256_file(path) != evidence["sha256"]:
                raise AssertionError(f"S2 source cache artifact hash mismatch: {name}")
        print(f"source cache reused key={index['cache_key']}", flush=True)
        return _load_bundle(cache_dir, manifest, output_dir)

    stage = SOURCE_ROOT / f".build-{index_key}-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    suvi_images_path = stage / "suvi_images.npy"
    gong_images_path = stage / "gong_images.npy"
    suvi_images = np.lib.format.open_memmap(
        suvi_images_path, mode="w+", dtype=np.float16,
        shape=(len(suvi_records), 2, IMAGE_H, IMAGE_W),
    )
    gong_images = np.lib.format.open_memmap(
        gong_images_path, mode="w+", dtype=np.float16,
        shape=(len(gong_records), 2, IMAGE_H, IMAGE_W),
    )
    decoded = {"suvi": 0, "gong": 0}
    errors = {"suvi": [], "gong": []}
    local_digests = {"suvi": [None] * len(suvi_records), "gong": [None] * len(gong_records)}

    for source_name, records, images in (
        ("suvi", suvi_records, suvi_images), ("gong", gong_records, gong_images),
    ):
        jobs = [(i, row) for i, row in enumerate(records)]
        context = mp.get_context("fork")
        with context.Pool(SOURCE_WORKERS) as pool:
            for n_done, result in enumerate(pool.imap_unordered(_decode_job, jobs, chunksize=8), 1):
                index, digest, frame, error = result
                if error is not None:
                    errors[source_name].append({"file": Path(records[index]["path"]).name, "error": error})
                    images[index] = 0
                else:
                    images[index] = frame
                    local_digests[source_name][index] = digest
                    decoded[source_name] += 1
                if n_done % 2000 == 0:
                    print(f"source decode {source_name} {n_done}/{len(records)} ok={decoded[source_name]}", flush=True)
        images.flush()

    suvi_valid = np.asarray([i for i, d in enumerate(local_digests["suvi"]) if d], dtype=np.int64)
    gong_valid = np.asarray([i for i, d in enumerate(local_digests["gong"]) if d], dtype=np.int64)
    if min(len(suvi_valid), len(gong_valid)) == 0:
        raise ValueError(f"source decode has zero valid frames: {decoded}")
    suvi_records = [suvi_records[i] | {"local_sha256": local_digests["suvi"][i]} for i in suvi_valid]
    gong_records = [gong_records[i] | {"local_sha256": local_digests["gong"][i]} for i in gong_valid]
    suvi_pixels_map = np.lib.format.open_memmap(
        stage / "suvi_pixels.npy", mode="w+", dtype=np.float16,
        shape=(len(suvi_valid), 2, IMAGE_H, IMAGE_W),
    )
    gong_pixels_map = np.lib.format.open_memmap(
        stage / "gong_pixels.npy", mode="w+", dtype=np.float16,
        shape=(len(gong_valid), 2, IMAGE_H, IMAGE_W),
    )
    for start in range(0, len(suvi_valid), 128):
        suvi_pixels_map[start:start + 128] = suvi_images[suvi_valid[start:start + 128]]
    for start in range(0, len(gong_valid), 128):
        gong_pixels_map[start:start + 128] = gong_images[gong_valid[start:start + 128]]
    suvi_pixels_map.flush()
    gong_pixels_map.flush()
    del suvi_images, gong_images, suvi_valid, gong_valid, suvi_pixels_map, gong_pixels_map
    del local_digests

    suvi_pixels = np.load(stage / "suvi_pixels.npy", mmap_mode="r")
    gong_pixels = np.load(stage / "gong_pixels.npy", mmap_mode="r")

    torch.set_num_threads(1)
    split_checkpoint = stage / "encoder_split.pt"
    final_checkpoint = stage / "encoder_final.pt"
    split_summary = _train_spatial_encoder(
        suvi_pixels, suvi_records, gong_pixels, gong_records,
        val_start_ns, split_checkpoint, device, seed=721,
    )
    final_summary = _train_spatial_encoder(
        suvi_pixels, suvi_records, gong_pixels, gong_records,
        cutoff_ns, final_checkpoint, device, seed=722,
    )
    split_time = {
        "suvi": _time_stats(suvi_records, val_start_ns),
        "gong": _time_stats(gong_records, val_start_ns),
    }
    final_time = {
        "suvi": _time_stats(suvi_records, cutoff_ns),
        "gong": _time_stats(gong_records, cutoff_ns),
    }
    for source_name, pixels, records in (
        ("suvi", suvi_pixels, suvi_records), ("gong", gong_pixels, gong_records),
    ):
        _embed_all(
            pixels, records,
            {"mean": split_summary[f"{source_name}_pixel_mean"], "scale": split_summary[f"{source_name}_pixel_scale"]},
            split_checkpoint, device, stage / f"{source_name}_embedding_split.npy",
        )
        _embed_all(
            pixels, records,
            {"mean": final_summary[f"{source_name}_pixel_mean"], "scale": final_summary[f"{source_name}_pixel_scale"]},
            final_checkpoint, device, stage / f"{source_name}_embedding_final.npy",
        )

    suvi_pixels_file = stage / "suvi_pixels.npy"
    gong_pixels_file = stage / "gong_pixels.npy"
    suvi_records_file = _encode_records(suvi_records)
    gong_records_file = _encode_records(gong_records)
    source_manifest = {
        "metadata_identity": meta_identity,
        "suvi_metadata_sha256": _sha256_file(SUVI_META),
        "gong_receipt_manifest_sha256": _sha256_file(GONG_MANIFEST),
        "source_code_sha256": code_hash,
        "availability_proxy_evidence": "S3 LastModified/observation end with local-vs-recorded object size equality for SUVI; issue+2h/HTTP Last-Modified and per-file SHA256 equality for GONG",
        "operational_proxy_evidence": True,
        "source_frame_counts": {"suvi": len(suvi_records), "gong": len(gong_records)},
        "decoded_frame_counts": decoded,
        "decode_error_counts": {k: len(v) for k, v in errors.items()},
        "decode_errors_sample": {k: v[:20] for k, v in errors.items()},
        "source_coverage": {
            "suvi_first_observation_utc": pd.Timestamp(min(r["obs_ns"] for r in suvi_records), tz="UTC").isoformat(),
            "suvi_last_observation_utc": pd.Timestamp(max(r["obs_ns"] for r in suvi_records), tz="UTC").isoformat(),
            "gong_first_observation_utc": pd.Timestamp(min(r["obs_ns"] for r in gong_records), tz="UTC").isoformat(),
            "gong_last_observation_utc": pd.Timestamp(max(r["obs_ns"] for r in gong_records), tz="UTC").isoformat(),
        },
        "metadata_counts": meta_counts,
        "split_encoder": split_summary,
        "final_encoder": final_summary,
        "split_time_normalization": {k: {"mean_ns": v[0], "scale_ns": v[1]} for k, v in split_time.items()},
        "final_time_normalization": {k: {"mean_ns": v[0], "scale_ns": v[1]} for k, v in final_time.items()},
        "records": {"suvi": suvi_records_file, "gong": gong_records_file},
    }
    local_content_hash = _json_hash({
        "suvi": [(r["path"], r["bytes"], r["local_sha256"]) for r in suvi_records],
        "gong": [(r["path"], r["bytes"], r["local_sha256"]) for r in gong_records],
    })
    cache_key = _json_hash({"metadata": meta_identity, "source_content": local_content_hash, "code": code_hash, "config": config})
    source_manifest["local_source_content_sha256"] = local_content_hash
    source_manifest["cache_key"] = cache_key
    source_manifest["cache_config"] = config
    (stage / "cache_manifest.json").write_text(json.dumps(source_manifest, sort_keys=True, indent=2) + "\n")

    # Raw spatial frames are the expensive staging input; retain only the
    # hash-pinned learned embeddings/checkpoints required by all seeds.
    del suvi_pixels, gong_pixels
    suvi_pixels_file.unlink()
    gong_pixels_file.unlink()
    shutil.rmtree(stage / "__pycache__", ignore_errors=True)
    artifacts = _artifact_hashes(stage)
    source_manifest["artifacts"] = artifacts
    (stage / "cache_manifest.json").write_text(json.dumps(source_manifest, sort_keys=True, indent=2) + "\n")
    final_dir = SOURCE_ROOT / cache_key
    if final_dir.exists():
        shutil.rmtree(stage)
    else:
        os.replace(stage, final_dir)
    index = {
        "cache_key": cache_key,
        "metadata_identity": meta_identity,
        "source_code_sha256": code_hash,
        "local_source_content_sha256": local_content_hash,
    }
    index_path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
    manifest = json.loads((final_dir / "cache_manifest.json").read_text())
    print(f"source cache complete key={cache_key} frames={decoded}", flush=True)
    return _load_bundle(final_dir, manifest, output_dir)


def _load_bundle(cache_dir: Path, manifest: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("encoder_split.pt", "encoder_final.pt"):
        shutil.copyfile(cache_dir / name, output_dir / name)
    for name, key in (("source_encoder_curves.json", "split_encoder"), ("source_encoder_final_curves.json", "final_encoder")):
        summary = manifest[key]
        (output_dir / name).write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    source_data = {}
    for source_name in ("suvi", "gong"):
        records = manifest["records"][source_name]
        for stage in ("split", "final"):
            source_data[f"{source_name}_embedding_{stage}"] = np.load(
                cache_dir / f"{source_name}_embedding_{stage}.npy", mmap_mode="r"
            )
        for stage in ("split_time_normalization", "final_time_normalization"):
            stats = manifest[stage][source_name]
            times = np.asarray([(r["obs_ns"] - stats["mean_ns"]) / stats["scale_ns"] for r in records], dtype=np.float32)
            source_data[f"{source_name}_times_{'split' if stage.startswith('split') else 'final'}"] = times
        source_data[f"{source_name}_available_ns"] = np.asarray([r["available_ns"] for r in records], dtype=np.int64)
        source_data[f"{source_name}_records"] = records
    source_data["manifest"] = manifest
    source_data["cache_dir"] = cache_dir
    return source_data


def load_source_bundle(train_validation_start, cutoff, official_end, device, output_dir):
    """Build/reuse hash-keyed spatial embeddings shared by all seed fits."""
    suvi_records, gong_records, meta_identity, counts = _load_records(pd.Timestamp(official_end))
    if not suvi_records or not gong_records:
        raise ValueError(f"causal source store is empty: SUVI={len(suvi_records)} GONG={len(gong_records)}")
    return _build_cache(
        suvi_records, gong_records, meta_identity, counts,
        _utc_ns(train_validation_start), _utc_ns(cutoff), _utc_ns(official_end),
        torch.device(device), Path(output_dir),
    )


class _TemporalPool(nn.Module):
    """Normalized content/time attention evaluated as a causal prefix scan."""

    def __init__(self, input_dim=LATENT_DIM + 1, output_dim=LATENT_DIM):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Tanh())
        self.score = nn.Linear(output_dim, 1)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings, time_position, prefix_count):
        features = torch.cat([embeddings, time_position[:, None]], dim=-1)
        score = torch.tanh(self.score(self.key(features))).squeeze(-1)
        # Bounded positive weights make prefix sums numerically stable without
        # a global softmax maximum that could depend on future frames.
        weight = torch.sigmoid(score)
        values = self.value(features)
        numerator = torch.cumsum(weight[:, None] * values, dim=0)
        denominator = torch.cumsum(weight, dim=0)
        index = (prefix_count - 1).clamp_min(0)
        pooled = numerator.index_select(0, index) / denominator.index_select(0, index).clamp_min(1e-8)[:, None]
        return pooled * (prefix_count > 0).to(pooled.dtype)[:, None]


class SourceTemporalAttention(nn.Module):
    """Separate temporal attention per stream over the shared spatial embeddings."""

    output_dim = 2 * LATENT_DIM

    def __init__(self):
        super().__init__()
        self.suvi_pool = _TemporalPool()
        self.gong_pool = _TemporalPool()

    def forward(self, source_context):
        (suvi_embedding, suvi_time, suvi_count,
         gong_embedding, gong_time, gong_count) = source_context
        suvi = self.suvi_pool(suvi_embedding, suvi_time, suvi_count)
        gong = self.gong_pool(gong_embedding, gong_time, gong_count)
        return torch.cat([suvi, gong], dim=-1)


def prepare_static_context(bundle, stage: str, device):
    """Move one seed-common frame bank to GPU once; later calls only gather counts."""
    if stage not in {"split", "final"}:
        raise ValueError(stage)
    context = []
    for source_name in ("suvi", "gong"):
        embeddings = np.asarray(bundle[f"{source_name}_embedding_{stage}"], dtype=np.float32)
        times = np.asarray(bundle[f"{source_name}_times_{stage}"], dtype=np.float32)
        context.extend([
            torch.from_numpy(embeddings.copy()).to(device=device),
            torch.from_numpy(times.copy()).to(device=device),
            bundle[f"{source_name}_available_ns"],
        ])
    return tuple(context)


def attach_seed_context(static_context, origin_ns, device):
    """Attach exact per-origin availability-prefix counts to the static banks."""
    result = []
    for embedding, times, available in (static_context[:3], static_context[3:]):
        counts = np.searchsorted(available, np.asarray(origin_ns, dtype=np.int64), side="right").astype(np.int64)
        result.extend([embedding, times, torch.from_numpy(counts).to(device=device)])
    return tuple(result)


def source_prefix_counts(bundle, stage: str, origins_ns):
    result = {}
    for source_name in ("suvi", "gong"):
        available = bundle[f"{source_name}_available_ns"]
        result[source_name] = np.searchsorted(available, np.asarray(origins_ns, dtype=np.int64), side="right").astype(np.int32)
    return result


def coverage_report(bundle, train_origins_ns, official_origins_ns):
    report = {}
    for label, origins in (("train", train_origins_ns), ("official", official_origins_ns)):
        per_source = {}
        for source_name in ("suvi", "gong"):
            available = bundle[f"{source_name}_available_ns"]
            counts = np.searchsorted(available, np.asarray(origins, dtype=np.int64), side="right")
            per_source[source_name] = {
                "origin_count": int(len(origins)),
                "covered_origins": int(np.count_nonzero(counts > 0)),
                "coverage_fraction": float(np.mean(counts > 0)),
                "frames_per_origin_min": int(counts.min()),
                "frames_per_origin_median": float(np.median(counts)),
                "frames_per_origin_max": int(counts.max()),
            }
        report[label] = per_source
    return report


def causal_poison_context(model, probe_context, device):
    """Future-frame invariance plus a valid past-frame positive control."""
    # The context is (embeddings, times, counts) for each stream. Future rows
    # are after their origin-specific prefix and must have no effect on either
    # the pooled branch representation or the final model output.
    model.eval()
    with torch.no_grad():
        original_representation = model.source_attention(probe_context)
        future_context = list(probe_context)
        for emb_index, count_index in ((0, 2), (3, 5)):
            changed = future_context[emb_index].clone()
            counts = future_context[count_index]
            for row, n in enumerate(counts.tolist()):
                if n < len(changed):
                    changed[n:] += 37.0
            future_context[emb_index] = changed
        future_representation = model.source_attention(tuple(future_context))
        future_invariant = torch.equal(original_representation, future_representation)

        positive_context = list(probe_context)
        changed_any = False
        for emb_index, count_index in ((0, 2), (3, 5)):
            changed = positive_context[emb_index].clone()
            counts = positive_context[count_index]
            for row, n in enumerate(counts.tolist()):
                if n > 0:
                    changed[n - 1] += 0.5
                    changed_any = True
                    break
            if changed_any:
                positive_context[emb_index] = changed
                break
        if not changed_any:
            raise AssertionError("source positive control has no admissible past frame")
        positive_representation = model.source_attention(tuple(positive_context))
        positive_changed = not torch.equal(original_representation, positive_representation)
        result = {
            "passed": bool(future_invariant and positive_changed),
            "future_frames_pool_invariant": bool(future_invariant),
            "admissible_past_positive_control_changed_pool": bool(positive_changed),
            "future_pool_max_abs_delta": float((original_representation - future_representation).abs().max().item()),
            "past_pool_max_abs_delta": float((original_representation - positive_representation).abs().max().item()),
        }
    if not result["passed"]:
        raise AssertionError(f"SUVI/GONG source sequence poison check failed: {result}")
    return result
