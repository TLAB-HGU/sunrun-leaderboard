"""Chronological, self-supervised full-field HUXt encoder for TD-WSAHUXT72."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FOLDS = pd.to_datetime([
    "2025-11-15", "2025-12-15", "2026-02-01", "2026-03-15", "2026-04-15", "2026-05-29",
], utc=True)
OFFICIAL_END = pd.Timestamp("2026-09-26T23:00:00Z")
LATENT = 16
MIN_FIT_SECONDS = 240.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


class FieldEncoder(nn.Module):
    """Lead-conditioned convolutional bottleneck with a full-field decoder."""

    def __init__(self, nrad: int):
        super().__init__()
        self.lead = nn.Embedding(72, 8)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1)
        self.to_z = nn.Linear(32, LATENT)
        self.from_z = nn.Linear(LATENT + 8, 24 * int(np.ceil(nrad / 8)) * 16)
        self.decode1 = nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1)
        self.decode2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.decode3 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.output = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.nrad = nrad

    def encode(self, fields: torch.Tensor, leads: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(fields))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x)).mean(dim=(-2, -1))
        z = torch.tanh(self.to_z(torch.cat([x, self.lead(leads)], dim=1)))
        return z

    def forward(self, fields: torch.Tensor, leads: torch.Tensor):
        z = self.encode(fields, leads)
        x = torch.cat([z, self.lead(leads)], dim=1)
        x = self.from_z(x).view(-1, 24, int(np.ceil(self.nrad / 8)), 16)
        x = F.gelu(self.decode1(x))
        x = F.gelu(self.decode2(x))
        x = F.gelu(self.decode3(x))
        reconstruction = self.output(x)[:, :, : self.nrad, :128]
        return z, reconstruction


def _read_origins(handle: h5py.File, ids: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(handle["fields"][int(i)], dtype=np.float32) for i in ids])


def _normalization(handle: h5py.File, train_ids: np.ndarray, seed: int, fold: int) -> tuple[float, float]:
    if len(train_ids) < 4:
        raise ValueError("causal encoder fold has fewer than four prior HUXt origins")
    rng = np.random.default_rng(seed * 1009 + fold)
    sample_n = min(96, len(train_ids))
    selected = np.sort(rng.choice(train_ids, size=sample_n, replace=False))
    total = 0.0
    squared = 0.0
    count = 0
    for index in selected:
        values = np.asarray(handle["fields"][int(index)], dtype=np.float64)
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        count += values.size
    mean = total / count
    variance = max(squared / count - mean * mean, 1e-8)
    return mean, float(np.sqrt(variance))


def _train_checkpoint(model: FieldEncoder, handle: h5py.File, origins: np.ndarray,
                      train_ids: np.ndarray, val_ids: np.ndarray, seed: int,
                      fold: int, device: torch.device, mean: float, std: float,
                      curve_path: Path, all_curves: list[dict]) -> tuple[FieldEncoder, dict]:
    if len(train_ids) < 4 or len(val_ids) < 4:
        raise ValueError(f"encoder fold {fold} lacks time-split train/validation origins")
    rng = np.random.default_rng(seed * 65537 + fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    leads = torch.arange(72, device=device).repeat(2)
    train_batch_origins = 2
    train_per_epoch = min(64, max(train_batch_origins, len(train_ids)))
    val_sample = np.sort(rng.choice(val_ids, size=min(12, len(val_ids)), replace=False))
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    epoch_rows = []
    started = time.monotonic()
    epoch = 0
    while epoch < 1000:
        epoch += 1
        model.train()
        order = rng.choice(train_ids, size=train_per_epoch, replace=len(train_ids) < train_per_epoch)
        train_losses = []
        for offset in range(0, len(order), train_batch_origins):
            batch_ids = order[offset:offset + train_batch_origins]
            if len(batch_ids) == 1:
                batch_ids = np.repeat(batch_ids, 2)
            values = _read_origins(handle, batch_ids)
            x = torch.as_tensor(values.reshape(-1, 1, values.shape[-2], values.shape[-1]), device=device)
            x = (x - mean) / std
            lead_ids = leads if len(batch_ids) == 2 else torch.arange(72, device=device)
            optimizer.zero_grad(set_to_none=True)
            _, reconstructed = model(x, lead_ids)
            loss = F.smooth_l1_loss(reconstructed, x, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for offset in range(0, len(val_sample), 2):
                batch_ids = val_sample[offset:offset + 2]
                if len(batch_ids) == 1:
                    batch_ids = np.repeat(batch_ids, 2)
                values = _read_origins(handle, batch_ids)
                x = torch.as_tensor(values.reshape(-1, 1, values.shape[-2], values.shape[-1]), device=device)
                x = (x - mean) / std
                _, reconstructed = model(x, leads)
                val_losses.append(float(F.smooth_l1_loss(reconstructed, x, beta=1.0).cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        row = {"fold": fold, "epoch": epoch, "train_loss": train_loss,
               "valid_loss": val_loss, "elapsed_seconds": time.monotonic() - started}
        epoch_rows.append(row)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 4 == 0 or epoch == 1:
            all_curves.append(row)
            dump(curve_path, all_curves)
        # Each seed accumulates >30 minutes across its chronological checkpoints;
        # per-checkpoint validation must also converge before moving to the next cutoff.
        if time.monotonic() - started >= MIN_FIT_SECONDS and stale >= 12:
            break
    if best_state is None:
        raise RuntimeError(f"encoder fold {fold} did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    duration = time.monotonic() - started
    checkpoint = {"fold": fold, "train_origins": int(len(train_ids)), "validation_origins": int(len(val_ids)),
                  "normalization_mean_kms": mean, "normalization_std_kms": std,
                  "epochs": epoch, "best_epoch": best_epoch, "best_valid_loss": best_loss,
                  "seconds": duration, "stopped_on": "time-split validation convergence after 240s minimum"}
    return model, checkpoint


def _encode_interval(model: FieldEncoder, handle: h5py.File, origins: np.ndarray,
                     z: np.ndarray, indices: np.ndarray, mean: float, std: float,
                     device: torch.device) -> None:
    model.eval()
    lead_ids = torch.arange(72, device=device).repeat(2)
    with torch.no_grad():
        for offset in range(0, len(indices), 2):
            batch_indices = indices[offset:offset + 2]
            batch_ids = batch_indices if len(batch_indices) == 2 else np.repeat(batch_indices, 2)
            values = _read_origins(handle, batch_ids)
            x = torch.as_tensor(values.reshape(-1, 1, values.shape[-2], values.shape[-1]), device=device)
            x = (x - mean) / std
            encoded = model.encode(x, lead_ids).reshape(2, 72, LATENT).cpu().numpy()
            z[batch_indices] = encoded[:len(batch_indices)].astype(np.float16)


def _encode_poison_fields(bundle: np.load, prefix: str, encoder: FieldEncoder,
                          mean: float, std: float, device: torch.device) -> dict[str, np.ndarray]:
    encoder.eval()
    leads = torch.arange(72, device=device)
    encoded = {}
    kinds = ("base", "future", "past") if prefix == "dense" else ("base", "future")
    with torch.no_grad():
        for kind in kinds:
            values = np.asarray(bundle[f"{prefix}_{kind}"], dtype=np.float32)
            x = torch.as_tensor(values[:, None, :, :], device=device)
            encoded[kind] = encoder.encode((x - mean) / std, leads).cpu().numpy().astype(np.float32)
    return encoded


def run(seed: int, manifest_path: Path, out_dir: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("TD-WSAHUXT72 field encoder requires a GPU")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    manifest = json.loads(manifest_path.read_text())
    field_path = Path(manifest["field_cache"])
    if sha256(field_path) != manifest["lead_fields_sha256"]:
        raise ValueError("HUXt lead-field cache SHA does not match its manifest")
    origins = np.load(manifest["origin_index"], allow_pickle=False)
    origins_ts = pd.to_datetime(origins, utc=True)
    cache_key = hashlib.sha256(json.dumps({
        "field_cache_key": manifest["cache_key"], "encoder_sha256": sha256(Path(__file__)),
        "torch": torch.__version__, "latent": LATENT, "seed": seed,
        "min_fit_seconds_per_fold": MIN_FIT_SECONDS, "folds": [str(x) for x in FOLDS],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_dir = out_dir / "td-encoder-cache" / cache_key / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    z_path = cache_dir / "td_z.npz"
    cache_receipt = cache_dir / "receipt.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if z_path.is_file() and cache_receipt.is_file():
        receipt = json.loads(cache_receipt.read_text())
        if receipt.get("passed") and sha256(z_path) == receipt.get("z_sha256"):
            shutil.copyfile(z_path, out_dir / f"td_z_seed{seed}.npz")
            shutil.copyfile(cache_dir / f"curves_s{seed}.json", out_dir / f"curves_s{seed}.json")
            shutil.copyfile(cache_receipt, out_dir / f"td_encoder_receipt_s{seed}.json")
            return receipt

    cutoff_list = list(FOLDS) + [pd.Timestamp("2026-09-27T00:00:00Z")]
    z = np.full((len(origins), 72, LATENT), np.nan, dtype=np.float16)
    model = FieldEncoder(int(manifest["field_shape"][2])).to(device)
    curves = []
    checkpoint_rows = []
    seed_started = time.monotonic()
    with h5py.File(field_path, "r") as handle:
        if handle["fields"].shape[0] != len(origins):
            raise ValueError("HUXt field cache and origin index lengths disagree")
        for fold, cutoff in enumerate(FOLDS):
            eligible = np.flatnonzero(origins < cutoff.value)
            if fold == 0:
                if len(eligible) < 20:
                    raise ValueError("initial chronological encoder window has too few prior origins")
                split = max(4, int(len(eligible) * .8))
                train_ids, val_ids = eligible[:split], eligible[split:]
            else:
                previous = FOLDS[fold - 1].value
                train_ids = np.flatnonzero(origins < previous)
                val_ids = np.flatnonzero((origins >= previous) & (origins < cutoff.value))
            mean, std = _normalization(handle, train_ids, seed, fold)
            checkpoint_path = cache_dir / f"encoder_cutoff_{cutoff.strftime('%Y%m%d')}.pt"
            checkpoint_key = hashlib.sha256(f"{cache_key}|{cutoff.isoformat()}".encode()).hexdigest()
            reused = False
            if checkpoint_path.is_file():
                try:
                    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    if saved.get("checkpoint_key") == checkpoint_key:
                        model.load_state_dict(saved["state_dict"])
                        mean, std = float(saved["mean"]), float(saved["std"])
                        row = {**saved["receipt"], "reused": True}
                        fold_curve = cache_dir / f"curves_fold_{fold:02d}.json"
                        if fold_curve.is_file():
                            curves.extend(json.loads(fold_curve.read_text()))
                        reused = True
                except Exception:
                    reused = False
            if not reused:
                curve_start = len(curves)
                model, row = _train_checkpoint(model, handle, origins, train_ids, val_ids,
                                               seed, fold, device, mean, std,
                                               out_dir / f"curves_s{seed}.json", curves)
                dump(cache_dir / f"curves_fold_{fold:02d}.json", curves[curve_start:])
                torch.save({"checkpoint_key": checkpoint_key, "state_dict": model.state_dict(),
                            "mean": mean, "std": std, "receipt": row}, checkpoint_path)
            row = {**row, "cutoff_utc": str(cutoff), "reused": reused}
            checkpoint_rows.append(row)
            # This cutoff's encoder represents origins until the next training cutoff.
            end_ns = cutoff_list[fold + 1].value
            interval = np.flatnonzero((origins >= cutoff.value) & (origins < end_ns))
            _encode_interval(model, handle, origins, z, interval, mean, std, device)
            np.savez_compressed(cache_dir / "td_z_partial.npz", origins_ns=origins, z=z)
            dump(out_dir / f"curves_s{seed}.json", curves + [{"checkpoint": row}])
            print("TD-WSAHUXT72 encoder checkpoint " + json.dumps({
                "seed": seed, "cutoff": str(cutoff), "train": len(train_ids), "valid": len(val_ids),
                "z_origins": len(interval), "epochs": row["epochs"], "seconds": row["seconds"],
                "best_valid_loss": row["best_valid_loss"], "reused": reused,
            }), flush=True)
    poison_path = Path(manifest["source_poison_fields_path"])
    if sha256(poison_path) != manifest["source_poison_fields_sha256"]:
        raise ValueError("production HUXt source poison field bundle does not match its state receipt")
    poison_bundle = np.load(poison_path, allow_pickle=False)
    proof_latents = {}
    for prefix in ("official", "dense"):
        origin_ns = int(poison_bundle[f"{prefix}_origin_ns"])
        eligible_folds = np.flatnonzero(FOLDS.asi8 <= origin_ns)
        if not len(eligible_folds):
            raise ValueError(f"{prefix} source poison probe predates the causal encoder folds")
        fold = int(eligible_folds[-1])
        cutoff = FOLDS[fold]
        checkpoint_path = cache_dir / f"encoder_cutoff_{cutoff.strftime('%Y%m%d')}.pt"
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        proof_model = FieldEncoder(int(manifest["field_shape"][2])).to(device)
        proof_model.load_state_dict(saved["state_dict"])
        proof_latents[prefix] = _encode_poison_fields(
            poison_bundle, prefix, proof_model, float(saved["mean"]), float(saved["std"]), device
        )
        del proof_model
    official_future_delta = float(np.max(np.abs(
        proof_latents["official"]["base"] - proof_latents["official"]["future"]
    )))
    dense_future_delta = float(np.max(np.abs(
        proof_latents["dense"]["base"] - proof_latents["dense"]["future"]
    )))
    dense_past_delta = float(np.max(np.abs(
        proof_latents["dense"]["base"] - proof_latents["dense"]["past"]
    )))
    poison_encoder = {
        "passed": bool(official_future_delta == 0.0 and dense_future_delta == 0.0 and dense_past_delta > 0.0),
        "official_future_latent_delta": official_future_delta,
        "dense_future_latent_delta": dense_future_delta,
        "dense_past_admissible_latent_delta": dense_past_delta,
        "state_poison_receipt": str(poison_path.parent / "source_poison.json"),
    }
    if not poison_encoder["passed"]:
        raise RuntimeError("lead-conditioned encoder failed the production-path SW-012 poison controls")
    np.savez_compressed(
        z_path, origins_ns=origins, z=z,
        poison_official_base=proof_latents["official"]["base"],
        poison_official_future=proof_latents["official"]["future"],
        poison_dense_base=proof_latents["dense"]["base"],
        poison_dense_future=proof_latents["dense"]["future"],
        poison_dense_past=proof_latents["dense"]["past"],
        poison_official_origin_ns=np.asarray(int(poison_bundle["official_origin_ns"]), dtype=np.int64),
        poison_dense_origin_ns=np.asarray(int(poison_bundle["dense_origin_ns"]), dtype=np.int64),
    )
    receipt = {
        "passed": True, "seed": seed, "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "encoder": "lead-conditioned 2D convolutional full-field autoencoder; 16 latent values per origin/lead",
        "training_objective": "self-supervised smooth-L1 reconstruction of complete HUXt radial x longitude fields; no target labels",
        "causal_folds": checkpoint_rows, "seed_training_seconds": float(time.monotonic() - seed_started),
        "minimum_total_training_seconds": MIN_FIT_SECONDS * len(FOLDS),
        "actual_seed_training_seconds": float(sum(row["seconds"] for row in checkpoint_rows)),
        "z_shape": [int(len(origins)), 72, LATENT], "z_dtype": "float16",
        "finite_origin_fraction": float(np.isfinite(z).all(axis=(1, 2)).mean()),
        "all_hux_t_origins": int(len(origins)), "official_origins": 2833,
        "field_cache_key": manifest["cache_key"], "encoder_cache_key": cache_key,
        "z_sha256": sha256(z_path), "source_poison_receipt": manifest["source_poison_sha256"],
        "source_poison_encoder": poison_encoder,
    }
    if receipt["actual_seed_training_seconds"] < 900:
        raise RuntimeError("seed field-encoder training did not reach the 30-minute neural budget floor")
    dump(cache_receipt, receipt)
    shutil.copyfile(z_path, out_dir / f"td_z_seed{seed}.npz")
    shutil.copyfile(out_dir / f"curves_s{seed}.json", cache_dir / f"curves_s{seed}.json")
    shutil.copyfile(cache_receipt, out_dir / f"td_encoder_receipt_s{seed}.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    print(json.dumps(run(args.seed, args.manifest, args.out), sort_keys=True), flush=True)
