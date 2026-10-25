"""Per-lead model vs constant-baseline reconstruction loss."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("/home/t-lab01/sunrun/research/e1466-epoch-repro-20261026")
STATE = ROOT / "run/state/tdwsahuxt-state/td-state-f0ebb023d70b732f128b18980a41d3cd1ba0723d9d2b4315d3b22f6ee7befcc1"
CKPT = ROOT / "run/encoder/td-encoder-cache/9f12cf61e159a71d22e473b8ab645eb63bad5264d48fb3d1b744c49d05e8042b/seed_0"
sys.path.insert(0, str(ROOT / "code"))
from td_field_encoder import FieldEncoder  # noqa: E402

FOLDS = pd.to_datetime(
    ["2025-11-15", "2025-12-15", "2026-02-01", "2026-03-15", "2026-04-15", "2026-05-29"],
    utc=True,
).as_unit("ns")
CUTOFF_FILES = [
    "encoder_cutoff_20251115.pt", "encoder_cutoff_20251215.pt", "encoder_cutoff_20260201.pt",
    "encoder_cutoff_20260315.pt", "encoder_cutoff_20260415.pt", "encoder_cutoff_20260529.pt",
]


def normalization(handle, train_ids, seed=0, fold=0):
    rng = np.random.default_rng(seed * 1009 + fold)
    sample_n = min(96, len(train_ids))
    selected = np.sort(rng.choice(train_ids, size=sample_n, replace=False))
    total = squared = count = 0.0
    for index in selected:
        values = np.asarray(handle["fields"][int(index)], dtype=np.float64)
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        count += values.size
    mean = total / count
    variance = max(squared / count - mean * mean, 1e-8)
    return mean, float(np.sqrt(variance))


def smooth_l1(pred, target, beta=1.0):
    diff = np.abs(pred - target)
    return np.where(diff < beta, .5 * diff ** 2 / beta, diff - .5 * beta)


def read_batch(handle, ids):
    return np.stack([np.asarray(handle["fields"][int(i)], dtype=np.float32) for i in ids])


def model_losses(model, handle, ids, mean, std):
    lead_loss = np.zeros(72, dtype=np.float64)
    lead_base = np.zeros(72, dtype=np.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ids), 2):
            batch_ids = ids[offset:offset + 2]
            if len(batch_ids) == 1:
                batch_ids = np.repeat(batch_ids, 2)
            values = read_batch(handle, batch_ids)
            x = (torch.as_tensor(values).reshape(-1, 1, values.shape[-2], values.shape[-1]) - mean) / std
            leads = torch.arange(72).repeat(2)
            _, reconstructed = model(x, leads)
            loss = smooth_l1(reconstructed.numpy(), x.numpy())  # [B*72,1,130,128]
            loss = loss.reshape(len(batch_ids), 72, -1).mean(axis=2)
            base = smooth_l1(np.zeros_like(x.numpy()), x.numpy()).reshape(len(batch_ids), 72, -1).mean(axis=2)
            lead_loss += loss[:len(batch_ids)].sum(axis=0)
            lead_base += base[:len(batch_ids)].sum(axis=0)
            count += len(batch_ids)
    return lead_loss / count, lead_base / count


def main() -> None:
    origins = np.load(STATE / "origins_ns.npy")
    rows = []
    with h5py.File(STATE / "lead_fields.h5", "r") as handle:
        for fold, cutoff in enumerate(FOLDS):
            if fold == 0:
                eligible = np.flatnonzero(origins < cutoff.value)
                split = max(4, int(len(eligible) * .8))
                train_ids, val_ids = eligible[:split], eligible[split:]
            else:
                previous = FOLDS[fold - 1].value
                train_ids = np.flatnonzero(origins < previous)
                val_ids = np.flatnonzero((origins >= previous) & (origins < cutoff.value))
            mean, std = normalization(handle, train_ids, seed=0, fold=fold)
            rng = np.random.default_rng(200 + fold)
            train_sample = np.sort(rng.choice(train_ids, size=min(32, len(train_ids)), replace=False))
            val_sample = np.sort(rng.choice(val_ids, size=min(32, len(val_ids)), replace=False))

            model = FieldEncoder(130)
            state = torch.load(CKPT / CUTOFF_FILES[fold], map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"])
            params = sum(p.numel() for p in model.parameters())
            train_loss, train_base = model_losses(model, handle, train_sample, mean, std)
            val_loss, val_base = model_losses(model, handle, val_sample, mean, std)
            rows.append({
                "fold": fold,
                "parameters": params,
                "train_model_mean": float(train_loss.mean()),
                "train_baseline_mean": float(train_base.mean()),
                "valid_model_mean": float(val_loss.mean()),
                "valid_baseline_mean": float(val_base.mean()),
                "train_loss_by_lead": train_loss.tolist(),
                "train_baseline_by_lead": train_base.tolist(),
                "valid_loss_by_lead": val_loss.tolist(),
                "valid_baseline_by_lead": val_base.tolist(),
            })
            print(f"fold {fold}: params={params:,} train model={train_loss.mean():.4f} base={train_base.mean():.4f} "
                  f"val model={val_loss.mean():.4f} base={val_base.mean():.4f}", flush=True)
    (ROOT / "model_lead_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("wrote", ROOT / "model_lead_diagnostics.json")


if __name__ == "__main__":
    main()
