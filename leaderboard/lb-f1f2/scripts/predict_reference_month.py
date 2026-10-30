#!/usr/bin/env python3
"""Replay the frozen E0472 model on a copied local v2 snapshot; no training/downloads."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_release import load_model
from model_release.export_availability import export_availability
from model_release.features import FeatureInputs
from model_release.strips import select_slots

MODEL_SHA256 = '2ae26f2c21a18fc6f59c330267b98e4414b64962380c94b41138ffebca2090d2'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-root', type=Path, required=True)
    parser.add_argument('--images-root', type=Path, required=True,
                        help='Root resolving metadata nc_path, e.g. /path/to/sun-img')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New local work directory')
    parser.add_argument('--month', default='2026-10')
    args = parser.parse_args()
    if sha256(args.model) != MODEL_SHA256:
        raise ValueError('Model SHA256 differs from the published E0472 seed-0 release')
    start = pd.Timestamp(args.month + '-01', tz='UTC') - pd.Timedelta(hours=1)
    month_end = (pd.Timestamp(args.month + '-01', tz='UTC')
                 + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=2))
    # Refuse reuse: every execution gets one immutable input copy and a fresh receipt.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / 'inputs'
    snapshot.mkdir()
    hashes = {}
    with (args.v2_root / '.refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        receipt = json.loads((args.v2_root / 'CHAMPION_READY.json').read_text())
        for relative, expected in receipt['files'].items():
            source, target = args.v2_root / relative, snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            actual = sha256(target)
            if actual != expected:
                raise ValueError(f'Generation receipt mismatch: {relative}')
            hashes[relative] = actual
        shutil.copyfile(args.v2_root / 'CHAMPION_READY.json', snapshot / 'CHAMPION_READY.json')
        paths = sorted((args.v2_root / 'donki').glob('WSAEnlilSimulations_*.json'))
        if not paths:
            raise ValueError('No local DONKI WSA-Enlil inputs')
        for source in paths:
            target = snapshot / 'donki' / source.name
            target.parent.mkdir(exist_ok=True)
            before = sha256(source)
            shutil.copyfile(source, target)
            if sha256(target) != before or sha256(source) != before:
                raise ValueError(f'DONKI input changed during copy: {source}')
            hashes[str(target.relative_to(snapshot))] = before
    model_path = args.output_dir / args.model.name
    model_hash = sha256(args.model)
    shutil.copyfile(args.model, model_path)
    if sha256(model_path) != model_hash:
        raise ValueError('Model changed during copy')
    availability = snapshot / 'suvi_availability.parquet'
    export_availability(snapshot / 'suvi-meta', args.images_root, availability)
    inputs = dict(raw_ace_dir=str(snapshot / 'ace-snapshot'),
                  speed_parquet=str(snapshot / 'ch/ace.parquet'),
                  ch_parquet=str(snapshot / 'ch/ch-hourly.parquet'),
                  suvi_parquet=str(snapshot / 'suvi-fusion/suvi-hourly-v2.parquet'),
                  suvi_manifest=str(availability), donki_dir=str(snapshot / 'donki'))
    history = pd.read_parquet(inputs['speed_parquet']).set_index('timestamp_utc').sort_index()
    history.index = pd.to_datetime(history.index, utc=True)
    if not history.index.is_unique:
        raise ValueError('Duplicate speed timestamps')
    last = min(history.index.max() - pd.Timedelta(hours=72), month_end)
    origins = pd.date_range(start, last, freq='h')
    if not len(origins):
        raise ValueError('No complete 72-hour windows in requested month')
    required = pd.date_range(origins.min() - pd.Timedelta(hours=167),
                             origins.max() + pd.Timedelta(hours=72), freq='h')
    if not np.isfinite(history.reindex(required).filled_speed_kms).all():
        raise ValueError('Missing input history or future observations')
    model = load_model(model_path)
    if model.experiment != 'xgb_d76_lead_moe_resid_dv_enlil':
        raise ValueError('Expected the frozen E0472 release model')
    # Set execution parallelism only; learned trees and feature arithmetic stay frozen.
    def limit_models(value):
        if hasattr(value, 'get_booster'):
            value.set_params(n_jobs=2, device='cpu')
        elif isinstance(value, dict):
            for item in value.values():
                limit_models(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                limit_models(item)
    limit_models(model.bundle['models'])
    with threadpool_limits(limits=2):
        source = FeatureInputs(inputs, include_enlil=True)
        selection = select_slots(origins, source.meta, source.block)
        valid = ~selection['missing']
        picked = source.meta.iloc[selection.loc[valid, 'slot_idx']]
        selected_origins = origins[valid.to_numpy()]
        if not ((picked.s3_modified.to_numpy() <= selected_origins.to_numpy()).all()
                and (picked.obs_end.to_numpy() <= (selected_origins - pd.Timedelta(hours=1)).to_numpy()).all()):
            raise ValueError('SUVI availability violation')
        print(f'Predicting {len(origins)} origins: {origins.min()} .. {origins.max()}', flush=True)
        predictions = model.predict(origins, inputs)
    predictions.to_parquet(args.output_dir / 'predictions.parquet', index=False)
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      purpose='October reference replay; not official leaderboard evaluation',
                      input_migration_status='pending_validation; explicitly approved reference use',
                      experiment='E0472', seed=0, model_sha256=model_hash,
                      model_config_sha256='b8ae672fdb6ff9a6dc85bc5c3f2d8b2acbc9207c98126d39ad1913f24581962c',
                      script_sha256=sha256(__file__), generation=receipt,
                      sources=hashes, suvi_availability_sha256=sha256(availability),
                      predictions_sha256=sha256(args.output_dir / 'predictions.parquet'),
                      origins=len(origins), first_origin_utc=origins.min().isoformat(),
                      last_origin_utc=origins.max().isoformat(),
                      last_actual_utc=(origins.max() + pd.Timedelta(hours=72)).isoformat(),
                      suvi_stale_origins=int(selection.img_stale.sum()),
                      contract='Original submitted model input timing, not the newer ACE+2h research contract',
                      observation_policy='Existing filled_speed_kms and was_missing; no additional filling',
                      runtime={'python': sys.version, 'threads': 2})
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ['origins', 'first_origin_utc', 'last_origin_utc', 'last_actual_utc']}), flush=True)


if __name__ == '__main__':
    main()
