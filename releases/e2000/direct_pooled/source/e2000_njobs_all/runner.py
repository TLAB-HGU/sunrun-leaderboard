"""Frozen single-seed architecture experiment; no official truth enters fitting."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import (CUTOFF, TRAIN_END, atomic_json, check_ref, digest, input_audit,
                     prediction_frame, reference, sha, split_masks, target_matrix, widen)
from .model import LoggedMultiOutputRegressor, ModelConfig, PooledRegressor

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent / 'cumulative_20261024'


def code_hashes():
    paths = [HERE / p for p in ('model.py', 'inputs.py', 'runner.py', 'launch.py')]
    paths += [CANONICAL / p for p in ('contracts.py', 'gate.py')]
    return {str(p): sha(p) for p in paths}


def metrics(y, pred, mask):
    if pred.shape != y.shape or mask.shape != y.shape or not mask.any():
        raise ValueError('metric shape/observation mismatch')
    se = np.square(pred - y)
    return {'mse': float(se[mask].mean()), 'observed_cells': int(mask.sum()),
            'mse_by_horizon': [float(se[:, h][mask[:, h]].mean()) for h in range(y.shape[1])]}


def curves(stage):
    # Standard-library PNG + labelled SVG: no plotting dependency installed here.
    import struct
    import zlib
    frames = [pd.read_csv(p) for p in sorted(Path(stage).glob('loss_history_s42*.csv'))]
    if not frames:
        raise ValueError('no native loss histories to plot')
    w, height, left, top, right, bottom = 900, 500, 80, 50, 860, 430
    pixels = np.full((height, w, 3), 255, dtype=np.uint8)
    maximum = max(float(np.nanmax(d[['train_mse', 'validation_mse']])) for d in frames)
    maximum = max(maximum, 1.)
    rounds = max(int(d.step.max()) for d in frames)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="80" y="25">{Path(stage).name}: blue=train, orange=validation</text>',
           '<text x="370" y="485">Boosting round</text>',
           '<text x="5" y="45">MSE (km/s)^2</text>']

    def line(x0, y0, x1, y1, color):
        count = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xx = np.rint(np.linspace(x0, x1, count)).astype(int)
        yy = np.rint(np.linspace(y0, y1, count)).astype(int)
        pixels[np.clip(yy, 0, height - 1), np.clip(xx, 0, w - 1)] = color

    line(left, top, left, bottom, (0, 0, 0))
    line(left, bottom, right, bottom, (0, 0, 0))
    svg.append(f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="black"/>')
    for d in frames:
        for name, color, css in [('train_mse', (31, 119, 180), '#1f77b4'),
                                 ('validation_mse', (255, 127, 14), '#ff7f0e')]:
            keep = np.isfinite(d[name])
            xs = left + (d.step[keep].to_numpy() - 1) / max(rounds - 1, 1) * (right - left)
            ys = bottom - d[name][keep].to_numpy() / maximum * (bottom - top)
            for i in range(1, len(xs)):
                line(xs[i - 1], ys[i - 1], xs[i], ys[i], color)
            points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{css}" opacity="0.35"/>')
    svg.extend([f'<text x="{left}" y="450">1</text>',
                f'<text x="{right}" y="450">{rounds}</text>',
                f'<text x="5" y="65">{maximum:.0f}</text>', '</svg>'])
    (Path(stage) / 'loss_history.svg').write_text('\n'.join(svg))

    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)

    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, height, 8, 2, 0, 0, 0))
           + chunk(b'tEXt', f'Description\0Blue=train; orange=validation; x=round 1..{rounds}; y=MSE 0..{maximum}'.encode())
           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    (Path(stage) / 'loss_history.png').write_bytes(png)


def load_data(spec):
    a = spec['artifacts']
    features = json.loads(check_ref(a['feature_names']).read_text())['features']
    origins, x = widen(pd.read_parquet(check_ref(a['train_features'])), features)
    y, mask = target_matrix(pd.read_parquet(check_ref(a['labels'])), origins)
    oo, ox = widen(pd.read_parquet(check_ref(a['official_features'])), features,
                   expected_origins=pd.date_range('2026-05-31T23:00Z', '2026-09-26T23:00Z', freq='h'))
    if list(x) != list(ox):
        raise ValueError('wide feature order differs')
    return origins, x, y, mask, oo, ox


def run(spec, out, *, arm, config=None):
    """Fit either matched pooled control or candidate; candidate always confirms."""
    audit = input_audit(spec)
    if not audit['passed']:
        raise ValueError('input preflight failed: ' + '; '.join(audit['errors']))
    if arm not in ('pooled', 'multioutput'):
        raise ValueError('unknown architecture')
    config = ModelConfig() if config is None else config
    if config.horizons != 72:
        raise ValueError('production experiment requires 72 horizons')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    origins, x, y, mask, official_origins, official_x = load_data(spec)
    train_mask, valid_mask = split_masks(origins, mask)
    tr, va = train_mask.any(axis=1), valid_mask.any(axis=1)
    context = {'code_hashes': code_hashes(), 'data_hashes': {'input_envelope': digest(spec)},
               'feature_names': list(x.columns), 'cutoff': TRAIN_END.isoformat(),
               'grid': {'origins': digest(origins.astype(str).tolist()), 'horizons': 72},
               'source_availability': spec['source_receipts'], 'stage': 'internal'}
    klass = LoggedMultiOutputRegressor if arm == 'multioutput' else PooledRegressor
    model = klass(config)
    directory = out / 'internal'
    print(f'{arm}: internal fit {tr.sum()} origins x {x.shape[1]} features; seed42', flush=True)
    model.fit(x.to_numpy()[tr], y[tr], observed_mask=train_mask[tr],
              validation=(x.to_numpy()[va], y[va], valid_mask[va]), output_dir=directory,
              context=context, validation_role='held_out')
    pred = model.predict(x.to_numpy()[va])
    result = metrics(y[va], pred, valid_mask[va])
    prediction_frame(origins[va], pred).to_parquet(out / 'internal_predictions_s42.parquet', index=False)
    result.update(arm=arm, seed=42, recipe=asdict(config),
                  train_target_cutoff=TRAIN_END.isoformat(), input_sha256=digest(spec),
                  feature_names=list(x.columns), code_hashes=code_hashes())
    atomic_json(out / 'internal.json', result)
    curves(directory)
    if arm == 'multioutput':
        # User explicitly requested final evaluation even for an internal negative.
        context = dict(context, cutoff=CUTOFF.isoformat(), stage='official',
                       grid={'origins': digest(official_origins.astype(str).tolist()), 'horizons': 72})
        final = klass(config)
        print('multioutput: final fit on all observed targets before 2026-05-29', flush=True)
        final.fit(x.to_numpy(), y, observed_mask=mask,
                  validation=(x.to_numpy()[va], y[va], valid_mask[va]), output_dir=out / 'official',
                  context=context, validation_role='overlapping_diagnostic')
        values = final.predict(official_x.to_numpy())
        # Checkpoint replay: re-enter fit, which must only verify/load bound receipts.
        restored = klass(config)
        restored.fit(x.to_numpy(), y, observed_mask=mask,
                      validation=(x.to_numpy()[va], y[va], valid_mask[va]), output_dir=out / 'official',
                      context=context, validation_role='overlapping_diagnostic')
        if not np.array_equal(values, restored.predict(official_x.to_numpy())):
            raise ValueError('final checkpoint predictions differ')
        frame = prediction_frame(official_origins, values)
        if len(frame) != 203976:
            raise ValueError('incomplete official grid')
        from scorer import scoring
        folds = pd.DataFrame({'origin_last_input_utc': official_origins.strftime('%Y-%m-%dT%H:%M:%SZ')})
        errors = scoring.validate(frame, folds)
        if errors:
            raise ValueError('official validator: ' + str(errors))
        tmp = out / 'predictions_official_s42.parquet.tmp'
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, out / 'predictions_official_s42.parquet')
        curves(out / 'official')
        atomic_json(out / 'confirmation.json', {
            'status': 'completed', 'seed': 42, 'horizon_models': 72, 'checkpoint_replay_exact': True,
            'validation_role': 'overlapping_diagnostic_not_selection',
            'prediction': reference(out / 'predictions_official_s42.parquet'),
            'input_sha256': digest(spec), 'code_hashes': code_hashes(),
        })
    return result
