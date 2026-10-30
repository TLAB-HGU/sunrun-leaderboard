"""GPU-only, resumable feat30 TCN experiment; official scores never select epochs."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import load_inputs
from model import TCN, weighted_mse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = 'e2000fs_feat30_pointweight4_tcn_lb120'
PARENT = ROOT / 'analysis/e2000fs_feat30_pointweight/runs/full_20261028_v1'
RECIPE = dict(seed=42, lookback_hours=120, kernel=3, dilations=[1, 2, 4, 8, 16, 32],
              channels=64, head=[128, 64, 1], optimizer='Adam', learning_rate=0.001,
              batch_size=256, max_epochs=100, patience=10,
              weighting='per-horizon target>=550:4 else:1; training mean normalized',
              selection='internal weighted MSE only', target='absolute speed; training z-score',
              input_lag_hours=0, target_maturity_hours=72)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def temporal_split(origins):
    start = pd.Timestamp('2025-12-04', tz='UTC')
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < start)
    val = np.flatnonzero((origins >= start) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))
    if not len(fit) or not len(val):
        raise ValueError('Empty temporal fit/validation split')
    assert (origins[fit] + pd.Timedelta(hours=72)).max() < origins[val].min()
    return fit, val


def normalization(inputs, fit):
    """Use only hours included in training windows, plus training future covariates."""
    positions = inputs['train']['positions'][fit]
    used = np.zeros(len(inputs['history_hourly']), dtype=bool)
    for end in positions:
        if end < 119:
            raise ValueError('Incomplete 120-hour training window')
        used[end - 119:end + 1] = True
    history = inputs['history_hourly'][used]
    future = inputs['train']['future'][fit].reshape(-1, 9)
    result = {}
    for tag, array in [('history', history), ('future', future)]:
        if np.isinf(array).any():
            raise ValueError(f'Infinite {tag} feature')
        median = np.nanmedian(array, axis=0)
        if not np.isfinite(median).all():
            raise ValueError(f'Entirely missing training feature in {tag}')
        filled = np.where(np.isfinite(array), array, median)
        result[tag] = dict(median=median.astype(np.float32),
                           mean=filled.mean(axis=0, dtype=np.float64).astype(np.float32),
                           std=np.maximum(filled.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32))
    y = inputs['train']['y'][fit]
    result['y_mean'], result['y_std'] = float(y.mean()), max(float(y.std()), 1e-6)
    result['weight_mean'] = np.where(y >= 550., 4., 1.).mean(axis=0).astype(np.float32)
    return result


def transform(array, stats):
    return ((np.where(np.isfinite(array), array, stats['median']) - stats['mean']) / stats['std']).astype(np.float32)


class Windows(Dataset):
    def __init__(self, history, stage, indices, stats):
        self.history, self.stage, self.indices, self.stats = history, stage, np.asarray(indices), stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        end = int(self.stage['positions'][i])
        if end < 119:
            raise ValueError('Incomplete history')
        h = transform(self.history[end - 119:end + 1], self.stats['history'])
        f = transform(self.stage['future'][i], self.stats['future'])
        actual = self.stage['y'][i]
        y = ((actual - self.stats['y_mean']) / self.stats['y_std']).astype(np.float32)
        w = (np.where(actual >= 550., 4., 1.) / self.stats['weight_mean']).astype(np.float32)
        return h, f, y, w


def loader(history, stage, indices, stats, shuffle=False, generator=None):
    return DataLoader(Windows(history, stage, indices, stats), batch_size=RECIPE['batch_size'],
                      shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True)


def validation(net, batches, device):
    net.eval()
    total = count = 0
    with torch.inference_mode():
        for arrays in batches:
            hist, future, target, weights = [a.to(device, non_blocking=True) for a in arrays]
            total += float(weighted_mse(net(hist, future), target, weights)) * target.numel()
            count += target.numel()
    return total / count


def train_stage(inputs, name, fit, query, stats, out, device, smoke, frozen_epochs=None):
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    recipe_id = digest(out / 'manifest.json')
    torch.manual_seed(RECIPE['seed'])
    torch.cuda.manual_seed_all(RECIPE['seed'])
    generator = torch.Generator().manual_seed(RECIPE['seed'])
    net = TCN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=RECIPE['learning_rate'])
    identity = dict(manifest_sha256=recipe_id, stage=name, fit_indices=fit.tolist(),
                    frozen_epochs=frozen_epochs)
    epochs = frozen_epochs if frozen_epochs is not None else (2 if smoke else RECIPE['max_epochs'])
    epoch, best, best_epoch, wait = 0, float('inf'), 0, 0
    latest = folder / 'latest.pt'
    curve = []
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Refusing changed checkpoint identity')
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, best, best_epoch, wait = saved['epoch'], saved['best'], saved['best_epoch'], saved['wait']
        curve = saved['curve']
        generator.set_state(saved['generator_state'])
        torch.set_rng_state(saved['rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    training = loader(inputs['history_hourly'], inputs['train'], fit, stats, True, generator)
    evaluating = None if name != 'internal' else loader(inputs['history_hourly'], inputs['train'], query, stats)
    for ep in range(epoch + 1, epochs + 1):
        if frozen_epochs is None and wait >= RECIPE['patience']:
            break
        net.train()
        total = count = 0
        started = time.monotonic()
        for arrays in training:
            hist, future, target, weights = [a.to(device, non_blocking=True) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(net(hist, future), target, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        train_mse = total / count * stats['y_std'] ** 2
        val_mse = validation(net, evaluating, device) * stats['y_std'] ** 2 if evaluating else None
        if val_mse is not None and not np.isfinite(val_mse):
            raise FloatingPointError('Nonfinite validation loss')
        improved = frozen_epochs is not None or val_mse < best
        if improved:
            best, best_epoch, wait = val_mse if val_mse is not None else train_mse, ep, 0
        else:
            wait += 1
        row = dict(epoch=ep, train_weighted_mse=train_mse, validation_weighted_mse=val_mse,
                   seconds=time.monotonic() - started)
        curve.append(row)
        state = dict(identity=identity, model=net.state_dict(), optimizer=optimizer.state_dict(),
                     stats=stats, epoch=ep, best=best, best_epoch=best_epoch, wait=wait,
                     generator_state=generator.get_state(), rng_state=torch.get_rng_state(),
                     cuda_rng_state=torch.cuda.get_rng_state_all(), curve=curve)
        if improved:
            save_checkpoint(folder / 'best.pt', state)
        save_checkpoint(latest, state)
        with (folder / 'loss.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(curve)
        write(out / 'status.json', dict(status='running', phase='training', stage=name,
                                       epoch=ep, max_epochs=epochs, best_epoch=best_epoch, **row_to_status(row)))
        print(f'{name} epoch={ep}/{epochs} train_wmse={train_mse:.4f} '
              f'val_wmse={val_mse} best_epoch={best_epoch} seconds={row["seconds"]:.2f}', flush=True)
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    if saved['identity'] != identity or saved['epoch'] != best_epoch:
        raise ValueError('Best checkpoint identity or selected epoch differs')
    net.load_state_dict(saved['model'])
    write(folder / 'receipt.json', dict(best_epoch=best_epoch, epochs_run=len(curve),
                                       checkpoint_sha256=digest(folder / 'best.pt'),
                                       validation_selection=name == 'internal', fit_count=len(fit)))
    return net, best_epoch


def row_to_status(row):
    return {key: value for key, value in row.items() if key != 'epoch'}


def predict(net, inputs, stage, indices, stats, device):
    net.eval()
    result = []
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for arrays in loader(inputs['history_hourly'], stage, indices, stats):
            hist, future = [a.to(device, non_blocking=True) for a in arrays[:2]]
            result.append(net(hist, future).cpu().numpy() * stats['y_std'] + stats['y_mean'])
    torch.cuda.synchronize()
    return np.clip(np.concatenate(result).astype(np.float64), 1., 3000.), time.monotonic() - started


def metrics(y, pred):
    error = (np.asarray(pred, dtype=np.float64) - y) ** 2
    high = y >= 550.
    return dict(mse=float(error.mean()),
                mse_lt550=float(error[~high].mean()) if (~high).any() else None,
                mse_ge550=float(error[high].mean()) if high.any() else None,
                n_origins=len(y), n_pairs=int(y.size),
                mse_by_band={f'h{h+1}-{h+24}': float(error[:, h:h+24].mean()) for h in (0, 24, 48)})


def frame(origins, pred):
    if pred.shape != (len(origins), 72) or not np.isfinite(pred).all() or not origins.is_unique:
        raise ValueError('Invalid prediction grid')
    return pd.DataFrame(dict(origin_last_input_utc=np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                             horizon_hours=np.tile(np.arange(1, 73), len(origins)), pred_kms=pred.ravel()))


def execute(out, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
                raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1 to use the authorized physical GPU 1')
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly one visible CUDA device required; no CPU fallback')
            device = torch.device('cuda:0')
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            print(f'CUDA physical=1 logical=0 device={torch.cuda.get_device_name(0)} torch={torch.__version__}', flush=True)
            write(out / 'status.json', dict(status='running', phase='load_inputs', smoke=smoke))
            inputs = load_inputs(out, smoke=smoke)
            fit, val = temporal_split(inputs['train']['origins'])
            manifest = dict(experiment=EXPERIMENT, smoke=smoke, recipe=RECIPE,
                            selected=inputs['selected'], varying=inputs['varying'],
                            source_sha256={str(p): digest(p) for p in sorted(HERE.glob('*.py'))},
                            input_identity=inputs['identity'], physical_gpu=1,
                            torch_version=torch.__version__, cuda_version=torch.version.cuda,
                            direct_parent_manifest_sha256=digest(PARENT / 'manifest.json'),
                            posthoc_official_exposed=True,
                            comparison_note='New architecture and 120h representation branch; no isolated model-only ablation')
            path = out / 'manifest.json'
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError('Refusing changed run identity')
            write(path, manifest)
            stats = normalization(inputs, fit)
            net, best_epoch = train_stage(inputs, 'internal', fit, val, stats, out, device, smoke)
            pred, _ = predict(net, inputs, inputs['train'], val, stats, device)
            np.savez_compressed(out / 'internal_predictions.npz', origins=inputs['train']['origins'][val].as_unit('ns').asi8, pred=pred)
            results = {'internal': metrics(inputs['train']['y'][val], pred)}
            write(out / 'selection.json', dict(best_epoch=best_epoch, criterion='internal weighted MSE',
                                              fit_last_target=str((inputs['train']['origins'][fit] + pd.Timedelta(hours=72)).max()),
                                              validation_first_origin=str(inputs['train']['origins'][val].min())))
            del net
            frames, ys, ps = [], [], []
            inference_seconds = 0.
            for name, cutoff in [('f1', '2026-02-26'), ('final', '2026-05-29')]:
                stage = inputs[name]
                train = np.flatnonzero(inputs['train']['origins'] + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))
                assert (inputs['train']['origins'][train] + pd.Timedelta(hours=72)).max() < stage['origins'].min()
                refit_stats = normalization(inputs, train)
                net, _ = train_stage(inputs, name, train, None, refit_stats, out, device, smoke, best_epoch)
                pred, elapsed = predict(net, inputs, stage, np.arange(len(stage['origins'])), refit_stats, device)
                inference_seconds += elapsed
                np.savez_compressed(out / f'{name}_predictions.npz', origins=stage['origins'].as_unit('ns').asi8, pred=pred)
                results[name] = metrics(stage['y'], pred)
                frames.append(frame(stage['origins'], pred))
                ys.append(stage['y'])
                ps.append(pred)
                write(out / 'results.json', results)
                del net
            combined = pd.concat(frames, ignore_index=True)
            if combined.duplicated(['origin_last_input_utc', 'horizon_hours']).any():
                raise ValueError('Duplicate combined grid')
            results['combined'] = metrics(np.vstack(ys), np.vstack(ps))
            if not smoke:
                assert len(combined) == 357840
                spec = importlib.util.spec_from_file_location('tcn_evaluation', ROOT / 'analysis/e2000fs_feat30_followups/evaluation.py')
                evaluation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(evaluation)
                # The pointweight parent embeds the full data/evaluation parent in its distribution reference.
                distribution = json.loads((ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1/manifest.json').read_text())
                results['combined']['official'] = evaluation.official_score(combined, distribution['parent'])
                np.testing.assert_allclose(results['combined']['mse'], results['combined']['official']['mse'], rtol=1e-10)
                reference = json.loads((PARENT / 'results.json').read_text())
                for name in results:
                    results[name]['delta_vs_pointweight4'] = {
                        key: results[name][key] - reference[name][key]
                        for key in ('mse', 'mse_lt550', 'mse_ge550')}
            temporary = out / 'predictions.parquet.tmp'
            combined.to_parquet(temporary, index=False)
            temporary.replace(out / 'predictions.parquet')
            write(out / 'results.json', results)
            write(out / 'timing.json', dict(inference_seconds=inference_seconds,
                                            inference_seconds_per_fold=inference_seconds / (len(combined) / 72),
                                            physical_gpu=1, max_memory_allocated_bytes=torch.cuda.max_memory_allocated()))
            write(out / 'status.json', dict(status='completed', smoke=smoke, best_epoch=best_epoch,
                                            rows=len(combined), predictions_sha256=digest(out / 'predictions.parquet')))
            print(f'COMPLETED smoke={smoke} rows={len(combined)} best_epoch={best_epoch}', flush=True)
        except Exception as exc:
            write(out / 'status.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    execute(args.out.resolve(), args.smoke)
