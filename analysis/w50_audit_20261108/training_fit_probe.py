"""Inference-only, in-sample capacity probe of frozen parent seed42 full models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PARENT = REPO / 'research/drop035_v650_confirm_20261108'
RUN = PARENT / 'runs/drop035_v650_s42'
INPUT = REPO / 'research/rerun_df_ffill_20261103/input'
CHECKPOINTS = (300, 600, 900)
LIMITS = [
    'Fixed seed42 descriptive inference; no fitting, candidate selection, or query scoring.',
    'Full models trained on every analyzed row; reference/calibration/offset subsets are all in-sample.',
    'Fixed-prefix trends are not future-generalization evidence or proof of a 50% improvement.',
    'This cannot separate model-class capacity from target noise or missing input information.',
    'Originally observed versus imputed labels describe the retained training target policy only.',
    'Spectral loss couples a complete 24h head; no spectral allocation to target bins or label masks is claimed.',
    'Source equality is against available saved snapshots; byte/time receipts do not prove immutable historical identity.',
]


def load_runner():
    path = REPO / 'research/w50_roundselect_20261108/experiment.py'
    spec = importlib.util.spec_from_file_location('training_probe_parent_loader', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def observed_mask(rows, origins, target, cutoff):
    """Validate exactly the training grid and unchanged originally observed labels."""
    rows = rows.copy()
    rows['origin'] = pd.to_datetime(rows.origin, utc=True)
    rows = rows.sort_values(['origin', 'h'], kind='stable').reset_index(drop=True)
    if target.shape != (len(origins), 72) or not np.isfinite(target).all():
        raise ValueError('Invalid prepared training target grid')
    if len(rows) != target.size or rows.duplicated(['origin', 'h']).any():
        raise ValueError('Original target keys have missing or duplicate rows')
    expected_origins = np.repeat(origins.asi8, 72)
    if (not np.array_equal(pd.DatetimeIndex(rows.origin).asi8, expected_origins)
            or not np.array_equal(rows.h.to_numpy(), np.tile(np.arange(1, 73), len(origins)))):
        raise ValueError('Original target keys differ from prepared training grid')
    if not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff, tz='UTC')).all():
        raise ValueError('Training targets violate original 144h cutoff')
    original = rows.y.to_numpy(dtype=np.float64).reshape(target.shape)
    if np.isinf(original).any():
        raise ValueError('Original targets contain infinities')
    observed = np.isfinite(original)
    if not np.array_equal(original[observed], target[observed]):
        raise ValueError('Originally observed training labels changed')
    return observed


def loss_arrays(prediction, target):
    """Exact retained objective potential, normalized by pairs only when reporting.

    The frequency-domain potential's derivative is .1 * real(IFFT(z*rho)).
    The supplied XGBoost Hessian is a surrogate, not this potential's Hessian.
    """
    error = prediction - target
    raw_weights = np.where(target >= 550., 4., 1.)
    weights = raw_weights / raw_weights.mean(axis=0)
    point = .5 * weights * error ** 2
    z = np.fft.fft(error, axis=1, norm='ortho')
    spectral = .1 * 50. ** 2 * (np.sqrt(1 + abs(z) ** 2 / 50. ** 2) - 1)
    return point, spectral


def metrics(error, point, gradient, mask):
    selected = np.broadcast_to(mask, error.shape)
    e = error[selected]
    if not len(e):
        return {'pairs': 0, 'rmse': None, 'mae': None, 'bias': None,
                'weighted_point_l2_per_pair': None, 'gradient_rms': None}
    return {'pairs': len(e), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae': float(np.abs(e).mean()), 'bias': float(e.mean()),
            'weighted_point_l2_per_pair': float(point[selected].mean()),
            'gradient_rms': float(np.sqrt(np.mean(gradient[selected].astype(np.float64) ** 2)))}


def summarize(prediction, target, observed, subsets, objective):
    error = prediction - target
    point, spectral = loss_arrays(prediction, target)
    gradient, _ = objective(prediction, None)
    result = {}
    bins = [('all', -np.inf, np.inf), ('lt400', -np.inf, 400), ('400_550', 400, 550),
            ('550_700', 550, 700), ('ge700', 700, np.inf)]
    for name, indices in subsets.items():
        row_mask = np.zeros((len(target), 1), bool)
        row_mask[indices] = True
        groups = {}
        for label, labels in [('all', np.ones_like(observed)), ('observed', observed), ('imputed', ~observed)]:
            groups[label] = {bin_name: metrics(error, point, gradient,
                                               row_mask & labels & (target >= low) & (target < high))
                             for bin_name, low, high in bins}
        result[name] = {'rows': len(indices), 'label_groups': groups,
                        'spectral_per_pair': float(spectral[indices].mean()),
                        'weighted_point_l2_per_pair': float(point[indices].mean()),
                        'retained_loss_per_pair': float(point[indices].mean() + spectral[indices].mean())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=INPUT / 'df_ffill_trainy.parquet')
    parser.add_argument('--original-labels', type=Path, default=INPUT / 'df_ffill.parquet')
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU inference requires CUDA_VISIBLE_DEVICES="" before startup')
    runner = load_runner()
    baseline, core = runner.parent_modules()
    sources = {}
    for name, current in [('baseline.py', Path(baseline.__file__)), ('core.py', Path(core.__file__))]:
        saved = RUN / name
        if current.read_bytes() != saved.read_bytes():
            raise ValueError(f'Saved parent source differs: {name}')
        sources[name] = {'current': core.file_identity(current), 'saved': core.file_identity(saved), 'equal_bytes': True}
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    names = runner.assert_parent_contract(core, config, 42)
    output = {'seed': 42, 'classification': 'saved-full-model training-only inference',
              'checkpoints': list(CHECKPOINTS), 'limits': LIMITS, 'source_receipts': sources,
              'input_receipts': {'prepared': core.file_identity(args.data), 'original_labels': core.file_identity(args.original_labels)},
              'script': core.file_identity(__file__), 'loader': core.file_identity(runner.__file__),
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgb.__version__},
              'features': names, 'config': config, 'periods': {}}
    for period in ('F1', 'F2', 'F3'):
        metadata = json.loads((RUN / f'{period}_metadata.json').read_text())
        if core.normalize_config(metadata['config']) != config or metadata['rounds'] != 900:
            raise ValueError('Parent configuration or training rounds changed')
        recorded_data = metadata['identity']['data']
        current_data = core.file_identity(args.data)
        if any(current_data[key] != recorded_data[key] for key in ('bytes', 'mtime_ns')):
            raise ValueError('Prepared training data byte/time receipt differs from frozen parent')
        train, target = baseline.load_grid(args.data, period, names, training=True)
        origins = train['origins']
        if (len(origins) != metadata['train_origins'] or str(origins.min()) != metadata['train_start']
                or str(origins.max()) != metadata['train_end'] or metadata['maturity_hours'] != 144):
            raise ValueError('Original training ranges or maturity differ')
        rows = pd.read_parquet(args.original_labels, columns=['origin', 'h', 'y'],
                               filters=[(f'fit_{period}', '==', True)], use_threads=False)
        observed = observed_mask(rows, origins, target, baseline.CUTOFFS[period])
        del rows
        reference, calibration, boundary = baseline.prior_split(origins)
        if len(calibration) < 336:
            raise ValueError('Insufficient original calibration rows')
        subsets = {'full': np.arange(len(origins)), 'reference': reference,
                   'calibration': calibration, 'offset_last336': calibration[-336:]}
        frozen = metadata['frozen_state']
        for label in ('reference', 'calibration'):
            indices = subsets[label]
            if (len(indices) != frozen[f'{label}_rows'] or str(origins[indices].min()) != frozen[f'{label}_start']
                    or str(origins[indices].max()) != frozen[f'{label}_end']):
                raise ValueError('Original reference/calibration subset differs')
        if (str(origins[calibration[-336]]) != frozen['offset_start']
                or str(origins[calibration[-1]]) != frozen['offset_end'] or frozen['offset_rows'] != 336):
            raise ValueError('Original offset subset differs')
        report = {'cutoff': baseline.CUTOFFS[period], 'maturity_hours': 144,
                  'split_boundary': str(boundary), 'subsets': {
                      name: {'rows': len(indices), 'start': str(origins[indices].min()), 'end': str(origins[indices].max())}
                      for name, indices in subsets.items()}, 'heads': []}
        for block, start in enumerate((0, 24, 48)):
            path = RUN / f'{period}_models/full/b{block}.ubj'
            model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
            model.load_model(path)
            model.set_param({'device': 'cpu', 'nthread': 1})
            x = core.design(train, start, 24)
            if model.num_boosted_rounds() != 900 or model.num_features() != 212 or x.shape[1] != 212:
                raise ValueError('Saved model tree count or input dimension differs')
            y = target[:, start:start + 24]
            objective = core.make_objective(y, config)
            head = {'block': block, 'leads': [start + 1, start + 24], 'model': core.file_identity(path),
                    'expanded_features': core.expanded_names(names, start, 24), 'checkpoints': {}}
            for rounds in CHECKPOINTS:
                prediction = model.inplace_predict(x, iteration_range=(0, rounds)).astype(np.float64)
                if prediction.shape != y.shape or not np.isfinite(prediction).all():
                    raise ValueError('Invalid saved-prefix predictions')
                head['checkpoints'][str(rounds)] = summarize(prediction, y, observed[:, start:start + 24], subsets, objective)
                print(json.dumps({'period': period, 'block': block, 'rounds': rounds, 'stage': 'training_inference'}), flush=True)
            report['heads'].append(head)
            del model, x
        output['periods'][period] = report
    args.out.mkdir(parents=True, exist_ok=True)
    core.write_json(args.out / 'training_fit_probe.json', output)
    lines = ['# Retained-parent training fit probe', '', *[f'- {item}' for item in LIMITS], '',
             '| Period | Head | Prefix | Full raw RMSE | Point L2/pair | Spectral/pair |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for period, report in output['periods'].items():
        for head in report['heads']:
            for rounds, values in head['checkpoints'].items():
                full = values['full']
                lines.append(f"| {period} | {head['block']} | {rounds} | {full['label_groups']['all']['all']['rmse']:.6f} | "
                             f"{full['weighted_point_l2_per_pair']:.6f} | {full['spectral_per_pair']:.6f} |")
    lines += ['', 'The JSON contains every fixed speed bin, original-label mask, and original row subset.',
              'Tail weights are normalized once across the full head training population, then retained for subset reports.',
              'Spectral potential is averaged over Fourier coefficients/pairs of whole 24h heads; its Hessian is not the training surrogate Hessian.']
    (args.out / 'training_fit_probe.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
