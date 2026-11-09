"""No-fit training-only correction-root sharing diagnostic on frozen parent models."""
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
CONFIG_RECEIPT = ROOT / 'output_sharing_probe/reconstructed-config.json'
FIT_RECEIPT = ROOT / 'training_fit_probe.json'
PREFIXES = (300, 900)
SEMANTICS = (
    '31 linear float64 quantiles at 1/32..31/32 of finite float32 X; min<q<=max; '
    'threshold is smallest float32>=q, preserving X<float64q; missing-left before '
    'missing-right, both only when NaNs exist; equivalent partitions deduplicated '
    'within each feature by finite-left count and missing direction; no missing-only end cuts.'
)
LIMITS = [
    'All contexts use full-model fitted training rows and fixed seed42 prefixes300/900; no future validation.',
    'This frozen-checkpoint correction-root probe does not replay historical splits, masks, or node memberships.',
    'The coarse31-quantile grid is not the actual256-bin sketch; no sampling, gamma/depth history or model-family comparison is inferred.',
    'A feature varying only in a rare tail can have no internal31-grid cuts; missing conflict on this grid does not rule out rare-event partition conflict.',
    'Common feasibility requires both child Hessians>=min_child_weight for every output; actual vector eligibility can differ.',
    'Gain uses the retained diagonal surrogate Hessian, not the exact decrease of the coupled spectral loss.',
    'No-split gain0 is relative to the unsplit correction-root Q(parent), not zero constant-correction potential.',
    'Positive regret is a local partition tradeoff, not proof of improvement, physical mechanism, or50% attainability.',
    'Zero positive available gain is uninformative. Concentration alone does not establish conflict.',
    'Exact maxima define R; selected output-regret sum equals R plus the tolerance-selected shared gap. Agreement is a separate tolerance diagnostic; leave-one-out uses full-context tolerance.',
    'No fitting, query target/observation access, scoring, prediction correction, submission or parent update.',
    'Byte/time and saved-source correspondence establish available-copy consistency, not immutable historical provenance.',
]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def identity(path):
    path = Path(path).resolve()
    value = path.stat()
    return {'path': str(path), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def validate_receipts(value):
    """Check every exact identity record without replacing any recorded value."""
    recorded = {}

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {'path', 'bytes', 'mtime_ns'}:
                path = item['path']
                if (not isinstance(path, str) or not Path(path).is_absolute()
                        or type(item['bytes']) is not int or type(item['mtime_ns']) is not int):
                    raise ValueError('Invalid postflight identity receipt')
                if path in recorded and recorded[path] != item:
                    raise ValueError(f'Conflicting duplicate postflight receipt: {path}')
                recorded[path] = item
            else:
                for nested in item.values():
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    for path, receipt in recorded.items():
        if identity(path) != receipt:
            raise ValueError(f'Postflight identity changed: {path}')
    return len(recorded)


def candidate_statistics(features, gradient, hessian, *, row_ids, derivative_row_ids):
    """Frozen ordered candidate statistics; no fitting or gain-based filtering."""
    x, g, h = np.asarray(features), np.asarray(gradient), np.asarray(hessian)
    ids, derivative_ids = np.asarray(row_ids), np.asarray(derivative_row_ids)
    if (x.ndim != 2 or g.ndim != 2 or h.shape != g.shape or len(x) != len(g)
            or not len(x) or not x.shape[1] or not g.shape[1]):
        raise ValueError('Feature/gradient/Hessian row and output dimensions must align')
    if x.dtype != np.float32 or g.dtype != np.float32 or h.dtype != np.float32:
        raise ValueError('Original features and callback derivatives must be float32')
    if (ids.shape != (len(x),) or derivative_ids.shape != ids.shape
            or not np.array_equal(ids, derivative_ids) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Feature and derivative row identities must match uniquely in order')
    if np.isinf(x).any() or not np.isfinite(g).all() or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError('Features allow NaNs only; derivatives require finite gradients/nonnegative Hessians')
    parent_g, parent_h = g.sum(axis=0, dtype=np.float64), h.sum(axis=0, dtype=np.float64)
    feature_ids, thresholds, directions, counts, left_g, left_h, reports = [], [], [], [], [], [], []
    levels = np.arange(1, 32, dtype=np.float64) / 32
    for feature in range(x.shape[1]):
        values = x[:, feature]
        finite_rows = np.flatnonzero(np.isfinite(values))
        missing_rows = np.flatnonzero(np.isnan(values))
        ordered = finite_rows[np.argsort(values[finite_rows], kind='stable')]
        finite = values[ordered]
        quantiles = np.unique(np.quantile(finite.astype(np.float64), levels, method='linear')) if len(finite) else np.empty(0)
        internal = quantiles[(quantiles > finite[0]) & (quantiles <= finite[-1])] if len(finite) else quantiles
        cuts = internal.astype(np.float32)
        below = cuts.astype(np.float64) < internal
        cuts[below] = np.nextafter(cuts[below], np.float32(np.inf))
        cuts = np.unique(cuts)
        if len(cuts) and (not np.isfinite(cuts).all() or (cuts <= finite[0]).any() or (cuts > finite[-1]).any()):
            raise ValueError('Canonical threshold ceiling failed to preserve internal finite partition')
        cg = np.vstack([np.zeros((1, g.shape[1])), np.cumsum(g[ordered], axis=0, dtype=np.float64)])
        ch = np.vstack([np.zeros((1, h.shape[1])), np.cumsum(h[ordered], axis=0, dtype=np.float64)])
        mg, mh = g[missing_rows].sum(axis=0, dtype=np.float64), h[missing_rows].sum(axis=0, dtype=np.float64)
        missing_directions = (True, False) if len(missing_rows) else (True,)
        seen = set()
        before = len(feature_ids)
        for cut in cuts:
            k = int(np.searchsorted(finite, cut, side='left'))
            for missing_left in missing_directions:
                key = (k, missing_left)
                if key in seen:
                    continue
                seen.add(key)
                count = k + (len(missing_rows) if missing_left else 0)
                feature_ids.append(feature)
                thresholds.append(cut)
                directions.append(missing_left)
                counts.append(count)
                left_g.append(cg[k] + (mg if missing_left else 0.))
                left_h.append(ch[k] + (mh if missing_left else 0.))
        reports.append({'feature': feature, 'finite_rows': len(finite), 'missing_rows': len(missing_rows),
                        'distinct_quantiles': len(quantiles), 'internal_quantiles': len(internal),
                        'canonical_thresholds': len(cuts), 'directions_considered': len(cuts) * len(missing_directions),
                        'duplicate_partitions_dropped': len(cuts) * len(missing_directions) - (len(feature_ids) - before),
                        'retained_candidates': len(feature_ids) - before})
    left_g = np.asarray(left_g, dtype=np.float64).reshape(-1, g.shape[1])
    left_h = np.asarray(left_h, dtype=np.float64).reshape(-1, g.shape[1])
    return {'feature_indices': np.asarray(feature_ids, dtype=np.int32),
            'thresholds': np.asarray(thresholds, dtype=np.float32), 'missing_left': np.asarray(directions, dtype=bool),
            'left_counts': np.asarray(counts, dtype=np.int64), 'right_counts': len(x) - np.asarray(counts, dtype=np.int64),
            'parent_gradient': parent_g, 'parent_hessian': parent_h,
            'left_gradient': left_g, 'left_hessian': left_h,
            'right_gradient': parent_g[None] - left_g, 'right_hessian': parent_h[None] - left_h,
            'feature_counts': reports}


def validate_reconstructed(reconstructed, baseline):
    if reconstructed['xgboost'] != '2.1.4' or reconstructed['trees_created'] != 0:
        raise ValueError('Expected pinned zero-tree reconstructed configuration receipt')
    expected = dict(baseline.PARAMS, seed=42, nthread=4)
    if reconstructed['parameters'] != expected:
        raise ValueError('Fit-time parameters differ from original parent parameters')
    effective = reconstructed['effective_train_parameters']
    for name, value in {'reg_alpha': 0., 'alpha': 0., 'max_delta_step': 0., 'gamma': 0.,
                        'min_split_loss': 0., 'reg_lambda': 5., 'lambda': 5., 'min_child_weight': 20.}.items():
        if float(effective[name]) != value:
            raise ValueError(f'Reconstructed effective default differs: {name}')
    return {'reg_alpha': 0., 'reg_lambda': 5., 'min_child_weight': 20., 'max_delta_step': 0.}


def validate_period(core, config, names, metadata, train, target, period, data_receipt, reconstructed):
    """Must succeed before opening any saved model for this period."""
    validate_reconstructed(reconstructed, core.baseline)
    if (metadata['parameters'] != reconstructed['parameters'] or core.normalize_config(metadata['config']) != config
            or metadata['rounds'] != 900 or metadata['smoke'] is not False or metadata['seed'] != 42
            or metadata['period'] != period):
        raise ValueError('Frozen fit-time metadata contract differs')
    expected_columns = {str(start): core.expanded_names(names, start, 24) for start in (0, 24, 48)}
    if (metadata['features'] != names or metadata['expanded_features'] != expected_columns
            or metadata['feature_count'] != 28 or metadata['expanded_feature_count'] != 212):
        raise ValueError('Frozen source/expanded feature contract differs')
    if any(metadata['identity']['data'][key] != data_receipt[key] for key in ('bytes', 'mtime_ns')):
        raise ValueError('Training data byte/time identity differs')
    origins = train['origins']
    cutoff = core.baseline.CUTOFFS[period]
    if (metadata['training_cutoff'] != cutoff or metadata['maturity_hours'] != 144
            or origins.tz is None or origins.hasnans or origins.has_duplicates or not origins.is_monotonic_increasing
            or not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(cutoff, tz='UTC')).all()):
        raise ValueError('Original maturity, cutoff or origin-order contract differs')
    if (len(origins) != metadata['train_origins'] or str(origins.min()) != metadata['train_start']
            or str(origins.max()) != metadata['train_end']):
        raise ValueError('Original full training ranges/counts differ')
    if (target.shape != (len(origins), 72) or not np.isfinite(target).all()
            or train['C'].shape != (len(origins), 20) or train['E'].shape != (len(origins), 72, 8)
            or train['C'].dtype != np.float32 or train['E'].dtype != np.float32):
        raise ValueError('Original full training matrices/targets differ')


def model_identity(path, fit_receipt, period, block):
    heads = fit_receipt['periods'][period]['heads']
    if len(heads) != 3 or heads[block]['block'] != block:
        raise ValueError('Sealed model receipt head inventory differs')
    current = identity(path)
    if current != heads[block]['model']:
        raise ValueError('Saved full model byte/time/path identity differs from sealed training probe')
    return current


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def flat_numeric(value, prefix=''):
    arrays = {}
    for key, item in value.items():
        name = f'{prefix}{key}'
        if isinstance(item, dict):
            arrays.update(flat_numeric(item, name + '__'))
        elif item is not None:
            arrays[name] = np.asarray(item)
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet')
    parser.add_argument('--out', type=Path, default=ROOT / 'output_sharing_probe')
    parser.add_argument('--device', choices=('cpu',), required=True)
    parser.add_argument('--threads', type=int, choices=(1,), required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or xgb.__version__ != '2.1.4':
        raise ValueError('Pinned XGBoost2.1.4 CPU1 requires CUDA_VISIBLE_DEVICES=""')
    if (args.out / 'report.json').exists() or (args.out / 'complete.json').exists() or (args.out / 'contexts').exists():
        raise ValueError('Existing completed or partial probe output must not be overwritten')
    runner = module(REPO / 'research/w50_roundselect/experiment.py', 'sharing_parent_loader')
    math_core = module(ROOT / 'output_sharing_probe_math.py', 'sharing_probe_math')
    baseline, core = runner.parent_modules()
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    names = runner.assert_parent_contract(core, config, 42)
    reconstructed = json.loads(CONFIG_RECEIPT.read_text())
    params = validate_reconstructed(reconstructed, baseline)
    sealed = json.loads(FIT_RECEIPT.read_text())
    if sealed['seed'] != 42 or sealed['features'] != names or sealed['config'] != config:
        raise ValueError('Sealed training probe parent contract differs')
    sources = {}
    for label, current in [('baseline.py', Path(baseline.__file__)), ('core.py', Path(core.__file__))]:
        saved = RUN / label
        if current.read_bytes() != saved.read_bytes():
            raise ValueError('Parent source bytes differ from saved source')
        sources[label] = {'current': identity(current), 'saved': identity(saved), 'equal_bytes': True}
    data_receipt = identity(args.data)
    if any(data_receipt[k] != sealed['input_receipts']['prepared'][k] for k in ('bytes', 'mtime_ns')):
        raise ValueError('Input identity differs from sealed training probe')
    result = {'status': 'complete', 'seed': 42, 'prefixes': list(PREFIXES), 'contexts': [],
              'versions': {'xgboost': xgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
              'semantics': SEMANTICS, 'limits': LIMITS, 'features': names, 'parent_config': config,
              'reconstructed_configuration': reconstructed, 'source_receipts': sources, 'data': data_receipt,
              'receipts': {name: identity(path) for name, path in [('controller', Path(__file__)),
                           ('math', ROOT / 'output_sharing_probe_math.py'),
                           ('parent_loader', REPO / 'research/w50_roundselect/experiment.py'),
                           ('configuration', CONFIG_RECEIPT),
                           ('sealed_training_probe', FIT_RECEIPT), ('strategy', ROOT / 'output_sharing_probe_strategy.md'),
                           ('mechanism_evidence', ROOT / 'output_sharing_mechanism_evidence.json')]}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'contexts').mkdir()
    unit = 0
    for period in ('F1', 'F2', 'F3'):
        metadata_path = RUN / f'{period}_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        train, target = baseline.load_grid(args.data, period, names, training=True)
        validate_period(core, config, names, metadata, train, target, period, data_receipt, reconstructed)
        origins = train['origins']
        for block, start in enumerate((0, 24, 48)):
            model_path = RUN / f'{period}_models/full/b{block}.ubj'
            model_receipt = model_identity(model_path, sealed, period, block)
            booster = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
            booster.load_model(model_path)
            booster.set_param({'device': 'cpu', 'nthread': 1})
            dimensions = json.loads(booster.save_config())['learner']['learner_model_param']
            if (booster.num_boosted_rounds() != 900 or booster.num_features() != 212
                    or int(dimensions['num_target']) != 24):
                raise ValueError('Saved full900 model dimensions differ')
            x = core.design(train, start, 24)
            if x.shape != (len(origins), 212):
                raise ValueError('Actual expanded matrix width differs')
            y = target[:, start:start + 24]
            objective = core.make_objective(y, config)
            for prefix in PREFIXES:
                prediction = booster.inplace_predict(x, iteration_range=(0, prefix))
                if prediction.shape != y.shape or not np.isfinite(prediction).all():
                    raise ValueError('Invalid frozen-prefix training prediction')
                # Complete domain/coupled head first; no subset or output renormalization.
                gradient, hessian = objective(prediction, None)
                candidates = candidate_statistics(x, gradient, hessian, row_ids=origins.asi8, derivative_row_ids=origins.asi8)
                gains = math_core.candidate_gains(candidates['parent_gradient'], candidates['parent_hessian'],
                                                 candidates['left_gradient'], candidates['left_hessian'], **params)
                diagnostic = math_core.sharing_diagnostics(gains['gains'], gains['common_feasible'])
                diagnostic['selected_regret_sum'] = float(diagnostic['output_regrets_at_selected'].sum())
                np.testing.assert_allclose(diagnostic['selected_regret_sum'],
                    diagnostic['regret']['raw'] + diagnostic['shared_selection_gap'],
                    rtol=0, atol=diagnostic['regret']['tolerance'])
                name = f'{period}_b{block}_prefix{prefix}'
                arrays = {key: value for key, value in candidates.items() if key != 'feature_counts'}
                arrays.update(flat_numeric(gains, 'gain__'))
                arrays.update(flat_numeric(diagnostic, 'diagnostic__'))
                arrays['training_origins'] = origins.asi8
                np.savez_compressed(args.out / 'contexts' / f'{name}.npz', **arrays)
                summary = {'context': name, 'period': period, 'head': block, 'leads': [start + 1, start + 24],
                           'prefix': prefix, 'origins': len(origins), 'training_start': str(origins.min()),
                           'training_end': str(origins.max()), 'cutoff': baseline.CUTOFFS[period], 'maturity_hours': 144,
                           'features': core.expanded_names(names, start, 24), 'model': model_receipt,
                           'metadata': identity(metadata_path), 'feature_cut_counts': candidates['feature_counts'],
                           'diagnostic': diagnostic, 'arrays': identity(args.out / 'contexts' / f'{name}.npz')}
                summary = json_safe(summary)
                core.write_json(args.out / 'contexts' / f'{name}.json', summary)
                result['contexts'].append(summary)
                unit += 1
                print(json.dumps({'stage': 'sharing_probe', 'period': period, 'head': block, 'prefix': prefix,
                                  'unit': unit, 'units': 18}), flush=True)
            del booster, x
    if unit != 18:
        raise ValueError('Incomplete fixed18-context probe')
    result['postflight_unique_receipts'] = validate_receipts(result)
    core.write_json(args.out / 'report.json', result)
    core.write_json(args.out / 'complete.json', {'status': 'complete', 'contexts': unit, 'no_fit': True,
                    'no_official_scoring': True, 'publish': 'not_required',
                    'publish_reason': 'Training-only diagnostic; no fitted model or submission candidate',
                    'postflight_unique_receipts': result['postflight_unique_receipts']})
    (args.out / 'README.md').write_text('# Frozen training-only output sharing probe\n\n' + SEMANTICS + '\n\n'
        + '\n'.join('- ' + limit for limit in LIMITS) + '\n\nreport.json contains all18 contexts; contexts/*.npz retain complete candidate statistics and diagnostics.\n')


if __name__ == '__main__':
    main()
