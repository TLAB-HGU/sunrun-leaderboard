"""Training-only common-support description and one fixed legacy weight audit."""
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
MIN_ORIGINS = 30
LIMITS = [
    'Fixed seed42 full900 raw predictions, observed training targets only for bias decomposition.',
    'Reference and calibration rows were both fitted by the full model; all results are in-sample.',
    'Coarse X-only common support does not identify full P(Y|X), conditional mapping drift, or causality.',
    'Weights are separately normalized over common-support observed pairs for each time group.',
    'Distinct-origin support is counted, but overlapping origins and target pairs are dependent.',
    'No tests, confidence intervals, new folds, fits, query scores, corrections, or parameter selection.',
    'Fixed legacy half-life180 and floor0.1 are audited, not proposed as an empirically selected policy.',
    'Recency multiplies original stage-specific tail-normalized point weights; no second tail normalization.',
    'Hessian masses describe the existing diagonal surrogate scaled by recency, not spectral cell losses.',
    'Raw input missingness is a proxy; availability and immutable historical provenance remain limited.',
]


def load_fit_probe():
    spec = importlib.util.spec_from_file_location('support_fit_probe', ROOT / 'training_fit_probe.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def align_raw_inputs(rows, origins):
    rows = rows.copy()
    rows['origin'] = pd.to_datetime(rows.origin, utc=True)
    if (rows.origin.duplicated().any() or not rows.h.eq(1).all()
            or len(rows) != len(origins) or not origins.isin(rows.origin).all()):
        raise ValueError('Raw input keys do not exactly match eligible training origins/h1')
    return rows.set_index('origin').reindex(origins)


def speed_bin(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), np.digitize(values, [400., 550., 700.]), 4).astype(int)


def strata_codes(v, recurrence, ch, missing_v, missing_age):
    if (np.isfinite(ch) & (ch < 0)).any():
        raise ValueError('Negative coronal-hole area outside fixed zero/positive/missing partition')
    ch_code = np.where(np.isfinite(ch), (ch > 0).astype(int), 2)
    base = speed_bin(v)[:, None] * 5 + speed_bin(recurrence)
    with_ch = base * 3 + ch_code[:, None]
    with_missing = (with_ch * 2 + missing_v[:, None].astype(int)) * 2 + missing_age[:, None].astype(int)
    return {'speed_recurrence': base, 'plus_ch': with_ch, 'plus_raw_missingness': with_missing}


def decomposition(error, observed, codes, older, recent):
    """Exact recent-minus-older bias identity on fixed common-support strata."""
    tables = []
    for group, indices in [('older', older), ('recent', recent)]:
        indices = np.asarray(indices)
        row, lead = np.nonzero(observed[indices])
        tables.append(pd.DataFrame({'group': group, 'stratum': codes[indices[row], lead],
                                    'origin': indices[row], 'error': error[indices[row], lead]}))
    table = pd.concat(tables, ignore_index=True)
    if not np.isfinite(table.error).all():
        raise ValueError('Nonfinite observed residual')
    summary = table.groupby(['stratum', 'group']).agg(pairs=('error', 'size'),
                  origins=('origin', 'nunique'), bias=('error', 'mean'))
    records, common = [], []
    for code in sorted(table.stratum.unique()):
        record = {'stratum_code': int(code)}
        for group in ('older', 'recent'):
            if (code, group) in summary.index:
                row = summary.loc[(code, group)]
                record[group] = {'pairs': int(row.pairs), 'origins': int(row.origins), 'bias': float(row.bias)}
            else:
                record[group] = {'pairs': 0, 'origins': 0, 'bias': None}
        record['common_support'] = all(record[g]['origins'] >= MIN_ORIGINS for g in ('older', 'recent'))
        records.append(record)
        if record['common_support']:
            common.append(record)
    totals = {g: int((table.group == g).sum()) for g in ('older', 'recent')}
    supported = {g: sum(r[g]['pairs'] for r in common) for g in totals}
    coverage = {}
    selected_codes = [r['stratum_code'] for r in common]
    for group, total in totals.items():
        all_rows = table[table.group == group]
        kept = all_rows[all_rows.stratum.isin(selected_codes)]
        coverage[group] = {'observed_pairs': total, 'supported_pairs': supported[group],
                           'pair_fraction': supported[group] / total if total else None,
                           'observed_origins': int(all_rows.origin.nunique()),
                           'supported_origins': int(kept.origin.nunique()),
                           'unrestricted_observed_bias': float(all_rows.error.mean()) if len(all_rows) else None}
    result = {'minimum_distinct_origins_per_group_stratum': MIN_ORIGINS,
              'coverage': coverage, 'strata': records, 'common_strata': len(common), 'decomposition': None}
    if not common:
        return result
    biases = {g: 0. for g in totals}
    within = composition = 0.
    for record in common:
        wo, wr = (record[g]['pairs'] / supported[g] for g in ('older', 'recent'))
        bo, br = (record[g]['bias'] for g in ('older', 'recent'))
        record.update(older_common_pair_weight=wo, recent_common_pair_weight=wr,
                      within=wr * (br - bo), composition=(wr - wo) * bo)
        biases['older'] += wo * bo
        biases['recent'] += wr * br
        within += record['within']
        composition += record['composition']
    delta = biases['recent'] - biases['older']
    np.testing.assert_allclose(within + composition, delta, atol=1e-10, rtol=1e-12)
    result['decomposition'] = {'older_bias': biases['older'], 'recent_bias': biases['recent'],
                               'recent_minus_older_bias': delta, 'within': within, 'composition': composition,
                               'identity_residual': within + composition - delta}
    return result


def describe(values):
    values = np.asarray(values, float)
    return {'count': int(values.size), 'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'quantiles_0_25_50_75_100': np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values.size else []}


def ess(values):
    values = np.asarray(values, float)
    return float(values.sum() ** 2 / np.sum(values ** 2)) if values.size and np.any(values) else None


def weight_audit(origins, target, observed):
    age = (origins.max() - origins).total_seconds().to_numpy() / 86400.
    raw = .1 + .9 * np.exp(-np.log(2.) * age / 180.)
    recency = raw / raw.mean()
    tail = np.where(target >= 550., 4., 1.)
    point = tail / tail.mean(axis=0)
    weighted_point = recency[:, None] * point
    hessian = point + .1
    weighted_hessian = recency[:, None] * hessian
    classes = {'all': np.ones_like(observed), 'ge550': target >= 550., 'ge700': target >= 700.,
               'observed': observed, 'imputed': ~observed,
               'old_ge550': (age[:, None] >= 180) & (target >= 550.),
               'old_ge700': (age[:, None] >= 180) & (target >= 700.)}
    result = {'formula': 'raw=0.1+0.9*exp(-ln(2)*age_days/180); normalized=raw/mean(raw)',
              'age_anchor': str(origins.max()), 'old_definition': 'age_days>=180, fixed half-life boundary',
              'age_days': describe(age), 'raw': describe(raw), 'normalized': describe(recency),
              'row_ess': ess(recency), 'row_count': len(origins), 'classes': {}}
    repeated_recency = np.broadcast_to(recency[:, None], target.shape)
    for name, mask in classes.items():
        rows = mask.any(axis=1)
        result['classes'][name] = {
            'origin_count_any_qualifying_lead': int(rows.sum()), 'pair_count': int(mask.sum()),
            'row_ess_recency_any_qualifying_lead': ess(recency[rows]),
            'flattened_pair_ess_recency_not_independent': ess(repeated_recency[mask]),
            'flattened_pair_ess_combined_point_not_independent': ess(weighted_point[mask]),
            'raw_row_weights': describe(raw[rows]), 'normalized_row_weights': describe(recency[rows]),
            'baseline_point_mass_per_lead': (point * mask).sum(axis=0).tolist(),
            'recency_point_mass_per_lead': (weighted_point * mask).sum(axis=0).tolist(),
            'baseline_hessian_mass_per_lead': (hessian * mask).sum(axis=0).tolist(),
            'recency_hessian_mass_per_lead': (weighted_hessian * mask).sum(axis=0).tolist(),
            'baseline_point_mass_fraction': float(point[mask].sum() / point.sum()),
            'recency_point_mass_fraction': float(weighted_point[mask].sum() / weighted_point.sum())}
    return result


def main(argv=None):
    fit = load_fit_probe()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=fit.INPUT / 'df_ffill_trainy.parquet')
    parser.add_argument('--original-labels', type=Path, default=fit.INPUT / 'df_ffill.parquet')
    parser.add_argument('--raw-inputs', type=Path, default=REPO / 'data-v2/input/df.parquet')
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU1 inference requires CUDA_VISIBLE_DEVICES=""')
    runner = fit.load_runner()
    baseline, core = runner.parent_modules()
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    names = runner.assert_parent_contract(core, config, 42)
    sources = {}
    for name, current in [('core.py', Path(core.__file__)), ('baseline.py', Path(baseline.__file__))]:
        saved = fit.RUN / name
        if current.read_bytes() != saved.read_bytes():
            raise ValueError('Current parent source differs from saved source')
        sources[name] = {'current': core.file_identity(current), 'saved': core.file_identity(saved), 'equal_bytes': True}
    output = {'limits': LIMITS, 'seed': 42, 'rounds': 900, 'minimum_origins': MIN_ORIGINS,
              'source_receipts': sources, 'script': core.file_identity(__file__),
              'input_receipts': {name: core.file_identity(path) for name, path in
                                 [('prepared', args.data), ('original_labels', args.original_labels), ('raw_inputs', args.raw_inputs)]},
              'stratum_codebook': {'speed': ['lt400', '400_550', '550_700', 'ge700', 'missing'],
                                   'ch': ['zero', 'positive', 'missing'],
                                   'speed_recurrence': 'v_bin*5+rec_mean_bin',
                                   'plus_ch': 'speed_recurrence*3+ch_bin',
                                   'plus_raw_missingness': '(plus_ch*2+raw_v_missing)*2+raw_img_age_missing'},
              'periods': {}}
    for period in ('F1', 'F2', 'F3'):
        meta = json.loads((fit.RUN / f'{period}_metadata.json').read_text())
        if core.normalize_config(meta['config']) != config or meta['rounds'] != 900:
            raise ValueError('Parent model contract differs')
        if any(core.file_identity(args.data)[k] != meta['identity']['data'][k] for k in ('bytes', 'mtime_ns')):
            raise ValueError('Training data receipt differs')
        train, target = baseline.load_grid(args.data, period, names, training=True)
        origins = train['origins']
        if (len(origins) != meta['train_origins'] or str(origins.min()) != meta['train_start']
                or str(origins.max()) != meta['train_end'] or meta['maturity_hours'] != 144):
            raise ValueError('Training origin contract differs')
        labels = pd.read_parquet(args.original_labels, columns=['origin', 'h', 'y'],
                                 filters=[(f'fit_{period}', '==', True)], use_threads=False)
        observed = fit.observed_mask(labels, origins, target, baseline.CUTOFFS[period])
        del labels
        raw = pd.read_parquet(args.raw_inputs, columns=['origin', 'h', 'v_lag0', 'img_age_h'],
                              filters=[(f'fit_{period}', '==', True), ('h', '==', 1)], use_threads=False)
        raw = align_raw_inputs(raw, origins)
        reference, calibration, boundary = baseline.prior_split(origins)
        for label, indices in [('reference', reference), ('calibration', calibration)]:
            frozen = meta['frozen_state']
            if (len(indices) != frozen[f'{label}_rows'] or str(origins[indices].min()) != frozen[f'{label}_start']
                    or str(origins[indices].max()) != frozen[f'{label}_end']):
                raise ValueError('Original temporal subsets differ')
        report = {'cutoff': baseline.CUTOFFS[period], 'split_boundary': str(boundary),
                  'reference_origins': len(reference), 'calibration_origins': len(calibration),
                  'weight_audits': {}, 'heads': []}
        for stage, indices in [('reference', reference), ('full', np.arange(len(origins)))]:
            report['weight_audits'][stage] = weight_audit(origins[indices], target[indices], observed[indices])
        v = train['C'][:, names['constant'].index('v_lag0')]
        ch = train['C'][:, names['constant'].index('ch_ch_t035_meridian15_equatorial_area')]
        for block, start in enumerate((0, 24, 48)):
            recurrence = train['E'][:, start:start + 24, names['varying'].index('rec_mean')]
            views = strata_codes(v, recurrence, ch, raw.v_lag0.isna().to_numpy(), raw.img_age_h.isna().to_numpy())
            path = fit.RUN / f'{period}_models/full/b{block}.ubj'
            model = xgb.Booster(params={'nthread': 1, 'device': 'cpu'})
            model.load_model(path)
            model.set_param({'nthread': 1, 'device': 'cpu'})
            if model.num_boosted_rounds() != 900 or model.num_features() != 212:
                raise ValueError('Saved full900 architecture differs')
            prediction = model.inplace_predict(core.design(train, start, 24), iteration_range=(0, 900))
            error = prediction.astype(float) - target[:, start:start + 24]
            if not np.isfinite(error).all():
                raise ValueError('Invalid raw full900 predictions')
            report['heads'].append({'block': block, 'leads': [start + 1, start + 24], 'model': core.file_identity(path),
                                    'views': {name: decomposition(error, observed[:, start:start + 24], code, reference, calibration)
                                              for name, code in views.items()}})
            print(json.dumps({'period': period, 'block': block, 'stage': 'training_support_inference'}), flush=True)
        output['periods'][period] = report
    args.out.mkdir(parents=True, exist_ok=True)
    core.write_json(args.out / 'temporal_support_probe.json', output)
    lines = ['# Training-only temporal common support', '', *[f'- {limit}' for limit in LIMITS], '',
             '| Period | Head | View | Older coverage | Recent coverage | Within | Composition |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for period, report in output['periods'].items():
        for head in report['heads']:
            for view, result in head['views'].items():
                parts = result['decomposition'] or {'within': None, 'composition': None}
                lines.append(f"| {period} | {head['block']} | {view} | {result['coverage']['older']['pair_fraction']} | "
                             f"{result['coverage']['recent']['pair_fraction']} | {parts['within']} | {parts['composition']} |")
    lines += ['', 'Full per-stratum counts, bias decomposition, and fixed stage-specific recency mass/ESS audits are in the JSON.',
              'Bias is raw prediction minus observed training target. No conclusion about stale conditional mapping follows from marginals alone.']
    (args.out / 'temporal_support_probe.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
