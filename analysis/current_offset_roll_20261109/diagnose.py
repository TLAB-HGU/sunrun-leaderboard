"""Replay the frozen published mean; no fitting, forecast export or publication."""
from __future__ import annotations

import importlib.util
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STORE = 'tlabtlab/sunrun-lb-store'
SPACE = 'tlabtlab/sunrun-leaderboard'
NAME = 'e2000fs_calib_offset_roll'
LEAD_BINS = [(1, 6), (7, 24), (25, 48), (49, 72)]
SPEED_BINS = [(0, 400), (400, 550), (550, 700), (700, np.inf)]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


official = module('current_official', ROOT / 'leaderboard/scorer/competition.py')
kernel = module('current_kernel', ROOT / 'research/post_ablation_five_axes_20261108/analysis_kernel.py')
ensemble = module('current_ensemble', ROOT / 'research/seed_mean3_board_20261109/ensemble.py')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def download(repo, path, revision, kind='dataset'):
    return Path(hf_hub_download(repo, path, repo_type=kind, revision=revision))


def normalized(frame):
    frame = frame[official.KEY + ['pred_kms']].copy()
    frame[official.KEY[0]] = pd.to_datetime(frame[official.KEY[0]], utc=True)
    if frame.duplicated(official.KEY).any():
        raise ValueError('Duplicate prediction keys')
    return frame.sort_values(official.KEY).reset_index(drop=True)


def from_array(origins, pred):
    return pd.DataFrame({official.KEY[0]: np.repeat(origins, 72),
                         official.KEY[1]: np.tile(np.arange(1, 73), len(origins)),
                         'pred_kms': pred.ravel()})


def energy_score(mse_by_lead, timse_by_lead):
    return float(np.mean((mse_by_lead * timse_by_lead) ** .25))


def decomposition(pred, truth, mse_weights, timse_weights, mask):
    valid = np.isfinite(truth) & np.broadcast_to(mask, truth.shape)
    error = (pred - truth)[valid]
    squared = np.where(valid, (pred - truth) ** 2, 0.)
    mse = (squared * mse_weights).sum(axis=0)
    timse = (squared * timse_weights).sum(axis=0)
    return {'n_pairs': int(valid.sum()), 'bias': float(error.mean()) if error.size else None,
                'mae': float(np.abs(error).mean()) if error.size else None,
                'pooled_rmse': float(np.sqrt(np.mean(error ** 2))) if error.size else None,
                'mse_contribution': float(mse.mean()), 'timse_contribution': float(timse.mean())}


def analyze(pred, context, score):
    truth = context['truth']
    mw, tw = kernel.metric_weights(truth, np.arange(len(truth)))
    sq = np.where(np.isfinite(truth), (pred - truth) ** 2, 0.)
    mse, timse = (sq * mw).sum(0), (sq * tw).sum(0)
    np.testing.assert_allclose(energy_score(mse, timse), score['total']['s'], atol=1e-8, rtol=0)
    masks = {}
    for low, high in SPEED_BINS:
        label = f'speed_{low}_{high}'
        masks[label] = (truth >= low) & (truth < high)
    speed_names = list(masks)
    for low, high in LEAD_BINS:
        masks[f'lead_{low}_{high}'] = (official.LEADS >= low) & (official.LEADS <= high)
    lead_names = [n for n in masks if n.startswith('lead_')]
    for period in ('F1', 'F2', 'F3'):
        masks[period] = (context['period'] == period)[:, None]
    groups = {name: decomposition(pred, truth, mw, tw, mask) for name, mask in masks.items()}
    for names in (speed_names, lead_names, ['F1', 'F2', 'F3']):
        for field, expected in [('mse_contribution', mse.mean()), ('timse_contribution', timse.mean())]:
            np.testing.assert_allclose(sum(groups[n][field] for n in names), expected, atol=1e-8, rtol=0)
    cross = {}
    for period in ('F1', 'F2', 'F3'):
        for lead in lead_names:
            for speed in speed_names:
                cross[f'{period}/{lead}/{speed}'] = decomposition(
                    pred, truth, mw, tw, masks[period] & masks[lead] & masks[speed])
    for field, expected in [('mse_contribution', mse.mean()), ('timse_contribution', timse.mean())]:
        np.testing.assert_allclose(sum(row[field] for row in cross.values()), expected, atol=1e-8, rtol=0)
    for name, mask in masks.items():
        kept = np.where(mask, 0., sq)
        floor = energy_score((kept * mw).sum(0), (kept * tw).sum(0))
        groups[name]['restricted_error_removal_s'] = floor
        groups[name]['conditional_s_reduction_pct'] = 100 * (1 - floor / score['total']['s'])
        groups[name]['mse_fraction'] = groups[name]['mse_contribution'] / float(mse.mean())
        groups[name]['timse_fraction'] = groups[name]['timse_contribution'] / float(timse.mean())
    # Same valid support for every tested shift; these are not fitted corrections.
    timing = {}
    for period, rows in [('total', np.ones(len(truth), bool))] + [
            (p, context['period'] == p) for p in ('F1', 'F2', 'F3')]:
        y = truth[rows, 12:60]
        timing[period] = {str(shift): float(np.sqrt(np.nanmean((pred[rows, 12+shift:60+shift] - y)**2)))
                          for shift in (-12, -6, 0, 6, 12)}
    adequate = np.isfinite(truth).sum(1) >= 48
    y = truth[adequate]
    p = np.where(np.isfinite(y), pred[adequate], np.nan)
    yrange, prange = np.nanmax(y, axis=1)-np.nanmin(y, axis=1), np.nanmax(p, axis=1)-np.nanmin(p, axis=1)
    nonflat = yrange > 0
    amplitude = {'n_origins': int(adequate.sum()), 'median_observed_range': float(np.median(yrange)),
                     'median_predicted_range': float(np.median(prange)),
                     'median_range_ratio': float(np.median(prange[nonflat]/yrange[nonflat])),
                     'median_absolute_peak_timing_h': float(np.median(np.abs(np.nanargmax(p, axis=1)-np.nanargmax(y, axis=1))))}
    targets = context['origins'].asi8[:, None] + official.LEADS[None, :] * 3600 * 10**9
    table = pd.DataFrame({'target_ns': targets.ravel(), 'truth': truth.ravel(),
                             'mse_contribution': (sq*mw/72).ravel(), 'timse_contribution': (sq*tw/72).ravel()})
    table = table[np.isfinite(table.truth)].groupby('target_ns').agg(
        truth=('truth', 'first'), forecast_pairs=('truth', 'size'),
        mse_contribution=('mse_contribution', 'sum'), timse_contribution=('timse_contribution', 'sum'))
    table['timestamp_utc'] = pd.to_datetime(table.index, utc=True)
    table = table.sort_values('timse_contribution', ascending=False)
    concentration = {str(k): float(table.timse_contribution.iloc[:k].sum()/timse.mean()) for k in (1, 5, 10, 24)}
    np.testing.assert_allclose(table.timse_contribution.sum(), timse.mean(), atol=1e-8, rtol=0)
    return {'groups': groups, 'cross': cross, 'amplitude': amplitude, 'fixed_shift_pooled_rmse': timing,
                'top_unique_target_timse_fractions': concentration,
                'mean_mse': float(mse.mean()), 'mean_timse': float(timse.mean())}, table


def main():
    api = HfApi()
    lock = OUT / 'snapshot.json'
    if not lock.exists():
        save('snapshot.json', {'store': STORE, 'space': SPACE,
             'store_revision': api.repo_info(STORE, repo_type='dataset').sha,
             'space_revision': api.repo_info(SPACE, repo_type='space').sha,
             'frozen_utc': datetime.now(UTC).isoformat()})
    snapshot = json.loads(lock.read_text())
    html = download(SPACE, 'index.html', snapshot['space_revision'], 'space').read_text()
    rows = [r for r in re.findall(r'<tr\b.*?</tr>', html, re.DOTALL)
            if re.search(r'<td[^>]*>\s*'+re.escape(NAME)+r'\s*</td>', r)]
    if len(rows) != 1:
        raise ValueError('Expected one exact visible row')
    ids = re.findall(re.escape(NAME)+r'--([a-f0-9]{64})', rows[0])
    if len(set(ids)) != 1:
        raise ValueError('Expected unique visible configuration identity')
    record_path = f'results/seongeun/{NAME}/{ids[0]}.json'
    record = json.loads(download(STORE, record_path, snapshot['store_revision']).read_text())
    save('submission_record.json', record)
    pred_path = 'submissions/'+record_path[len('results/'):-5]+'.parquet'
    stored_path = download(STORE, pred_path, snapshot['store_revision'])
    stored = normalized(pd.read_parquet(stored_path))
    obs_path = download(STORE, 'competition-observations.parquet', snapshot['store_revision'])
    context = official.prepare_context(official.load_observations(obs_path))
    # Also prove the observation table has not changed since original submission.
    historical = download(STORE, 'competition-observations.parquet', record['competition_dataset_revision'])
    pd.testing.assert_frame_equal(pd.read_parquet(obs_path), pd.read_parquet(historical))
    frames, no_offset_frames, scores, arrays = [], [], {}, {}
    input_paths = [Path(__file__), ROOT/'leaderboard/scorer/competition.py',
                   ROOT/'research/post_ablation_five_axes_20261108/analysis_kernel.py',
                   ROOT/'research/seed_mean3_board_20261109/ensemble.py', stored_path, obs_path, historical]
    for seed, run in zip((42, 43, 44), record['config']['data']['constituent_runs'], strict=True):
        if not run.endswith(f'_s{seed}'):
            raise ValueError('Constituent order mismatch')
        pieces = []
        for period in ('F1', 'F2', 'F3'):
            path = ROOT/run/f'{period}_predictions.npz'
            with np.load(path) as z:
                pieces.append(from_array(pd.to_datetime(z['origins'], utc=True), z['pred']))
            input_paths.append(path)
        frame = pd.concat(pieces, ignore_index=True)
        frames.append(frame)
        no_offset_parts = []
        for period in ('F1', 'F2', 'F3'):
            path = ROOT/run/f'{period}_no_offset_predictions.npz'
            with np.load(path) as z:
                no_offset_parts.append(from_array(pd.to_datetime(z['origins'], utc=True), z['pred']))
            input_paths.append(path)
        no_offset_frames.append(pd.concat(no_offset_parts, ignore_index=True))
        input_paths.extend(ROOT/run/p for p in ('experiment.py', 'baseline.py', 'core.py'))
        arrays[str(seed)] = official.align_predictions(frame, context)
        scores[str(seed)] = official.score_competition(frame, context)
    mean, _ = ensemble.average_predictions(frames)
    pd.testing.assert_frame_equal(normalized(mean)[official.KEY], stored[official.KEY])
    np.testing.assert_array_equal(normalized(mean).pred_kms.to_numpy(), stored.pred_kms.to_numpy())
    arrays['published_mean3'] = official.align_predictions(stored, context)
    scores['published_mean3'] = official.score_competition(stored, context)
    for scope in ('total', 'f1', 'f2', 'f3'):
        actual, expected = scores['published_mean3'][scope], record['competition'][scope]
        for key in ('s', 'timse', 'rmse', 'trmse', 'bias', 'n_pairs', 'n_replaced', 'replacement_fraction', 'n_origins'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-8, rtol=0)
        pd.testing.assert_frame_equal(pd.DataFrame(actual['by_lead']), pd.DataFrame(expected['by_lead']),
                                      check_exact=False, atol=1e-8, rtol=0)
    if scores['published_mean3']['version'] != record['competition']['version']:
        raise ValueError('Scorer version mismatch')
    results = {}
    for name, arr in arrays.items():
        if not (np.isfinite(arr) & (arr > 0)).all() or scores[name]['total']['n_replaced']:
            raise ValueError('Unexpected prediction replacement')
        results[name], table = analyze(arr, context, scores[name])
        if name == 'published_mean3':
            table.to_csv(OUT/'target_contributions.csv', index=False)
        for scope, pop in context['scopes'].items():
            y = context['truth'][pop['selected']]
            sq = np.where(np.isfinite(y), (arr[pop['selected']]-y)**2, 0.)
            metrics = kernel.scores_from_weights(sq[None], *kernel.metric_weights(y, np.arange(len(y))))
            for key, value in metrics.items():
                np.testing.assert_allclose(value[0], scores[name][scope][key], atol=1e-8, rtol=0)
    no_offset_mean, _ = ensemble.average_predictions(no_offset_frames)
    offset = {}
    for name, frame in list(zip(('42', '43', '44'), no_offset_frames, strict=True)) + [('published_mean3', no_offset_mean)]:
        no_score = official.score_competition(frame, context)
        no_pred = official.align_predictions(frame, context)
        if no_score['total']['n_replaced']:
            raise ValueError('No-offset diagnostic has replacements')
        no_diagnostic, _ = analyze(no_pred, context, no_score)
        offset[name] = {'no_offset_scores': no_score,
            'with_minus_without_s': {scope: scores[name][scope]['s']-no_score[scope]['s']
                                  for scope in ('total', 'f1', 'f2', 'f3')},
            'no_offset_speed_groups': {k: v for k, v in no_diagnostic['groups'].items() if k.startswith('speed_')}}
    save('offset_effect.json', offset)
    save('official_scores.json', scores)
    save('diagnostics.json', results)
    save('verification.json', {'score_parity_atol': 1e-8, 'all_scopes_and_leads': True,
         'full_key_equal_mean_exact': True, 'observations_equal_submission_revision': True,
         'energy_partitions_reproduce': True, 'n_origins': len(context['origins']),
         'n_valid_pairs': int(np.isfinite(context['truth']).sum()), 'no_replacements': True,
         'no_training': True, 'no_fitted_parameters': True, 'no_forecast_export': True, 'no_publication': True,
         'space_row_html': rows[0], 'record_path': record_path, 'prediction_path': pred_path,
         'warning': 'Fixed-sample descriptive analysis; restricted removal S is not attainable model performance.'})
    save('provenance.json', {'snapshot': snapshot, 'commit': subprocess.check_output(
         ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'python': platform.python_version(),
         'numpy': np.__version__, 'pandas': pd.__version__, 'inputs': [{'path': str(p), 'bytes': p.stat().st_size,
         'mtime_ns': p.stat().st_mtime_ns} for p in input_paths], 'observed_output_utc': datetime.now(UTC).isoformat()})
    print(json.dumps({'stage': 'complete', 'scores': {n: s['total']['s'] for n, s in scores.items()},
                         'concentration': results['published_mean3']['top_unique_target_timse_fractions']}))


if __name__ == '__main__':
    main()
