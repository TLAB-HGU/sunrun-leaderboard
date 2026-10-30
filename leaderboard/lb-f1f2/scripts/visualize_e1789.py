#!/usr/bin/env python3
"""Render the submitted E1789 grid with the established monthly GIF renderer.

Evaluation diagnostics are from the official/test grid, not training losses.
E1789 is a fixed forecast combination with no training iterations.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

import visualize_enlil_validation as renderer

renderer.EXPERIMENTS['E1789'] = {
    'model': 'e1789_core5_prediction_mean',
    'title': 'E1789 Core5 (3-seed prediction mean)',
    'model_config_sha256': '0daed7e8ff871de91e68c48983ad5031c2e27dd6894bfb6e8dfd757247675503',
    'expected_mse': 3177.232883758485,
    'seeds': [0, 1, 2],
    'aggregation': 'equal prediction mean of original Core5 seed forecasts',
}


def diagnostics(output):
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = manifest['sources']
    pred = pd.read_parquet(sources['predictions']['file'])
    truth = pd.read_parquet(sources['truth']['file'])
    for data in (pred, truth):
        data[renderer.ORIGIN] = pd.to_datetime(data[renderer.ORIGIN], utc=True)
    data = pred.merge(truth, on=renderer.KEY, validate='one_to_one').sort_values(renderer.KEY)
    data['squared_error'] = (data.pred_kms - data.target_kms) ** 2
    overall = float(data.squared_error.mean())
    renderer.require(abs(overall - renderer.EXPERIMENTS['E1789']['expected_mse']) < 1e-7,
                     'Official/test score differs from published score')
    per_horizon = data.groupby('horizon_hours').squared_error.mean()
    per_origin = data.groupby(renderer.ORIGIN).squared_error.mean()
    per_horizon.rename('official_test_mse_kms2').to_csv(output / 'evaluation_mse_by_horizon.csv')
    per_origin.rename('official_test_mse_kms2').to_csv(output / 'evaluation_mse_by_origin.csv')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(per_horizon.index, per_horizon.values, color='#df8300', linewidth=2)
    axes[0].set(xlabel='Forecast lead (hours)', ylabel='MSE ((km/s)^2)', xlim=(1, 72),
                title='Official/test MSE by forecast horizon')
    axes[1].plot(per_origin.index, per_origin.values, color='#346a99', linewidth=.6, alpha=.7,
                 label='Each origin (72-hour forecast)')
    axes[1].plot(per_origin.index, per_origin.rolling(24, min_periods=24).mean(),
                 color='#b14f00', linewidth=1.3, label='Trailing 24-origin mean (display only)')
    axes[1].set(xlabel='Forecast origin (UTC)', ylabel='MSE ((km/s)^2)',
                title='Official/test MSE through the evaluation period')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(overall, color='#666666', linestyle='--', linewidth=1)
        ax.grid(alpha=.2)
    fig.suptitle(f'E1789 Core5 — official/test error diagnostics | MSE {overall:.6f}\n'
                 'Not train/validation learning curves: fixed combination, no fitting', fontsize=12)
    fig.savefig(output / 'evaluation_error.png', dpi=140)
    plt.close(fig)
    audit = {
        'experiment': 'E1789', 'train_validation_learning_curve': 'not_applicable',
        'reason': 'Fixed combination of stored forecasts; no fit, optimizer or epochs in E1789.',
        'constituents': ['E1604-public', 'E1593', 'E1595', 'E0472', 'E1769'],
        'constituent_weights': [2, 1, 2, 1, 1],
        'constituent_iteration_loss_histories': 'Not found in inspected original run artifacts.',
        'retraining_performed': False, 'evaluation_scope': 'Previously exposed official/test period only',
        'prediction_aggregation': 'Mean of original seed 0/1/2 forecasts',
        'mse': overall, 'origins': len(per_origin), 'horizons': len(per_horizon),
        'pairs': len(data), 'truth_missing_policy': 'Original scorer forward-filled targets included',
        'display_smoothing': 'Trailing 24-origin arithmetic mean only; never applied to forecasts',
    }
    (output / 'loss_history_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for name in ['evaluation_error.png', 'evaluation_mse_by_horizon.csv',
                 'evaluation_mse_by_origin.csv', 'loss_history_audit.json']:
        path = output / name
        manifest.setdefault('diagnostic_outputs', []).append(
            {'file': name, 'sha256': renderer.sha256(path), 'bytes': path.stat().st_size})
    manifest['training_loss_status'] = 'Not applicable; E1789 performs no training.'
    manifest['entrypoint_script_sha256'] = renderer.sha256(__file__)
    manifest['notes'][0] = 'Displayed history is 168 hours of observations, not the full input history of all five constituent models.'
    manifest['evaluation_scope'] = 'Historical official/test comparison; not training validation or current-contract certification'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    renderer.main()
    import sys
    index = sys.argv.index('--output-dir')
    output = Path(sys.argv[index + 1])
    diagnostics(output)
