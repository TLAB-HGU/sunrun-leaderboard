#!/usr/bin/env python3
"""Plot already computed train/validation MSE on a common stage axis."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob('curves_s*.csv'))
    if len(files) != 3:
        raise ValueError('Exactly three seed curve files required')
    frames = [pd.read_csv(path) for path in files]
    for data in frames:
        if not np.array_equal(data['round'], np.arange(301)):
            raise ValueError('Expected stages 0..300 without missing rounds')
        if not np.isfinite(data[['train_mse', 'validation_mse']]).all().all():
            raise ValueError('Missing or non-finite train/validation losses')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    train = np.stack([data.train_mse for data in frames])
    valid = np.stack([data.validation_mse for data in frames])
    for i, (ax, data) in enumerate(zip(axes.flat, frames)):
        ax.plot(data['round'], data.train_mse, color='#1764a0', label='Train MSE')
        ax.plot(data['round'], data.validation_mse, color='#d96b12', label='Validation MSE')
        ax.set_title(f'Seed {i}: same frozen model, same boosting stage')
    ax = axes.flat[3]
    ax.plot(frames[0]['round'], train.mean(0), color='#1764a0', label='Mean train MSE')
    ax.plot(frames[0]['round'], valid.mean(0), color='#d96b12', label='Mean validation MSE')
    ax.fill_between(frames[0]['round'], train.min(0), train.max(0), color='#1764a0', alpha=.12)
    ax.fill_between(frames[0]['round'], valid.min(0), valid.max(0), color='#d96b12', alpha=.12)
    ax.set_title('Mean of per-seed MSE (band = seed min..max)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set(xlabel='Boosting rounds completed', ylabel='Raw residual-head MSE ((km/s)^2)')
    fig.suptitle(f'{args.model} — train and validation loss together\n'
                 'Retrospective frozen-model staged replay; historical fold 3, leads 25–72 h', fontsize=13)
    fig.savefig(args.directory / 'train_validation_loss.png', dpi=150)
    plt.close(fig)
    pd.DataFrame({'round': frames[0]['round'], 'train_mse': train.mean(0),
                  'validation_mse': valid.mean(0)}).to_csv(args.directory / 'curves_mean.csv', index=False)
    result = {'model': args.model, 'type': 'Retrospective frozen-model staged reevaluation, not original logged curves',
              'same_metric': 'Raw residual-head MSE for train and validation',
              'plot_sha256': hashlib.sha256((args.directory / 'train_validation_loss.png').read_bytes()).hexdigest(),
              'final_mean_train_mse': float(train.mean(0)[-1]),
              'final_mean_validation_mse': float(valid.mean(0)[-1]),
              'seed_curves': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (args.directory / 'plot_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
