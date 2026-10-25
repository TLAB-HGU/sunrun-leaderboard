"""Plot measured calibration curves; no fitting or selection logic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.calibration / 'selection.json').read_text())
    selected = selection['selected_n_estimators']
    rows = []
    for path in sorted((args.calibration / 'checkpoints').glob('*/curve.json')):
        rows.append(json.loads(path.read_text()))
    assert len(rows) == 9
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for row in rows:
        head = [(1, 6), (7, 24), (25, 72)].index(tuple(row['band']))
        ax = axes[row['seed'], head]
        x = np.arange(1, len(row['train_rmse']) + 1)
        ax.plot(x, np.square(row['train_rmse']), color='#1764a0', label='Train MSE')
        ax.plot(x, np.square(row['valid_rmse']), color='#d96b12', label='Inner validation MSE')
        ax.axvline(selected, color='#555555', linestyle='--', linewidth=1,
                   label=f'Frozen common budget = {selected}')
        ax.set(title=f"Seed {row['seed']}, prediction h{row['band'][0]}–{row['band'][1]}",
               xlabel='Boosting rounds', ylabel='MSE ((km/s)^2)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Actual train / inner-validation histories: 9 fixed-budget direct XGBoost fits\n'
                 'Train targets before 2025-02-26; inner validation origins March–April 2025', fontsize=13)
    fig.savefig(args.output / 'inner_train_validation_by_head.png', dpi=130)
    plt.close(fig)
    data = pd.read_csv(args.calibration / 'pooled_curves.csv')
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(data['round'], data.train_mse_descriptive, color='#1764a0', label='Training-pool MSE (descriptive)')
    ax.plot(data['round'], data.validation_mean_seed_mse, color='#d96b12', label='Inner validation MSE (selection metric)')
    for seed in range(3):
        ax.plot(data['round'], data[f'validation_seed{seed}_mse'], color='#d96b12', alpha=.16, linewidth=.7)
    ax.axvline(selected, color='#444444', linestyle='--', label=f'Frozen budget = {selected}')
    ax.set(title='Preregistered common-round selection: pooled SSE/count, then mean across seeds',
           xlabel='Boosting rounds', ylabel='MSE ((km/s)^2)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(args.output / 'inner_common_round_selection.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'selected_round': selected, 'plotted_head_seed_curves': len(rows), 'smoothing': False}))


if __name__ == '__main__':
    main()
