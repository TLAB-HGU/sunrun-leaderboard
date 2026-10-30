"""Export saved training traces, without calling fit/predict or reading targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def extract(root, output):
    import joblib
    sys.path.insert(0, str(root / 'leaderboard'))
    runs = root / 'research/expos/runs'
    source_manifest = root / 'research/forecast-fivepct-20261023/E1733-OFFICIAL-INPUT-MANIFEST.json'
    mapping = json.loads(source_manifest.read_text())['sources']['E1725']
    assert mapping['official_experiment'] == 'E1759'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_manifest, output / 'original-official-input-manifest.json')
    shutil.copyfile(runs / 'E1759/official_member_receipt.json', output / 'official_member_receipt.json')
    records = []
    for seed in range(3):
        parent = 'E1745' if seed < 2 else 'E1759'
        folder = runs / parent / 'E1725'
        model_path = folder / f'model_s{seed}.joblib'
        receipt_path = folder / f'receipt_s{seed}.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['passed'] and receipt['member'] == 'E1725'
        assert sha(model_path) == receipt['outputs'][model_path.name]
        if seed < 2:
            reuse = runs / 'E1759' / f'reuse_s{seed}.json'
            r = json.loads(reuse.read_text())
            assert r['passed'] and r['source_experiment'] == parent
            assert sha(receipt_path) == r['source_receipt_sha256']
            assert r['output_sha256'] == receipt['outputs'][f'predictions_official_s{seed}.parquet']
            shutil.copyfile(reuse, output / reuse.name)
        assert mapping['seeds'][str(seed)]['sha256'] == receipt['outputs'][f'predictions_official_s{seed}.parquet']
        model = joblib.load(model_path)
        trace = model.trace
        assert len(trace) == model.rounds == 300 and model.gamma == 2
        assert [x['round'] for x in trace] == list(range(300))
        assert all(x['after'] <= x['before'] + 1e-6 for x in trace)
        assert all(abs(a['after'] - b['before']) < 1e-6 for a, b in zip(trace, trace[1:]))
        rows = int(sum(model.group_counts))
        assert rows == receipt['training_rows']
        dump(output / f'training_trace_s{seed}.json', trace)
        with (output / f'training_trace_s{seed}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        shutil.copyfile(receipt_path, output / receipt_path.name)
        records.append({'seed': seed, 'fit_experiment': parent,
                        'official_member_experiment': 'E1759', 'family': 'E1725',
                        'model_sha256': sha(model_path), 'receipt_sha256': sha(receipt_path),
                        'trace_sha256': sha(output / f'training_trace_s{seed}.json'),
                        'training_rows': rows, 'rounds': len(trace),
                        'initial_objective': trace[0]['before'], 'final_objective': trace[-1]['after']})
    engine = root / 'leaderboard/experiments/solar_groupmean_anchor_20261023/engine.py'
    assert sha(engine) == receipt['initial_pins'][str(engine)]
    shutil.copyfile(engine, output / 'original_objective_engine.py')
    dump(output / 'provenance.json', {
        'lineage': 'E1789 -> E1769 -> E1725 family -> E1759 official member',
        'training_loss': '0.5 * (sum_i r_i^2 + (gamma-1) * sum_g n_g * mean(r_g)^2)',
        'gamma': 2, 'residual': 'prediction minus training residual target',
        'groups': '(source_fold, forecast lead)',
        'units': 'Unnormalized summed custom training objective; not MSE',
        'per_iteration_validation_loss': 'Not recorded; no validation curve is fabricated',
        'fit_or_predict_called': False, 'official_source_manifest_sha256': sha(source_manifest),
        'models': records,
    })


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    provenance = json.loads((output / 'provenance.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for record in provenance['models']:
        seed = record['seed']
        path = output / f'training_trace_s{seed}.json'
        assert sha(path) == record['trace_sha256']
        trace = json.loads(path.read_text())
        x = np.arange(301)
        y = np.array([trace[0]['before']] + [row['after'] for row in trace])
        axes[0].plot(x, y / record['training_rows'], label=f'Seed {seed}')
        axes[1].plot(x, y / y[0], label=f'Seed {seed}')
    axes[0].set(xlabel='Boosting rounds completed (0 = before fitting)',
                ylabel='Custom training objective / training rows',
                title='Saved objective trace (not MSE)')
    axes[1].set(xlabel='Boosting rounds completed', ylabel='Objective / initial objective',
                title='Relative decrease from original saved trace')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('E1769 constituent E1725 — actual official-fit training traces\n'
                 '300 rounds × 3 seeds; per-round validation loss was not recorded', fontsize=12)
    fig.savefig(output / 'training_loss.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['extract', 'plot'], required=True)
    args = parser.parse_args()
    if args.mode == 'extract':
        extract(args.root.resolve(), args.output)
    else:
        plot(args.output)
