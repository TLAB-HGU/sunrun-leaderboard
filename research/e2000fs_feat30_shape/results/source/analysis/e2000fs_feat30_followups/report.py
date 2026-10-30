"""Per-lane static report. Reads completed inference; does not select candidates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'e2000fs_dirmo_24_feat30/logs/full_01/artifacts'
CASES = ['2026-03-20T12:00:00Z', '2026-04-08T08:00:00Z', '2026-06-10T20:00:00Z',
         '2026-07-20T07:00:00Z', '2026-08-16T17:00:00Z']


def render(directory, data, smoke=False):
    directory = Path(directory)
    result = json.loads((directory / 'results.json').read_text())
    selection = json.loads((directory / 'selection.json').read_text())
    lines = ['# ' + directory.name, '', 'Post-hoc feat30 diagnostic; no upload or parent promotion.', '',
             'Frozen internal selection: ' + selection['selected'], '',
             '| Stage | MSE | Peak MAE | Positive 6h ramp MAE | CSI | FP | FN |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ['f1', 'final', 'combined', 'parent']:
        if stage not in result:
            continue
        m = result[stage]
        fields = [m.get(k) for k in ['mse', 'peak_mae', 'ramp6_mae', 'csi', 'fp', 'fn']]
        lines.append('| ' + stage + ' | ' + ' | '.join('N/A' if v is None else f'{v:.4f}' for v in fields) + ' |')
    if 'probability_combined' in result:
        lines.extend(['', 'Probability metrics (all cases, never-crossing included):', '',
                      '```json', json.dumps(result['probability_combined'], indent=2), '```'])
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if smoke:
        return
    origins = data['f1']['origins'].append(data['official']['origins'])
    truth = np.vstack([data['f1']['Y'], data['official']['Y']])
    parent = np.vstack([np.load(PARENT / f'{s}_predictions.npy') for s in ['f1', 'final']])
    pred = np.vstack([np.load(directory / f'{s}_predictions.npz')['pred'] for s in ['f1', 'final']])
    selected = [(str(pd.Timestamp(c)), origins.get_loc(pd.Timestamp(c))) for c in CASES]
    # Add misses and false alarms chosen from the PARENT, not candidate performance.
    for tag, mask in [('parent_missed_peak', (truth.max(1) >= 550) & (parent.max(1) < 550)),
                      ('parent_false_peak', (truth.max(1) < 550) & (parent.max(1) >= 550))]:
        indices = np.flatnonzero(mask)
        if len(indices):
            discrepancy = np.abs(parent[indices].max(1) - truth[indices].max(1))
            i = indices[np.argmax(discrepancy)]
            selected.append((f'{tag}: {origins[i]}', i))
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, len(selected) * 2.8), squeeze=False)
    for ax, (label, i) in zip(axes[:, 0], selected):
        ax.plot(np.arange(1, 73), truth[i], label='truth', color='black')
        ax.plot(np.arange(1, 73), parent[i], label='frozen feat30', alpha=.8)
        ax.plot(np.arange(1, 73), pred[i], label=directory.name, color='tab:orange')
        ax.axhline(550, color='gray', linestyle=':')
        ax.axvline(24.5, color='gray', alpha=.25); ax.axvline(48.5, color='gray', alpha=.25)
        ax.set_title(label); ax.set_ylabel('km/s'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory / 'cases.png', dpi=130); plt.close(fig)
