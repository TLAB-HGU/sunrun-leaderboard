"""Render frozen results; no candidate selection or publication."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASES = ['2026-03-20T12:00:00Z', '2026-04-08T08:00:00Z', '2026-06-10T20:00:00Z',
         '2026-07-20T07:00:00Z', '2026-08-16T17:00:00Z']


def render(directory, data, smoke):
    directory = Path(directory)
    r = json.loads((directory/'results.json').read_text())
    s = json.loads((directory/'selection.json').read_text())
    lines = ['# feat30 agnostic v2: ' + directory.name, '',
             'Post-hoc analysis; official periods already exposed. No promotion/upload.', '',
             f'Validation choice: {s["selected"]}; preregistered research confirmation: {s["research_arm"]}.', '',
             '| Set | MSE | Peak MAE (all windows) | 6h ramp MAE | Unique TP | Unique FP | Unique FN | Unique timing MAE |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in [('candidate', r['combined']), ('parent', r['parent_combined'])]:
        vals = [m.get(k) for k in ['mse','peak_mae','ramp6_mae','unique_tp','unique_fp','unique_fn','unique_timing_mae_h']]
        lines.append('| ' + name + ' | ' + ' | '.join('N/A' if v is None else f'{v:.4f}' for v in vals) + ' |')
    lines += ['', 'Exact UTC timestamps are deduplicated; nearby peak times are not physical-event clustered.',
              'Matched-only errors are reported alongside misses and false alarms.', '']
    (directory/'REPORT.md').write_text('\n'.join(lines))
    if smoke:
        return
    oo = data['f1']['origins'].append(data['official']['origins'])
    y = np.vstack([data['f1']['Y'], data['official']['Y']])
    p = np.vstack([np.load(directory/f'{stage}_predictions.npz')['pred'] for stage in ['f1','final']])
    base = np.vstack([np.load(directory/f'{stage}_predictions.npz')['parent'] for stage in ['f1','final']])
    examples = [(str(pd.Timestamp(t)), oo.get_loc(pd.Timestamp(t))) for t in CASES]
    for tag, mask in [('parent_missed', (y.max(1)>=550)&(base.max(1)<550)),
                      ('parent_false', (y.max(1)<550)&(base.max(1)>=550))]:
        choices=np.flatnonzero(mask)
        if len(choices):
            index = choices[np.argmax(np.abs(base[choices].max(1)-y[choices].max(1)))]
            examples.append((f'{tag}: {oo[index]}',index))
    fig, axs=plt.subplots(len(examples),1,figsize=(11,2.8*len(examples)),squeeze=False)
    for ax,(label,i) in zip(axs[:,0],examples):
        ax.plot(np.arange(1,73),y[i],color='black',label='truth')
        ax.plot(np.arange(1,73),base[i],label='frozen feat30')
        ax.plot(np.arange(1,73),p[i],label=directory.name,color='tab:orange')
        ax.axhline(550,color='gray',ls=':');ax.set_title(label);ax.set_ylabel('km/s');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(directory/'cases.png',dpi=130);plt.close(fig)
