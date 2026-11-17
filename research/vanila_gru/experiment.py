"""vanila_gru — a plain GRU on 5-day hourly sequences, unweighted MSE, mean of seeds 42/43/44.

PyTorch auxiliary-model experiment (replacement ablation of vanila_mse: architecture + input representation).
Input (data-v2/input/df_ts.parquet): for each origin one (120 h x 38 channel) tensor of ACE in-situ series, the
27-day recurrence, ENLIL arrival features and the longitude sweep of one SUVI frame. A two-layer GRU reads the
sequence and a linear head maps its last hidden state to the 72 leads. The vanila_mse protocol is retained: prior/full
split, training-only frozen per-lead offset and clipping.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_gru/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_gru'
HYPERPARAMETERS = dict(hidden_size=128, num_layers=2, dropout=0.1, head='linear(last hidden state -> 72 leads)')


class GRU(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.gru = torch.nn.GRU(channels, HYPERPARAMETERS['hidden_size'], HYPERPARAMETERS['num_layers'],
                                batch_first=True, dropout=HYPERPARAMETERS['dropout'])
        self.head = torch.nn.Linear(HYPERPARAMETERS['hidden_size'], seq.H)

    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='gru', build_model=GRU, hyperparameters=HYPERPARAMETERS)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
