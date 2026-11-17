"""vanila_gru — GRU on 5-day hourly sequences plus the XGBoost summary features, unweighted MSE, seeds 42/43/44.

PyTorch auxiliary-model experiment (replaced on the board 2026-11-17; the first version is described in README.md).
Inputs: the 120 h x 38 channel tensor from data-v2/input/df_ts.parquet (ACE in-situ series, 27-day recurrence, ENLIL
arrival features, longitude sweep of one SUVI frame) and the 31 summary features of the XGBoost row vanila_mse
(23 per-origin constants, 8 per-lead features x 72 leads; research/_module/summary.py). A two-layer GRU (hidden 128,
dropout 0.1 on the input, 0.5 between the layers) reads the sequence; the flattened summary features pass one hidden
layer (128, ReLU); both are concatenated, dropped out (0.5) and mapped to the 72 leads. The forecast is the last
observed speed plus that output (persistence residual). vanila_mse protocol: prior/full split, training-only frozen
per-lead offset, clip.

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

from _module import seq, summary  # noqa: E402

EXPERIMENT = 'vanila_gru'
CONSTANTS, LEAD_FEATURES = len(summary.CONSTANT), len(summary.VARYING)
HYPERPARAMETERS = dict(hidden_size=128, num_layers=2, input_dropout=0.1, dropout=0.5, head_dropout=0.5,
                       head='linear([last hidden state, summary hidden] -> 72 leads)',
                       output='last observed speed + network output', summary_hidden=128)


class GRU(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        hidden = HYPERPARAMETERS['hidden_size']
        self.input_dropout = torch.nn.Dropout(HYPERPARAMETERS['input_dropout'])
        self.gru = torch.nn.GRU(channels, hidden, HYPERPARAMETERS['num_layers'], batch_first=True,
                                dropout=HYPERPARAMETERS['dropout'])
        self.summary = torch.nn.Sequential(
            torch.nn.Linear(CONSTANTS + seq.H * LEAD_FEATURES, HYPERPARAMETERS['summary_hidden']), torch.nn.ReLU())
        self.head_dropout = torch.nn.Dropout(HYPERPARAMETERS['head_dropout'])
        self.head = torch.nn.Linear(hidden + HYPERPARAMETERS['summary_hidden'], seq.H)

    def forward(self, x, constant, lead):
        state = self.gru(self.input_dropout(x))[0][:, -1]
        extra = self.summary(torch.cat([constant, lead.flatten(1)], 1))
        return self.head(self.head_dropout(torch.cat([state, extra], 1)))


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='gru', build_model=GRU, hyperparameters=HYPERPARAMETERS,
                   persistence_residual=True, summary=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
