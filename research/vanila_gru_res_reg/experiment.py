"""vanila_gru_res_reg — vanila_gru with a persistence residual output and dropout, unweighted MSE, seeds 42/43/44.

PyTorch auxiliary-model experiment. Same input, loss and protocol as vanila_gru (120 h x 38 channel tensor from
data-v2/input/df_ts.parquet, prior/full split, frozen per-lead offset, clip). Two changes, both chosen on F1 training
data only: (1) the forecast is the last observed speed plus the network output, so short leads start from
persistence; (2) dropout on the input (0.1), between the GRU layers (0.5) and before the head (0.5) against the
long-lead overfitting.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_gru_res_reg/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_gru_res_reg'
HYPERPARAMETERS = dict(hidden_size=128, num_layers=2, input_dropout=0.1, dropout=0.5, head_dropout=0.5,
                       head='linear(last hidden state -> 72 leads)', output='last observed speed + network output')


class GRU(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(HYPERPARAMETERS['input_dropout'])
        self.gru = torch.nn.GRU(channels, HYPERPARAMETERS['hidden_size'], HYPERPARAMETERS['num_layers'],
                                batch_first=True, dropout=HYPERPARAMETERS['dropout'])
        self.head_dropout = torch.nn.Dropout(HYPERPARAMETERS['head_dropout'])
        self.head = torch.nn.Linear(HYPERPARAMETERS['hidden_size'], seq.H)

    def forward(self, x):
        return self.head(self.head_dropout(self.gru(self.input_dropout(x))[0][:, -1]))


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='gru', build_model=GRU, hyperparameters=HYPERPARAMETERS,
                   persistence_residual=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
