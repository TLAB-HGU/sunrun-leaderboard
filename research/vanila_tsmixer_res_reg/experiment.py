"""vanila_tsmixer_res_reg — vanila_tsmixer with a persistence residual output and dropout 0.7, unweighted MSE,
seeds 42/43/44.

PyTorch auxiliary-model experiment. Same input, loss and protocol as vanila_tsmixer (120 h x 38 channel tensor from
data-v2/input/df_ts.parquet, torchtsmixer.TSMixer, prior/full split, frozen per-lead offset, clip). Two changes, both
chosen on F1 training data only: (1) the forecast is the last observed speed plus the network output, so short leads
start from persistence; (2) dropout 0.3 -> 0.7 against the long-lead overfitting.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_tsmixer_res_reg/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from torchtsmixer import TSMixer  # noqa: E402

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_tsmixer_res_reg'
HYPERPARAMETERS = dict(num_blocks=2, ff_dim=64, dropout_rate=0.7, activation_fn='relu', normalize_before=True,
                       norm_type='batch', output_channels=1)


class Model(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.net = TSMixer(sequence_length=seq.LOOKBACK, prediction_length=seq.H, input_channels=channels,
                           **HYPERPARAMETERS)

    def forward(self, x):
        return self.net(x)[:, :, 0]


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='tsmixer', build_model=Model, hyperparameters=HYPERPARAMETERS,
                   libraries=('pytorch-tsmixer',), persistence_residual=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
