"""vanila_tsmixer — TSMixer on 5-day hourly sequences, unweighted MSE, mean of seeds 42/43/44.

PyTorch auxiliary-model experiment (replacement ablation of vanila_mse: architecture + input representation).
Input (data-v2/input/df_ts.parquet): for each origin one (120 h x 38 channel) tensor of ACE in-situ series, the
27-day recurrence, ENLIL arrival features and the longitude sweep of one SUVI frame. Two mixer blocks alternate
time-mixing and feature-mixing MLPs (Chen et al. 2023, arXiv 2303.06053) and a temporal projection gives the 72-lead
speed forecast. Implementation: the `pytorch-tsmixer` package (torchtsmixer.TSMixer).
The vanila_mse protocol is retained: prior/full split, training-only frozen per-lead offset and clipping.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_tsmixer/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from torchtsmixer import TSMixer  # noqa: E402

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_tsmixer'
HYPERPARAMETERS = dict(num_blocks=2, ff_dim=64, dropout_rate=0.3, activation_fn='relu', normalize_before=True,
                       norm_type='batch', output_channels=1)


class Model(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.net = TSMixer(sequence_length=seq.LOOKBACK, prediction_length=seq.H, input_channels=channels,
                           **HYPERPARAMETERS)

    def forward(self, x):
        return self.net(x)[:, :, 0]


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='tsmixer', build_model=Model, hyperparameters=HYPERPARAMETERS,
                   libraries=('pytorch-tsmixer',))


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
