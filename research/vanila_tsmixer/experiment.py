"""vanila_tsmixer — TSMixer on 5-day hourly sequences plus the XGBoost summary features, unweighted MSE,
seeds 42/43/44.

PyTorch auxiliary-model experiment (replaced on the board 2026-11-17; the first version is described in README.md).
Inputs: the 120 h x 38 channel tensor from data-v2/input/df_ts.parquet (ACE in-situ series, 27-day recurrence, ENLIL
arrival features, longitude sweep of one SUVI frame) and the 31 summary features of the XGBoost row vanila_mse
(research/_module/summary.py). The extended TSMixer with auxiliary information (Chen et al. 2023, arXiv 2303.06053;
torchtsmixer.TSMixerExt from pytorch-tsmixer 0.2.0) takes the 8 per-lead features as known future features and the 23
constants as static features (two mixer blocks, ff_dim 64, hidden 64, dropout 0.7, batch norm). The forecast is the
last observed speed plus its output (persistence residual). vanila_mse protocol: prior/full split, training-only
frozen per-lead offset, clip.

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
from torchtsmixer import TSMixerExt  # noqa: E402

from _module import seq, summary  # noqa: E402

EXPERIMENT = 'vanila_tsmixer'
CONSTANTS, LEAD_FEATURES = len(summary.CONSTANT), len(summary.VARYING)
HYPERPARAMETERS = dict(num_blocks=2, ff_dim=64, dropout_rate=0.7, activation_fn='relu', normalize_before=True,
                       norm_type='batch', output_channels=1, hidden_channels=64, extra_channels=LEAD_FEATURES,
                       static_channels=CONSTANTS)


class Model(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.net = TSMixerExt(sequence_length=seq.LOOKBACK, prediction_length=seq.H, input_channels=channels,
                              **HYPERPARAMETERS)

    def forward(self, x, constant, lead):
        history = lead.new_zeros(len(x), seq.LOOKBACK, LEAD_FEATURES)
        return self.net(x, history, lead, constant)[:, :, 0]


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='tsmixer', build_model=Model, hyperparameters=HYPERPARAMETERS,
                   libraries=('pytorch-tsmixer',), persistence_residual=True, summary=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
