"""vanila_itransformer — iTransformer on 5-day hourly sequences plus the XGBoost summary features, unweighted MSE,
seeds 42/43/44.

PyTorch auxiliary-model experiment (replaced on the board 2026-11-17; the first version is described in README.md).
Inputs: the 120 h x 38 channel tensor from data-v2/input/df_ts.parquet (ACE in-situ series, 27-day recurrence, ENLIL
arrival features, longitude sweep of one SUVI frame) and the 31 summary features of the XGBoost row vanila_mse
(research/_module/summary.py). As in Liu et al. 2024 (arXiv 2310.06625) every channel's 120-step series is one
variate token; each per-lead summary feature's 72-lead series is one more token and the 23 constants are one token.
A 2-layer post-norm Transformer encoder (d_model 32, d_ff 64, 4 heads, dropout 0.1, GELU) attends over the 47 tokens
and the ACE speed token is projected to the 72 leads. The forecast is the last observed speed plus that output
(persistence residual). vanila_mse protocol: prior/full split, training-only frozen per-lead offset, clip.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_itransformer/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from _module import seq, summary  # noqa: E402

EXPERIMENT = 'vanila_itransformer'
CONSTANTS = len(summary.CONSTANT)
HYPERPARAMETERS = dict(d_model=32, e_layers=2, n_heads=4, d_ff=64, dropout=0.1, activation='gelu',
                       instance_norm=False,
                       output='last observed speed + linear projection of the ace_speed_kms variate token',
                       summary_tokens='8 per-lead variate tokens (Linear(72, d_model)) + 1 constant token')


class ITransformer(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.speed = speed
        width, dropout = HYPERPARAMETERS['d_model'], HYPERPARAMETERS['dropout']
        self.embedding = torch.nn.Linear(seq.LOOKBACK, width)
        self.lead_embedding = torch.nn.Linear(seq.H, width)
        self.constant_embedding = torch.nn.Linear(CONSTANTS, width)
        self.dropout = torch.nn.Dropout(dropout)
        layer = torch.nn.TransformerEncoderLayer(width, HYPERPARAMETERS['n_heads'], HYPERPARAMETERS['d_ff'], dropout,
                                                 activation=HYPERPARAMETERS['activation'], batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, HYPERPARAMETERS['e_layers'], norm=torch.nn.LayerNorm(width),
                                                   enable_nested_tensor=False)
        self.projection = torch.nn.Linear(width, seq.H)

    def forward(self, x, constant, lead):
        tokens = torch.cat([self.embedding(x.transpose(1, 2)), self.lead_embedding(lead.transpose(1, 2)),
                            self.constant_embedding(constant)[:, None]], 1)
        return self.projection(self.encoder(self.dropout(tokens))[:, self.speed])


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='itransformer', build_model=ITransformer,
                   hyperparameters=HYPERPARAMETERS, persistence_residual=True, summary=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
