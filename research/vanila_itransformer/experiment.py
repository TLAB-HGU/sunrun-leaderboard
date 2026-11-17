"""vanila_itransformer — iTransformer on 5-day hourly sequences, unweighted MSE, mean of seeds 42/43/44.

PyTorch auxiliary-model experiment (replacement ablation of vanila_mse: architecture + input representation).
Input (data-v2/input/df_ts.parquet): for each origin one (120 h x 38 channel) tensor of ACE in-situ series, the
27-day recurrence, ENLIL arrival features and the longitude sweep of one SUVI frame. As in Liu et al. 2024
(arXiv 2310.06625) each channel's whole 120-step series is embedded as one variate token, an unmodified Transformer
encoder attends over the 38 variate tokens, and a linear projection of the ACE speed token gives the 72 leads.
The blocks are torch.nn.TransformerEncoder (post-norm, GELU), the same structure as the official implementation
(thuml/iTransformer, model/iTransformer.py). The paper's per-window instance normalization is off: it would zero the
mask channels and remove the level of the covariates, and the inputs are already standardized on the training span.
The vanila_mse protocol is retained: prior/full split, training-only frozen per-lead offset and clipping.

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

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_itransformer'
HYPERPARAMETERS = dict(d_model=128, e_layers=2, n_heads=4, d_ff=256, dropout=0.1, activation='gelu',
                       instance_norm=False, output='linear projection of the ace_speed_kms variate token -> 72 leads')


class ITransformer(torch.nn.Module):
    def __init__(self, channels, speed):
        super().__init__()
        self.speed = speed
        width, dropout = HYPERPARAMETERS['d_model'], HYPERPARAMETERS['dropout']
        self.embedding = torch.nn.Linear(seq.LOOKBACK, width)
        self.dropout = torch.nn.Dropout(dropout)
        layer = torch.nn.TransformerEncoderLayer(width, HYPERPARAMETERS['n_heads'], HYPERPARAMETERS['d_ff'], dropout,
                                                 activation=HYPERPARAMETERS['activation'], batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, HYPERPARAMETERS['e_layers'], norm=torch.nn.LayerNorm(width),
                                                   enable_nested_tensor=False)
        self.projection = torch.nn.Linear(width, seq.H)

    def forward(self, x):
        tokens = self.encoder(self.dropout(self.embedding(x.transpose(1, 2))))   # (B, variates, d_model)
        return self.projection(tokens[:, self.speed])


SPEC = seq.SeqSpec(experiment=EXPERIMENT, family='itransformer', build_model=ITransformer,
                   hyperparameters=HYPERPARAMETERS)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
