"""vanila_itransformer_res_reg — vanila_itransformer with a persistence residual output and a narrower encoder,
unweighted MSE, seeds 42/43/44.

PyTorch auxiliary-model experiment. Same input, loss and protocol as vanila_itransformer (120 h x 38 channel tensor
from data-v2/input/df_ts.parquet, variate tokens, post-norm Transformer encoder, prior/full split, frozen per-lead
offset, clip). Two changes, both chosen on F1 training data only: (1) the forecast is the last observed speed plus
the network output, so short leads start from persistence; (2) d_model 128 -> 32 and d_ff 256 -> 64 against the
long-lead overfitting.

Usage (DL: GPU only; the shared runner is research/_module/seq.py):
  CUDA_VISIBLE_DEVICES=0 python experiment.py --device cuda --threads 2    # all seeds, F1/F2/F3, mean3, score
  CUDA_VISIBLE_DEVICES=0 python experiment.py --predict-only               # inference from weights/
  CUDA_VISIBLE_DEVICES=0 python experiment.py --smoke --out DIR            # 2-epoch wiring check
Outputs: weights/F{1,2,3}/s{seed}.npz in this folder; results/{s42,s43,s44,mean3}_{predictions.parquet,score.json}
and logs/ (*.jsonl, F{n}_s{seed}_metadata.json) in research/result/vanila_itransformer_res_reg/ (or DIR/ with --out DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from _module import seq  # noqa: E402

EXPERIMENT = 'vanila_itransformer_res_reg'
HYPERPARAMETERS = dict(d_model=32, e_layers=2, n_heads=4, d_ff=64, dropout=0.1, activation='gelu',
                       instance_norm=False,
                       output='last observed speed + linear projection of the ace_speed_kms variate token')


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
                   hyperparameters=HYPERPARAMETERS, persistence_residual=True)


if __name__ == '__main__':
    seq.main(SPEC, __doc__, HERE)
