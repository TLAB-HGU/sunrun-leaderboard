"""Regularized causal TCN with uncompressed current features at its forecast head."""
import numpy as np
import torch
from torch import nn


ARCH = {
    'history_hours': 120,
    'history_channels': 30,
    'future_channels': 9,
    'horizons': 72,
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16, 32],
    'width': 32,
    'dropout': 0.2,
    'current_channels': 21,
    'current_path': 'last history row, selected current features, directly concatenated',
    'receptive_field': 127,
    'pool': 'last_timestep',
    'head_hidden': [64, 32],
    'target': 'standardized absolute speed',
}


class CausalBlock(nn.Module):
    """One left-padded convolution; output t only uses inputs at or before t."""

    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.pad = (ARCH['kernel_size'] - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, ARCH['kernel_size'], dilation=dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(ARCH['dropout'])

    def forward(self, x):
        return self.dropout(self.act(self.conv(nn.functional.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    """Encode each origin once, then condition its 72 forecasts on known inputs.

    history: [batch, 120, 30], oldest hour first and origin hour last.
    future: [batch, 72, 9], horizon 1 first; every value is known at origin.
    Returns standardized absolute target predictions with shape [batch, 72].
    """

    def __init__(self, current_indices):
        super().__init__()
        indices = tuple(current_indices)
        if (len(indices) != ARCH['current_channels']
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)
                or len(set(indices)) != len(indices)
                or any(i < 0 or i >= ARCH['history_channels'] for i in indices)
                or 0 not in indices):
            raise ValueError('current_indices must contain 21 unique channel indices in 0..29, including v_lag0 at 0')
        self.register_buffer('current_indices', torch.tensor(indices, dtype=torch.long))
        widths = [ARCH['history_channels']] + [ARCH['width']] * len(ARCH['dilations'])
        self.encoder = nn.Sequential(*[
            CausalBlock(widths[i], widths[i + 1], dilation)
            for i, dilation in enumerate(ARCH['dilations'])
        ])
        self.head = nn.Sequential(
            nn.Linear(ARCH['width'] + ARCH['current_channels'] + ARCH['future_channels'] + 1, 64),
            nn.ReLU(), nn.Dropout(ARCH['dropout']), nn.Linear(64, 32),
            nn.ReLU(), nn.Dropout(ARCH['dropout']), nn.Linear(32, 1),
        )
        self.register_buffer('horizon', torch.arange(1, 73, dtype=torch.float32).view(1, 72, 1) / 72)

    def encode(self, history):
        if history.ndim != 3 or history.shape[1:] != (120, 30) or not len(history):
            raise ValueError('history must have nonempty shape [batch, 120, 30]')
        return self.encoder(history.transpose(1, 2))[:, :, -1]

    def forward(self, history, future):
        context = self.encode(history)
        if future.shape != (len(history), 72, 9):
            raise ValueError('future must have shape [batch, 72, 9] matching history')
        context = context[:, None, :].expand(-1, 72, -1)
        current = history[:, -1, self.current_indices][:, None, :].expand(-1, 72, -1)
        horizon = self.horizon.expand(len(history), -1, -1)
        return self.head(torch.cat((context, current, future, horizon), dim=-1)).squeeze(-1)


def point_weights(y):
    """Normalize 4x high-speed weights separately over each training horizon.

    Call on the entire stage training target matrix before forming minibatches.
    Thresholding uses physical km/s targets, before target standardization.
    """
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 72 or not len(y) or not np.isfinite(y).all():
        raise ValueError('targets must have finite nonempty shape [origins, 72]')
    weights = np.where(y >= 550., 4., 1.)
    return (weights / weights.mean(axis=0, keepdims=True)).astype(np.float32)


def weighted_mse(pred, target, weights):
    """Elementwise weighted MSE; weights are already training-set normalized."""
    if pred.ndim != 2 or pred.shape[1] != 72 or not len(pred):
        raise ValueError('predictions must have nonempty shape [batch, 72]')
    if target.shape != pred.shape or weights.shape != pred.shape:
        raise ValueError('targets and weights must match prediction shape')
    return (weights * (pred - target).square()).mean()
