"""Training-only frozen calibration: prior/calibration split, per-lead offset, physical clipping."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import H
from .table import PURGE_HOURS

POLICY = 'training-only frozen calibration; official labels used only for scoring'
ALPHA = .25
OFFSET_WINDOW = 336


def prior_split(origins):
    """Reference rows end 144 h before the last three months; those months calibrate the offset."""
    origins = pd.DatetimeIndex(origins)
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError('Training origins must be unique and sorted')
    boundary = origins.max() - pd.DateOffset(months=3)
    calibration = np.flatnonzero(origins >= boundary)
    reference = np.flatnonzero(origins + pd.Timedelta(hours=PURGE_HOURS) <= boundary)
    if len(reference) < 2 or len(calibration) < 48:
        raise ValueError('Insufficient strictly prior reference/calibration support')
    return reference, calibration, boundary


def component_values(raw):
    return np.clip(raw, 1., 3000.)


def estimate_offset(prior, truth, window=OFFSET_WINDOW, alpha=ALPHA, smoke=False):
    if prior.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != H:
        raise ValueError('Prior predictions and training truth require a shared 72-lead grid')
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite calibration values')
    rows = window
    if rows > len(truth) and not smoke:
        raise ValueError('Insufficient training calibration rows for frozen window')
    rows = min(rows, len(truth))
    residuals = truth[-rows:] - component_values(prior)[-rows:]
    return alpha * residuals.mean(0), rows


def apply_offset(raw, offset):
    corrected = np.clip(component_values(raw) + offset[None, :], 1., 3000.)
    if not np.isfinite(corrected).all():
        raise ValueError('Nonfinite final forecasts')
    return corrected


def forecast_values(raw):
    """Full-model forecast clipped to the physical range [1, 3000] km/s (no calibration offset)."""
    values = np.clip(raw, 1., 3000.)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite final forecasts')
    return values
