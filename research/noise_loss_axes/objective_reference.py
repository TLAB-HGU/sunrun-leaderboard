"""CPU reference losses for conditional distributional boosting.

Rows are independent forecast cells; every parameter may depend on features and
horizon. Arrays have shape (n, p), or (p,) for one cell. The returned curvature is
a positive diagonal Fisher-style approximation, NOT the exact observed Hessian:
normal location/log-scale use (1 / sigma**2, 2), and mixture components use
responsibility-weighted complete-data curvature. Cross-parameter terms and the
mixture missing-information term are omitted. This makes the approximation
usable by a diagonal-Hessian tree learner without claiming exact Newton steps.

External weights multiply the whole NLL, gradient, and curvature. Target-based
weights change the estimated distribution; they are not variance corrections.
Lognormal NLL includes log(y), so reported scores are on the original y scale.
"""
from __future__ import annotations

import numpy as np

PARAMETER_COUNTS = {"gaussian": 2, "lognormal": 2, "mixture": 5}
LOG_2PI = np.log(2.0 * np.pi)


def _parameters(theta, family):
    if family not in PARAMETER_COUNTS:
        raise ValueError(f"Unknown family: {family}")
    values = np.asarray(theta, dtype=np.float64)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != PARAMETER_COUNTS[family]:
        raise ValueError(f"theta must have shape (n, {PARAMETER_COUNTS[family]})")
    if not np.isfinite(values).all():
        raise ValueError("Distribution parameters must be finite")
    return values, scalar


def _scales(rho, sigma_floor):
    if not np.isfinite(sigma_floor) or sigma_floor <= 0:
        raise ValueError("sigma_floor must be finite and positive")
    lower = np.log(sigma_floor)
    effective = np.maximum(rho, lower)
    # The floor is part of the loss, so its derivative is zero below the floor.
    # At the kink we choose the zero subgradient. A curvature floor keeps the
    # learner's numerical denominator positive, but does not change the gradient.
    active = rho > lower
    return effective, np.exp(-2.0 * effective), active


def nll_and_grad(theta, y, family, *, weights=None, mask=None,
                 sigma_floor=1e-4, hessian_floor=1e-8):
    """Return (per-cell NLL, gradient, positive approximate diagonal Hessian).

    gaussian/lognormal: theta=(mu, log_sigma).
    mixture: theta=(logit_pi1, mu1, mu2, log_sigma1, log_sigma2), where
    pi1=sigmoid(logit_pi1). Component means need not be ordered.

    y, weights, mask accept scalars or vectors of length n. Nonfinite labels and
    explicitly masked cells contribute exactly zero. Nonpositive *active*
    lognormal labels are errors. Zero-weight rows are excluded. Inactive rows may
    contain invalid labels, but all parameters must remain finite. A 1-D theta
    returns a scalar loss and 1-D gradient/Hessian.
    """
    theta, scalar = _parameters(theta, family)
    if not np.isfinite(hessian_floor) or hessian_floor <= 0:
        raise ValueError("hessian_floor must be finite and positive")
    n = len(theta)
    labels = np.broadcast_to(np.asarray(y, dtype=np.float64), (n,))
    weight = np.ones(n) if weights is None else np.broadcast_to(
        np.asarray(weights, dtype=np.float64), (n,))
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    valid = np.isfinite(labels) & (weight > 0)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), (n,))
    if family != "gaussian" and (labels[valid] <= 0).any():
        raise ValueError("Lognormal targets must be positive")
    safe = np.where(valid, labels, 1.0)
    z = safe if family == "gaussian" else np.log(safe)
    grad, curvature = np.zeros_like(theta), np.zeros_like(theta)
    if family in ("gaussian", "lognormal"):
        rho, inv_var, active = _scales(theta[:, 1], sigma_floor)
        residual = z - theta[:, 0]
        squared = residual**2 * inv_var
        loss = 0.5 * (LOG_2PI + squared) + rho
        grad[:, 0] = -residual * inv_var
        grad[:, 1] = (1.0 - squared) * active
        curvature[:, 0] = inv_var
        curvature[:, 1] = 2.0 * active
    else:
        eta = theta[:, 0]
        log_pi = np.column_stack((-np.logaddexp(0., -eta),
                                  -np.logaddexp(0., eta)))
        rho, inv_var, active = _scales(theta[:, 3:5], sigma_floor)
        residual = z[:, None] - theta[:, 1:3]
        squared = residual**2 * inv_var
        log_joint = log_pi - 0.5 * (LOG_2PI + squared) - rho
        log_density = np.logaddexp(log_joint[:, 0], log_joint[:, 1])
        responsibility = np.exp(log_joint - log_density[:, None])
        pi = np.exp(log_pi)
        loss = -log_density
        grad[:, 0] = pi[:, 0] - responsibility[:, 0]
        grad[:, 1:3] = -responsibility * residual * inv_var
        grad[:, 3:5] = responsibility * (1.0 - squared) * active
        curvature[:, 0] = pi[:, 0] * pi[:, 1]
        curvature[:, 1:3] = responsibility * inv_var
        curvature[:, 3:5] = 2.0 * responsibility * active
    if family != "gaussian":
        loss += z
    loss = np.where(valid, loss * weight, 0.)
    grad = np.where(valid[:, None], grad * weight[:, None], 0.)
    curvature = np.where(valid[:, None],
                         np.maximum(curvature, hessian_floor) * weight[:, None], 0.)
    if not all(np.isfinite(value).all() for value in (loss, grad, curvature)):
        raise FloatingPointError("Nonfinite distribution loss or derivative")
    if scalar:
        return loss[0], grad[0], curvature[0]
    return loss, grad, curvature


def predict_mean(theta, family, *, sigma_floor=1e-4):
    """Return the arithmetic conditional mean, without positivity clipping.

    Gaussian means may be negative. Any forecast-domain clipping is a separate
    readout policy and must be reported by the experiment runner.
    """
    theta, scalar = _parameters(theta, family)
    if family == "gaussian":
        mean = theta[:, 0].copy()
    elif family == "lognormal":
        rho, _, _ = _scales(theta[:, 1], sigma_floor)
        mean = np.exp(theta[:, 0] + 0.5 * np.exp(2.0 * rho))
    else:
        rho, _, _ = _scales(theta[:, 3:5], sigma_floor)
        log_pi = np.column_stack((-np.logaddexp(0., -theta[:, 0]),
                                  -np.logaddexp(0., theta[:, 0])))
        component_log_mean = theta[:, 1:3] + 0.5 * np.exp(2.0 * rho)
        mean = np.exp(np.logaddexp(log_pi[:, 0] + component_log_mean[:, 0],
                                  log_pi[:, 1] + component_log_mean[:, 1]))
    if not np.isfinite(mean).all():
        raise FloatingPointError("Nonfinite distribution mean")
    return mean[0] if scalar else mean
