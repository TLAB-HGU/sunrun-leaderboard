"""QXGB arm: convolution-smoothed quantile regression inside XGBoost (Yao & Franklin, arXiv:2608.15290).

One dense conditional quantile grid per 24 h block, fitted *jointly* with multi-output (vector-leaf) trees, so the
quantile surfaces are coupled and cannot cross. The board point forecast is the predictive mean read out of the
estimated CDF, E[Y] = integral_0^1 Q(tau) dtau, not a separate mean model.

Convolution smoothing (He et al. 2023; paper sec. 3.1.5): with u = y - yhat and kernel K, bandwidth h,
    loss   l_h(u) = (rho_tau * K_h)(u)          (globally convex, twice continuously differentiable)
    grad   l_h'(u) = tau - Kcal_h(u)            Kcal_h(u) = Kcal(u/h), Kcal = integral of K
    hess   l_h''(u) = (1/h) K(u/h)              <- non-zero everywhere: XGBoost keeps its second-order splits
Gaussian kernel:  grad = tau - Phi(-u/h),   hess = exp(-u^2/(2h^2)) / (h sqrt(2 pi)).
Laplace  kernel:  grad = tau - 1/2 exp(u/h) (u<0) or tau - 1 + 1/2 exp(-u/h) (u>=0),  hess = exp(-|u|/h)/(2h).

Targets are Z-scaled before fitting (paper sec. 3.3): both kernels use exponentials, and an unscaled residual makes
them overflow (hessian -> 0 in the tails, which turns the Newton leaf step -G/(H+lambda) into a blow-up; observed
locally as anti-monotone quantiles). max_delta_step additionally caps the per-output leaf value.

Speed: a block is one 240-column vector-leaf fit (24 horizons x 12 levels) by default; QXGB_TREES/`rounds` cut trees
for smoke runs only.
"""
from __future__ import annotations

import os

import numpy as np

# paper grid {0.05,...,0.95} plus 0.01/0.99 so the CDF mean readout is not truncated in the 1% tails of a fat right
# tail (the paper's 0.05..0.95 grid covers 90% of the mass, which is not enough for a mean/MSE target).
LEVELS = np.array([0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99])
BANDWIDTH = 0.5  # in standardised units; paper's tuned value on standardised targets
KERNELS = ("gaussian", "laplace")


def _kernel(name):
    if name == "gaussian":
        from scipy.special import ndtr

        def tail(u):
            return ndtr(u)

        def density(u):
            return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    elif name == "laplace":

        def tail(u):
            return np.where(u < 0.0, 0.5 * np.exp(u), 1.0 - 0.5 * np.exp(-u))

        def density(u):
            return 0.5 * np.exp(-np.abs(u))
    else:
        raise ValueError(f"unknown kernel {name}")
    return tail, density


def objective(levels, bandwidth, kernel, mu, sd, horizons):
    """Custom XGBoost objective over the 1-D output vector [level0 x horizons, level1 x horizons, ...]."""
    tail, density = _kernel(kernel)
    tau = np.repeat(levels, horizons)  # column -> quantile level
    floor = 1e-6

    def obj(pred, matrix):
        t = matrix.get_label().astype(np.float64).reshape(pred.shape)
        r = t - pred.astype(np.float64)  # residual u = y - yhat, standardised units (labels are already scaled)
        z = r / bandwidth
        # The paper's l_h'(u) = tau - Kcal(-u/h) is the derivative w.r.t. u = y - yhat; XGBoost minimises in yhat, so
        # the gradient it needs is -l_h'(u). Hessian: d2L/dyhat2 = K(u/h)/h (the two sign flips cancel).
        grad = tail(-z) - tau[None, :]
        hess = np.maximum(density(z) / bandwidth, floor)
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


def labels(y_block, levels):
    """Tile the standardised 24 h target across the quantile levels: column order matches objective()."""
    return np.hstack([y_block] * len(levels)).astype(np.float32)


def cdf_mean(q, levels):
    """E[Y] from the estimated quantile function: integral_0^1 Q(tau) dtau, trapezoid with constant end extrapolation.

    ``q`` is (n, horizons, levels): tau runs on the last axis.
    """
    ext = np.concatenate([[0.0], levels, [1.0]])
    qq = np.concatenate([q[:, :, :1], q, q[:, :, -1:]], axis=2)
    return np.sum(np.diff(ext)[None, None, :] * 0.5 * (qq[:, :, 1:] + qq[:, :, :-1]), axis=2)


def exceedance(q, levels, threshold):
    """P(Y > threshold) per (n, horizon): linear interpolation of the estimated inverse CDF, clamped outside the grid."""
    flat = q.reshape(-1, q.shape[-1])
    out = np.array([np.interp(threshold, row, levels) for row in flat])
    return 1.0 - out.reshape(q.shape[:2])


def crossing_rate(q):
    """Share of prediction vectors whose sorted-by-tau quantiles are not monotone (paper eq. 4)."""
    return float(np.mean((np.diff(q, axis=1) < 0.0).any(axis=1)))


def predict_cdf(model, xq, n, horizons, levels, mu, sd):
    """Unscale and (diagnostically) order the predicted quantiles; returns (n, horizons, levels)."""
    raw = model.inplace_predict(xq).astype(np.float64)
    q = raw.reshape(len(xq), len(levels), horizons).transpose(0, 2, 1) * sd + mu
    return np.sort(q, axis=2)


def fit_block(x, y, xq, params, rounds, threads, levels=LEVELS, bandwidth=BANDWIDTH, kernel="gaussian"):
    """Fit one 24 h block; returns the predictive mean and the ordered CDF array for diagnostics."""
    import xgboost as xgb

    mu, sd = float(np.mean(y)), float(np.std(y))
    yb = ((y - mu) / sd).astype(np.float32)
    lab = labels(yb, levels)
    dt = xgb.DMatrix(x, label=lab.reshape(-1), nthread=threads)
    obj = objective(levels, bandwidth, kernel, mu, sd, y.shape[1])
    model = xgb.train(dict(params, nthread=threads, base_score=0.0), dt, num_boost_round=rounds, obj=obj, verbose_eval=False)
    q = predict_cdf(model, xq, len(xq), y.shape[1], levels, mu, sd)
    return cdf_mean(q, levels), q


def arm_qxgb(st, blocks, threads, rounds=None):
    """`qxgb` arm body: same block/query layout as the other arms, point forecast = CDF mean.

    QXGB_LEVELS ("0.05,0.25,...") sets the quantile grid, QXGB_ROUNDS the boosting rounds and QXGB_ETA the learning
    rate: the grid size multiplies the per-round cost, because a vector-leaf split is scored over every output column.
    """
    levels = np.array([float(v) for v in os.environ.get("QXGB_LEVELS", "").split(",") if v.strip()]) if os.environ.get("QXGB_LEVELS") else LEVELS
    rounds = int(os.environ.get("QXGB_ROUNDS", 900)) if rounds is None else rounds
    kernel = os.environ.get("QXGB_KERNEL", "gaussian")
    bandwidth = float(os.environ.get("QXGB_BANDWIDTH", BANDWIDTH))
    eta = os.environ.get("QXGB_ETA")
    import arms as A

    pred = np.full((len(st.q_origins), A.H), np.nan)
    crossing = {}
    for b in blocks:
        s_ = A.sl(b)
        over = dict(max_delta_step=1.0)
        if eta is not None:
            over["learning_rate"] = float(eta)
        params = A.params_of(threads, **over)
        mean, q = fit_block(st.x[b], st.y[:, s_], st.xq[b], params, rounds, threads, levels=levels, bandwidth=bandwidth, kernel=kernel)
        pred[:, s_] = mean
        crossing[b] = crossing_rate(q)
    print(f"qxgb kernel={kernel} h={bandwidth} levels={len(levels)} rounds={rounds} eta={eta} crossing={crossing}", flush=True)
    return np.clip(pred, 1.0, 3000.0)
