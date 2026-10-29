"""HL-Gauss histogram regression with exact target-shift posterior correction.

Adaptations of the source methods for a CPU XGBoost DirMO-24 lane:

* Imani et al. (arXiv:2402.13425) and Farebrother et al. (arXiv:2403.03950):
  replace the scalar regression head with a K-bin categorical head trained by
  cross-entropy against Gaussian-smoothed (HL-Gauss) soft labels, and read the
  point forecast off as the mean of the predicted histogram.  The original
  papers use a neural network and uniform bins; here the head is a per-horizon
  XGBoost `multi:softprob` model on the frozen 237-column block design and the
  bins are a fixed non-uniform grid.
* ReTaSA (arXiv:2401.16410), Liu et al. (arXiv:2609.15785) and Prior2Posterior
  (arXiv:2412.16540): under target shift with p(x|y) invariant, training with a
  weight w that depends on the label alone learns p_w(y|x) proportional to
  w(y) p(y|x); dividing the learnt class probabilities by w and renormalising
  recovers p(y|x) exactly.  The bin grid keeps 550 km/s on an edge so every bin
  has a single scalar weight and the correction is exact for the discretised
  target.  The continuous-target density-ratio machinery of ReTaSA/Liu et al.
  is therefore replaced by a finite-bin version, not reproduced.
"""

import numpy as np
from scipy.special import ndtr

# Non-uniform grid: 16 bins, 550 km/s is an exact edge (index 7) so the
# pointwise fast-wind weight is constant inside every bin.  Widths are fine
# (30-50 km/s) over the dense 240-640 km/s range and at the 550 metric
# boundary, and coarse (70-150 km/s) in a tail whose conditional spread is
# already 80-120 km/s.
EDGES = np.array([240., 290., 340., 390., 440., 490., 520., 550.,
                  580., 610., 640., 680., 730., 800., 900., 1000., 1150.])
THRESHOLD = 550.
# index of the first bin that lies entirely at or above THRESHOLD
FAST_BIN = int(np.flatnonzero(EDGES[:-1] == THRESHOLD)[0])
SIGMA_SCALE = 0.5
SIGMA_MIN = 10.
SIGMA_MAX = 50.


def bin_index(y):
    """Index of the bin containing each value; out-of-range values clip."""
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('targets must be finite')
    return np.clip(np.searchsorted(EDGES, y, side='right') - 1, 0, len(EDGES) - 2)


def sigma(y):
    """Local Gaussian bandwidth: half the width of the bin holding y, clipped."""
    index = bin_index(y)
    width = EDGES[index + 1] - EDGES[index]
    # np.clip drops the NaN that np.clip(x, None, 50) would keep
    return np.minimum(np.maximum(SIGMA_SCALE * width, SIGMA_MIN), SIGMA_MAX)


def soft_labels(y):
    """HL-Gauss label: Gaussian probability mass of each bin around y.

    Values outside the grid are pinned to the nearest grid bound first, so an
    out-of-support target still yields an honest one-sided soft label instead of
    an empty row.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y):
        raise ValueError('targets must be a nonempty 1-d array')
    if not np.isfinite(y).all():
        raise ValueError('targets must be finite')
    pinned = np.clip(y, EDGES[0], EDGES[-1])
    band = sigma(pinned)[:, None]
    high = ndtr((EDGES[None, 1:] - pinned[:, None]) / band)
    low = ndtr((EDGES[None, :-1] - pinned[:, None]) / band)
    q = np.clip(high - low, 0., None)
    total = q.sum(axis=1, keepdims=True)
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError('degenerate soft label; check the bin grid support')
    return (q / total).astype(np.float32)


def bin_weight(factor):
    """Per-bin training weight, a function of the label only (exactly invertible)."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('factor must be finite and at least one')
    weights = np.ones(len(EDGES) - 1)
    weights[FAST_BIN:] = float(factor)
    return weights


def point_weights(y, factor):
    """Mean-one sample weights implied by the per-bin weight of each target."""
    if factor == 1:
        return None
    weights = bin_weight(factor)[bin_index(y)]
    return weights / weights.mean()


def softmax(margins):
    """Row softmax over the class axis of an (n, K) raw-margin matrix."""
    z = np.asarray(margins, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(EDGES) - 1:
        raise ValueError('margins must be (n, K) with K = len(EDGES) - 1')
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def soft_cross_entropy(targets, margins):
    """Mean weighted cross-entropy of soft labels against softmax margins."""
    p = softmax(margins)
    return float(np.mean(-np.sum(np.asarray(targets) * np.log(np.maximum(p, 1e-12)), axis=1)))


class LabelRegistry(dict):
    """Maps a live DMatrix to its (soft-label matrix, sample weights).

    XGBoost's low-level ``obj`` hook receives ``(margins, dmat)``, so the soft
    labels and the sample weights are looked up by DMatrix identity and never
    have to be encoded in a 1-d DMatrix label.  The DMatrix label itself is the
    row index.  Weights are applied inside the objective on purpose: xgboost
    2.1.4 ignores ``DMatrix.weight`` on the custom-objective path (verified by
    ``test_pointwise_weights_reach_a_custom_objective``), so relying on the
    built-in weighting would silently train an unweighted model.
    """

    def __init__(self):
        super().__init__()
        self._held = []

    def add(self, dmat, targets, weights=None):
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != len(EDGES) - 1:
            raise ValueError('targets must be (n, K) with K = len(EDGES) - 1')
        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError('soft targets must be finite and nonnegative')
        index = np.asarray(dmat.get_label())
        if len(index) != len(targets):
            raise ValueError('one soft label row per DMatrix row is required')
        if (index != np.arange(len(targets))).any():
            raise ValueError('DMatrix label must be the row index')
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(targets),) or not np.isfinite(weights).all():
                raise ValueError('weights must be one finite value per row')
            if (weights <= 0).any():
                raise ValueError('weights must be positive')
        if dmat.get_weight() is not None and len(dmat.get_weight()):
            raise ValueError('do not also set DMatrix.weight; it is ignored under a custom objective')
        self[id(dmat)] = (targets, weights)
        self._held.append(dmat)
        return dmat

    def __call__(self, dmat):
        try:
            return self[id(dmat)]
        except KeyError as error:
            raise KeyError('DMatrix was not registered with soft labels') from error


def class_objective(lookup, hessian_floor=1e-6):
    """Soft-label categorical cross-entropy gradient/Hessian for XGBoost.

    Signed as ``obj(margins, dmat)``.  Hessians are the usual diagonal softmax
    majoriser, floored to stay positive, and both gradient and Hessian carry the
    sample weight of the row.
    """

    def objective(margins, dmat):
        targets, weights = lookup(dmat)
        p = softmax(margins)
        gradient = p - targets
        hessian = np.maximum(p * (1. - p), hessian_floor)
        if weights is not None:
            gradient = gradient * weights[:, None]
            hessian = hessian * weights[:, None]
        return gradient, hessian

    return objective


def representatives(values, floor=1.):
    """Per-bin representative speed used by the mean read-out.

    The training conditional mean inside each bin, falling back to the bin
    midpoint for empty bins.  Computed from unweighted training targets only, so
    both arms of the study share exactly the same read-out grid.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    index = bin_index(values)
    midpoint = (EDGES[:-1] + EDGES[1:]) / 2.
    out = midpoint.copy()
    for k in range(len(midpoint)):
        mask = index == k
        if mask.any():
            out[k] = values[mask].mean()
    out = np.maximum(out, floor)
    if not np.isfinite(out).all() or np.any(np.diff(out) <= 0):
        raise ValueError('bin representatives must be finite and increasing')
    return out


def histogram_mean(margins, reps, weights=None):
    """Point forecast: mean of the predicted histogram, optionally un-weighted.

    ``weights`` is the per-bin training weight.  Dividing the predicted class
    probabilities by it and renormalising inverts the target shift exactly when
    the weight is a function of the label only.
    """
    reps = np.asarray(reps, dtype=np.float64)
    if reps.shape != (len(EDGES) - 1,):
        raise ValueError('representatives must have one value per bin')
    p = softmax(margins)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != p.shape[1:] or (w <= 0).any():
            raise ValueError('bin weights must be positive with one value per bin')
        p = p / w[None, :]
        p = p / p.sum(axis=1, keepdims=True)
    return p @ reps, p
