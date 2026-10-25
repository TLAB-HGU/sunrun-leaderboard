"""Diagonal-surrogate tree partitions with exact coupled anchor leaf updates."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import xgboost as xgb

ROUNDS, RATE, LAMBDA = 300, .03, 10.


def groups_index(groups):
    groups = np.asarray(groups)
    if groups.ndim != 2 or groups.shape[1] != 2 or not len(groups):
        raise ValueError('groups must be nonempty source_fold,lead pairs')
    unique, inverse, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse, counts.astype(float)


def gradient_loss(residual, inverse, counts, gamma):
    r = np.asarray(residual, dtype=float)
    means = np.bincount(inverse, weights=r, minlength=len(counts))/counts
    grad = r+(gamma-1)*means[inverse]
    diagonal = 1+(gamma-1)/counts[inverse]
    loss = .5*(r@r+(gamma-1)*np.sum(counts*means**2))
    return grad, diagonal, float(loss)


def leaf_update(residual, leaves, inverse, counts, gamma, lam=LAMBDA):
    """Accumulate M' H M and M' H r without constructing an n-by-n H."""
    ids, position = np.unique(leaves, return_inverse=True)
    width = len(ids)
    group_leaf = np.bincount(inverse*width+position, minlength=len(counts)*width).reshape(len(counts), width)
    mean_r = np.bincount(inverse, weights=residual, minlength=len(counts))/counts
    matrix = np.diag(np.bincount(position, minlength=width).astype(float)+lam)
    matrix += (gamma-1)*(group_leaf.T/counts)@group_leaf
    rhs = np.bincount(position, weights=residual, minlength=width)+(gamma-1)*group_leaf.T@mean_r
    beta = np.linalg.solve(matrix, -rhs)
    if not np.isfinite(beta).all():
        raise ValueError('nonfinite coupled leaf solve')
    return ids, beta, beta[position], matrix, rhs


class AnchorBoost:
    def __init__(self, gamma, seed, rounds=ROUNDS, rate=RATE, min_child_weight=50):
        if gamma not in (1, 2) or not 0 < rate <= 1 or rounds < 1:
            raise ValueError('invalid fixed anchor recipe')
        self.gamma, self.seed, self.rounds, self.rate = gamma, seed, rounds, rate
        self.min_child_weight = min_child_weight
        self.trees, self.trace = [], []

    def fit(self, x, y, groups):
        x, y = np.asarray(x), np.asarray(y, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite or unmatched observed training data')
        self.group_keys, inverse, counts = groups_index(groups)
        if len(inverse) != len(y):
            raise ValueError('group row mismatch')
        self.group_counts = counts.astype(int)
        self.feature_count = x.shape[1]
        self.trees, self.trace = [], []
        prediction = np.zeros(len(y), dtype=float)
        matrix = xgb.DMatrix(x, label=np.zeros(len(y)), weight=1+(self.gamma-1)/counts[inverse], nthread=1)
        for iteration in range(self.rounds):
            residual = prediction-y
            grad, diagonal, before = gradient_loss(residual, inverse, counts, self.gamma)
            matrix.set_label(-grad/diagonal)
            tree = xgb.train(dict(objective='reg:squarederror', tree_method='hist', device='cpu',
                max_depth=3, max_bin=256, eta=1., min_child_weight=self.min_child_weight, subsample=.8,
                colsample_bytree=.9, reg_lambda=LAMBDA, nthread=1, base_score=0.,
                seed=int(self.seed*1000003+iteration)), matrix, num_boost_round=1)
            leaves = tree.predict(matrix, pred_leaf=True).astype(np.int64).reshape(-1)
            ids, beta, direction, _, rhs = leaf_update(residual, leaves, inverse, counts, self.gamma)
            prediction += self.rate*direction
            _, _, after = gradient_loss(prediction-y, inverse, counts, self.gamma)
            if not np.isfinite(after) or after > before+1e-10*max(1., before):
                raise ValueError('true anchor objective increased')
            self.trees.append((tree, ids, beta))
            self.trace.append(dict(round=iteration, before=before, after=after, leaves=len(ids),
                                   leaf_gradient_norm=float(np.linalg.norm(rhs))))
        self.training_prediction = prediction
        return self

    def predict(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.feature_count or not np.isfinite(x).all():
            raise ValueError('invalid prediction features')
        matrix = xgb.DMatrix(x, nthread=1)
        prediction = np.zeros(len(x), dtype=float)
        for tree, ids, beta in self.trees:
            leaves = tree.predict(matrix, pred_leaf=True).astype(np.int64).reshape(-1)
            positions = np.searchsorted(ids, leaves)
            if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], leaves):
                raise ValueError('unbound leaf identifier')
            prediction += self.rate*beta[positions]
        return prediction
