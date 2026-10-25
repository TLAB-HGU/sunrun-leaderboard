"""CPU adaptation of "Stable Update of Regression Trees" (arXiv:2402.13655).

Paper: Morten Blorstad, Berent A. S. Lunde, Nello Blaser, *Stable Update of
Regression Trees*, first submitted 2024-02-21.

Mechanism kept from the paper
-----------------------------
The paper replaces the squared-error loss of a regression tree by a stable loss

    L_SL(y_i, f_1(x_i); f_0(x_i)) = (y_i - f_1(x_i))^2
                                    + gamma_i (f_0(x_i) - f_1(x_i))^2 ,
    gamma_i = alpha + beta * phi(x_i) ,

where f_0 is the tree fitted on the earlier data, phi(x_i) = c / (n(x_i) *
sigma^2_{w_0(x_i)} + eps) is the uncertainty-weighted regularization (UWR)
term, c is the average response variance across leaves and n(x_i) sigma^2 is
the leaf-level estimate of the prediction variance.  The tree is grown with the
second-order split gain using g_i = -2 (y_i + gamma_i f_0(x_i)) and
h_i = 2 (1 + gamma_i), and for squared error the leaf value is

    w_d = sum_{i in d} (y_i + gamma_i f_0(x_i)) / sum_{i in d} (1 + gamma_i) .

For squared error this is exactly a *weighted* least-squares problem with
pseudo-response z_i = (y_i + gamma_i f_0(x_i)) / (1 + gamma_i) and weight
s_i = 1 + gamma_i, because s_i (z_i - w)^2 has the same gradient and Hessian as
L_SL at f_1 = w.  The implementation therefore hands (z_i, s_i) to the same
XGBoost learner the frozen baseline uses, which makes the adaptation exact in
the loss/gain sense rather than an approximation of it.

How the paper is adapted for this project (stated, not hidden)
-------------------------------------------------------------
1.  The paper updates a single regression tree; this project needs a direct
    1-72 h forecaster competitive with a 900-tree boosted baseline.  The stable
    loss is applied to a boosted ensemble of regression trees.  Each boosting
    tree is fitted on the pseudo-response/weight pair above, so every tree's
    leaf values and split gains use the stable-loss gradient/Hessian.
2.  The paper's uncertainty sigma^2_{w_0} is the Huber-sandwich estimate on the
    leaves of f_0 with the Cox-Ingersoll-Ross split correction
    (1 + E[max_j B_j])/2.  For a 900-tree ensemble the per-tree leaf sandwich is
    not the variance of the ensemble prediction, and evaluating it on 900
    partitions is not tractable on the CPU budget.  This adaptation estimates
    the same quantity - the local variance of the response around f_0 - with a
    dedicated shallow variance tree fitted on the squared residuals of f_0, and
    uses its leaf size as n(x).  c is the mean predicted variance on the f_0
    training rows, so phi is on the same O(1) scale as the paper's (there
    n(x) sigma^2_{w_0} ~ Var(y|x)), and beta keeps its meaning.  The CIR
    correction is not applied (factor >= 1 in the paper); we use factor 1.
3.  The paper updates f_0 on D_0 with f_1 on D_1 = D_0 + new data.  Here the
    training prefix is replayed chronologically: f_0 is fitted on the older
    share of the eligible rows (the first 1 - 1/blocks of the unique origins)
    and the updated model on all eligible rows.  The final model is a single
    new tree ensemble; f_0 only enters the training objective.  No output is
    blended with f_0 and no residual correction is applied.
4.  The paper tunes (alpha, beta) on a grid and reports a Pareto front of
    predictability vs stability.  This adaptation pre-registers the paper's
    Pareto-efficient setting (alpha, beta) = (0.2, 0.6) for the UWR arm (its
    pure-UWR sibling (0, 0.6) is the primary arm here; see DIRECTION), and
    treats constant regularization (alpha > 0, beta = 0) as an internal
    diagnostic only.  Nothing is selected on the protected evaluation.

The class is estimator-shaped (fit/predict) so contracts.ContractModel keeps
the target codec, checkpoint identity and prediction guards unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from contracts import ContractModel, TargetSpec  # noqa: F401  (documented dependency)

STABLE_KEYS = ('stable_update', 'stable_alpha', 'stable_beta', 'stable_blocks',
               'stable_eps_rel', 'stable_gamma_cap', 'stable_var_depth',
               'stable_var_min_leaf')
UPDATES = ('off', 'uwr', 'constant', 'combined')
CURVE_MARKS = (1, 25, 100, 200, 400, 600, 900)


def _frame(X):
    """Feature block as a float64 C-order array (XGBoost casts to float32 itself)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _subsample(n, cap=20000, seed=0):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


class StableUWRBoost:
    """Boosted regression trees trained with the paper's stable (UWR) loss."""

    def __init__(self, seed=0, n_jobs=1, n_estimators=900, learning_rate=0.03,
                 max_depth=3, subsample=0.8, colsample_bytree=0.9,
                 min_child_weight=20.0, reg_lambda=5.0, update='off', alpha=0.0,
                 beta=0.6, blocks=2, eps_rel=0.01, gamma_cap=10.0,
                 var_depth=6, var_min_leaf=300, max_bin=256):
        if update not in UPDATES:
            raise ValueError('unknown stable update mode: ' + str(update))
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.update = update
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.blocks = int(blocks)
        self.eps_rel = float(eps_rel)
        self.gamma_cap = float(gamma_cap)
        self.var_depth = int(var_depth)
        self.var_min_leaf = int(var_min_leaf)
        self.max_bin = int(max_bin)

    # -- learner -----------------------------------------------------------
    def _learner(self):
        return XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, reg_lambda=self.reg_lambda,
            objective='reg:squarederror', tree_method='hist', device='cpu',
            max_bin=self.max_bin, random_state=self.seed, n_jobs=self.n_jobs)

    def _curve(self, booster, X, y):
        n = len(y)
        if n == 0:
            return []
        idx = _subsample(n, seed=self.seed)
        sub = X[idx]
        target = y[idx]
        out = []
        for mark in sorted({m for m in CURVE_MARKS if m <= self.n_estimators} | {self.n_estimators}):
            pred = booster.predict(sub, iteration_range=(0, min(mark, self.n_estimators)))
            out.append({'trees': int(min(mark, self.n_estimators)),
                        'train_mse': float(np.mean((pred - target) ** 2))})
        return out

    # -- fit / predict -----------------------------------------------------
    def fit(self, X, y, groups=None):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError('stable update needs a finite target vector')
        self.n_features_in_ = int(X.shape[1])
        self.receipt_ = {'update': self.update, 'alpha': self.alpha, 'beta': self.beta,
                         'blocks': self.blocks, 'eps_rel': self.eps_rel,
                         'gamma_cap': self.gamma_cap, 'n_estimators': self.n_estimators}
        if self.update == 'off':
            # gamma == 0 everywhere.  The recurrence then has no dependence on
            # f_0 (z_i = y_i, s_i = 1), so the update schedule collapses to a
            # single plain fit on the eligible prefix: the declared no-penalty
            # control of the same data and schedule.
            model = self._learner()
            model.fit(X, y)
            Xa = _frame(X)
            self.fitted_ = [model]
            self.receipt_.update(gamma_mean=0.0, gamma_p50=0.0, gamma_p95=0.0,
                                 gamma_max=0.0, weight_mean=1.0, n_snapshots=1,
                                 training_curve=self._curve(model, Xa, y),
                                 train_mse=float(np.mean((model.predict(Xa) - y) ** 2)))
            return self

        if groups is None:
            raise ValueError('stable update requires chronological origin groups')
        origins = pd.DatetimeIndex(groups)
        if len(origins) != len(y):
            raise ValueError('group vector length mismatch')
        unique = np.sort(pd.unique(origins))
        if self.blocks < 2 or len(unique) < self.blocks:
            raise ValueError('stable update needs at least two non-empty time blocks')
        edge = int(np.floor(len(unique) * (self.blocks - 1) / self.blocks)) - 1
        if edge < 0 or edge >= len(unique) - 1:
            raise ValueError('degenerate update boundary')
        boundary = unique[edge]
        old = origins <= boundary
        new = ~old
        if old.sum() < 1000 or new.sum() < 1000:
            raise ValueError('update block too small')

        Xa = _frame(X)
        base = self._learner()
        base.fit(Xa[old], y[old])
        prior = base.predict(Xa)
        residual2 = (y[old] - prior[old]) ** 2
        variance = DecisionTreeRegressor(max_depth=self.var_depth,
                                         min_samples_leaf=self.var_min_leaf,
                                         random_state=self.seed)
        variance.fit(Xa[old], residual2)
        v = np.clip(variance.predict(Xa), 0.0, None)
        c = float(np.mean(v[old]))
        eps = self.eps_rel * c
        phi = c / (v + eps)
        gamma = np.clip(self.alpha + self.beta * phi, 0.0, self.gamma_cap)
        weight = 1.0 + gamma
        pseudo = (y + gamma * prior) / weight

        final = self._learner()
        final.fit(Xa, pseudo, sample_weight=weight)
        self.fitted_ = [base, final]
        self.boundary_ = pd.Timestamp(boundary)
        self.receipt_.update(
            n_snapshots=2, boundary=str(pd.Timestamp(boundary)),
            last_origin=str(pd.Timestamp(unique[-1])), blocks_registered=self.blocks,
            n_old=int(old.sum()), n_new=int(new.sum()),
            gamma_mean=float(gamma.mean()), gamma_p50=float(np.median(gamma)),
            gamma_p95=float(np.percentile(gamma, 95)), gamma_max=float(gamma.max()),
            weight_sum=float(weight.sum()), weight_mean=float(weight.mean()),
            phi_mean=float(phi.mean()), phi_p95=float(np.percentile(phi, 95)),
            phi_max=float(phi.max()), variance_mean=float(v.mean()),
            c=float(c), eps=float(eps),
            base_train_mse=float(np.mean((prior[old] - y[old]) ** 2)),
            base_curve=self._curve(base, Xa[old], y[old]),
            training_curve=self._curve(final, Xa, y),
            train_mse=float(np.mean((final.predict(Xa) - y) ** 2)),
            objective=float(np.mean(weight * (pseudo - final.predict(Xa)) ** 2)),
            var_tree_depth=int(variance.get_depth()),
            var_tree_leaves=int(variance.get_n_leaves()))
        return self

    def predict(self, X):
        if not getattr(self, 'fitted_', None):
            raise ValueError('estimator is not fitted')
        return np.asarray(self.fitted_[-1].predict(_frame(X)), dtype=np.float64)

    def receipt(self):
        import copy
        return copy.deepcopy(self.receipt_)
