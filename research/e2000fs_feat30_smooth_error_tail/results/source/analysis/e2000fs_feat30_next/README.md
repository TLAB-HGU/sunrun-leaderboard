# Smooth error-tail follow-up

One fixed direct LightGBM arm, using the same frozen HPO parameters, features,
targets and purged chronological validation as the pointwise4 experiment.

Inspired by Ramaswamy & Prashanth (2025), *Optimizing Shortfall Risk Metric for
Learning Regression Models*, https://arxiv.org/abs/2505.17777 . This is a smooth
surrogate adaptation, not the paper's exact shortfall-risk/bisection algorithm.
The fixed 100 km/s cutoff and 50 km/s smoothing scale are engineering choices.

For error e=prediction-truth, loss is
`0.5*e² + 1.5*50²*softplus((e²-100²)/50²)`.
The gradient multiplier approaches 1 for small errors and 4 for large errors of
either sign. False high predictions and missed high predictions both receive
increased emphasis. Targets are centered on each training set's mean to provide
the custom objective's intercept; that stored mean is added at inference.

Train internal, F1 and F2 in order. This is one preregistered arm with no parameter
selection. Internal acceptance failure does not suppress the fixed F1/F2 report.
Report ordinary MSE against L2 and high MSE against pointwise4 and blockwise4.
Both low and high conditions must pass. No automatic promotion or upload.
Official-exposed follow-up and point-estimate comparisons are explicitly disclosed.

Run `/home/t-lab01/sunrun/.venv/bin/python -u analysis/e2000fs_feat30_next/run.py`.
`--smoke` runs all 72 horizons in all three stages with three trees and subsampled
real data. Full output is `runs/full_20261028_v1/`. The manifest pins code, helper,
reference and data hashes. Model checkpoints pin stage, horizon, training indices,
model, mean offset and loss history. Resume refuses changed identities or files.
