# Training-only loss-distribution audit

No fitting, evaluation truth, target-policy change or new experiment. This
extends the earlier imputed-label count with F1-training-only distributions.
All2,037,384 rows have origin+144h<=2025-12-29 and targets before that cutoff.
Original observed labels and every non-y key match prepared data exactly.

668,736rows(32.8233%) have forward-filled labels. With the existing550km/s
tail multiplier4 they represent33.6333% of diagonal loss weight, before
horizon normalization and spectral coupling. This is substantial exposure to
synthetic repeated targets, not proof that dropping or changing them helps.

Observed/imputed label mean438.81/448.20km/s; fraction>=550 is13.16/14.89%,
fraction>=700 is2.35/1.92%. The simple claim that fill always lowers speed is
contradicted by these means. Extreme tail frequency differs, but this alone
does not explain the model's evaluation-event attenuation.

Imputed source age: median1h,p904h,p9911h,max30h. Missing fractions at target
UTC03/09/15/21 are81.43/93.98/94.49/85.35%, much higher than surrounding hours.
Non-random clock dependence is already present before the benchmark period;
it is not a phenomenon discovered solely from exposed F1/F2/F3 errors.

Inference: training's supervised distribution and official observed-only loss
distribution differ. Whether this costs skill is untested; do not interpret
descriptive shifts as an achievable50% reduction. The existing user-confirmed
training-fill policy remains unchanged, and is not silently replaced by this
diagnostic or the ongoing image-age experiment. A future permitted causal
comparison would need preregistration and one isolated training-loss change.

Reproduction: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python analysis/w50_audit_20261108/training_imputation_diagnostic.py`.
Detailed row/lead/hour/source identities are in the adjacent JSON. No official
observation file or query y is read by this script.
