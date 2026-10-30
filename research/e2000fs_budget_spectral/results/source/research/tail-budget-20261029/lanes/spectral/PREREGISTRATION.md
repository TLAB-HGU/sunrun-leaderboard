# Frozen single candidate, before any candidate score
Direct parent: e2000fs_dirmo_24_feat30_pointweight4/run.py and runs/full_v1/manifest.json.
Frozen input/source hashes will be checked against this manifest at runtime.
Three 24-output XGBoost vector trees; original 237 columns/block, depth3,
900 rounds, eta0.03, seed42, subsample0.8, colsample0.7, min_child_weight20,
reg_lambda5, scalar unweighted block mean initialization. CPU3 and BLAS1.

For each 24-vector e=p-y, F is the complete complex 24-point unitary DFT
(np.fft.fft(norm='ortho')). w_ih=(4 if y_ih>=550 else1)/training_mean_h(w).
Single per-row loss (sum convention to match the parent gradient exactly):
 L = 0.5 sum_h w_ih e_ih² + lambda s² sum_k [sqrt(1+|F e|_k²/s²)-1].
Fixed lambda=0.10, s=50 km/s. Spectral loss is dimensionless after division by
s², then scaled back by s²; complete FFT has no rFFT multiplicity ambiguity.
Complex radial pseudo-Huber uses the complex residual, not magnitude differences;
it preserves phase alignment to y. No output correction or target transform.
The point4 anchor is unchanged including DC. All frequencies have equal coefficient.
FreDF (https://arxiv.org/abs/2402.02399) motivates time/frequency alignment;
this is a differentiable GBDT adaptation, not its neural architecture or a general
MSE/likelihood bias theorem. Literature evidence is the assigned LITERATURE.md.

Let z=Fe, rho=(1+|z|²/s²)^(-1/2).
 g = w*e + lambda Re(F* (rho*z)).
 Hspec = lambda Re[F* blockdiag(rho I2-rho³ vv^T/s²) F], v=(Re z,Im z).
Its eigenvalues are nonnegative and bounded by lambda (F unitary).
Return diagonal Hessian majorizer diag(w)+lambda I. This dominates actual full
Hessian in PSD order, retains no off-diagonal entries and deliberately overstates
curvature as residual grows. Save actual finite-difference Hessian/majorizer gap
receipts on fixed synthetic residuals, and training residual rho summaries per fit.
Lambda0 returns the exact parent gradient/Hessian; s0 is invalid, not a no-op.
Squared unitary Fourier error equals time L2 (Parseval) and is excluded.

Both parent control (lambda0) and candidate use identical origin+144h<=cutoff
training rows, enforcing every target+72h<=cutoff. ACE lag0. No old model reuse.
D1 2024-12-04..2025-02-22; D2 2025-03-04..2025-05-22;
D3 2025-09-04..2025-11-22; C 2025-12-04..2026-02-22 (ends 23h).
F1 cutoff2026-02-26, F2 cutoff2026-05-29. Fixed900 rounds, no early stopping.
Internal signal: pooled D1-D3 total improves and high-speed nonworsens, >=2/3
windows same direction, C same direction. Diagnostic only; never skips official.
Official target: total<3500 AND actual>=550 MSE<6000. Transferred only if internal
positive and paired official total improves/high-speed nonworsens. F1/F2 separate.
No official-score selection. One candidate, one official submission, no --replace.

Smoke plumbing repair: high-speed count0 yields None and must be false for
internal-positive checks, rather than comparing None. No candidate constants changed.
Smoke_v1 retained; rerun smoke_v2 before full launch.
