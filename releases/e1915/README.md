# E1915 — frozen stable-UWR direct forecast

Source: Morten Blorstad, Berent A. S. Lunde, Nello Blaser, *Stable Update of
Regression Trees*, arXiv:2402.13655. CPU tabular adaptation: 111 existing base111
features, three direct horizon heads, nine boosted fits of 900 rounds each (an
earlier chronological snapshot and the final single learner per head), seeds
0/1/2, training targets strictly before 2026-05-29 UTC.

The stable (uncertainty-weighted) penalty g_i = beta * phi(x_i) pulls each row
toward the earlier snapshot in proportion to the inverse local residual variance,
beta = 0.6, alpha = 0, gamma cap 10 (never reached). The earlier snapshot only
constrains the training objective: the submitted forecast is the single final
ensemble, with no output correction, blending or residual head.

The user explicitly requested this leaderboard entry after both recorded
pre-official gates had failed: ROLL3 pooled MSE -1.21% with a 72h-block bootstrap
95% CI of [-3.85%, +1.39%], and the arrival gate showed no timing gain (71 events,
capped-MAE difference CI -0.48..+0.42h). This is a submitted candidate, not an
adopted champion, and not a gate pass. The accepted research parent (E1914) and
all recorded failures are unchanged. The administrative execution that produced
these official-grid forecasts is E1940; the scientific identity stays E1915.

## Reproduce the submitted forecast

From the repository root:

    python scripts/seongeun/e1915_stable_uwr_prediction_mean/408fec8d05d37a2cf470955176e15c4d2442297e6f80073898884a80a3e9acbb.py --output predictions.parquet

Requires NumPy, pandas and pyarrow. The script verifies the configuration and the
three seed archives and reproduces their predeclared equal prediction mean. It
contains no labels or observations. This is fixed-grid forecast replay, not
end-to-end retraining. The archived `source/` modules preserve the registered
training and prediction implementation; `official-verification.json` records the
checkpoint-to-output equivalence, the 900-round evidence and the causal cutoff.

## Timing

The leaderboard timing is measured from actual predictions of nine loaded CPU
ensembles plus the equal mean of three seeds over all 2,833 origins. Data loading,
model loading/initialization, training, feature preparation and output writing are
excluded. The replay script's cached averaging time is reported separately and is
never presented as model inference time.

## Limits

Inherited base111 source availability receipts and the ACE observation+2h contract
are reused; publication does not independently certify every historical source
issue timestamp. Stability regularization helped long horizons (49-72h -1.88%,
25-48h -1.19%) and hurt the 1-6h band (+2.92%) on the exposed ROLL3 window, and no
official-window tuning was performed.
