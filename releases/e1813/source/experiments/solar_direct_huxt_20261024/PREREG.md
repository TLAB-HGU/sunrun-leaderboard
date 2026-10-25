# D_DIRECTHUXT_20261024 — pre-registration

## User direction (2026-10-24)

Direct forecasting only: no post-hoc bias/amplitude/time correction, no residual head on a
frozen champion, no forecast blending/stacking, no ensemble/expert mixture of forecast
outputs. The frozen champion is an evaluation comparator only. A direct model may change its
input representation, training objective and dynamics.

## Problem

The strongest direct control (E1782, champion-family base without its k-NN residual) is
3775.61 official seed-mean, 15.06 % above champion E0472 (3281.31) and 17.1 % above the public
champion E1604 (3222.88). The measured HUXt trajectory/ring cache
(`experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz`, derived read-only from the
E1466 causal TD-WSA HUXt state) is not used by that base. The champion's own HSS bump uses only
six coarse HUXt summaries and only through a post-hoc path.

## Hypothesis

A single direct 3-head XGBoost trained on speed/ACE/CH/SUVI/ENLIL features **plus** the causal
HUXt Earth trajectory and the HUXt radial-longitude ring at the target horizon, and trained on
the persistence anomaly `y - v_lag0` rather than `y`, improves the direct base on all three
pre-official periods and on the official window without any correction, residual, blend or
ensemble of forecast outputs.

Mechanism: the HUXt trajectory is a physical propagation of the WSA boundary; the ring at the
target horizon encodes whether Earth sits on the rising/falling edge of a stream. The
persistence-anomaly target removes the dominant predictable level from the learning problem and
lets the trees spend capacity on the 25–72 h departure.

## Inputs and causality

- ACE/OMNI features and labels: `store/ch-v1/ace.parquet` under the existing ACE +2 h rule;
  training labels only with `target_t < 2026-05-29`.
- CH: `store/ch-v1/ch-hourly.parquet` ffill <= 6 h.
- SUVI: the existing d1 longitude-strip module and its proxy rule (unchanged).
- ENLIL: `store/headroom-audit/donki/WSAEnlilSimulations_*.json`, only runs with
  `modelCompletionTime <= origin`.
- HUXt: `experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz`, derived read-only from
  the E1466 causal TD-WSA state; the E1466 state enforces SW-012 map availability
  `max(obs+2h, Last-Modified) <= origin` for every boundary substep. This is the same
  "operational proxy evidence" already accepted for E1466/E1590/E1604. Cache SHA is recorded in
  the run receipt.
- No new data, no downloads, no GPU, no DL.

## Model

- 3 heads: `h<=6`, `7<=h<=24`, `h>25` (the champion-family split), one XGBoost per head;
  `members="d33"` (one short and one mid level member plus one long event member) — **no member
  stitching, no mixture, no seed averaging inside a prediction**.
- depth 3, 900 trees, lr 0.03, subsample 0.8, colsample 0.9, min_child_weight 20, lambda 5,
  CPU `n_jobs=4`.
- Target: `y - v_lag0`; prediction = model output + `v_lag0`.
- 3 genuine seeds (XGB random_state and the E0421 event-seed schedule).
- Features: `existing_all` (h + speed lags/rolls + rec26/27/28 + ACE plasma/EPAM + CH) +
  17 SUVI strips + 6 ENLIL + `h` + 14 HUXt columns (trajectory at the target lead, trajectory
  mean/sd/min/max/argmax, ring at Earth ±0/1/2 cells, ring max/min/std).

## Gates (fixed before any official score)

Pre-official (dev 2025-08-04..11-27, selection 2025-12-04..2026-02-25, audit
2026-03-04..05-28): candidate must improve on the matched direct control E1782 on all three
periods by >= 2 % each in MSE.

Official: one `expos lb-eval` with 3 seeds. Report total seed-mean, four high cells, F1, and the
gap to the public champion 3222.883 / 5 % target 3061.739. Adoption only if the unchanged
leaderboard gate passes (seed-mean below current best, every seed below it, all four high cells
not worse, F1 >= champion) and an independent review accepts. No upload.

## Falsification

If the HUXt block does not improve all three pre-official periods, or if the official seed-mean
is not below E1782 3775.61, the hypothesis is false and the direction stops. A gain that exists
only on one period is a representation artefact, not evidence.

## Recovery note (E1801)

E1800 completed dev/selection/audit diagnostics and died before the official grid (process gone,
no run record; likely memory pressure building the full 2,833-origin grid at once). E1801
re-runs **official only** with the identical model and features, predicting the official grid in
240-origin chunks. The three diagnostic periods are not recomputed; their E1800 outputs are
kept only as diagnostics and are not used for the 5 % claim. The full-period leaderboard total
remains the only performance evidence.

## Recovery note (E1809)

E1808's official path fitted the model on `y` but added `v_lag0` at prediction time, producing
a ~+400 km/s offset (official MSE 191507; the diagnostic periods were unaffected because
`fit_predict` applied the anomaly target correctly). E1809 fixes only that target line:
the official training frame is now `y - v_lag0`, identical to the pre-registered and
diagnostic-period model. No feature, model or split change; checkpoints restart empty.
