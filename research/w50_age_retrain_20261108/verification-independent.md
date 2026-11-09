# Independent fresh-fit verification

## Verdict

- **PARTIAL overall:** the bounded implementation and artifact audit **PASS**; the claimed 50% official-S reduction **FAILS**. Historical provenance and generalization remain incompletely proven.
- **Do not adopt or publish this treatment as an improvement.** No blocking implementation defect was found in the audited training, calibration, prediction, or scoring paths. This is a negative result for the fixed age-repair hypothesis, not completion of the wider research objective.

The acceptance criteria were three fresh CPU-only fits (seeds 42/43/44), preservation of the parent feature/configuration contract except causal `img_age_h` replacement, correct official cutoffs and 144-hour maturity/purge, reproducible saved models/calibration/predictions, official scorer parity, and the preregistered statistical/adoption gates. W is official S; the fixed reference is 97.49244208059969 and the 50% target is 48.746221040299844.

## Evidence

Fresh checks were executed locally without fitting models, computing hashes, installing packages, or editing data, models, scorer, or runtime state. Detailed per-period results are in [verification-independent.json](verification-independent.json).

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider research/w50_age_retrain_20261108/test_experiment.py` — **8 passed in 1.03s**. Tests cover causal forward propagation, no backfill, conflicting raw ages, parent-input immutability, query-label rejection, matched configurations, and fresh prior/full fit routing.
- [experiment.py](experiment.py), [parent-to-retrain.diff](parent-to-retrain.diff), and parent `core.py`/`baseline.py` — only the age loader changes the input matrix. Configuration contains no `reuse_run`; both prior and full stages call `xgb.train`. The shared runner retains the spectral/tail objective and frozen training-only offset procedure. The experimental entrypoint contains one architecture, with only the three registered seeds varying.
- Direct `Path.read_bytes()` equality — both current parent shared files match the corresponding snapshots in **all three** `../drop035_v650_confirm_20261108/runs/drop035_v650_s{42,43,44}/` directories. All recorded source sizes also match: core 20,521 bytes, baseline 6,702 bytes. This strengthens current-code parity without claiming immutable historical provenance. Seed 43/44 copied snapshot mtimes differ from the recorded remote nanoseconds by less than 100 ns; those are not asserted equal.
- Fresh actual-matrix checks — reloaded all three training/query grids, compared repaired ages against an independent latest-observed-frame `searchsorted` construction, and asserted exact equality (including NaNs) for every other constant and varying input. All three seeds' saved feature receipts match these recomputed receipts. Every metadata/configuration pair preserves the parent's 28 source features, ordered 212-column block designs, parameters, seed, periods, row counts, and training/query boundaries.

| Period | Training origins | Changed training ages | Query origins | Changed query ages |
|---|---:|---:|---:|---:|
| F1 | 28,297 | 523 | 2,089 | 63 |
| F2 | 30,457 | 586 | 2,113 | 21 |
| F3 | 32,665 | 604 | 2,113 | 104 |

Each training grid retains 1,270 leading missing ages; none are backward-filled. Query ages have no missing values. All training targets were finite. Training origins plus 144 hours are at or before the respective cutoffs (2025-12-29, 2026-03-29, 2026-06-29). Prior fitting origins plus 144 hours are at or before the calibration boundary. Query loaders return no labels.

- Fresh saved-model replay — loaded **54 boosters**, 18 per seed, covering prior/full stages and all three blocks in F1/F2/F3. Every booster has **900 rounds and 212 input columns**. Independently reconstructed prior predictions, final raw predictions, and the offset `0.25 * mean(last336 training_truth - clipped_prior_prediction)`; applied the documented clipping and offset. **Maximum absolute differences are exactly 0** for prior predictions, offsets, raw full predictions, and final predictions in all nine seed-period combinations. Calibration saved labels/origins equal the reloaded training rows. All nine pre-existing model-replay receipts agree with the fresh replay.
- Lifecycle — all three `train.exit` values are 0; each log contains 18 completed fit units, six per period, followed by completion. Models, period predictions, metadata, local scores, and `complete.json` exist; completion marks publication `not_required`. Every launch uses `--device cpu --threads 4`, and every `train.sh` exports `CUDA_VISIBLE_DEVICES=""`. The recorded live-process check also reports invisible CUDA for all three processes. These are historical launch/process receipts, not a fresh inspection of already-ended processes.
- Fresh official rescoring — directly loaded [../../leaderboard/scorer/competition.py](../../leaderboard/scorer/competition.py), version `lead-timse-6h-v1`, and the frozen `../offset_roll_complete_ablation_20261106/competition-observations.parquet`. Reassembled every control/treatment parquet from its period NPZ files and required frame equality. All six runs reproduce every stored total/F1/F2/F3 S, RMSE, TRMSE, and TIMSE within 1e-8. **No scope uses replacement forecasts.** Pooled MAE was independently recomputed.

| Seed | Parent S | Fresh age-repair S | Paired reduction |
|---|---:|---:|---:|
| 42 | 97.49244208059969 | 97.82503916944195 | −0.341152% |
| 43 | 97.87370833197167 | 97.92266669105470 | −0.050022% |
| 44 | 98.07451343718657 | 97.95989993386527 | +0.116864% |

Treatment mean S is **97.90253526478732** versus parent mean **97.81355461658598**. Treatment sample SD is 0.0696477723 and worst S is 97.9598999339. Mean paired relative reduction is **−0.0914366%**, with 95% interval **[−0.3279122%, +0.0558201%]**. Two seeds worsen; every treatment score is worse than the fixed best-single reference and far above 48.7462210403.

- [results.json](results.json), `analysis.exit`, and `bootstrap.npz` — analysis exited 0. Reviewed the period-stratified circular 168-hour bootstrap and its per-draw empirical TIMSE quantiles; recomputed the saved 2,000-draw interval endpoints, reproduced the first five deterministic draws exactly, and checked bootstrap point scores against fresh official scores. This was **not** a second full 2,000-draw execution. Candidate and 50% score gates fail; auxiliary deterioration gates pass. A passing auxiliary gate does not reverse the failed improvement gate.

## Gaps

- The requested hash evidence remains unfulfilled because earlier automatic approval review rejected hashes. **No hashes were computed or bypass attempted.** Bytes/mtime and direct current-file equality are narrower evidence.
- Fresh remote provenance has `git_commit: null`; the source commit references a dirty workspace. Local parent source files are untracked, so an empty git diff alone proves nothing about historical immutability. Matching current parent snapshots, saved metadata, and exact replay do not independently reconstruct every historical source/data state.
- Historical public availability at forecast issue time and independent corroboration of all upstream observations/features remain incomplete. Causal array indexing proves a temporal computation rule, not publication-time availability. The raw age series itself is taken as supplied.
- Repeated F1/F2/F3 exposure and earlier leaderboard-driven feature selection remain selection bias. The bootstrap conditions on these periods and three fitted seeds; it is not evidence of unseen future performance. External SOTA comparisons remain tied to the recorded snapshot rather than a new live-board audit.
- Strategy and other progress documents may still contain earlier pending-state sections; final outcome/failure diagnosis is the leader's integration responsibility. This report changes no campaign gate or research-goal state.

## Risks

The negative result supports failure of material benefit under this fixed age-repair configuration. It does not prove that corrected ages are universally useless, that every upstream observation is valid, or that training has reached a global optimum. No additional fit should be chosen by iterating against these official scores. Retain the parent baseline; complete the failure diagnosis and preregister a distinct evidence-supported hypothesis before another experiment. Any independent future validation proposal must respect the user's existing evaluation policy rather than silently introducing new selection folds.

Stop condition for this review is met: the assigned implementation/artifact checks and official-score replay are complete, unresolved evidence is explicitly bounded, and no performance success or global-goal completion is claimed.
