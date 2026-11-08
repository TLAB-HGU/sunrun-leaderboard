# Original A minus the two 3/3-improving features: fresh confirmation

Frozen before training, 2026-11-08 KST. The user explicitly requests retraining
and leaderboard publication of the configuration containing only the two
feature removals whose individual LOCO results improved at all three seeds.

## Hypothesis and exact intervention

Parent A is the original `e2000fs_calib_offset_roll`, not A+B1, the pruned model,
or the previously published eight-removal/Hessian/alpha combination. Remove
exactly `ch_ch_t035_equatorial_area` and `strip_v650_dark`. The explicit retained
feature set is the other 28 original source features. The parent feature list
and actual expanded columns are verified before launch and in every period.

Everything else remains original A: XGBoost depth3, learning rate .03, 900
rounds, subsample .8, colsample .7, min-child-weight20, lambda5, 3x24h joint
outputs, tail weight4 at550, full spectral gradient/Hessian coefficient .1 and
scale50, training-only horizon offset alpha .25 over the last336 calibration
rows, component/final clipping [1,3000]. No B1, no fresh features, no changed
objective, no seed selection, no previous prediction/model reuse.

The individual removals had mean delta S -1.454302 and -0.495362 versus A,
respectively, with all three seeds negative. Neither passed the original
bootstrap/BH contribution gate. They were selected after viewing official
results. Their combination is an **exploratory user-selected combination**;
individual effects cannot be added and the repeated official periods are not
independent generalization evidence. This campaign tests their joint behavior,
not independent causal attribution or a new literature-supported architecture.

Existing mechanism/evidence sources:
- `../offset_roll_complete_ablation_20261106/strategy.md` and
  `analysis/results.json`: frozen parent, LOCO design and three-seed observations.
- `../post_ablation_five_axes_20261108/strategy.md`: explains why unconfirmed
  contribution cannot justify wholesale pruning. The failed final-pruned
  configuration is not used as a parent.
- https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html
  supports the existing vector-leaf joint-output implementation; it does not
  predict that removing these two features will improve solar-wind scores.

## Fitting and evaluation

Fresh seeds42/43/44 each cover official F1/F2/F3 only, under the predecessor's
frozen observation revision and official scorer. Cutoffs are 2025-12-29,
2026-03-29 and 2026-06-29. Enforce origin+144h label maturity and the same purge
before the training-only last-three-month calibration holdout. Official truth
enters scoring only; no internal experiment-selection folds or post-hoc fitting.

Compare every seed with its matching original A, separately report adopted
A+B1 and external-board reference, and report S/RMSE/TRMSE/TIMSE/MAE, period
scores, mean/standard deviation/worst over seeds. A performance-candidate gate
requires mean S at least1% below A, all seeds improving, paired168h block
bootstrap2000 CI upper<0, MAE/RMSE deterioration<=1% and period S deterioration
<=2%. The external SOTA-2% goal is a separate comparison. A result not meeting
these gates is reported without promotion; do not tune or revert removals.

User requests publication regardless of the gate outcome. Publish all three
seed results and their preregistered fixed arithmetic prediction mean with
weights1/3. Never choose the best seed or fit ensemble weights to official
targets. Mark the prediction mean as a three-model ensemble, distinct from the
mean of three scalar S scores and the single-model results.

## Resources, lifecycle and completion

Inspect actual local/golab2 workload and RAM before admission. GBM CPU only with
`CUDA_VISIBLE_DEVICES=''`, four threads per job, at most one new campaign job
locally and two on golab2 when both fit the measured budget. Include other campaigns in the six-workload/24-thread host limit,
require at least11GiB MemAvailable before admission to reserve6GiB for an
estimated5GiB job, and check current CPU load. Preserve running experiments.
Do not reserve resources with placeholder processes.

Use independent exact-owned tmux sessions and logs. Record launch metadata,
source snapshots/diff, Git commit/dirty state, library versions, data byte size,
row counts/ranges/columns, source features, expanded columns, parameters and seed.
Do not compute file hashes. Emit JSON period/block/unit/units records and
F1/F2/F3 predictions/metadata, train.exit and validated completion receipts.
The existing sunrun monitor must discover the standard layout unchanged.

Register all four intended leaderboard entries, verify each local official
score against submit dry-run, publish replay and pinned architecture sources,
submit, and verify stored scores and visible Space rows. Completion requires
three normal training exits, all artifacts and reproducible scores, four
verified publications and architecture links. Preserve any failure logs and
classify hypothesis failure, implementation/protocol failure and resource
failure separately. A submitted command/tmux session is not completion.

## Post-run record

Pending fresh fits. Append actual score comparisons, adoption decision and
publication evidence after validation. Existing results must never be described
as fresh training results for this campaign.
