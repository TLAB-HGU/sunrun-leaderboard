# Output-sharing history and the block-size confound

Bounded repository lookup by native explore specialist, preserved by root.
No current-treatment predictions/truth/scores, fitting or next-experiment selection.
Stored historical results below are not newly rescored comparisons.

CurrentD has28source/212expanded inputs, three independent24lead vector-leaf
XGBoost boosters, depth3,900rounds, tail4+spectral+offset. Within a vector tree,
outputs share partitions with output-specific leaf values; separate boosters
share no partitions. Every output gets the same20constant+24×8varying input row.
Exact upstream split-gain aggregation and whether long-lead gradients dominate
are unverified; do not present that mechanism as fact.

Changing block_size24→12 also changes212→116input columns, FFT width, scalar
block mean base_score and number of fitted boosters. It cannot isolate split
sharing while retaining the original input information/loss. A naive rerun
would violate the intended single-change interpretation.

Existing completed block12/72 tests are on the older30source parent, changing
237→129/669inputs and spectral width. Block12seeds42/43/44 S=100.1522/99.9536/
99.9811 vs older baseline98.7627/99.9322/100.2128; all exits0. Block72scores=
102.7592/102.2466/102.2035, all exits0. These are confounded historical negatives.

Scalar mean_control42/43 are completed,44had no exit when inspected. Its30feature
+B1 configuration replaces retained tail+spectral with unweightedMSE. Legacy
single-lead e2000_direct_s42 has exit0 and216saved models, but1530inputs and a
different squared-error fitting/postprocessing recipe. A dev runner exposes
one_output_per_tree without a clean execution receipt. No completed clean
currentD output-sharing-only comparison was established by this lookup.

The older short-lead strategy already proposed shared-split competition as
unverified; the actual subsequent intervention was persistence blending.
These findings do not establish that another topology improves S or enables50%.
The next hypothesis still depends on the current complete quantitative diagnosis.

Concrete source references and per-run evidence are in output_structure_history.json.
