# Publication packaging plan

2026-11-12: user requested publication as `current_offset_roll_direct_s`.
The selected architecture is arm B (direct S only) from
`research/current_offset_roll_direct_s_20261109`, seeds 42/43/44.
This is packaging of completed runs, not a new hypothesis or training run.

Preserve all 31 source / 215 expanded features, data fill, cutoff, seed,
900-round model configuration, prior/full split, clipping and frozen offset.
Remove unrelated objective branches from the self-contained script. Reuse the
saved 54 boosters and nine frozen calibration states. Verify objective values,
gradients and curvature proxies against the original pure-S callback before
inference; then require bitwise equality to all nine saved prediction matrices.
Create the fixed arithmetic mean of all three seeds without fitting weights,
and score it with the official scorer and frozen official observations.
No new model fitting is authorized for this packaging task.

## Packaging outcome

Pure-S value/gradient/curvature parity passed on three independent finite matrices.
All 54 booster byte arrays and nine frozen calibration states are unchanged.
Raw-table inference reproduced all three seed forecasts exactly, with original
seed scores unchanged. Equal-weight mean3 Total S is 97.53022862821581,
with no persistence replacements. Full training was not rerun. See
verification.json and provenance.json for detailed evidence. Publication itself
is handled by the leader after this local package handoff.
