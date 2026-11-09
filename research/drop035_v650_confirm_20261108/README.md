# Fresh original-A model with the two 3/3-improving removals

Remove only `ch_ch_t035_equatorial_area` and `strip_v650_dark`, retain the other
28 features and all original-A model/calibration settings. No B1. Fresh
seeds42/43/44, official F1/F2/F3, CPU only. Read `strategy.md` for the frozen
protocol and the limits of the post-LOCO exploratory combination.

Training, collection and publication use separate durable services and exact
tmux identities. Existing campaigns are preserved. Publication includes all
three single-seed models and their fixed equal-weight prediction mean. The
single-model scores and ensemble score are reported separately.

Inspect `scheduler-state.json`, `remote-state.json`, `runtime.json`,
`controller-state.json` and `runs/*/train.log`. A running process or submitted
command does not mean completion. `campaign-complete.json` must be backed by
the four verified leaderboard submissions, matching official scores and pinned
architecture links.

After all three fresh runs validate:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python research/drop035_v650_confirm_20261108/analyze_results.py
```

This read-only-to-models analysis writes paired official score comparisons and
uncertainty. No score-selected tuning, restoration or new experiment is run.
Publication is explicitly authorized regardless of whether the joint candidate
meets the improvement gate; promotion to the cumulative baseline is separate.
