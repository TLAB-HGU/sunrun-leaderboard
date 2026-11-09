# Audit-first S50 study

Start with [report.md](report.md); the user's W means official S.
The goal is unachieved. `results.json` contains the completed three-seed
inference-only age-repair diagnostic and 2000 temporal bootstrap replicates.
`verification.json` records tests, real exits, TUI checks and remaining gaps.

Audit scripts and compact evidence are in
`../../analysis/w50_audit_20261108/`; raw tables, existing model bundles and
large NPZ/Parquet files remain local, listed by `artifact-manifest.json`.
No existing training campaign, official scorer or raw dataset was changed.

Reproduce the read-only audits from repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/data_probe.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/metric_reproduce.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/residual_diagnostics.py --bootstrap 2000
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/event_probe.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python research/w50_understanding_20261108/analyze_results.py
```

These regenerate diagnostic outputs, never fit a predictor. Board lookup is a
separate read-only `metric_board_snapshot.py` command and may show a newer
revision; keep the frozen snapshot when reproducing the historical comparison.

The age experiment uses `age_repair.main(seed, experiment_name)` from three
single-experiment wrappers in `runs/age_repair_s{42,43,44}/experiment.py`.
Its fixed trees/offsets come from the existing two-removal configuration.
Remote full boosters are copied into `frozen_models/`; seed42 uses the original
local models. Do not silently substitute a different model or retrain it.
For a new execution allocate a fresh output folder, launch.json and tmux session
under the same registry/TUI contract; the original run logs and receipts are
research evidence and should be retained.

The required data is the exact input and observation files named in
`age_repair.py` and the parent model metadata. Their acquisition/generated-data
contract is in `data-v2/README.md`; source science and availability limitations
are in the report. Missing historical availability and independent calibrated
data cannot be repaired by rerunning this script.

`single-column-source.diff` isolates the actual one-line input intervention;
`parent-to-diagnostic.diff` shows how the full training runner differs from the
frozen-booster diagnostic. Full source diffs were added after execution; the
prospective strategy and run-level semantic diff were saved beforehand.
