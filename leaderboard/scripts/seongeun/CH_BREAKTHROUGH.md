# Frozen CH breakthrough submissions

These three scripts refit the frozen GPU-HPO configurations; they never search
parameters or read official scoring truth. Inputs are numerical ACE history,
explicit single-Fe195 coronal-hole geometry/quality, and (where applicable)
26/27/28-day target-relative recurrence. No image neural network is used.

Run from this repository with Python 3.12 and `requirements.txt` (XGBoost 2.1.4).
Download the official `folds.parquet` as described in the root README, then:

```bash
.venv-fusion/bin/python scripts/seongeun/<experiment>/<config_sha256>.py \
  --folds folds.parquet --output predictions.parquet --device auto \
  --ace store/ch-v1/ace.parquet --ch-hourly store/ch-v1/ch-hourly.parquet \
  --raw-pattern '/path/to/pinned-raw-ace/year=*/part.parquet'
```

Experiments: `xgb_ch_recurrence_physics_official` (combined numerical features),
`xgb_ch_recurrence_official` (recurrence+CH), and
`xgb_ch_physics_quality_official` (physical ACE+CH quality).

The configuration embedded in each script pins input and implementation hashes,
training cutoff `2026-05-31T23:00:00Z`, seed 42, and a 6,000-origin training cap.
All labels end at or before the cutoff; each forecast uses only issuance-known
inputs. `--max-folds 3` is a smoke option only, not a submission.

The live raw ACE 2026 file advanced after the experiment. Before publication,
an updated raw snapshot was pinned and all 203,976 forecast values per model
were reproduced against the approved original result (without changing frozen
parameters). Both raw-version hashes are disclosed in the configurations.
Use the matching pinned inputs, not an arbitrary later live download; mismatched
hashes fail closed. Input data and prediction artifacts are not in GitHub.

The output manifest records source/input/prediction hashes, actual device,
runtime versions, and measured inference time. Submission timing excludes
loading, training, feature-matrix construction, and serialization; the older
pipeline timing is reported separately.

The official interval was previously inspected, and ACE operational vintage is
unverified. Scores are exploratory, not an untouched test claim. None of these
models met the 50%-MSE-improvement and peak-event acceptance gates.
