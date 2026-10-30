# Frozen V-JEPA 2.1 video leaderboard entry

This entry adds frozen GOES-18 SUVI Fe195 video embeddings to the existing ACE, coronal-hole, physical-quality and 26/27/28-day recurrence residual XGBoost forecast. The fixed encoder is V-JEPA 2.1 ViT-B/384, using 16 frames spaced three hours apart through each origin. Training labels end at 2026-05-31T23:00:00Z. The model outputs 72 hourly ACE speed forecasts at each of the 2,833 supplied evaluation origins. The official scoring truth is not used for feature selection or fitting.

The entry was submitted at the user's request as a leaderboard comparison. In retrospective development/selection experiments, adding video features lowered selection MSE by 5.675% but reduced peak precision and recall, so the earlier experiment kept the numerical baseline. This is an exploratory leaderboard submission, not a validated peak-skill improvement. Historical ACE vintage and SUVI product arrival time are not fully verified.

The registered script at `scripts/seongeun/vjepa_frozen_video_official/<config-sha256>.py` checks its filename, source/config/input/fold/reference hashes, and downloads only the pinned public upstream source and checkpoint when a local frozen reference run is not supplied. It requires local pinned ACE, CH, raw ACE, and hourly SUVI metadata/images. The numerical inputs are described in the repository README. The environment is pinned in `requirements.lock.txt` (Python 3.12, CUDA 12.6 wheels).

From the repository root:

```bash
uv venv --python 3.12 .venv-vjepa-fusion
uv pip install --python .venv-vjepa-fusion/bin/python -r experiments/vjepa_frozen_fusion/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv-vjepa-fusion/bin/python scripts/seongeun/vjepa_frozen_video_official/<config-sha256>.py \
  --folds /path/to/pinned/folds.parquet \
  --run-dir /path/to/new-output-directory \
  --root "$PWD" \
  --metadata-root /path/to/suvi_hourly \
  --images-root /path/to/suvi-images \
  --output /path/to/predictions.parquet
```

The script builds an isolated official run, uses the fixed preprocessing and model configuration, checks saved-model replay and future-input invariance, records the 72-hour forecast timing, and writes `official_manifest.json`. `--reference-run /path/to/frozen-development-run` reuses a verified local reference checkpoint and immutable frame cache. `--mode prepare`, `--mode extract`, and `--mode predict` allow restart without changing the frozen configuration. New output directories must be used for new input snapshots.
