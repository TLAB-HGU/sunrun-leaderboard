# E1733 fixed-grid reproduction

Run `python scripts/seongeun/e1733_anchor_mean/6e56d82f07d0d0bc3b0b954a01e15f1c745fd762b998a9010da0a57f765b7c23.py --output predictions.parquet --timing timing.json` from the release repository root. Requires NumPy, pandas and pyarrow.

Frozen member forecasts and their source hashes are retained; the script performs the registered combination and equal averaging across three seeds. Assets contain no truth labels. This is fixed-grid reproduction, not retraining or forecasting new origins.

Timing measures combination and seed averaging after loading/warmup; it excludes upstream inference, training and feature preparation. Upstream causality is asserted only after separate review by the manifest producer. The locality/power tests cover the combination only.
