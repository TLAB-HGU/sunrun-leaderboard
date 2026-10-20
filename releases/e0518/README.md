# E0518: enlil_latest_arrival — archived forecast ensemble

This is an equal-weight ensemble of the original three frozen member forecasts, uploaded at the user's request as an original-contract retrospective comparator. It is an archive and ensemble reproduction, not a standalone upstream model release or a predictor for unseen origins. The member forecasts are final outputs of the original experiment; they are disclosed as cached inputs rather than represented as fitted model weights.

The SHA-named script verifies configuration and input hashes and reproduces all 203,976 output rows by averaging the three member forecasts. The only measured operation is cached forecast averaging. **The reported time excludes all upstream model inference, fitting, features and I/O and is not end-to-end forecasting latency.** No deep learning training or inference is executed, including for frozen TabPFN-derived results.

Original contract and limitations: Historical E0472 origin-hour ACE contract, not ACE+2; training cutoff May29. Latest same-CME ENLIL arrival revision/run count; completion <= origin. Original poison passes. Retrospective comparator.

The original internal adoption gate failed. This entry is not a claim of current-contract promotion, independent replication, or out-of-sample improvement. `provenance.json` records the registered original command and source hashes, prediction hashes, evaluation and audit evidence. The bundle contains only predictions and forecast origins, not target labels. Run the SHA-named script with `--output predictions.parquet --timing timing.json`.
