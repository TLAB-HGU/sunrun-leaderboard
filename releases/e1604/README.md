# E1604 HSSPEAK25 original-contract comparator

Equal prediction mean of three frozen E0472-based forecasts with a learned HSS peak correction from lead 25. This release trains the original two HistGradientBoosting heads on frozen prepared preforecast features and labels, then reproduces the correction. Upstream E0472 and HUXt inference is outside this artifact-conditioned release and outside its timing.

The original recipe uses origin-hour ACE observations, not the current research requirement of a two-hour availability delay. Its training labels extend through 2026-05-31 23:00 UTC (the first forecast origin), rather than the stricter targets-before-May29 cutoff. Every label precedes the forecast targets. This entry is a retrospective comparator under its original contract and must not be presented as passing the current research contract. Its internal four-cell adoption gate failed.

Run the SHA-named script with `--output predictions.parquet --timing timing.json`. It verifies input hashes and configuration identity, rebuilds both peak heads, and reports mean batched head/correction/ensemble inference time per 72-hour forecast, excluding training, I/O, feature generation, and upstream inference. The release contains prepared features, base forecasts, and preforecast training targets, but no official-window target labels.
