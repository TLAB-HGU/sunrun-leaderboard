# E1666 chronological delayed-LHX release

Reference E1666; registered official run E1669. This is one explicitly user-requested submission. The roll3 confidence-interval gate failed (MSE -5.713%, CI [-12.792%, +1.888%]); arrival passed (71 events, capped-MAE difference -1.095 h, CI [-2.015, -0.231] h). Upload does not make this a gate-passing champion. The exposed historical folds are not fresh holdout data.

## Reproduce

From the repository root, install NumPy, pandas and pyarrow, then run:

```sh
python scripts/seongeun/e1666_chronological_delayed_lhx/b50029f51a5ae9cfe1aaed236602437d24c1ebb1e6aa21f6dbdba2a331fcc819.py --output predictions_e1666.parquet --timing timing_e1666.json
```

Each prepared_sN.npz contains only control forecasts, five prepared features, frozen per-lead regression coefficients and UTC origin timestamps. There are no target labels. The script verifies the canonical configuration identity and every asset SHA-256, applies `control + einsum('nhk,hk->nh', design, beta)` per seed, and takes the equal seed mean. It writes the complete 2,833 x 72 forecast grid.

This release reproduces the submitted fixed grid from frozen intermediate assets. It cannot forecast arbitrary new origins or retrain the upstream models. The feature preparation and upstream frozen base/HUXt/LHX generation are not included. Source availability uses ACE observation +2h and LHX model cutoff +45 days; fit targets precede 2026-05-29 UTC. Source receipt hashes in config preserve linkage to the registered run; private research paths and target arrays are excluded.

## Runtime disclosure

The recorded timing uses `time.perf_counter()` for 20 repetitions over all 2,833 origins after one warmup. It includes all three residual predictions and their equal mean. Loading, initialization, feature preparation, fitting, saving and upstream base/HUXt/LHX prediction generation are excluded. This is residual-head inference time, not end-to-end forecasting latency. A rerun writes new measured timing without changing the submitted metadata.
