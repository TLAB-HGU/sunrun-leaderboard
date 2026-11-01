# e2100_tda_pi_suvi72 — arXiv:2610.35549 on GOES-18 SUVI Fe195

Replication of *Explainable Prediction of High-Speed Solar Wind Streams driven by Coronal Holes via
Topological Data Analysis* (arXiv:2610.35549v1) on this workspace's data, plus the leaderboard
submission that the paper's representation supports.

## What the paper does

* Input: full-disk SDO/AIA 193 Å images, hourly, 2010-05-18..2024-06-30 (71,976 labelled frames).
* Preprocess: clip at 7757.31 -> rescale to [0,1] -> `asinh(I/a)/asinh(1/a)` with `a = 0.005` ->
  rescale to [-1,1] -> invert (`max - I`) so coronal holes become peaks.
* Topology: cubical complex, sublevel filtration on the inverted image, H0 + H1 persistence diagrams.
* Vectorization: Betti curves, persistence landscapes, silhouettes, persistence images (14x14, best).
* Target: binary HSS at t0 + 4 days — `y=1` if the target time is in an HSS interval and
  V > 500 km/s, `y=0` if outside an HSS interval and V < 400 km/s, otherwise excluded.
* CV: 20 contiguous chunks, 4 chunks per fold as test, 90-day purge, 5 folds, pooled OOF scores.
* Result: LogReg + persistence image, AUC 0.8204, TSS 0.4238 (prevalence baseline AUC 0.5 / F1 0.1956).

## What we have instead

| requirement | ours |
|---|---|
| AIA 193 Å hourly full-disk | **GOES-18 SUVI L1b Fe195**, 1280x1280, 2.5"/px, hourly, 35,094 frames 2022-08-22..2026-09-27 |
| OMNI hourly solar wind | `data-v2/collectors/nrt/omni/hourly/omni2_hourly.parquet`, 2000-2026 |
| HSS interval catalogue | rebuilt by peak filtering (6-h median, `find_peaks(height=500, prominence=60, distance=24h)`, window -54h/+36h around each peak) |
| 14 years / full solar cycle | 4.1 years, SC25 rise to decline |

## Deviations (all forced, all recorded)

1. **Channel and instrument.** SUVI Fe195 instead of AIA 193 Å: same Fe XII band, but 2.5"/px instead of
   0.6"/px and a different radiometric scale.
2. **Normalization constants.** The paper's clip 7757.31 and `a = 0.005` are AIA data numbers; SUVI is
   radiance in W m-2 sr-1 (disk p99.5 ~ a few units). The clip is replaced by the per-frame disk
   99.5th percentile; `a = 0.005` is kept because it acts on the [0,1] image.
3. **HSS catalogue.** The paper's list (GFZ doi:10.5880/gfz.2.7.2024.001) is not in this workspace;
   a peak-filter catalogue is rebuilt from OMNI. Label counts therefore differ.
4. **CV geometry.** 20 chunks x 75 days instead of 20 x 9 months, with a 45-day purge (a 90-day purge
   leaves too little training data in 4.1 years). Inflation of the top/bottom chunks and the
   `min_persistence`/`sigma` choices are unspecified in the paper and were fixed here.
5. **Persistence image implementation.** `persim.PersistenceImager.fit_transform` re-estimates the
   (birth, persistence) range per image and produces a variable-size grid; Eq. (2) is implemented
   directly on a fixed 14x14 grid instead.

## Leaderboard target

The leaderboard scores a 72-hour speed curve over 6,315 origins, not a 4-day binary event, so the
paper's representation is carried over with its best-performing linear model family: ridge regression
per horizon on 392 persistence-image features + 20 causal ACE lag/window features, one fit per period
(cutoffs 2025-12-29 / 2026-03-29 / 2026-06-29), training-label maturity target + 72h.

The paper's own task (4-day HSS classification) is run as the internal result and recorded in
`results.json` and in the leaderboard description. It never gates the official forecast.

## Run

```bash
python build_cache.py                 # 35k frames -> features/part-*.parquet, resumable
python run_all.py --skip-cache        # internal paper task + official 454,680-row forecast
python -m pytest test_pipeline.py -q  # causality / admissibility / shape regression tests
```

Artefacts: `features/` (persistence images, one row per frame), `predictions.parquet`,
`results.json`, `meta.yaml`.
