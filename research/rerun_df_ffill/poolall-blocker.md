
# 2026-11-03 bounded recovery: exact legacy inventory found

The earlier conclusion that only ten arm names survived was incorrect. The full ordered **81-arm inventory is recovered** from `docs/explainers/data/calib5.json` → `arms.names`. Its provenance is `docs/explainers/code/extract_calib5.py`: it reads `research/e2000fs_calib5-20261031/runs/poolall_v1/weights.npz`, extracts `arms`, asserts shape `(4970, 81)`, reconstructs predictions, and writes the names. `docs/visualizations/e2000fs_calib5_detpool_all/legacy_81arm_weights.csv` independently lists all 81 shortened labels. `pooling.py:POOLALL_ARMS` freezes the recovered raw names without computing hashes or reading old forecasts.

## Searches and evidence

- `git ls-tree -r --name-only b930cc556b626a633c58d8fd74680cbf80f27008 research/e2000fs_calib5-20261031` contains only `pool_all.py`; no run weights or results are committed at that revision.
- `309a44ce7` experiment-results archive has no calib5/poolall entries.
- Filename history led to visualization commit `c482c10e2`, which contains the legacy 81-label CSV, modern weights receipt, and documentation. These files are also present in the current tree.
- The original discovery function sorts `research/**/forecast.npz`, excludes publication paths and the GPD base, checks matching origin grid, shape, finite predictions, and prediction-content duplicates. It does **not** select by submission date. Reconstructing the set from leaderboard timestamps alone would be invalid. The recovered weights metadata avoids that ambiguity.
- 65 of the 81 paths have a direct matching experiment source in `source_archive`; the remaining 16 are calibration subarms.
- Shared calib3 `lanes3.py` / archived `e2000fs_calib3_era.py` supplies extrap, affine, restack, bag3, era, and GPD recipes. It does **not** define the required `monotone_v1` recipe. Current source search, archived-source search at `b930cc556`, and path history found only score/weight references to this arm, not its fitting algorithm or settings. `research/notes/tail-550-frontier-20261031.md` mentions its score, which is insufficient to recreate its training recipe.

## Important submitted-model distinction

The visualization README and `weights_receipt.json` explicitly show that the actual F1/F2/F3 submitted predictions were a **16-arm refit stack**, while the leaderboard replay-script/config still pointed to the legacy 81-arm program. The receipt records recomputed-stack equality with float32 max difference 0.0. The exact 16-arm order is in `code/pool_weights.py:ALLOWED` and the receipt's `arm_order`. This is evidence of a distinct submitted variant, not authorization to silently shrink an 81-arm request.

## Remaining execution gap

The inventory is no longer a blocker. Exact legacy-81 execution still needs the verified `monotone_v1` recipe and newly trained, prior-only calibration predictions plus current-query predictions for every frozen arm, including all shared calibration subarms. Existing original prediction arrays cannot be reused. No smaller pool has been substituted; `calib5_detpool_all` remains fail-closed with this corrected reason.

## Ordered recovered legacy arms

1. `e2000fs_budget_complexity/results/forecast.npz`
2. `e2000fs_budget_covshift/results/forecast.npz`
3. `e2000fs_budget_primaldual/results/forecast.npz`
4. `e2000fs_budget_spectral/results/forecast.npz`
5. `e2000fs_calib-20261030/runs/control_144h_v1/forecast.npz`
6. `e2000fs_calib-20261030/runs/offset_v1/forecast.npz`
7. `e2000fs_calib-20261030/runs/pool_v1/forecast.npz`
8. `e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz`
9. `e2000fs_calib2-20261031/runs/capacity_depth5_v1/forecast.npz`
10. `e2000fs_calib2-20261031/runs/detroute_v1/forecast.npz`
11. `e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz`
12. `e2000fs_calib3-20261031/runs/affine_v1/forecast.npz`
13. `e2000fs_calib3-20261031/runs/bag3_v1/forecast.npz`
14. `e2000fs_calib3-20261031/runs/era_v1/forecast.npz`
15. `e2000fs_calib3-20261031/runs/extrap_v1/forecast.npz`
16. `e2000fs_calib3-20261031/runs/gpd_v1/forecast.npz`
17. `e2000fs_calib3-20261031/runs/monotone_v1/forecast.npz`
18. `e2000fs_calib3-20261031/runs/restack_v1/forecast.npz`
19. `e2000fs_calib4-20261031/runs/era_v1/forecast.npz`
20. `e2000fs_calib4-20261031/runs/mixture_v1/forecast.npz`
21. `e2000fs_dirmo_12/results/forecast.npz`
22. `e2000fs_dirmo_24/results/forecast.npz`
23. `e2000fs_dirmo_24_feat30/results/forecast.npz`
24. `e2000fs_dirmo_24_feat30_lgbm/results/forecast.npz`
25. `e2000fs_dirmo_24_feat30_lgbm_HPO/results/forecast.npz`
26. `e2000fs_dirmo_24_feat30_pointweight4/results/forecast.npz`
27. `e2000fs_dirmo_24_feat30_pointweight4_last24/results/forecast.npz`
28. `e2000fs_dirmo_24_feat30_pointweight4_last48/results/forecast.npz`
29. `e2000fs_dirmo_6/results/forecast.npz`
30. `e2000fs_dirrec/results/forecast.npz`
31. `e2000fs_dirrecmo_12/results/forecast.npz`
32. `e2000fs_dirrecmo_24/results/forecast.npz`
33. `e2000fs_dirrecmo_6/results/forecast.npz`
34. `e2000fs_feat30_agnostic_v2_context/results/forecast.npz`
35. `e2000fs_feat30_agnostic_v2_event/results/forecast.npz`
36. `e2000fs_feat30_extreme/results/forecast.npz`
37. `e2000fs_feat30_extreme_histprior/results/forecast.npz`
38. `e2000fs_feat30_gaussian_pilot/results/forecast.npz`
39. `e2000fs_feat30_hyper_gain/results/forecast.npz`
40. `e2000fs_feat30_lgbm_extreme4/results/forecast.npz`
41. `e2000fs_feat30_lognormal_crps/results/forecast.npz`
42. `e2000fs_feat30_lognormal_nll/results/forecast.npz`
43. `e2000fs_feat30_mc_calibration/results/forecast.npz`
44. `e2000fs_feat30_pointweight4/results/forecast.npz`
45. `e2000fs_feat30_pointweight4_tcn_lb120/results/forecast.npz`
46. `e2000fs_feat30_pointweight4_tcn_lb120_regskip/results/forecast.npz`
47. `e2000fs_feat30_probtraj/results/forecast.npz`
48. `e2000fs_feat30_shape/results/forecast.npz`
49. `e2000fs_feat30_smooth_error_tail/results/forecast.npz`
50. `e2000fs_feat30_smooth_wcrps/results/forecast.npz`
51. `e2000fs_feat30_tailcal_crps/results/forecast.npz`
52. `e2000fs_feat30_transfer_center/results/forecast.npz`
53. `e2000fs_feat30_transfer_recency/results/forecast.npz`
54. `e2000fs_feat30_transfer_recent_window/results/forecast.npz`
55. `e2000fs_feat30_wavelet/results/forecast.npz`
56. `e2000fs_kumo_direct/results/forecast.npz`
57. `e2000fs_kumo_dirrec/results/forecast.npz`
58. `e2000fs_kumo_pooled/results/forecast.npz`
59. `e2000fs_kumo_recursive/results/forecast.npz`
60. `e2000fs_mimo/results/forecast.npz`
61. `e2000fs_point4_high_anchor/results/forecast.npz`
62. `e2000fs_point4_high_regret_guard/results/forecast.npz`
63. `e2000fs_point4_leaf_refit/results/forecast.npz`
64. `e2000fs_point4_linear_leaf/results/forecast.npz`
65. `e2000fs_point4_stable_retrain/results/forecast.npz`
66. `e2000fs_point4_teacher_trust/results/forecast.npz`
67. `e2000fs_realmlp_direct/results/forecast.npz`
68. `e2000fs_realmlp_dirrec/results/forecast.npz`
69. `e2000fs_realmlp_pooled/results/forecast.npz`
70. `e2000fs_realmlp_recursive/results/forecast.npz`
71. `e2000fs_recmo_12/results/forecast.npz`
72. `e2000fs_recmo_24/results/forecast.npz`
73. `e2000fs_recmo_6/results/forecast.npz`
74. `e2000fs_rectify/results/forecast.npz`
75. `e2000fs_rectifymo_24/results/forecast.npz`
76. `e2000fs_recursive/results/forecast.npz`
77. `e2000fs_stratify_dirrecmo_dirrecmo_24/results/forecast.npz`
78. `e2000fs_tabiclv2_direct/results/forecast.npz`
79. `e2000fs_tabiclv2_dirrec/results/forecast.npz`
80. `e2000fs_tabiclv2_pooled/results/forecast.npz`
81. `e2000fs_tabiclv2_recursive/results/forecast.npz`

# Resolution: rerun the documented actual 16-arm submission

The parent selected the documented actual submitted model after identity verification. The current inventory and `docs/visualizations/e2000fs_calib5_detpool_all/leaderboard-result.json` both identify **seongeun / e2000fs_calib5_detpool_all / 20261031T172348000Z**. Their total MSE 4154.1526325545565 and F1/F2/F3 MSE values 6007.057035049542 / 3600.168705922258 / 2876.277923686607 exactly match the receipt's reconstruction. The receipt records `recomputed_stack_equals_submission=true` and `max_abs_difference_float32=0.0`. Evidence is pinned to commit `c482c10e2720e03157a512d347c559f9e319e82c`. The three LightGBM component kernels are recovered from `research/e2000fs_f123-20261031/arms.py` at `6b85acc0b00d731bf31b6885e0b847c18ee395d0`.

`pooling.py` now executes this **actual 16-arm model**, not a guessed reduction and not a claim of 81 newly trained arms. Its exact order is: pointweight4, point4_144h, tweedie15, mixture, era, gpd, spectral, dirmo24, detroute, offset_roll, pool_bodytail, detpool_era, detpool_mixture, lgbm_extreme4, lgbm_point4, lgbm_recent. The original 81 names remain frozen in `POOLALL_ARMS` for audit; the missing monotone recipe affects only the obsolete legacy variant.

All primitive models are newly fitted to the requested snapshot. Inner offset/pooling calibration uses an earlier training quarter with a 144-hour purged reference fit; primitive models are refitted on the outer reference set to generate honest forecasts for the latest prior training quarter. Inner postprocessor states remain frozen when generating those outer forecasts. Original NNLS/ridge/simplex/tail-guard equations then fit the 16 outer weights on that latest prior quarter. Final component models are refitted on all eligible training rows; every postprocessor and NNLS state remains frozen for official queries. Query labels are never read. Both named pointweight4 and point4_144h arms are retained; the common current 144-hour fit policy can make their forecasts coincide.

Verification: 27 tests passed, covering exact original formulas, exact 16-arm order and baseline identity, nested prior cutoffs, query-target poisoning, all primitive families, and LightGBM scalar parity. Actual filled-data F1 smoke completed successfully with eight query origins × 72 horizons; all component boosters and frozen calibration state were written under `/tmp/sunrun-pooling-actual16-filled-smoke`. No full training or publication was performed by this slice.
