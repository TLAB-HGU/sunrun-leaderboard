# Continuous-Relevance cohort evaluation

Status: complete. Official scorer `lead-timse-6h-v1`; no replaced predictions.

Point4 comparison: `e2000fs_calib_offset_roll`. Local best anchor: `e2000fs_calib_offset_roll_scap`. External SOTA was not independently rechecked.

| Family | Mean3 S | Seed mean S | Seed SD S | Pareto all six | Practical 1% | Statistical all six |
|---|---:|---:|---:|---|---|---|
| e2000fs_calib_offset_roll_relevance_cdf_lead | 98.287253 | 98.384697 | 0.268060 | False | False | False |
| e2000fs_calib_offset_roll_relevance_cdf_pool | 98.228562 | 98.333044 | 0.359905 | False | False | False |
| e2000fs_calib_offset_roll_relevance_direct_s | 97.127946 | 97.289536 | 0.675924 | False | False | False |
| e2000fs_calib_offset_roll_relevance_fs | 96.487802 | 96.613988 | 0.165095 | False | False | False |
| e2000fs_calib_offset_roll_relevance_mm | 97.213157 | 97.372904 | 0.399488 | False | False | False |

Bootstrap: 2000 paired circular seven-day draws stratified by period; exact quantile/metric recomputation. Holm family covers candidate mean3 versus E0 for six metrics.

Full period, seed, auxiliary MAE/MSE/bulk/tail metrics, confidence intervals, and completion provenance are in report.json and results/.

No automatic promotion or publication. Failure classifications require training diagnostics; this report establishes endpoint gate outcomes only.
