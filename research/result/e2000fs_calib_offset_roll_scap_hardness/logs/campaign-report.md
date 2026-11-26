# SEP/InHaR/CPC cohort evaluation

Status: complete. Official scorer `lead-timse-6h-v1`; no replaced predictions.

Point4 comparison: `e2000fs_calib_offset_roll`. Local best anchor: `e2000fs_calib_offset_roll_scap`. External SOTA was not independently rechecked.

| Family | Mean3 S | Seed mean S | Seed SD S | Pareto all five | Practical 1% | Statistical all five |
|---|---:|---:|---:|---|---|---|
| current_offset_roll_cpc4 | 97.530244 | 97.718944 | 0.314485 | False | False | False |
| current_offset_roll_cpc4_tail | 97.530244 | 97.718944 | 0.314485 | False | False | False |
| current_offset_roll_cpc_s4 | 97.530761 | 97.726998 | 0.310136 | False | False | False |
| e2000fs_calib_offset_roll_scap_hardness | 96.089937 | 96.233509 | 0.198040 | False | False | False |
| e2000fs_calib_offset_roll_scap_hardness_null | 95.617803 | 95.746660 | 0.228144 | False | False | False |
| e2000fs_calib_offset_roll_scap_sep_asym | 96.714419 | 97.025876 | 0.510061 | False | False | False |
| e2000fs_calib_offset_roll_scap_sep_sym | 95.189664 | 95.514377 | 0.485333 | False | False | False |

Bootstrap: 2000 paired circular seven-day draws stratified by period; exact quantile/metric recomputation. Holm family covers prespecified methodological comparisons for five primary metrics.

Full period, seed, auxiliary MAE/MSE/bulk/tail metrics, confidence intervals, and completion provenance are in report.json and results/.

No automatic promotion or publication. Failure classifications require training diagnostics; this report establishes endpoint gate outcomes only.
