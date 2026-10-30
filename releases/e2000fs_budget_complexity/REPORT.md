# e2000fs_budget_complexity

EDS-inspired train-only OOF-error × causal-context density weighting; bounded and group-mass preserving. No simplex selection. Timing-correct point4 control uses identical origin+144h<=cutoff rows.

| Stage | Control total | Candidate total | Delta | Control ≥550 | Candidate ≥550 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 5508.9518 | 5520.1140 | +11.1621 | 15962.236790371097 | 16103.6198136185 | 141.3830232474029 |
| D2 | 5559.5161 | 5666.8015 | +107.2854 | 18172.798302918032 | 18619.665934481265 | 446.86763156323286 |
| D3 | 8538.2251 | 8716.8080 | +178.5830 | 21326.45460398794 | 21878.74247579893 | 552.2878718109896 |
| pooled_dev | 6531.3045 | 6629.9502 | +98.6457 | 18995.525422946026 | 19414.171431413706 | 418.6460084676801 |
| C | 7766.3282 | 7927.4369 | +161.1088 | 11646.714216961327 | 12027.724991638492 | 381.0107746771646 |
| f1 | 4056.9298 | 4046.7036 | -10.2262 | 5257.235575971638 | 5447.018090543465 | 189.78251457182705 |
| final | 3069.7383 | 3149.1822 | +79.4439 | 6292.477756630086 | 6738.035311329446 | 445.55755469935957 |
| combined | 3494.2108 | 3535.0983 | +40.8875 | 5769.472791661479 | 6085.812644067584 | 316.3398524061049 |

Internal signal: {"positive": false, "development_windows_positive": 0, "confirmation_positive": false, "official_unconditional": true}

Official outcome: {"target": false, "transferred": false, "smoke": false}
Historical published parent (older maturity mask): total3556.1984 / ≥5505923.5239; not the paired control.

Artifacts: /home/t-lab01/sunrun/analysis/e2000fs_budget_complexity/runs/full_v1
Manifest SHA256: 3268b99f0a0fd24df5892a0a998926ba7fc7505c04e6735d049e1d7993687639
Per-round losses: losses/*.csv (including auxiliary estimator); final loss/convergence: convergence.json.
Hashes: manifest.json, model_receipts.json; grouped mass/ESS and auxiliary cost: weights/*.
Submission receipt: pending; training/evaluation alone is not submission.
Limitations: previously exposed Top30 and evaluation periods; no fresh-holdout or future guarantee. Scientific outcome is separate from publication infrastructure status.
