# e2000fs_budget_complexity

EDS-inspired train-only OOF-error × causal-context density weighting; bounded and group-mass preserving. No simplex selection. Timing-correct point4 control uses identical origin+144h<=cutoff rows.

| Stage | Control total | Candidate total | Delta | Control ≥550 | Candidate ≥550 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 1801.5871 | 1795.8954 | -5.6916 | None | None | None |
| D2 | 1164.0136 | 1172.7815 | +8.7679 | 16376.11059843709 | 16368.45327499646 | -7.657323440629625 |
| D3 | 9133.4624 | 9132.2945 | -1.1679 | 24199.725307187135 | 24196.81417611904 | -2.9111310680928 |
| pooled_dev | 4033.0210 | 4033.6571 | +0.6361 | 24149.734158888412 | 24146.792700712507 | -2.9414581759046996 |
| C | 30894.5407 | 30902.3003 | +7.7596 | 37737.416814681674 | 37746.7305967113 | 9.313782029625145 |
| f1 | 5886.1946 | 5972.5124 | +86.3178 | None | None | None |
| final | 2191.9328 | 2185.5175 | -6.4153 | None | None | None |
| combined | 4039.0637 | 4079.0149 | +39.9512 | None | None | None |

Internal signal: {"positive": false, "development_windows_positive": 1, "confirmation_positive": false, "official_unconditional": true}

Official outcome: {"target": false, "transferred": false, "smoke": true}
Historical published parent (older maturity mask): total3556.1984 / ≥5505923.5239; not the paired control.

Artifacts: /home/t-lab01/sunrun/analysis/e2000fs_budget_complexity/runs/smoke_v1
Manifest SHA256: e1d0ef10fd6fe48ac746f41cc1256939d44c12896c1c15fa9d55b121b6b6a543
Per-round losses: losses/*.csv (including auxiliary estimator); final loss/convergence: convergence.json.
Hashes: manifest.json, model_receipts.json; grouped mass/ESS and auxiliary cost: weights/*.
Submission receipt: pending; training/evaluation alone is not submission.
Limitations: previously exposed Top30 and evaluation periods; no fresh-holdout or future guarantee. Scientific outcome is separate from publication infrastructure status.
