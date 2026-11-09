# Post-fit descriptive diagnosis

- Exposed benchmark: repeated leaderboard selection prevents unseen-future inference.
- Descriptive matched-seed differences are not causal attribution or event labels.
- Timing diagnostics use future truth; no shift is chosen or applied and no correction is fitted.
- Regional MSE/TIMSE energies are additive within a partition; S is nonlinear and is not additive.
- Strata overlap across families; do not sum energies across different families.
- No-offset forecasts isolate saved pre-offset output algebra, not a separately trained intervention.
- Input issue-time availability and historical byte-identity limitations remain unresolved.

| Seed | Final ΔS | No-offset ΔS | ΔMSE energy | ΔTIMSE energy |
| --- | ---: | ---: | ---: | ---: |
| 42 | 1.45162612 | 1.12146328 | 149.22954396 | 485.02216339 |
| 43 | 1.66149915 | 1.30141840 | 172.23486120 | 572.39136897 |
| 44 | 1.64773207 | 1.35633323 | 133.56555898 | 747.62758161 |

All differences are treatment minus control. See diagnosis.json for every fixed stratum,
all five timing shifts, peak/range summaries, offset replay and official score parity.
No candidate selection, causal mechanism conclusion, or forecast correction is made.
