# Training-only temporal common support

- Fixed seed42 full900 raw predictions, observed training targets only for bias decomposition.
- Reference and calibration rows were both fitted by the full model; all results are in-sample.
- Coarse X-only common support does not identify full P(Y|X), conditional mapping drift, or causality.
- Weights are separately normalized over common-support observed pairs for each time group.
- Distinct-origin support is counted, but overlapping origins and target pairs are dependent.
- No tests, confidence intervals, new folds, fits, query scores, corrections, or parameter selection.
- Fixed legacy half-life180 and floor0.1 are audited, not proposed as an empirically selected policy.
- Recency multiplies original stage-specific tail-normalized point weights; no second tail normalization.
- Hessian masses describe the existing diagonal surrogate scaled by recency, not spectral cell losses.
- Raw input missingness is a proxy; availability and immutable historical provenance remain limited.

| Period | Head | View | Older coverage | Recent coverage | Within | Composition |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| F1 | 0 | speed_recurrence | 0.9999591968989643 | 1.0 | -4.003095050964376 | 2.0507322417401523 |
| F1 | 0 | plus_ch | 0.904763162000312 | 0.9708324289125244 | -3.5934807887476286 | 1.6603898992856148 |
| F1 | 0 | plus_raw_missingness | 0.8725167112700565 | 0.9447851096158021 | -3.7542244911539178 | 1.8953946739605276 |
| F1 | 1 | speed_recurrence | 0.9999736039488493 | 1.0 | -13.582552186796043 | 3.4000725311291697 |
| F1 | 1 | plus_ch | 0.9046742607305946 | 0.9700955319206517 | -12.414818710703633 | 2.5228233938466 |
| F1 | 1 | plus_raw_missingness | 0.8732893559123555 | 0.934264295959514 | -12.88355327681591 | 2.7912166238429137 |
| F1 | 2 | speed_recurrence | 0.9953876485882319 | 0.9985119047619048 | -17.292725701061595 | 0.9695984879887255 |
| F1 | 2 | plus_ch | 0.9000753134863908 | 0.9727272727272728 | -16.920039874597627 | 0.2528482318971136 |
| F1 | 2 | plus_raw_missingness | 0.8775316364613215 | 0.9323051948051948 | -16.328928497687738 | -0.2915690760374192 |
| F2 | 0 | speed_recurrence | 0.9835717370420313 | 0.9963716327652556 | -2.0397264365998318 | 1.784082545047667 |
| F2 | 0 | plus_ch | 0.8753073689220274 | 0.9705332600329851 | -2.439360752049551 | 2.0668595661383344 |
| F2 | 0 | plus_raw_missingness | 0.7932850356503882 | 0.8824079164376031 | -3.3898718553161906 | 2.2238882925071 |
| F2 | 1 | speed_recurrence | 0.9962189651069531 | 1.0 | -4.506833057320465 | 2.9657180479408374 |
| F2 | 1 | plus_ch | 0.9068776936568512 | 0.9747184048910799 | -5.417978816800155 | 3.512852143243263 |
| F2 | 1 | plus_raw_missingness | 0.7590030935740034 | 0.8729310677205254 | -7.542858061002228 | 3.071700683799883 |
| F2 | 2 | speed_recurrence | 0.9956904718889217 | 0.9977138764942434 | -3.3965659776121138 | 1.6981391804703503 |
| F2 | 2 | plus_ch | 0.9031744213202667 | 0.9652949925632127 | -4.267359735612787 | 1.7556810356737398 |
| F2 | 2 | plus_raw_missingness | 0.7999043795909914 | 0.8807359665069134 | -5.054250936789305 | 1.30401284964001 |
| F3 | 0 | speed_recurrence | 0.963883687833205 | 0.9940849274182607 | -2.637435572008969 | 0.7697131173830479 |
| F3 | 0 | plus_ch | 0.8355531688551937 | 0.9717813051146384 | -2.5195230239767614 | 0.9618737510779172 |
| F3 | 0 | plus_raw_missingness | 0.7926966968929109 | 0.9473341473341473 | -2.270449758723984 | 0.776142503105866 |
| F3 | 1 | speed_recurrence | 0.9639324452863156 | 0.9947032459595274 | -1.4992221176037384 | 0.9925393092530296 |
| F3 | 1 | plus_ch | 0.8357344350232898 | 0.9691430123590927 | -1.8444100607894067 | 1.2475478458403662 |
| F3 | 1 | plus_raw_missingness | 0.7920675873724776 | 0.941871519760967 | -0.9363732114603216 | 1.0360288371535957 |
| F3 | 2 | speed_recurrence | 0.96359563033303 | 0.9942961132085721 | 1.0572169032387617 | 0.5843611241408284 |
| F3 | 2 | plus_ch | 0.8803932210583016 | 0.9834315669391857 | 0.6969868611245529 | 0.7460308525242516 |
| F3 | 2 | plus_raw_missingness | 0.7967480504217455 | 0.9460846891381699 | 0.8888323578680796 | 0.47281511786418634 |

Full per-stratum counts, bias decomposition, and fixed stage-specific recency mass/ESS audits are in the JSON.
Bias is raw prediction minus observed training target. No conclusion about stale conditional mapping follows from marginals alone.
