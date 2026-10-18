# E1532 frozen-feature residual head ensemble

Three equally weighted seeds (0, 1, 2). Coefficients are frozen from six preofficial folds using observed targets strictly before 2026-05-29. No official scores were used to choose a seed or ensemble weight.

Inputs are frozen causal forecasts: E1149 champion, E1506 OMNI2-trained long-history expert, and E1466 GONG/PFSS/WSA/HUXt. ACE features use observations at least two hours old. OMNI2 is training only. The design array contains HUXt minus champion, trailing ACE/HUXt bias, LHX minus champion path mean, and an intercept.

The SHA-named script verifies every input hash and regenerates all 203,976 official predictions from prepared features and the fitted coefficients. This is artifact-conditioned reproduction of the residual model. The prepared forecasts are not final E1532 predictions. Raw-data training and upstream forecast generation are outside this release.

Inference timing is one 72-hour forecast from all three heads plus equal averaging, measured over all 2,833 origins. Upstream E1149/E1506 inference and HUXt simulation are excluded because their outputs are frozen input features; this is not an end-to-end pipeline latency.

Roll6 validation is cross-fold rather than chronological: each held-out fold uses the other five folds. Its -11.30% result must not be interpreted as a prospective backtest. Official coefficients use only preofficial labels.
