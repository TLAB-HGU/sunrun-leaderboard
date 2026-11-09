# Training-subset mechanism diagnostic, F1/F2 only

These are existing pre-cutoff fitting subsets, not independent evaluation.
They do not alter the current first-minimum rule or identify the next experiment.

|Period|Seed|Selected counts|Stopping RMSE 900 → selected|Stopping ≥700 bias 900 → selected|Later offset raw RMSE 900 → selected|
|---|---|---|---|---|---|
|F1|42|315/152/135|98.838 → 91.237|-190.287 → -175.544|88.045 → 85.015|
|F2|42|892/98/68|85.509 → 82.441|-174.473 → -190.658|63.144 → 61.621|
|F1|43|282/159/133|98.519 → 91.484|-189.698 → -174.918|88.750 → 84.881|
|F2|43|867/99/74|85.063 → 82.580|-170.096 → -186.043|63.032 → 61.074|
|F1|44|304/150/108|98.207 → 91.411|-189.172 → -176.393|89.386 → 85.455|
|F2|44|888/121/76|84.713 → 82.392|-176.177 → -188.840|63.980 → 61.239|

Confirmed: selecting prefixes lowers native RMSE on the selection subset; that
alone is expected from the selection rule and is not evidence of useful transfer.
Later offset rows were excluded from round selection but participate in fitting
the unchanged offsets. Their raw-prior errors remain fitting diagnostics.

Inference to test after all forecasts freeze: regularization may suppress rare
fast-wind amplitude while reducing broad RMSE. F2 already shows greater negative
≥700 bias in the selected prior despite lower RMSE. Full models additionally fit
these intervals, so prior-stage behavior cannot prove final official degradation.
The strong lead/period variation in selected counts does not support a universal
fixed short-round prescription. Neither observation is used to retune this run.

Official S effects and adoption remain pending all three completed runs.
