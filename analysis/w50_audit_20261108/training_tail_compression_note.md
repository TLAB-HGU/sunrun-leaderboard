# In-sample tail compression: extraction from the existing probe

This uses only the already frozen seed42 training-only probe and its fixed300/900
prefixes, all nine heads, originally observed labels, and pre-existing speed bins.
No fitting, new cutoff, candidate selection, official truth or active-treatment
score access. The full model was fitted on every reported row.

|Period|Lead range|Observed >=700 pairs|High bias300|High bias900|High RMSE900|Low bias900|
|---|---|---:|---:|---:|---:|---:|
|F1|1–24h|10553|-57.11|-38.27|58.29|18.10|
|F1|25–48h|10777|-93.94|-61.04|80.54|38.32|
|F1|49–72h|10848|-116.39|-73.89|94.37|50.31|
|F2|1–24h|11688|-57.99|-39.40|61.77|18.63|
|F2|25–48h|11688|-100.74|-67.47|87.48|41.56|
|F2|49–72h|11688|-121.35|-78.39|99.01|53.70|
|F3|1–24h|11952|-60.23|-41.52|64.37|19.00|
|F3|25–48h|11952|-103.89|-69.99|90.17|41.93|
|F3|49–72h|11997|-123.30|-79.94|100.69|53.23|

Bias is prediction minus target, km/s. Low means y<400; high means y>=700.
Pair counts repeat physical target hours across origins and are not independent
event counts. This is a single-seed descriptive result with no generalization CI.

All nine heads underpredict the observed high tail and overpredict the low tail
even in-sample. The compression therefore already exists before applying the
models to future benchmark observations; it is not exclusively a future-only
phenomenon. Nevertheless high-tail RMSE improves at the longer fixed prefix in
all nine heads, consistent with continued useful training fit over this range.

Neither finding identifies the cause. Finite tree/round capacity, the retained
loss/regularization, conditional uncertainty, missing predictive information and
training-label quality can all contribute. This does not prove an optimizer
defect, justify increasing rounds indefinitely, rule out information limits, or
show how much official S could improve. Training and evaluation distributions,
offset application and observed support differ, so their biases cannot be
subtracted to obtain a causal generalization gap.

The current temporal-weight trial remains unchanged. Its completed fixed
diagnostics must distinguish high/low, lead, period and no-offset behavior before
the next hypothesis is chosen. Source receipts and exact numbers are in the
adjacent JSON; the original probe artifacts were not modified.
