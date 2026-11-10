# Current published offset-roll diagnostic protocol

Freeze the live Space and store revisions before reading outcomes. Select the
current row by its exact visible name, member and code-link configuration ID;
fail on ambiguity. Reuse frozen revisions on subsequent runs. Download its
stored forecast, record and official observations without invoking publication.

Replay the unchanged root official scorer and compare all stored Total/F1/F2/F3
lead metrics at absolute tolerance 1e-8. Compare every stored forecast key/value
against the existing equal mean of local seeds 42/43/44. No new forecast is saved.

Use the existing metric_weights implementation, independently checked against
the official scorer for every scope. Predefined descriptive partitions: periods,
leads 1–6/7–24/25–48/49–72, truth <400/400–550/550–700/>=700 km/s, and their
cross-product. Sum linear MSE and TIMSE energy with unchanged official masks,
quantiles and equal-lead weights; require partitions to reproduce totals. Rank
unique target hours, preserving forecast-pair multiplicity. Restricted removal
of each fixed speed/lead/period group's error is an algebraic sensitivity only;
not a forecast, achievable bound, fit, model selection or improvement claim.

Amplitude uses origins with >=48 valid targets and identical valid lead support
for truth and predictions. Compare fixed shifts -12/-6/0/+6/+12 on common leads
13–60, without selecting a correction. Report period-specific behavior and peak
timing ambiguity. Analyze the current mean plus constituent seeds; parent audits
are contextual evidence only. Do not use benchmark truth to fit any component.

Completion: exact current score reproduction, no unreported replacements,
full-key mean identity, energy accounting, targeted missingness/tie tests,
ranked report separating observed failure modes from unknown causal mechanisms.
No training, experiment admission, remote process, scorer edit or publication.

Before reading its outcomes, also replay the existing current-run no-offset
artifacts for all three seeds. Compare with-offset and no-offset S by scope,
and the same predefined speed strata. This isolates the already-saved additive
postprocessing effect without training or producing a new forecast file.
