# Training-era issued CME archive: feasibility inventory

Retrieved2026-11-08 from the official DONKI-API notifications endpoint. This
is a data audit, not a model experiment. No forecast model was fitted and no
official target column or observation store was read. The ongoing image-age
retraining configuration remains unchanged.

41monthly requests cover2022-08 through2025-12. All41 returned message lists,
with no retrieval/parse errors. Preserved raw API responses and URL/status/
receipt-time/byte-size metadata live in notification_archive/. After requiring
reported issue time<2025-12-29,1,174 distinct CME messages remain. 882 have a
single event identifier and unambiguous labeled forecast speed, half-angle
and HEEQ direction; these describe806distinct CME events. Others are retained
as unparsed/ambiguous, never assigned guessed kinematics.

Extraction requires an explicit Summary section and unique labeled values,
and excludes Notes. Four tests check original official-message values,
seconds-vs-minute issue time, late API issue times, ambiguity and exclusion
of retrospective-note values. Both the raw response and parsed records remain
available for independent review. API time and body time are combined by
taking the later value; no earlier time is inferred from simulation completion.

On28,297F1-training input origins,22,802(80.58%) have at least one unambiguous
issued summary in the preceding72h. This is only source exposure over the
forecast length, not a selected propagation window, Earth-impact determination
or demonstration that a message was received operationally. Parsed speed
quantiles(min/median/p90/p99/max) are403/790/1372/2137.83/3180km/s.

The inventory changes the next decision: a timestamped-message input pathway
has substantial training-era coverage, so the information question need not
depend solely on unverified current ENLIL versions. It does not yet justify a
new model or adoption. After the current controlled experiment finishes,
diagnose its outcome before defining another single change. A possible future
addition must retain every baseline column, use only already-issued messages,
distinguish updates of the same event, preserve ambiguity/missingness, and
handle CME direction and Earth arrival separately from launch speed.

Important limits: current official archival bodies plus issue timestamps are
not contemporaneous consumer receipts or proof of immutable historical body
versions. The882 accepted messages are a strict subset, not a complete event
catalog. Multiple-event/ensemble messages are often ambiguous. Absence of an
accepted record does not mean no eruption. No association with simulation44038
or retrospective prime/IPS metadata is assumed. This cannot establish50%
attainability or remove the incumbent's existing availability gaps.

Reproduce with `notification_archive_probe.py`; cached responses are reused,
so rerunning preserves the snapshot and does not silently refresh raw data.
Detailed coverage/records/receipts are in notification_archive/inventory.json.
No file hashes were generated; the previously documented automatic approval
gap still applies. Official issue-time records remain separate from forecast
truth and are not used to tune against F1/F2/F3 outcomes here.
