# Research: seven-message historical capture coverage pilot

## Request Type / Direct Answer

Bounded source-gate data audit, 2026-11-09. **Zero of seven selected messages had
a returned successful HTML capture before its stated forecast Earth-arrival
time.** Three had later indexed captures; four returned empty filtered indexes
for both checked URL forms. No qualifying body was recovered, no message was
admitted, and the existing `blocked_source` gate remains unchanged.

## Preselection and method

The selection was exactly the seven explicit Earth-arrival messages in the
already completed deterministic 16-message training-era sample. Their IDs,
issued fields and selection timestamp were saved in `capture_coverage/preselected.json`
before network queries. No scientific targets or realized arrivals were used.

For each message, query Wayback CDX for the exact official URL in the prior
sample. If it supplies no candidate before the forecast arrival, check at most
one known legacy variant by changing `ccmc.gsfc.nasa.gov` to
`kauai.ccmc.gsfc.nasa.gov`. All seven needed that second query. Filters require
HTTP 200 and `text/html`; the date bound is through 2025-12-28, with at most
20 index rows and a 200,000-byte response cap per request. Timestamp order was
checked for returned capture rows. These are indexed captures, not inferred
publication times.

[Internet Archive's official documentation](https://archive.org/developers/tutorial-compare-snapshot-wayback.html)
explains the CDX capture timestamp, original URL, MIME type and HTTP status.
The [Availability API](https://archive.org/help/wayback_api.php) returns a closest
capture rather than necessarily one before a requested time; this pilot instead
used explicit CDX timestamps and compared them directly with the issued forecast.

## Results

All times below are UTC. “Empty” means no matching rows in these bounded,
filtered exact-URL queries; it does not mean no archive exists anywhere.

|Message|Issued forecast Earth arrival|Earliest returned successful HTML capture|Before forecast arrival?|
|---|---|---|---|
|20230101-AL-001|2023-01-02 20:00|2025-01-12 19:59:52|No|
|20230704-AL-001|2023-07-07 04:27|Empty, both URL forms|No candidate|
|20230706-AL-001|2023-07-09 07:59|Empty, both URL forms|No candidate|
|20240101-AL-001|2024-01-02 18:00|Empty, both URL forms|No candidate|
|20240412-AL-001|2024-04-14 16:20|Empty, both URL forms|No candidate|
|20241002-AL-001|2024-10-05 03:32|2024-11-11 23:32:30|No|
|20241002-AL-002|2024-10-04 20:40|2024-11-12 00:48:03|No|

All **14 requests returned HTTP 200**: 11 empty indexes and 3 indexes with
successful HTML captures. There were **zero access/parse failures, zero byte-cap
truncations and zero indexes reaching the 20-row limit**. All captures shown
came from the legacy-host variant.

There were no qualifying positive candidates, so this pilot fetched no new
HTML bodies. The separate preceding immutability audit already recovered the
late November 11 body for `20241002-AL-001`; that does not change its lateness.
Successful indexed HTML alone would not suffice: a positive candidate would
still require fetching its exact captured payload and checking the notification
ID, issue time and target-qualified Summary fields against the preselected record.
Admission would start at `max(verified capture time, reported issue time)`, never
at an earlier assumed issue time.

## Evidence and verification

`capture_coverage_audit.json` contains all query URLs, returned indexes, timestamps,
comparisons and counts. `capture_coverage/` preserves the preselection and each
raw CDX response plus receipt metadata (query/resolved URL, HTTP status/headers,
start/end receipt times and size). Seven-message uniqueness, all 14 preserved
response/receipt pairs, zero positive candidates and absence of truncation/errors
were checked successfully. No fitting, delay fitting, score reads or campaign
changes occurred.

## Caveats / Reusable Takeaway

This small sample does not establish corpus-wide absence or sufficient coverage.
Redirects, other MIME types, alternate IDs/routes, unindexed captures and other
receipt channels were not surveyed. Earliest returned capture is not original
publication, immediate public archive access or consumer receipt. The official
archive documentation is unversioned; fresh query behavior supplies the concrete
retrieval evidence here.

The pilot removes a specific near-term justification for using this Wayback
route to admit these seven messages at useful pre-arrival origins. It supplies
**no new proof to clear `blocked_source` and no basis for another kinematics run**.
The next source-gate action needs a concrete earlier body/receipt archive or newly
evidenced early-capture URL, followed by exact-payload and coverage checks. These
results do not warrant a large crawl of the same route on their own, and do not
prove that a different historical evidence channel is impossible.
