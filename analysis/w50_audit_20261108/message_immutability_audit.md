# Research: notification-body immutability and historical capture evidence

## Request Type / Direct Answer

Bounded source-gate audit, retrieved 2026-11-09. **Inconclusive for immutability
at issue; axis8 remains `blocked_source`.** Official documentation inspected does
not affirm that a sent notification's body is immutable. An independent historical
capture does exist for the selected training-era message, but it is too late to
justify using that body at its original issue-time forecast origins. This is not
proof that early evidence cannot be found elsewhere or for other messages.

Only `message_immutability_audit.md/json` were written. No scientific targets,
scores, model fits, delay fitting, experiment launch or gate bypass occurred.

## Official Docs Evidence

|Source|Affirmative evidence|What it does not establish|
|---|---|---|
|[Notification guide](https://ccmc.gsfc.nasa.gov/DONKI/guide/notifications/)|Full message/metadata; notification ID and version ID links; unique notification page.|No stated write-once body guarantee after sending, no documented complete body-edit history.|
|[Event guide](https://ccmc.gsfc.nasa.gov/DONKI/guide/events/)|Each event version can be viewed with a version picker.|Event versioning is not a guarantee that notification bodies or all embedded linked data are historically immutable.|
|[DONKI system description](https://ccmc.gsfc.nasa.gov/tools/DONKI/)|Public web/API access; internal DONKI-Editor data entry portal.|Editor existence does not prove either that sent bodies are editable or that they are locked.|
|[OpenAPI, version 2.0](https://ccmc.gsfc.nasa.gov/DONKI-API/v3/api-docs)|Notifications endpoint retrieves notifications “sent from DONKI”; fields include message ID, issue time, URL and body.|No documented original-body endpoint, notification revision/as-of query, immutable signature, or contemporaneous receipt field.|

The guide/API semantics support a **reported issue/sent timestamp**. They do not
bind the bytes returned today to that earlier timestamp. A `/1` URL, a new ID for
an update, or current UI/API equality does not independently prove historical
body identity. Absence of a guarantee in these bounded sources is not evidence
that modification actually occurred.

## Supporting Evidence: one independent archive mechanism

Internet Archive's [official snapshot documentation](https://archive.org/developers/tutorial-compare-snapshot-wayback.html)
defines a snapshot as the content captured when the crawler visits, and CDX
provides capture timestamp, original URL, MIME type and status. Its
[Availability API documentation](https://archive.org/help/wayback_api.php)
returns the closest capture; that is not necessarily a capture before a requested
origin. Therefore this audit used an explicit CDX date bound and inspected the
returned capture timestamps.

One distinct URL was probed (HTTP/HTTPS variants canonicalized to the same URL):
`https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/Alert/33664/1`.
An unfiltered exact-URL CDX query, bounded through 2025-12-28 and limited to 20
rows, returned four captures:

- 2024-11-11 23:32:30 UTC
- 2024-12-11 22:14:16 UTC
- 2025-04-21 12:45:49 UTC
- 2025-09-08 06:14:34 UTC

The [earliest returned capture](https://web.archive.org/web/20241111233230id_/https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/Alert/33664/1)
returned HTTP 200 and contains the actual message body, not merely a script shell:
message `20241002-AL-001`, reported issue `2024-10-02T02:15:47Z`, forecast source
speed 487 km/s and Earth-arrival prediction `2024-10-05T03:32Z`.
The HTML-decoded, whitespace-normalized **Summary** equals the locally stored
2026 snapshot's Summary. This is historical agreement at the **November 11
capture**, not byte-for-byte HTTP identity, proof of no intervening changes,
original publication time, or consumer receipt at issue. Notes and linked pages
were not used as scientific evidence or feature inputs.

The capture lag is **981.2786 hours (40 days 21 hours 16 minutes 43 seconds)**.
It is after the predicted arrival. Thus this observation cannot admit the message
as an advance warning for October 2–5 origins. “Earliest returned” is bounded to
this index query; it is not a proof of the earliest public appearance or complete
Internet Archive coverage. Exact query/result, capture URL/headers/raw HTML and
Summary comparison are preserved in the companion JSON.

## Version Note / Caveats

The current DONKI schema identifies itself as version 2.0; documentation pages
have no inspected immutable release identifier. Internet Archive's general API
page carries an old 2013 update label; the retrieval behavior above was checked
fresh rather than assumed from that page. One message supplies no corpus coverage
estimate, and no near-issue independent receipt was recovered in this bounded
probe. No code repository or third-party claim is needed for this finding.

## Reusable Takeaway: conditional capture admission and smallest next check

**Yes, conditionally:** for an exact body independently captured at time `c`, a
research reconstruction can admit that captured body only at origins no earlier
than `max(c, reported_issue_time)`, provided the archived payload and its timestamp
are verified. A later changed body needs its own capture evidence. Use the
captured content itself, not current linked data. This replaces an assumed issue
clock with a demonstrated capture clock for eligible records; it does not prove
body immutability or authorize a blanket source-gate change.

A crawler capture supports that the server served the content by `c`; it does not
by itself establish when the archive made its copy publicly retrievable, or when
an operational consumer ingested it. Claims about actual consumer replay still
need receipt/ingest evidence. Current local captures from November 2026 cannot
retroactively admit messages into 2022–2025 origins.

The smallest useful next observation is a **bounded, preselected training-era
sample of these same notification URLs**, checking whether exact recoverable
bodies have captures before their forecast-arrival/useful lead-time windows.
Report message count, recoverability and capture-versus-issue/forecast times only;
do not associate scientific outcomes or fit a delay. Even a positive result
supports only those capture-gated records. The present late capture keeps the
existing source gate unresolved; no duplicate kinematics experiment is warranted.
