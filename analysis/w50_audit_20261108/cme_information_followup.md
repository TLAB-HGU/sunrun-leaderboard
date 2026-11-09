# Research: origin-available CME information and DONKI time semantics

Retrieved 2026-11-08 using Exa search/fetch, followed by direct retrieval of the
official OpenAPI schema. Request type: conceptual documentation, implementation
reference, and scientific evidence. This follow-up does not fit models, read
benchmark target arrays, refresh repository source data, or choose an experiment.
The existing baseline's kp/pending summary is supplied by the parent audit.

## Direct answer

CME speed, direction, width and modeled propagation contain physically relevant
information about transient speed enhancements that a geomagnetic Kp summary
does not uniquely preserve. That is a mechanism-based hypothesis, not evidence
that adding these fields improves this benchmark or can halve official S.
The immediate prerequisite is a version-and-publication audit: current DONKI
JSON is not a historical as-of snapshot. In particular, model completion,
measurement time at a solar radius, database submission and notification issue
are different times.

A concrete official record proves the distinction: ENLIL/41054/1 completed at
2025-09-05 16:13 UTC but version1 was submitted at16:55 UTC,42 minutes later.
It explicitly says no notification had been sent. Using completion time as
public availability is therefore unsupported even for this documented example
([version page](https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/WSA-ENLIL/41054/1)).
This example does not estimate the delay distribution of the full corpus.

## Official field and version evidence

The [official OpenAPI document](https://ccmc.gsfc.nasa.gov/DONKI-API/v3/api-docs)
identifies itself as DONKI-API version2.0 on retrieval. The legacy
`/DONKI/WS/get/` service and new `/DONKI-API/get/` schema are separate interfaces;
do not assume a historical downloaded response has every field in the current
schema. Important schema evidence is preserved in `cme_information_sources.json`.

|Field or concept|Verified meaning and evidence|Causal use limitation|
|---|---|---|
|`cmeStartTime` / CME `startTime`|CME event start identifier/time; model input is distinct from simulation completion. The official model/version pages show both.|Not a database publication or receipt time. Do not infer when a forecast operator had sufficient images.|
|`speed`|CME analysis speed in km/s. Version pages explicitly label it; the model uses coronagraph-derived kinematics to initialize a disturbance.|Not the ACE bulk speed at a future target hour, and not a measurement of the eventual L1 shock speed.|
|`latitude`,`longitude`,`halfAngle`|Official analysis pages label direction in HEEQ and half angular width in degrees; leading-edge vs shock-front feature type is separately recorded.|Keep coordinate/sign convention and feature type; a wide shock-front fit is not interchangeable with a bulk leading-edge fit.|
|`speedMeasuredAtHeight`|Official analysis page labels “Speed measured at height (Rs)”. It is a heliocentric height in solar radii.|It is not a timestamp. No generic `measuredAt` timestamp was found in the inspected CMEAnalysis schema; require a mapping before treating such a local name as a clock.|
|`time21_5`|Official UI labels Time@21.5; WSA-ENLIL+Cone inserts the CME disturbance at21.5 solar radii. Papers describe transit time starting at that inner boundary.|A physical/model reference time, not the analysis submission or release time. Its value can precede when the analyst made the estimate.|
|`modelCompletionTime`|Simulation completion; `/get/WSAEnlilSimulations` documentation explicitly says date filtering is **via model completion time**.|Neither the API date filter nor this field proves origin-time availability.|
|`submissionTime`,`versionId`|Present in current CMEAnalysisObject and CMEJSON schemas; official event guide supports viewing every version; concrete analysis pages list per-version submission times.|Supports a version audit, but does not by itself prove a particular consumer received the payload then. The reviewed WSAEnlilObject schema has neither of these fields.|
|`messageIssueTime`|Current notification schema describes notifications sent from DONKI; the notification payload includes ID, issue time, URL and body.|Use only information actually included in that versioned message. A link may later resolve to updated data. Preserve the issued body.|
|`isMostAccurate` / prime|Official reference: most accurate analysis **per measurement type**; more than one true measurement can exist for a CME.|A current prime flag is a current assessment. It cannot retrospectively select the best measurement at all earlier forecast origins.|
|`levelOfData`|Official analysis UI:0 real-time;1 real-time checked by supervising forecaster;2 retrospective science analysis.|Neither0 nor1 alone supplies publication timing. Level2 is explicitly retrospective and cannot automatically enter origin-available features.|

Sources for the table:
[official data reference](https://ccmc.gsfc.nasa.gov/DONKI/guide/data-reference/),
[official event/version guide](https://ccmc.gsfc.nasa.gov/DONKI/guide/events/),
[official notification guide](https://ccmc.gsfc.nasa.gov/DONKI/guide/notifications/),
[official analysis version example](https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/CMEAnalysis/46603/2),
[Wold2018 model and verification sections](https://doi.org/10.1051/swsc/2018005).
The2026 analysis example was inspected for field labels and revision mechanics,
not matched to benchmark truth or used for a feature/parameter decision.

The current CMEAnalysis endpoint defaults to `mostAccurateOnly=true`,
`completeEntryOnly=true`, and `version=Latest`. Its schema documents a version
parameter but does not enumerate an as-of timestamp query or all accepted
version values. The ENLIL endpoint exposes only startDate/endDate query parameters
in this schema. Thus a request bounded by dates is not an as-of query.

## Direct evidence of retrospective enrichment

A currently returned [official legacy ENLIL response for2025-01-01–02](https://ccmc.gsfc.nasa.gov/DONKI/WS/get/WSAEnlilSimulations?startDate=2025-01-01&endDate=2025-01-02)
contains a simulation completed January2 and an input's `ipsList` linking
observed shocks dated January3–4. Wold2018 independently states that some
observed ICME arrivals were added post-event and corrected against catalogs.
Consequently, future observed arrivals and link relationships in current JSON
must never be inherited as forecast features merely because their containing
simulation has an earlier completion time.

Even an explicit `/.../1` event page can display prediction-error annotations
or linked information added later; the inspected41054/1 page includes such PE
annotations. A version-specific page is evidence for the submitted forecast
fields, not proof that every dynamically embedded annotation was visible then.
Do not extract PE, observed IPS, retrospective link existence or updated notes
into fitting inputs. No numeric PE was used in this research.

## Physical mechanism and validated implementation scope

[Mays et al.2015](https://arxiv.org/abs/1504.04402v2),
DOI10.1007/s11207-015-0692-1, describes the real-time CCMC implementation:
coronagraph speed/location/width initialize a hydrodynamic CME-like disturbance
at21.5 solar radii in the WSA background wind. It does not simulate eruption
initiation. Ambient wind, CME geometry, preceding transients and assumed
density affect propagation. Fast-CME modeled speeds can be overestimated;
the cone treatment lacks the magnetic structure of an actual ejecta.

[Wold et al.2018](https://doi.org/10.1051/swsc/2018005) validates the operational
CCMC WSA-ENLIL+Cone implementation over2010–2016, mostly ENLIL2.7/a4b1 with
WSA2.2; an earlier subset used ENLIL2.6/a3b2. For273 correctly predicted arrivals
across Earth and STEREO, arrival-time MAE was10.4±0.9h, with mean bias−4.0h.
These are conditional event-arrival metrics, not hourly speed S and not a2026
skill estimate. Sources of error include input parameters, background wind,
prior CMEs, and assumed cloud properties. The paper explicitly notes that
many slow or out-of-ecliptic CMEs may not be modeled: absent ENLIL coverage is
not a reliable negative eruption indicator.

The same paper says Kp is estimated from model outputs using a coupling
function and assumed magnetic clock-angle scenarios90/135/180 degrees. Therefore
Kp is not an invertible encoding of the local speed waveform. This supports
testing whether retained CME kinematics or a versioned model speed timeline
adds information, while leaving the expected gain unknown.

ENLIL's physical simulation produces time-dependent plasma fields; the inspected
DONKI WSAEnlilObject JSON schema contains arrival/duration/Kp summaries and CME
inputs but no predicted speed waveform. Official event pages link timeline GIFs,
and [CCMC API documentation](https://ccmc.gsfc.nasa.gov/tools/API/) points to ISWA
HAPI/file services. The existence of services does not establish that a complete,
immutable, machine-readable **as-issued** timeline archive is available for all
training and evaluation origins. That dataset retrieval/version audit remains
unresolved. No digitization of plots or retroactive timeline substitution is
recommended here.

These are original-paper validations of an operational implementation. This
lane did not execute ENLIL, reproduce those papers or verify an open-source
forecast implementation at a pinned commit; no OSS reproducibility claim is made.

## Evidence needed before a causal input experiment

1. For each CME/simulation, preserve event ID, exact analysis/version, measurement
   feature and data level, input kinematics, model version/run ID, submission
   time, notification issue/body and any consumer receipt or archive capture.
2. Resolve event versions valid at each origin, rather than using the current
   prime flag or latest payload. Require all contributing information to be
   available by that origin; retain unknown availability explicitly.
3. Strong evidence is a timestamped archived response/notification containing
   the exact values. A version submission record supports a conservative
   research reconstruction but should remain distinct from proven public
   exposure and consumer receipt. A later capture cannot prove earlier access.
4. Whitelist forecast fields. Exclude observed IPS, PE, retrospective event
   links, post-event annotations and level2 analyses. Do not use F1/F2/F3 labels
   to choose among competing measurements or to estimate delay/propagation
   coefficients. Any coefficients must be fixed physically or fitted pre-cutoff.

No new policy is adopted by this proposed protocol. If issue-time evidence
cannot be recovered, label an experiment as a retrospective-information
sensitivity; it cannot support a real-time deployability claim or erase the gap.

## Falsifiable hypothesis and next decision

Hypothesis: among origins with a documented pre-origin CME analysis, source
kinematics and Earth-impact geometry supply transient amplitude/timing
information absent from kp/pending summaries. Necessary observations are the
versioned coronagraph analysis and, for waveform use, as-issued ENLIL output.
It applies after sufficient observations/analysis become available, not before
the eruption or before those products arrive. Null effects can arise from poor
coverage, measurement uncertainty, already-redundant inputs or model misuse.

Refutation for a preregistered frozen learning configuration requires valid
availability/matrix checks and a matched-seed controlled addition that fails
the existing improvement/uncertainty criteria. Separate amplitude from arrival
timing changes without fitting target-aligned shifts. A benefit restricted to
records whose versions were created after origin is evidence against a causal
improvement claim. This report fixes neither a model family nor an experiment
count and asserts no50% attainable bound.

The decision-changing unresolved question is whether an adequate **as-issued
archive of the exact kinematics/model outputs** covers the important forecast
origins. Current JSON plus `modelCompletionTime` alone cannot answer it.

## Bounded follow-up: January18 event and simulation44038/1

The leader supplied current local simulation44038/1 values: speed1820km/s,
half-angle53degrees, HEEQ longitude−2/latitude−6, time21.5=January18 19:55,
model completion21:35 and predicted arrivalJanuary19 20:17. Its exact version1
submission time remains **unverified**: Exa timed out on the legacy version
page, direct retrieval of the legacy host failed, and the new official route
returned a JavaScript page without exposed event text. No timestamp is invented.

The [official notifications endpoint](https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications?startDate=2026-01-18&endDate=2026-01-19&type=CME)
does supply preserved issued message bodies for CME2026-01-18T18:09:00-CME-001.
The following are forecast statements in those messages, not observed outcomes:

|Message and issue time from its body|Issued CME kinematics|Issued Earth arrival prediction|
|---|---|---|
|[20260118-AL-005](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/44037/1), January18 21:39:21UTC|1431km/s;half-angle41degrees;HEEQ−27/−3|January20 02:18UTC ±7h|
|[20260119-AL-001](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/44046/1), January19 07:24:38UTC|1473km/s;half-angle46degrees;HEEQ−22/−6|January20 01:07UTC ±7h|
|[20260119-AL-004](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/44056/1), January19 17:22:57UTC|24-member beta ensemble; no single1820km/s analysis asserted|January20 00:07–03:47UTC;average01:50UTC|

The first two messages link output filenames beginning respectively
`20260118_201400` and `20260118_201700`, not the44038 reference time19:55.
Their bodies do not identify simulation44038. The ensemble message links
`2026-01-18_ncmes1_sims24_LAHAINA093`. Thus the differing values may represent
different fits/runs; they do **not** establish that44038 was revised or leaked.
They establish the exact stated forecast information issued at those later
timestamps. The public issue record is useful availability evidence; consumer
receipt remains a distinct fact. It does not authorize1820km/s at earlier origins.

The same issued bodies identify flare activity2026-01-18T17:27:00-FLR-001 and
state that the flare peaked18:09. This further supports treating18:09 as a peak
reference, not proof that all earlier origins lacked eruption information.
Any local diagnostic stratum retaining “before18:09” must keep that limitation.

Full selected message bodies and API URL are preserved in
`cme_information_sources.json` under `target44038_followup`. Current prime
flags, observed IPS links, and retrospective prediction errors were excluded
from all causal recommendations. The unresolved44038 submission timestamp is
an explicit evidence gap, not a reason to substitute a different event's time.
