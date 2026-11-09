# Research: What issued CME information survives beyond Kp/pending?

## Request Type

Bounded mechanism and implementation-reference audit. Exa search/fetch and official
schema retrieval on 2026-11-09; stored notification snapshot retrieved 2026-11-08.
No fitting, model selection, evaluation targets, observed arrivals, or plot
extraction. The failed tree-count-selection result (three-seed mean 99.4005 versus
97.8136) and the 28-feature baseline are supplied by the leader. They motivate
an input-information audit, not another optimizer change. The campaign's fixed
97.492442 comparator and 50% reduction objective are not attainability claims.

## Direct Answer

**Audit target-qualified issued Earth-arrival forecasts and their revisions before
introducing another model or constructing a propagation transform.** A small
training-era sample already proves that source speed and forecast arrival can
change while the issued Kp range stays fixed. It does **not** prove that this
information is absent from the baseline's target-aligned `pending` representation,
or that it improves hourly speed forecasts. That requires a source-to-feature
comparison with the existing Kp/pending construction.

The messages can carry an explicit predicted arrival clock without providing a
physical initial-condition clock. None of the 16 sampled summaries supplies a
speed reference height or Time@21.5. Therefore do not manufacture a DBM start
state from a CME ID timestamp, event-start label, message issue time, or movie
filename. Missing reference epochs remain missing. As-issued products cannot
repair forecast origins before those products were issued.

## Official Docs Evidence

- [DONKI OpenAPI](https://ccmc.gsfc.nasa.gov/DONKI-API/v3/api-docs), retrieved schema
  version **2.0**: `/get/notifications` retrieves notifications sent from DONKI
  within a maximum 60-day window. `NotificationObject` exposes `messageType`,
  `messageID`, `messageURL`, `messageIssueTime`, and string `messageBody`.
  It does not supply typed Earth-arrival, speed-reference-epoch or duration fields;
  a body parser needs an explicit grammar and abstention policy.
- [Official notification guide](https://ccmc.gsfc.nasa.gov/DONKI/guide/notifications/):
  a notification opens its full message/metadata, with notification/version links.
  This does not document historical body immutability or contemporaneous consumer
  receipt. Existing max(API issue time, body issue time) handling remains necessary,
  but cannot establish those missing properties.

The exact schema subset and selected Summary bodies are preserved in
`cme_feature_design_evidence.json`. Notes were excluded before inspection.

## Supporting Examples: deterministic training-era sample

Selection fixed before reading bodies: first two already accepted messages in
each of January/April/July/October 2023 and 2024, using inventory order. This is
16 messages from the existing 882-message accepted subset, not a representative
prevalence estimate. No evaluation truth or current-event association was used.

|Summary content|Count / 16|Interpretation|
|---|---:|---|
|Labeled source speed, width, HEEQ direction and event-start time|16|Selection already required unambiguous kinematics; event time is not a boundary time.|
|Explicit predicted arrival at **NASA missions near Earth**|7|Target-qualified forecast information; not observed arrival.|
|No explicit Earth arrival|9|Other-target forecasts appear; absence is not an explicit Earth miss.|
|Explicit Time@21.5 or speed reference height|0|Sample does not justify a physical propagation initial state.|
|Numeric local Earth-arrival speed or disturbance duration|0|Source speed cannot supply a local speed waveform by renaming it.|

The seven Earth-arrival statements use this precise inspected syntax:

> Simulations indicate that the **[leading edge / flank]** of the CME **[will / may]**
> reach **NASA missions near Earth** at about **[ISO timestamp]**
> **(plus minus 7 hours)**.

The phrase `NASA missions near Earth` is essential. For example,
[20230103-AL-001](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/23110/1)
forecasts Parker Solar Probe arrival; extracting its first timestamp as an Earth
arrival would be wrong. The Summary can contain several target-specific times
even when it contains only one CME. The stated ±7 hours is forecast text, not an
empirically calibrated Gaussian standard deviation or coverage guarantee.

The decisive same-event example is entirely within the training era:

|Field|[20241002-AL-001](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/33664/1)|[20241002-AL-002](https://ccmc.gsfc.nasa.gov/DONKI/view/Alert/33676/1)|
|---|---|---|
|Conservative issue time UTC|2024-10-02 02:15:47|2024-10-02 13:17:33|
|CME event ID|2024-10-01T23:09:00-CME-001|Same; explicitly updates AL-001|
|Source speed km/s|487|594|
|Half-angle degrees|39|38|
|HEEQ longitude/latitude degrees|−17/−5|−19/−10|
|Earth leading-edge forecast UTC|2024-10-05 03:32|2024-10-04 20:40|
|Maximum Kp range|4–6|4–6|

Arrival moves 6 hours 52 minutes earlier; Kp range remains identical. This is
**direct evidence of information loss in Kp alone**, not proof of new information
relative to `pending` or ACE. The second message's values must not appear before
13:17:33. Both versions must remain available for an as-of reconstruction.
The companion JSON records all 16 IDs, URLs, cached monthly files, original API
URLs, receipt metadata, exact Summary text and audit classifications.

## Source-Reference Evidence: physical meaning and required observations

|Proposed information|Primary evidence|Permitted interpretation and missing condition|
|---|---|---|
|Source speed relative to ambient wind|[Vršnak et al. 2013](https://doi.org/10.1007/s11207-012-0035-4); [author DBM equations, §1.1](https://oh.geof.unizg.hr/DBM/docs/DBM.pdf)|Drag acceleration is `−γ(v−w)|v−w|`: fast structures decelerate and slow ones accelerate. Source speed gives potential amplitude/propagation information, **not** local 1-AU plasma speed. Physical propagation needs co-referenced `t0, R0, v0`, ambient wind `w`, and drag assumptions `γ`; message issue time only gates availability.|
|Direction and width|[Dumbović et al. 2021, DBM tools](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2021.639986/full); [author documentation §1.2](https://oh.geof.unizg.hr/DBM/docs/DBM.pdf)|Apex/flank geometry changes which part of a CME reaches a target and its time/speed. Direction and width are conditional impact descriptors; simple angular overlap is not guaranteed Earth impact. Preserve HEEQ convention, target direction, and measurement/front type. A shock fit and ejecta leading edge are not interchangeable.|
|Issued modeled arrival|[Wold et al. 2018, §2 and §3](https://www.swsc-journal.org/articles/swsc/full_html/2018/01/swsc170034/swsc170034.html)|ENLIL inserts disturbances at 21.5 solar radii into a structured background wind. An already-issued, explicitly Earth-qualified arrival carries the forecaster/model's propagation estimate. `target_time − issued_predicted_arrival` is an auditable relative clock without inventing `t0` or solving DBM; it is not a certain onset or fitted time shift.|
|Amplitude omitted by Kp summary|[Wold et al. 2018, §5](https://www.swsc-journal.org/articles/swsc/full_html/2018/01/swsc170034/swsc170034.html)|Kp estimates compress modeled fields through coupling and 90/135/180-degree clock-angle scenarios, with rounded extrema. Kp cannot uniquely reconstruct the speed waveform. Source speed/geometry may add signal, but mass/density, background wind, interactions and model errors still matter.|
|Uncertainty and model limitations|[Mays et al. 2015, v2](https://arxiv.org/abs/1504.04402v2); [Wold et al. 2018](https://www.swsc-journal.org/articles/swsc/full_html/2018/01/swsc170034/swsc170034.html)|Arrival spread depends on input distributions and background conditions; ENLIL cone CMEs lack an internal magnetic field. Wold's historical 10.4±0.9 h MAE is conditional on 273 correctly predicted arrivals across three targets, not hourly-speed skill or validation of each message's ±7-hour range.|

The DBM documentation explicitly links drag to cross section, mass and ambient
wind density. Having a width and source speed does not identify those quantities.
The 2021 author paper also warns that DBM physically describes the magnetic
structure and is often used only as a proxy for shock arrival. Training-time ACE
measurements could be an **assumed** ambient proxy; they are not automatically the
wind along the CME's future trajectory. No coefficient, background proxy, or
propagation-model choice is adopted here.

## Version Note / Verified Implementations

Wold verifies an operational CCMC implementation from 2010–2016, mostly ENLIL
2.7/a4b1 with WSA 2.2, with an earlier ENLIL 2.6/a3b2 subset. Mays' version is
arXiv v2 dated 2015-05-11. These support mechanism and limitations, not current
2026 accuracy. The 2021 DBM author paper documents five author-developed tools;
the [Hvar author service](https://oh.geof.unizg.hr/DBM/dbm.php) exposes explicit
initial distance/speed and ambient/drag inputs. Its example defaults are not
parameters selected for this task. Its interface text inconsistently calls Rs
“Sun diameter”; the author equation document defines solar **radius**, so the
form should not be copied uncritically.

No source-code dependency was adopted, downloaded or executed. No repository
implementation is asserted verified at a pinned SHA; the verified implementation
evidence here is the original operational/author validation, sufficient for this
input-audit decision. No third-party tutorial is used as authority.

## Caveats / Ambiguity Flags

Current official archival bodies and original issue labels are useful evidence
but do not prove that bodies were immutable or received by a consumer at issue
time. Keep that uncertainty separate from the parsing audit; a retrospective
reconstruction cannot silently become a deployability claim. Linked current
analyses, prime flags, observed IPS, prediction-error annotations, Notes, and
later model plots cannot fill missing issued fields. The sample does not resolve
multi-event or ensemble attribution, corpus-wide arrival coverage, or the exact
information already encoded by baseline pending/Kp.

## Reusable Takeaway: next audit and a falsifiable cumulative question

1. Extend the **data audit**, before any model run: classify Summary-only Earth
   arrival, explicit Earth non-impact, other-target-only, missing, multiple and
   ambiguous statements. Do not treat the latter categories as zero/no CME.
   Require an unambiguous event association; retain multi-event/ensemble messages
   as unsupported until their event-to-target attribution is independently clear.
   Reject duplicate/conflicting target timestamps instead of choosing one.
2. Preserve exact message/event IDs and update links. Resolve the latest eligible
   message at each origin using conservative issue time; never use a later update
   to fill an earlier record. Missing Earth-arrival text in an update requires
   explicit carry-forward semantics or abstention, not silent inheritance.
   Keep arrival time, front/flank wording and stated uncertainty together.
3. Compare this issued clock and geometry with the **existing source-to-feature
   mapping** for Kp/pending on training inputs only. Count coverage, duplicates,
   version changes and whether target-relative timing is already preserved. No
   target association or F1/F2/F3 coefficient/feature selection is needed for this
   audit. This is the outstanding repo-local handoff, not completed in this lane.
4. Only if nonredundant, sufficiently attributable inputs survive, a future
   preregistered A+B can ask: **does retaining issued arrival/geometry/kinematic
   information improve the frozen baseline beyond its current Kp/pending summary?**
   Preserve all 28 baseline features explicitly; keep fit/evaluation protocol and
   matched seeds fixed; use the project's existing adoption criteria. The audit
   decides B's supported content, not a preset model family or experiment count.

A null controlled result after availability and matrix checks refutes the tested
encoding's incremental utility; it does not show all CME information is useless.
A gain requiring pre-issue values, invented reference epochs, or future revisions
refutes the causal claim. No improvement magnitude or 50% goal feasibility is
inferred from the literature or this sample.

## Narrow addendum: top-level arrival semantics and resolved baseline mapping

Follow-up requested by leader, 2026-11-09. **Do not interpret an `impactList`
containing only other spacecraft as evidence that the top-level arrival is
non-Earth, and do not require an Earth entry in that list.**

The official OpenAPI v2.0 schema gives `estimatedShockArrivalTime` only
`type: string`, without a target description, in both `WSAEnlilObject` and
`WSAEnlilJSON`. `Impact` separately has `location`, `arrivalTime`,
`isGlancingBlow`, and `isMinorImpact`. The
[official data reference](https://ccmc.gsfc.nasa.gov/DONKI/guide/data-reference/)
does not define the top-level field. Thus a **universal schema-level Earth
semantic guarantee was not found**. That absence does not establish generic or
arbitrary-target semantics.

There is stronger official presentation evidence for an **Earth-specific
convention**. The training-era
[WSA-ENLIL/33848/1 page](https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/WSA-ENLIL/33848/1)
explicitly displays:

> Earth Impact: (Glancing Blow) Earth Shock Arrival Time = 2024-10-11T04:00Z.
> Other Location(s) Impact: STEREO A ... 2024-10-10T23:43Z;
> Mars ... 2024-10-12T12:00Z.

The [matching official API response](https://ccmc.gsfc.nasa.gov/DONKI-API/get/WSAEnlilSimulations?startDate=2024-10-08&endDate=2024-10-08)
identifies `WSA-ENLIL/33848/1`, with top-level
`estimatedShockArrivalTime="2024-10-11T04:00Z"`, `isEarthGB=true`, and an
`impactList` containing **only Mars and STEREO A** at their separately labeled
times. This directly corroborates the Earth interpretation for this record and
explains why an Earth entry need not appear in `impactList`; one record does not
prove every historical payload is correctly populated. The JSON addendum stores
only the relevant forecast fields. No observed arrival, prediction-error
annotation, prime flag or current evaluation association was used.

The leader's repo explorer has now resolved the earlier feature-redundancy gap:
the current 28-feature baseline discards numeric arrival time and forecast-horizon
alignment. `pending` retains a coarse `dt >= −6h` eligibility condition, so it is
not wholly independent of arrival timing. These are supplied repo findings,
not independently reinspected in this external-evidence lane. The existing
completion-floor-plus-one-hour availability rule also differs from notification
issue gating.

The next input audit can therefore test explicit **issued arrival relative to
target time**, with update provenance and source-change accounting, while
preserving the baseline. Notification Earth-qualified wording offers explicit
target attribution and issue evidence; it does not establish immutable bodies or
consumer receipt. Do not label the existing baseline target-contaminated merely
because `impactList` omits Earth. No model, coefficient or expected gain is selected.
