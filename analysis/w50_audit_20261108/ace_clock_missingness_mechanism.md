# Research: ACE/SWEPAM fixed-clock missingness

## Request Type

Bounded data-validity and official-source research. Local retrieval date: 2026-11-09; direct mirror receipt: 2026-11-08 20:30 UTC. No fitting, evaluation/query or treatment access, experiment selection, or target/feature/filter/model/policy change.

## Direct Answer

**The clock-dependent target gaps are already encoded in the upstream monthly real-time product; local hour-binning did not create them. The physical/producer cause remains unresolved.**

The descriptive window was frozen **before raw-value inspection** to [2025-01-01 00:00, 2025-02-01 00:00) UTC, one original training month. All 744 hourly timestamp rows exist uniquely. At UTC03, UTC09 and UTC15, every one of the 31 records is status9 (“no data”) with speed −9999.9. At UTC21, 16 records are status9/sentinel, four have positive speed but status1, and 11 are nominal. The cached source is byte-identical to a fresh 55,191-byte retrieval from the [NASA mirror](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/202501_ace_swepam_1h.txt). The unchanged official observation parser matches the direct rule on all 744 records.

This excludes a local absent-row or target-hour-shift explanation for this month. It does **not** distinguish absent instrument measurements from telemetry loss, hourly-product logic, or a persisted upstream archive artifact. Status9 describes the product record; it does not by itself prove the spacecraft failed to measure the solar wind.

## Frozen Data and Parser Evidence

[Scope frozen before inspection](ace_clock_missingness/frozen_scope.json); [counts and mirror receipt](ace_clock_missingness/january_counts.json); [cached header](ace_clock_missingness/cached_header.txt); [retrieved monthly source](ace_clock_missingness/remote_202501_ace_swepam_1h.txt).

| January 2025 count | Result |
|---|---:|
| Expected / present / unique hourly timestamps | 744 / 744 / 744 |
| Nonhour timestamps / absent timestamps | 0 / 0 |
| Valid target under existing rule | 494 (66.40%) |
| Invalid target under existing rule | 250 (33.60%) |
| Speed sentinel −9999.9 | 141 |
| Positive finite speed rejected by nonzero status | 109 |
| Status0 with nonpositive/nonfinite speed | 0 |
| Official parser mismatches | 0 |
| Current mirror status/speed differences | 0 |

The 141 sentinels comprise 132 status9 and nine status3 records. The 109 rejected positive speeds comprise 106 status1 and three status6 records. No narrower meaning for codes1,3,6 is inferred from the header: it groups codes1–8 as “bad data record.”

| UTC hour | Records | Nominal / valid | Status9 + sentinel | Other invalid |
|---|---:|---:|---:|---:|
| 03 | 31 | 0 | 31 | 0 |
| 09 | 31 | 0 | 31 | 0 |
| 15 | 31 | 0 | 31 | 0 |
| 21 | 31 | 11 | 16 | 4 status1 |
| Other 20 hours combined | 620 | 483 | 23 | 114 |

Thus 113/124 records at the four named hours are invalid versus 137/620 at other hours. This is a fixed-window description, not an independent-event significance test or a revised estimate for the complete training period. The earlier sealed audit's full F1-training figures remain unchanged: 668,736/2,037,384 filled labels and UTC03/09/15/21 missing fractions81.43/93.98/94.49/85.35%. Forecast rows repeat physical target hours; this month counts unique hourly source records.

The supplied repository context distinguishes the paths:

- `leaderboard/scorer/competition_data.py:16–39`: named month, original label timestamp, speed retained only if status==0 and finite speed>0. This existing function was called on the frozen January payload only; its downloader was not called.
- `data-v2/notebooks/raw_to_input_df.ipynb`, code cell4: targets use the same unshifted status0/finite-positive rule. Features mask sentinel/nonfinite/nonzero-status values and map labels to floor(time,1h)+1h. For these exact-hour rows that is a one-to-one timestamp shift, not an hourly aggregation. Feature gaps therefore move one hour later; target gaps do not.
- Notebook duplicate selection is “last source file,” while the official parser uses the named month. This source has no duplicate labels; this task did not rescan other months. The earlier sealed exhaustive equality audit supplies the broader context.
- `data-v2/fetch.py:145–168` preserves existing rows and adds missing timestamp keys. That can retain early records in general, but is not a sufficient explanation for this month's local pattern: the present upstream file has exactly the same bytes.

The cached and retrieved header says `:Issued: 2025 Feb 01 0510 UT`. The HTTP Last-Modified is 2025-02-01 04:10 UTC. These are distinct metadata fields, not an independently reconstructed measurement time or proof of historical receipt availability.

## Official Docs Evidence

**Actual product.** The [January source header](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/202501_ace_swepam_1h.txt) identifies NOAA SWPC as preparer, ACE SWEPAM as instrument, bulk speed in km/s, and “Hourly Averaged Real-time Bulk Parameters of the Solar Wind Plasma.” It defines status0 nominal,1–8 bad record,9 no data and speed/density sentinel−9999.9. NASA/SolarSoft hosting does not convert this into calibrated Level2 science data.

**Producer versus science archive.** [NOAA's ACE real-time product page](https://www.spaceweather.gov/products/ace-real-time-solar-wind) distinguishes real-time hourly/minute products from ACE Science Center science products, and says the displayed real-time data have no official archive, while historical copies are available through NOAA/SolarSoft. This distinction limits claims of historical availability and calibration.

**Instrument sampling and mode gaps.** [McComas et al.,1998, SWEPAM instrument paper](https://izw1.caltech.edu/ACE/ASC/docs/swepam-ssr-paper.pdf), sections5.2 and5.4, describes 64-second modes and a nominal normal cycle of one search mode followed by 30 solar-wind-ion tracking modes, with command-adjustable cycle length. RTSW is supplied only in the tracking mode, nominally30/31 data sets. Section5.4 also describes RTSW as a reduced subset of angular samples and cautions that its derived moments differ from complete science products.

This gives a documented reason for short mode-related gaps in the original design. It does **not** describe one missing hourly product at03/09/15/21UTC, and command-adjustable1998 modes cannot establish2025 operation. The nominal31×64s cycle is approximately33minutes, not six hours.

**Calibrations and quality.** [SWEPAM team Level2 release notes](https://izw1.caltech.edu/ACE/ASC/DATA/level2/swepam/swepam_release_notes) document weekly onboard CEM gain calibrations, later calibration reprocessing, pointing dependence and historically problematic penetrating-particle/energy-selection intervals. They also describe a full-energy-range mode once each half-hour during discussed high-activity cases. None establishes a fixed six-hour calibration outage. Assigning this clock pattern to calibration would be unsupported.

**Publication schedule.** NOAA's page contains a **July30,1998** announcement: monthly hourly files were recreated every three hours beginning0010UT, with final previous-month creation on the first day of the new month. This is a historical *file publication* schedule, not an observation-quality schedule. It neither matches the asserted six-hour pattern nor establishes the2025 processing cadence. The retrieved header's0510 issue time also cautions against treating that old schedule as a current invariant.

**Other documented problems.** The same NOAA page describes historical low-speed flagging, occasional tracking/transmission dropouts, a2006 processing update and a2012 pointing change. Those establish possible classes of RTSW quality issues, not the cause of this January2025 clock pattern.

## Exact Averaging-Time Support

“Hourly averaged” is explicitly documented. **The reviewed RTSW header and producer pages do not establish whether labelT represents [T−1h,T), [T,T+1h), or an interval centered onT.** Nor did they establish the precise weighting, minimum valid sample coverage, or minute-status-to-hour-status reduction rule.

The [ACE Science Center Level2 description](https://izw1.caltech.edu/ACE/ASC/level2/swepam_l2desc.html) lists64-second ion, hourly and other averaging products and references Level2 timing documentation. That is a different product; its timestamp conventions cannot simply be imported into this NOAA RTSW monthly stream.

The code's feature timestamp shift is an implementation fact. It is not evidence of producer integration boundaries. Nominal hourly cadence, or a guessed shift, also cannot prove calibrated ACE–Wind agreement: instrument response, product quality, actual integration support, spacecraft position and solar-wind propagation still matter.

## Mechanism Assessment

| Candidate explanation | What is established |
|---|---|
| Local missing timestamp rows | Refuted in the frozen source: all744 rows exist. |
| Target hour-binning creates the four-hour pattern | Refuted for the inspected path: target labels are unshifted, and sentinel/status gaps exist beforehand. |
| Quality mask alone discards all clock-hour values | Insufficient:03/09/15 already contain status9/sentinels. The mask additionally rejects109 numeric records overall. |
| Local insert-only cache uniquely introduced the gaps | Not supported for this month: current upstream is byte-identical. Earlier upstream history remains unknown. |
| Instrument search/track cycle | Documented historical mechanism for short gaps; no matching2025 UTC schedule. |
| Weekly gain calibration | Documented, but does not explain six-hourly missingness. |
| Telemetry/receiving schedule | General historical dropouts documented; matching2025 schedule unresolved. |
| Hourly aggregation/publication/archive artifact | Plausible unresolved class; no producer algorithm/log evidence assigning the cause. |

## Version Note

Documentation was retrieved on2026-11-09 local time. The source payload concerns January2025 and is marked issued2025-02-01. The instrument paper is1998; NOAA announcements cited span1998–2012; Level2 release notes include2021 reprocessing, and the Level2 description is marked updatedOctober2025. These dates are preserved rather than silently treating all material as current operational specifications.

## Source-Reference Evidence

Repository references are supplied-context parser facts, not a new migration or implementation review. Official excerpts, URLs and historical caveats are retained in [official_doc_receipts.json](ace_clock_missingness/official_doc_receipts.json). No third-party source or OSS implementation is needed to substantiate the counts/product semantics.

## Caveats / Ambiguity Flags

The precise unresolved producer question is:

> For NOAA ACE RTSW202501_ace_swepam_1h.txt, why do all03/09/15UTC rows carry status9 and why is21UTC frequently status9? What exact accumulation interval does labelT represent, what sample-coverage threshold and minute-to-hour quality rule apply, and did2025 telemetry, processing or archive schedules omit those bins?

A producer specification, processing code or operational log is needed to distinguish those mechanisms. A separately scoped comparison with same-month minute records could test whether the gaps arise before or during hourly production; no minute-data download, outreach or new scope was undertaken here. Failure to find an explanation in this bounded search is not proof none exists.

Status1–8 numeric speeds are not automatically usable merely because positive; the requested quality rule remains intact. No calibrated truth replacement, revised filters, target imputation change or model improvement is inferred.

## Reusable Takeaway

The fixed-clock missingness is **upstream-product missingness confirmed for a frozen training month**, faithfully propagated by the local target parser. Its instrument/telemetry/aggregation cause and exact hourly support are not established. Preserve the existing policy and report those producer questions explicitly.

