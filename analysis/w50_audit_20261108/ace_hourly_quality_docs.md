## Research: ACE SWEPAM hourly status reduction

### Request Type

Bounded implementation-reference and producer-documentation lookup. Read the existing mechanism report, January aggregation report, and its independent verification before searching. Retrieval completed 2026-11-08 21:54:05 UTC (2026-11-09 KST).

### Direct Answer

**No exact hourly S-reduction algorithm or January 2025 producer implementation/version was found in this bounded search.** Neither “take the last/end minute,” “reject if any minute is bad,” nor another exact reduction rule is established by the reviewed official sources. The hourly accumulation interval, valid-sample threshold, sample weighting, and relationship to the published rounded minute values also remain undocumented in those sources.

The supplied counterexample is specific: January 1, 2025 at 03:00 has hourly S=9/sentinels; minutes 03:00–03:58 are nominal and 03:59 has S=9. Both an end-minute rule and an any-bad-minute rule could produce that outcome, among other mechanisms. It does not select an algorithm. The independently verified 744-hour correspondence study also did not establish simple averaging of the accepted published minute values at display precision. No further numeric scan was performed here.

### Official Docs Evidence

- [NOAA SWPC ACE Real-Time Solar Wind](https://www.spaceweather.gov/products/ace-real-time-solar-wind) distinguishes **1-minute/5-minute averaged real-time data**, linked to the ACE/daily service, from **1-hour averaged real-time data**, linked to ACE2/monthly. It separately links ACE Science Center science products. Thus the named hourly product and the “1-minute averaged Real-time Bulk Parameters” product are distinct published outputs. The page does **not** state that the former is computed by applying an arithmetic mean and a particular S reduction to the latter's text rows.
- The same page says NOAA funded instrument teams to provide real-time processing algorithms. Its **June 1, 2006** announcement states SWEPAM processing was updated at 0000 UTC to improve speed, density and temperature agreement with ACE Science Center data. This supplies a dated processing-change fact, **not** an hourly algorithm, software release identifier, or evidence of the code deployed in January 2025.
- Its **July 30, 1998** announcement says SEC creates monthly files of hourly averages, recreates the current month every three hours beginning 0010 UT, and creates the previous month for the last time on the first day of the next month. This documents a historical **file-generation schedule**, not averaging support, coverage requirements or status propagation. It cannot be promoted into a 2025 operational guarantee.
- [ACE Science Center RTSW Format](https://izw1.caltech.edu/ACE/ASC/SC_Telem/rtsw.html), explicitly **Version 2/2/96**, describes a 434-bit/s telemetry layout with a 16-second major frame and SWEPAM ion data created specifically for RTSW. The reviewed layout contains no ground-produced hourly average or hourly S rule. Its packet/frame timing cannot determine the hourly product's timestamp support.
- [NCEI SWPC Products and Data](https://www.ncei.noaa.gov/products/space-weather/partners/swpc-products-and-data) says NCEI archives some SWPC products. The retrieved excerpt supplied no ACE hourly producer specification. Extraction was capped at 16,000 characters; this is not an exhaustive statement about that page or NCEI holdings.

### Version Note

The investigated numerical context remains **January 2025**. Documentation was accessed during **2026-11-08 21:52–21:53 UTC**. The explicit documentation versions above are **1996**, **1998**, and **2006**, not a frozen 2025 producer build. A current web page containing an old announcement does not make the announcement a current processing specification. No release number, repository, full SHA, or deployed build for the hourly generator was located.

### Supporting Examples

The January 1 example is supplied, independently verified local evidence from [ace_probe-verification.md](ace_hourly_aggregation_probe/ace_probe-verification.md). It is illustrative evidence of minute support beside an invalid hourly record, not official documentation of the reduction rule.

### Source-Reference Evidence

No upstream hourly producer implementation was located. Search results included pysatSpaceWeather loaders/cleaning helpers; those are downstream consumers of published status flags and were not accepted as evidence of NOAA generation logic. No moving-branch code is cited as a producer specification.

### OSS Reference Implementations

None used: no located OSS implementation was established to be the NOAA hourly producer.

### Supplemental Evidence

None used. Incidental numeric text-product excerpts returned by search were not analyzed; numeric rows are omitted from saved search receipts.

### Caveats / Ambiguity Flags

| Question | Finding |
|---|---|
| Last/end-minute S rule? | Not established. |
| Any-bad-minute / worst-status rule? | Not established. |
| Exact alternative S rule? | Not located. |
| Hour label denotes start/end/center? | Not established for this NOAA RTSW stream. |
| Minimum coverage, weighting, source sample precision? | Not located. |
| Producer code and January 2025 version? | Not located. |
| Cause of fixed six-hour pattern? | Unresolved; no new cause claim. |

The negative finding is limited to **three focused Exa searches and four successful document fetches across three URLs** (the NOAA page was also fetched directly to inspect links). No explicit numeric product download, fit, scorer call, F-period truth access, outreach, or policy change occurred. General calibration/search-track/telemetry explanations were not reopened. The first attempted local verification path was absent; the actual nested verification report was then located and read.

### Reusable Takeaway

Keep **hourly producer status semantics unresolved**. The verified minute/hour mismatch narrows the evidence but does not authorize calling the hourly flag “last minute,” “any bad minute,” or “all minutes absent.” A producer specification or versioned implementation tied to the January 2025 product is still needed. Existing target/feature treatment remains unchanged.

Machine-readable findings: [ace_hourly_quality_docs.json](ace_hourly_quality_docs.json). Search/document receipts: [ace_hourly_quality_docs/](ace_hourly_quality_docs/).

