# Research: ACE SWEPAM Level 2 crosscheck for January 19–20, 2026

## Request Type

Bounded scientific data validation, retrieved November 8, 2026, 13:38–13:40 UTC. Read-only with respect to research data, models and scorer; writes confined to this report and `calibration_crosscheck/` evidence snapshots. No fitting, training, installs, hashes or scoring-policy changes.

## Direct Answer

**The three high-impact hourly RTSW values are verified in the public monthly product, but independent science-Level-2 corroboration is unavailable for this event in the checked official archives.** The instrument-team listing ends in 2024; NASA CDAWeb independently advertises coverage ending July 9, 2024 and returns no data for January 19–20, 2026. Thus this check cannot classify these specific rows as calibrated, erroneous or averaging artifacts.

| ACE RTSW UTC label | Bulk speed, km/s | Status | Science L2 comparison |
|---|---:|---:|---|
| 2026-01-19 22:00 | 1080.3 | 0 | Unavailable |
| 2026-01-19 23:00 | 1044.3 | 0 | Unavailable |
| 2026-01-20 00:00 | 1035.6 | 0 | Unavailable |

The NOAA discussion remains qualitative support for a real extreme wind following reported recovery at 21:15 UTC. It is not an independent, calibrated hourly series. These **labels** follow the reported recovery; the inspected RTSW documentation does not establish the precise averaging support of each label, so “every contributing observation is after recovery” is not proved. The evidence does not justify attributing the dominant residual to contamination, rejecting these rows, or replacing official scorer truth.

## Official Docs Evidence

- [ASC SWEPAM data selector](https://izw1.caltech.edu/ACE/ASC/level2/lvl2DATA_SWEPAM.html) lists latest 64-second yearly file `swepam_data_64sec_year2024.hdf` and latest hourly Bartels file `swepam_data_1hr_2603.hdf`, explicitly labeled **166/2024–192/2024** (June 14–July 10, 2024). Its full-range option ends earlier at day 165/2024. The generated November 8, 2026 page date is not observation coverage or a January 2026 data release.
- [ASC instrument-team directory](https://izw1.caltech.edu/ACE/ASC/DATA/level2/swepam/) confirms these are the latest listed yearly/hourly files. The latest relevant listing entries are dated November 11, 2025; that is a file modification date, not a 2025 observation interval. No January 2026 HDF was downloaded because none was listed.
- [NASA CDAWeb AC_H0_SWE metadata](https://cdaweb.gsfc.nasa.gov/hapi/info?id=AC_H0_SWE) identifies `Vp` as Solar Wind Bulk Speed, km/s, fill `-1.0E31`, and advertises `stopDate=2024-07-09T23:59:13Z`. The [CDF archive](https://cdaweb.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0/) lists year directories only through 2024; its 2026 child URL returned HTTP404. A narrowly scoped [January 19–21 HAPI request](https://cdaweb.gsfc.nasa.gov/hapi/data?id=AC_H0_SWE&parameters=Vp&time.min=2026-01-19T00:00:00Z&time.max=2026-01-21T00:00:00Z&format=json) returned the text `1201`, `OK - no data for time range`. The HTTP200 body is malformed JSON, so it was preserved as raw response text and not treated as a successful scientific-data result.
- [SWEPAM L2 description](https://izw1.caltech.edu/ACE/ASC/level2/swepam_l2desc.html), last updated October 2025, documents 64-second ion and hourly/daily/Bartels averages, proton speed in km/s, and `-9999.9` as bad/missing ASC values. [Release notes](https://izw1.caltech.edu/ACE/ASC/DATA/level2/swepam/swepam_release_notes) describe detector-gain reprocessing and penetrating-radiation effects. These establish why science reprocessing matters; historical quality statistics do not quantify January 2026 accuracy.
- [ASC timing definition](https://izw1.caltech.edu/ACE/ASC/level2/timing_data.html), updated October 10, 2024, explicitly defines ASC L2 timestamps as **start of integration period in UTC** and documents an approximately five-second spacecraft-to-Earth clock-calibration offset for products other than MAG one-second data. This is an ASC format statement; do not silently transfer it to CDAWeb epochs or NOAA RTSW hourly labels without product-specific metadata.
- [January 2026 RTSW monthly product](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/202601_ace_swepam_1h.txt), issued **2026-02-01 05:10 UT**, contains the three exact nominal-status rows above. Its header calls them “Hourly Averaged Real-time Bulk Parameters.” [RTSW README](https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/README), dated December 27, 2010, defines 0 nominal, 1–8 bad and 9 no data; it describes preliminary, unrevised operational lists and hourly updates at :10. Neither inspected source specifies hourly interval start/end/center, weighting, minimum sample count or minute-level quality aggregation. This is the precise aggregation evidence gap.
- [NOAA January 20 discussion](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/discussion/2026/01/202601201230forecast_discussion.txt), already preserved in `event_noaa_snapshot.md`, reports compromised ACE/DSCOVR speed followed by recovery at January 19 21:15 and about 1135 km/s, decreasing to 900–950 km/s by January 20 03:15. It does not provide the independent versioned hourly values needed here.

## Version Note

No applicable January 2026 science L2 filename, revision or quality flags were obtainable. This is a documented archive-availability limitation, not evidence that such data cannot exist elsewhere or be produced later. `retrieval_manifest.json` preserves each requested/final URL, retrieval UTC, status, response byte count and server Last-Modified where available. The RTSW filename has no explicit revision identifier; its issue header is retained. No older science file was substituted for event coverage.

## Caveats / Ambiguity Flags

1. Reprocessed ACE is a different processing product from the same spacecraft/instrument, not an independent instrument measurement. Even a future exact match would have that limitation.
2. [NASA high-resolution OMNI documentation](https://omniweb.gsfc.nasa.gov/html/HROdocum.html) defines bow-shock arrival-time tags rather than ACE observation-time tags. No OMNI values were downloaded or timestamp-subtracted in this check. A future comparison must retain spacecraft identity, shift and averaging metadata.
3. Status0 is a nominal operational flag, not an uncertainty bound. The NOAA recovery statement supports the event's physical plausibility but does not certify each minute, hourly aggregate or numerical value.
4. No numerical calibration discrepancy, noise floor, contamination fraction or attainable score improvement can be estimated from zero matched L2 samples.

## Reusable Takeaway

Keep the scorer unchanged. Mark these three rows **RTSW-source verified; calibrated corroboration unresolved because public ACE science L2 coverage stops in 2024**. The next decisive evidence is an actually released, versioned January 2026 science product plus documented hourly support, or independently calibrated spacecraft data with explicit physical-time alignment. Do not expand this absence into a claim of target error.

## Artifacts and Verification

`calibration_crosscheck/` contains official metadata/doc snapshots, the monthly RTSW file, raw no-data response, `retrieval_manifest.json`, 48 extracted January 19–20 rows in `rtsw_event_rows.json`, and machine-readable `findings.json`. Fresh Python standard-library checks passed: all three specified speeds/statuses match exactly, CDAWeb coverage/no-data evidence agrees, and every saved successful response has the recorded byte count. Numeric L2 pairs: **0**. No tests involving model fitting or scorer mutation were run.
