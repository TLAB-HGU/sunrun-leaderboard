# Research: DSCOVR plasma crosscheck for January 19–20, 2026

## Request Type

Bounded independent-instrument observation check. Retrieved November 8, 2026. No model fitting, target alignment fitting, raw/scorer changes, installations or hashes. Evidence writes are confined to this report and `dscovr_crosscheck/`.

## Direct Answer

**The retrieved DSCOVR archive does not supply trustworthy corroboration or refutation of ACE's sustained post-21:15 high speeds.** The two public files are explicitly operational real-time-processing Level2 products, not demonstrated retrospective science recalibration. Their finite speeds are roughly 287–600 km/s during the examined interval, but **706 of707 finite positive speed records carry `reduced_proton_quality_flag=1`**, defined by the file as “resulting science data subject to high uncertainties.” The remaining finite record also fails the conservative quality screen. There are **zero clear-quality comparison records**.

This is a material discrepancy with the NOAA bulletin's generalized statement that “ACE and DSCOVR data became reliable once again” at21:15. That statement is **not numerically reproduced by this specific archived DSCOVR f1m stream**. We cannot resolve whether the bulletin referenced another processed feed, an operational interpretation, or an overgeneralized recovery statement. The discrepancy must remain explicit; it is not evidence that the ACE peaks are erroneous.

## Official Docs Evidence

- [NOAA NCEI DSCOVR product page](https://www.ncei.noaa.gov/products/deep-space-climate-observatory-dscovr) calls the downloadable products **operational data products**, identifies f1m as Faraday Cup one-minute averages and fc1 as Level1 scientific units, and describes DSCOVR's L1 location approximately1.5million km from Earth. These instrument/level labels do not certify correction of this event's contamination.
- The old [NGDC January2026 directory](https://www.ngdc.noaa.gov/dscovr/data/2026/01/) redirects to [NOAA's new archive](https://archive.data.noaa.gov/satellite-spaceweather#DSCOVR/). Its public directory API resolves the product path `DSCOVR/DSCOVR/FC/f1m/2026/01/`; listing snapshots preserve this discovery chain.
- [January19 f1m file](https://archive.data.noaa.gov/satellite-spaceweather/DSCOVR/DSCOVR/FC/f1m/2026/01/oe_f1m_dscovr_s20260119000000_e20260119235959_p20260120020420_pub.nc.gz): **45,193bytes compressed**; created2026-01-20T02:04:20.439Z.
- [January20 f1m file](https://archive.data.noaa.gov/satellite-spaceweather/DSCOVR/DSCOVR/FC/f1m/2026/01/oe_f1m_dscovr_s20260120000000_e20260120235959_p20260121020337_pub.nc.gz): **40,711bytes compressed**; created2026-01-21T02:03:37.137Z. Both downloads and uncompressed payloads are below10MB.
- Embedded metadata in both files: title **DSCOVR Faraday Cup Level2 One Minute Averages**; source **DSCOVR Faraday Cup Level1**; history **DSCOVR real-time telemetry processing system**; processing description **1-minute average of L1 cycle peaks using Huber M-estimator**; algorithmVersion **1.5**, algorithmDate **2016-07-26**. Referenced documents are the Faraday Cup Level1B ATBD and Time Series Data Averages ATBD v2.4. Full embedded metadata is saved alongside each file.
- Embedded `overall_quality` means0 normal,1 suspect,2 error. It is insufficient alone: **679 of707 finite records are overall_quality0**, while706 carry the specific reduced-proton-quality flag. Embedded flag descriptions, not inferred heuristics, establish the uncertainty warning. Missing speed is−99999; units arekm/s; advertised valid range is189–1111km/s. A declared range is not proof of correctness within it or of calibration at the extremes.
- [NOAA January20 discussion](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/discussion/2026/01/202601201230forecast_discussion.txt), preserved separately in `event_noaa_snapshot.md`, documents a shared intense-proton contamination concern for ACE and DSCOVR. Independent instruments can share the same environmental failure mechanism; neither independence nor an operational quality label makes this a calibrated reference.

## Fixed-Window Results

Windows use the pre-existing bulletin's recovery time and03:15 transition, ending12:00UTC to bound the subsequent interval. They are label-time diagnostic summaries, not propagated comparisons or official hourly aggregates. Values below are **descriptive, unvalidated instrument outputs**, not assertions of the true wind speed.

| DSCOVR UTC labels, start inclusive/end exclusive | Labels | Finite speed | Reduced-quality finite | Descriptive min/median/max, km/s | Clear-quality records |
|---|---:|---:|---:|---|---:|
| Jan19 21:15–Jan20 00:00 |165|161|160|287.4 /501.5 /568.6|0|
| Jan20 00:00–03:15 |195|195|195|466.0 /477.9 /505.8|0|
| Jan20 03:15–12:00 |525|351|351|465.4 /499.9 /599.5|0|

No finite output exceeds900km/s in these windows. The conservative diagnostic screen requires finite positive speed, overall_quality0, and zero fill, reduced-proton-quality, bad-speed, old-packet and low-velocity-sample-count flags. It is explicitly an audit screen, **not a new official acceptance rule**. The absence of clear rows is already driven by the instrument's own quality warnings; no filter was optimized against ACE or model error.

## Version, Timing and Position Note

- Filename processing timestamps and full metadata pin the actual operational products retrieved. No independently calibrated/reprocessed January2026 release was established. “Level2” here describes averaging of operational Level1 cycle peaks; it is not equivalent to the hand-validated ACE science L2 product investigated earlier.
- `time` is described as “date and time for each observation,” in milliseconds since1970-01-01T00:00:00Z, Gregorian. Each file has1440 consecutive one-minute labels from00:00 through23:59. Metadata states one-minute Huber averaging but does **not** identify interval start/end/center at the individual-record level. No assertion of exact one-minute integration boundaries is made.
- The f1m products do not provide the spacecraft ephemerides needed for ACE–DSCOVR physical alignment. Only the generic L1 location is established. **Event-specific positions and separation are unmeasured in this check; no propagation shift was applied.** Given zero clear-quality rows, fetching orbit files would not repair the primary evidence gap and was not pursued.
- The ACE RTSW hourly support remains unknown, as documented in `calibration_crosscheck.md`. No exact same-clock row differences, calibration offsets or relative arrival errors were calculated. No OMNI values or bow-shock timestamps were used, and no target-fitted time alignment was attempted.

## Caveats / Ambiguity Flags

The archive's low uncertain outputs conflict with the bulletin's recovery narrative but cannot discriminate instrument contamination, peak-identification failure, processing-feed differences or physical spacecraft differences. Peak-unresolved/unbounded flags occur in subsets and reduced-proton-quality is pervasive; assigning a unique cause would exceed the evidence. These files are not a scientifically valid substitute for official ACE truth. An event-specific validated reprocessing or independent reliable observation remains necessary.

## Reusable Takeaway

**The apparent “independent DSCOVR check” fails its own quality gate.** Preserve the genuine-extreme-event evidence from NOAA while narrowing its recovery claim: the retrieved DSCOVR f1m archive does not validate it. Do not explain ACE's large residual by subtracting these suspect speeds, and do not change targets, weights or scorer policy. Stop the bounded search with this precise quality/provenance gap.

## Artifacts and Verification

`dscovr_crosscheck/` contains the two compressed event files, embedded metadata snapshots, official archive listings, `retrieval_manifest.json`, `event_extract.csv`, `findings.json` and the reproducible `summarize.py` diagnostic. Executed successfully using existing `.venv` NumPy/SciPy, with no installs. Fresh checks verify2880 consecutive minute labels, km/s units, original response byte counts and speed consistency with the recorded GSE velocity norm (maximum absolute difference0.098km/s, consistent with displayed component precision). That consistency checks parsing, not instrument accuracy. No raw/scorer/model files were modified.
