# Research: independent Wind/SWE crosscheck, January 18–22, 2026

## Request Type / Direct Answer

Bounded independent-instrument observational-validity audit, retrieved
2026-11-09. **Wind provides limited positive corroboration of very high solar-wind
speeds:** three native proton-speed fits pass the predeclared conservative,
documented quality screen and report 1,000.6–1,308.9 km/s. This is independent
Wind/SWE evidence, not mixed-spacecraft OMNI. It supports the high-speed-event
interpretation but **does not verify a sustained hourly ~1,000 km/s plateau,
ACE's exact three hourly values, or absence of shared energetic-particle bias**.

The precise product label is **Validated Summary Data**, with the producer
calling this the primary science/definitive nonlinear analysis. No unsupported
“Level 2 calibrated truth” label is applied. There are **zero defensible paired
hourly ACE–Wind calibration comparisons**, because ACE's exact averaging support
remains unresolved. No ACE data, new experimental predictions, or scores were
opened in this audit; official observations and the scorer remain unchanged.

## Frozen Protocol and Quality Gate

Before product/data retrieval, `wind_crosscheck/protocol.json` fixed
**2026-01-18 00:00 inclusive through 2026-01-23 00:00 exclusive**. This window
comes from the already exposed event diagnosis and is not blind. Before
scientific-value extraction, `quality_gate.json` fixed the main screen:

- `fit_flag == 10`, officially “Solar wind parameters OK — no action necessary.”
- Finite, nonfill, positive proton speed and finite GSE velocity components within
  their actual daily CDF variable validity ranges; finite, nonnegative speed sigma.
- Separate descriptive results for documented proton-valid flags 2–10; never
  select the screen by agreement with ACE. Unknown flags are excluded.
- Native full-window/daily UTC summaries; no gap filling, fitted lag, selected
  temporal alignment, propagation coefficient or calibrated offset.

Protocol/quality-file timestamps preceding extraction were verified. The
optional vector-norm conversion was fixed and only used as a parsing consistency
check; primary reported speeds use the official scalar proton-speed field.

## Official Docs and Original Instrument Evidence

- [NASA Wind data sources](https://wind.nasa.gov/data_sources.php) identifies
  **WI_H1_SWE** as “Bi-Maxwellian, Anisotropic Analysis of Protons and Alphas
  [92 sec] — PRIMARY SCIENCE DATA,” derived by nonlinear fits to ion peaks.
- [SPDF product README](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/00readme.txt)
  calls the corresponding nonlinear anisotropic parameters **definitive**, using
  processing introduced in 2012, and distinguishes this product from superseded
  SWE processing. The [producer format/quality documentation](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/documentation_jck.txt)
  defines the fit flag, start-of-spectrum time, peak-charge-flux time and one-sigma
  fitting/propagated uncertainties. Its historical yearly record counts are not
  counts for this 2026 event.
- [CDAWeb HAPI metadata](https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H1_SWE)
  advertises coverage through **2026-10-09 23:59:16 UTC**, scalar proton speed and
  GSE components in km/s, fit flags, uncertainties and Wind positions. Its
  `fit_flag` unit shorthand is `10=He_OK, >1=P_OK`; the detailed documented table
  stops at 10, so this shorthand is not used to accept an unknown value of 11.
- [Original Kasper et al. (2006) paper](https://doi.org/10.1029/2005JA011442),
  *Physics-based tests to identify the accuracy of solar wind ion measurements:
  A case study with the Wind Faraday Cups*
  ([NASA record](https://ntrs.nasa.gov/citations/20070010017)), develops
  physics-based accuracy checks using ion differential flow and other plasma
  constraints. Historical accuracy estimates from its sample are **not** an
  event-specific uncertainty bound for January 2026 or evidence that SPE effects
  are absent.

An older CDAWeb SWE quality page discusses a bit-coded `iqual_core` and earlier
KP/nonlinear products. It is not the current WI_H1_SWE `fit_flag` definition;
those bit masks and historical recommendations were not transferred here.

## Actual Product and Native Results

The [official daily archive](https://cdaweb.gsfc.nasa.gov/pub/data/wind/swe/swe_h1/2026/)
provided `wi_h1_swe_20260118_v01.cdf` through
`wi_h1_swe_20260122_v01.cdf`, totaling **3,214,610 bytes**. All report version
**01**, `Source_name=Wind`, `Data_type=VS>Validated Summary Data`, and generation
**2026-04-17 09:19:46**. Retrieval receipts preserve URLs, UTC fetch time, server
Last-Modified, byte count and local file mtime. These are retrospective files,
not proof of what was available to a forecaster in January.

|UTC date|Native records|Accepted flag-10 records|Native speed min / median / max, km/s|Accepted ≥1,000 km/s|
|---|---:|---:|---|---:|
|Jan 18|521|111|534.2 / 576.1 / 722.6|0|
|Jan 19|649|264|410.6 / 430.6 / 1,308.9|2|
|Jan 20|606|222|772.6 / 843.2 / 1,000.6|1|
|Jan 21|691|357|575.3 / 729.8 / 809.1|0|
|Jan 22|628|256|480.3 / 543.5 / 610.6|0|
|Full fixed window|3,095|1,210|410.6 / 587.9 / 1,308.9|3|

These are sample distributions, **not time-weighted averages or duration
coverage**. Five accepted records exceed 900 km/s; the three ≥1,000 records are:

|Wind spectrum-start UTC|Proton speed, km/s|Reported one-sigma speed uncertainty, km/s|Fit flag|
|---|---:|---:|---:|
|Jan 19 20:37:41.944|1,308.855|69.029|10|
|Jan 19 20:42:40.812|1,217.768|51.619|10|
|Jan 20 06:58:17.701|1,000.639|47.539|10|

Reported uncertainties arise from fitting/propagation, not a complete systematic
error budget. The two highest samples precede the previously documented NOAA
21:15 ACE/DSCOVR recovery statement; that statement is not applied as a Wind
quality filter or translated between spacecraft clocks/locations. The third
value's uncertainty also makes a sharp physical threshold at exactly 1,000
inappropriate. The evidence is about magnitude, not a precisely timed threshold.

## Quality, Time and Position Limits

**Unknown flag 11:** 399 records have `fit_flag=11` (72 on Jan 19, 326 on Jan 20,
one on Jan 22). This exceeds the CDF `VALIDMAX=10` and is absent from the inspected
fit-flag table. They were excluded from both screens. This is a substantial
quality-documentation gap during the disturbance. No interpolation across those
records, assertion of sustained speed, or inference that excluded values are
wrong is justified. The separate predeclared 2–10 screen accepts 2,567 records;
its less restrictive output is preserved in JSON, not used to replace the main
screen after seeing values.

**Actual fill values:** the daily CDF speed/components use approximately
`−1e31` fill, whereas the current HAPI metadata reports `99999.9` for those
fields. The conversion reads fill/validity attributes from each downloaded CDF;
it does not apply the HAPI fill convention to a different representation.
The strict screen excludes five flag-10 rows with invalid numerical fields.

**Timing:** CDF Epoch explicitly marks the **beginning of each SWE spectrum**.
Peak charge-flux time is a separate field. Global resolution is “92 seconds,”
but actual returned timestamp gaps have median **100.671 s** and maximum
**2,409.818 s**. This sampled, quality-filtered product is not a continuous series
of identical 92-second averages. Its detailed within-spectrum sampling kernel
and the ACE hourly interval support are not jointly established. Therefore no
same-hour calibration difference, hourly resampling or optimized lag was computed.

**Position:** accepted Wind rows have GSE coordinates spanning
X **229.99–236.61 Re**, Y **−94.75 to −87.76 Re**, Z **6.11–7.80 Re**.
These are actual Wind ephemerides in the product. Simultaneous ACE positions
were not obtained, so inter-spacecraft separation and convection delay are not
quantified. Different flux tubes and spatial structure remain possible; the
same UTC label need not describe identical plasma. SPE can also affect both
instruments, so independence is useful corroboration rather than truth status.

## Reusable Takeaway / Verification

The previous ACE/DSCOVR checks lacked usable independent numerical corroboration.
This check adds **three documented-quality Wind native fits at very high speed**,
while showing that most of the relevant interval cannot support a strong
continuous-speed claim under the inspected quality documentation. It favors
keeping a real high-speed disturbance in the diagnosis; it does not license
replacing official targets, declaring ACE exactly accurate, or attributing all
large errors to instrument contamination. Further event-specific accuracy claims
would require resolving flag 11 and obtaining defensible shared support/geometry,
not fitting an alignment to model residuals.

`wind_crosscheck/` preserves the frozen protocol, quality gate, five CDFs, official
metadata/docs, receipts, native CSV and executable retrieval/conversion scripts.
The existing environment supplied cdflib **1.3.16** and NumPy **2.4.6**; no package
was installed. Fresh checks verified byte counts, 3,095 unique in-window epochs,
quality-screen counts, three high-speed rows and scalar-speed/vector-norm agreement
within **0.000067 km/s**. That last check validates extraction, not instrument
accuracy. The complete machine-readable audit is `wind_independent_crosscheck.json`.
Historical issue-time feature availability and model-goal attainability remain
unresolved by this retrospective observational check.
