# Research: WI_H1_SWE scalar fit_flag 11

## Request Type / Direct Answer

Bounded official-product/version audit, retrieved 2026-11-09 KST.
**No authoritative, version-matched definition of scalar `fit_flag=11` was
found.** Its proton-velocity meaning remains unresolved. This is bounded negative
retrieval evidence, not proof that the value is physically bad or that no
producer definition exists.

The frozen flag-10 primary screen, previous Wind report, native summary and
observational conclusions were preserved. No flag-11 records were newly accepted;
no additional scientific samples, model predictions or scores were read.

## Exact Product Evidence

The already downloaded `wi_h1_swe_20260118_v01.cdf` through
`wi_h1_swe_20260122_v01.cdf` identify version **01**, **Validated Summary Data**,
generation **2026-04-17 09:19:46**, and a `MODS` statement that the analysis package
was revised in **March 2012**. All five files' embedded `fit_flag` metadata has
`VALIDMAX=10` and describes codes **0–10 only**. The existing native summary
records **399 scalar 11s**: Jan 19 =72, Jan 20 =326, Jan 22 =1.

This is an actual data/metadata inconsistency. The creation date does not tell
us which newer fitting or export branch assigned 11, and the 2012 `MODS` entry
does not prove the code was unchanged since 2012. No inference about the speeds
or fit residuals of those rows was used to assign semantics.

## Official Docs Evidence and Bounded Retrieval

|Source|What was established|Limit|
|---|---|---|
|[Producer-written SPDF fit guide](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/documentation_jck.txt)|Kasper/Stevens document, created June 28, 2012; HTTP Last-Modified August 9, 2012; explicit 10→0 table.|No code 11 or 2026 revision contract.|
|[All-orbit fit guide](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/all_orbit_phases/documentation_jck.txt)|Separately checked official directory's guide has the same 2012-era 0–10 taxonomy.|No newer definition recovered.|
|[Official current directory](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/) and [all-orbit directory](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/all_orbit_phases/)|Listings include 2026 scientific files but point to the inspected generic documentation.|Presence of current data does not update the linked taxonomy; scientific files were not downloaded again.|
|[File-specific information](https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/file_specific_info.txt)|Contains historical processing/count summaries.|No 2026 flag dictionary or scalar-11 definition.|
|[Current HAPI metadata](https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H1_SWE)|Shorthand `10=He_OK, >1=P_OK`; description highlights 10 as SW parameters OK.|Generic inequality does not resolve an out-of-range, undefined value against exact CDF metadata.|
|[MIT producer-linked page](https://web.mit.edu/space/www/wind_data.html#Protons)|Legacy preliminary KP data page; Exa exposed a plot index ending in 2015.|Not an updated WI_H1_SWE nonlinear-fit flag contract.|
|Mission-linked `ftp://space.mit.edu/pub/plasma/wind/vho/wind_swe_fc_apbimax/`|Direct retrieval failed: **No route to host**.|Access failure, not evidence its contents lack an answer.|
|[Mission-linked CfA page](https://www.cfa.harvard.edu/shocks/wi_data/)|Direct HTTP **403**; Exa fetch also timed out.|No claim about inaccessible documentation contents.|

Exact query URLs, receipt times, Last-Modified values, response sizes and small
raw documents/listings are in `wind_crosscheck/flag11_docs/`. No response reached
the retrieval size cap. Exa searches located no explicit scalar-11 producer
meaning; downstream readers and unrelated electron-fit quality schemes were
not treated as definitions of this ion product.

## Historical Bit-Code Confusion

The [older SWE quality page](https://cdaweb.gsfc.nasa.gov/wind_swe_quality.html)
says explicitly that its flags are set in the **KP** analysis and are numerical
sums of bit values. Its **bit 11 adds 1024** and describes cup-2-only parabolic
fits. That is not scalar nonlinear `fit_flag=11`. Nor can 11 be interpreted as
“10 plus 1” in the categorical nonlinear-fit table. A recent HTTP Last-Modified
on that older page does not establish applicability to the current product.

Likewise, current HAPI's `>1=P_OK` shorthand is insufficient to certify 11 when
the same product's exact CDF valid range ends at 10. These sources conflict in
scope/completeness; resolving them by arithmetic or by choosing the permissive
interpretation would invent a quality definition.

## Source-Reference Evidence / Version Note

No producer flag-setting implementation with an identified 2026 version and an
explicit scalar-11 branch was located. No pinned implementation is therefore
claimed to establish semantics. A downstream reader that merely preserves the
integer or accepts `flag>1` would demonstrate that reader's choice, not the
producer's validity contract. The authoritative evidence presently available is
the exact version-01 metadata and producer-written 2012 taxonomy, which leave 11
undefined.

## Reusable Takeaway

The precise unresolved question is:

> Which analysis/export code version and condition assigned scalar `fit_flag=11`
> in the April 17, 2026 WI_H1_SWE version-01 files? Does that condition certify or
> qualify proton bulk velocity and its sigma separately from density,
> temperature and alpha parameters? Is `VALIDMAX=10` stale, or is 11 an invalid
> export value?

The smallest decisive next evidence is a version-matched producer dictionary,
release note, or the actual flag-assignment/export branch with its
parameter-specific validity contract. No message was sent to a person, and no
new acceptance policy was adopted. Until that evidence exists, retain the prior
flag-10 screen and its limited native-speed corroboration; neither promote nor
condemn flag-11 speeds based on this documentation gap.

Verification confirmed both retrieved fit guides enumerate exactly 0–10, all
five cached CDF metadata records set VALIDMAX 10, the prior count totals 399,
and all saved response byte counts match. Only this report, its JSON and the
new `flag11_docs/` evidence were written.
