# Data validity audit — 2026-11-08

Scope: source inspection and read-only numerical checks; no model fitting, source modifications, publication, or evaluation-policy changes. The initial request identified adopted A+B1; the leader subsequently identified the current best single run as `drop035_v650_confirm_20261108` seed42. That campaign reuses the same prepared input (`prepare.py:29`) and retains `img_age_h`; the data findings apply to both. Metrics/ranking are owned by the independent metric audit.

## Findings ordered by decision relevance

|Finding|Status|Evidence and implication|
|---|---|---|
|Historical feature availability is unverified|Confirmed gap, not proof all rows leak|`data-v2/README.md:46,64,73,81–89`: no availability delay processing, latest downloaded values, no version history; SUVI upload-time filter removed. Observation-before-origin alone cannot certify real-time availability.|
|Stale SUVI age is misrepresented after feature filling|Confirmed implementation defect|Raw missing `img_age_h` is filled like any scalar (`prepare.py:51–68`). 711 unique origins have a carried image but reported age <1h; source-age reconstruction reaches 83.69h. Of 6,315 official-period query origins, 188 have this condition, max actual age 66.76h.|
|Scoring truth is internally consistent with local raw quality masking|Confirmed by exhaustive probe|2,816,784 raw-df targets match named-month SWEPAM status0 finite-positive values, mismatch 0. All 7,296 frozen official observation rows match local raw, mismatch 0. This establishes internal consistency, not instrument accuracy.|
|Training cutoff/target alignment passes|Confirmed by exhaustive probe|All target timestamps equal origin+h; 144h maturity violations 0 for all F periods. Query y unchanged by preprocessing, observed y unchanged, non-target metadata identical.|
|Target missingness is substantial and strongly clock dependent|Confirmed|2026 Jan–Sep: 4,583/6,552 valid hours; UTC03/09/15/21 valid rates 34.43/20.15/22.71/23.08%. A missing-at-random interpretation is unsupported.|
|Training loss includes imputed truth while scoring excludes missing truth|Confirmed policy difference|763,854 train rows receive forward-filled y; official missing truth remains excluded. Causal fill is not future leakage but can flatten transients and change the loss distribution. Improvement from changing this policy is unknown and would require separate preregistration.|
|Official science-grade calibration validation absent|Unresolved|Current chain uses ACE real-time monthly SWEPAM, SUVI L1b radiance/DQF and DONKI. No independent L2/calibrated cross-instrument validation was found in this inspected chain.|

## Exact prediction and temporal contract

Target is ACE SWEPAM bulk speed in km/s at label time `T=origin+h`, h=1…72. It is an L1 spacecraft speed target, not an independently propagated Earth-arrival label. `leaderboard/scorer/competition_data.py:16–39` uses the named calendar month, `status==0`, finite speed >0; invalid truth is NaN. The model input has hourly query origins; the scorer separately determines the official evaluation grid.

The input notebook bins each observed source time as `floor(time,1h)+1h`. ACE label O−1h enters origin O; SUVI uses its observation end in [O−1h,O), ENLIL uses model completion time in that interval. Recurrence reads speed at target time minus 26/27/28 days from the binned ACE stream, which stays before origin for all 72 leads. This is a causal timestamp ordering, subject to publication-delay caveats.

|Period|Last training origin|Last training target|Cutoff|Query origins|Raw query missing-target rate|
|---|---|---|---|---:|---:|
|F1|2025-12-23 00Z|2025-12-26 00Z|2025-12-29 00Z|2,089|29.45%|
|F2|2026-03-23 00Z|2026-03-26 00Z|2026-03-29 00Z|2,113|31.08%|
|F3|2026-06-23 00Z|2026-06-26 00Z|2026-06-29 00Z|2,113|31.06%|

All training starts 2022-10-01 00Z. Origin +144h ≤ cutoff leaves targets (up to +72h) at least72h before cutoff. Later-period training may legitimately contain previous official periods under this fixed rolling protocol; it does not make F2/F3 statistically independent untouched tests.

`post_ablation_five_axes_20261108/baseline.py:25–40` loads y only for training and checks 144h maturity. `common.py:138–176` fits prior calibration strictly inside training, uses training-only offsets, and checks query truth is None. `baseline_policy.py:89–96` separately checks short-blend calibration maturity. These checks oppose direct query-y fitting in the inspected incumbent path; they do not certify every historical experiment.

## Source and quality chain

1. `data-v2/fetch.py` downloads ACE monthly TXT, GOES18 Fe195 L1b NetCDF, DONKI WSA-Enlil simulations. ACE rows and simulation IDs are insert-only. Existing records are not revised or versioned. An exception in `metric/metric.ipynb` may overwrite current SWEPAM monthly data (`README.md:89`); it was not executed here.
2. `data-v2/notebooks/raw_to_input_df.ipynb`, code cells4/6/8/10/12/14, parses ACE status fields and product-specific sentinels; builds ACE rolling summaries; reads SUVI radiance, masks DQF!=0, nonfinite/negative pixels and off-disk pixels; uses DONKI model completion timestamps. Its manifest explicitly says no availability delays.
3. `research/rerun_df_ffill_20261103/prepare.py:32–68,198` forwards each feature independently over sorted origin within each lead. Duplicate train/query values must agree. Leading NaNs remain; no maximum age is imposed. This can also preserve stale lead-dependent strip values without reprojecting them to the later origin; that is a distinct physical-feature question, not included in the confirmed age-only defect.
4. `fill_train_y.py:22–39` forwards labels by physical target_time, changes only train y, and guards observed/query labels. The timeline includes both sets, but all admissible source timestamps remain before the relevant training cutoff. No evidence of future truth being pulled backward was found.
5. The official observation builder uses the same real-time SWEPAM product and quality filter. Status0 does not establish absence of instrument bias. `analysis/A0003/README.md:36` already explicitly limits earlier work: final calibrated L2 accuracy and historical publication availability were not validated.

`research/ace_1m_to_hourly_20261108/strategy.md` records an existing operational study: monthly products update at :10 and observed delays made the latest hour unavailable; one-minute SWEPAM/MAG and five-minute EPAM/SIS delays were empirically bounded. Its estimates are existing study evidence, not a fresh external validation in this audit. Its September exploratory observations also require exposure disclosure.

## Most decisive next controlled test

Pre-register a single-feature **replacement/bugfix ablation**: compute `img_age_h = origin − last_actual_SUVI_obs_end` from the raw missingness timeline, preserving every other input and all fitted settings. Compare A with A+repair over matched seeds and official periods. Do not change strip projection, fill limit or quality thresholds simultaneously. A frozen-model inference-only replay is the cheapest mechanism probe, but ages beyond 1h are outside the original age support; a negative replay cannot rule out benefit after consistently retraining on corrected age. Any complete retraining must retain seed-matched A and preserve model artifacts.

Before any deployment-valid claim, replay features against actual receipt/publication timestamps. For ACE monthly inputs at exact-hour origins, a fixed one-extra-hour withholding sensitivity is a valid preregistered availability stress test, not an asserted reconstruction of the unknown historical publication schedule. DONKI revision history and SUVI upload timestamps require independent archived as-of records. Science-grade ACE SWEPAM/MAG or a properly time-aligned independent spacecraft reference could assess observation error, but must remain a separate diagnostic and must not silently replace official scoring truth.

## Earlier failures and interpretation limits

`analysis/A0003/README.md` documents nonstationary level/variance and variable 27-day correlation, together with quality-policy sensitivity. This supports time-stratified analysis, not a particular model choice or a 50% attainable bound. `research/e2000fs_calib11g_rawdf/strategy.md` records historical old/new input incompatibilities (status filtering, one-hour ACE shift, SUVI recomputation, origin spacing) and a failed port improvement. Its earlier tuning protocol is historical, not adopted here. `analysis/A0015_loc_information/README.md` reports location-based residual correction worsening all leads; location alone is not demonstrated missing predictive information.

No present evidence establishes what fraction of error is physically irreducible, timing error versus amplitude error, or whether correcting data defects can halve official S. Those need the independent residual analysis and controlled tests.

## Validation and artifacts

`data_probes.json` records exhaustive alignment, cutoff, raw/official equality and stale-age results plus commit `d64c40339d770803cf46415bd383971dd30f00cd` and input byte/mtime identities. `data_quality.json` records status counts and per-hour valid fractions. Both probes executed with exit0; they used pandas/numpy and the official observation parser, without model fitting. The precise reconstruction is: de-duplicate h1 raw rows by origin; derive source time `origin−raw_img_age_h`; forward-fill source timestamp; subtract it from current origin; compare against the prepared filled age. Query rate uses only non-null F1/F2/F3 period rows.

The requested file-digest command was rejected by the automatic PreToolUse hook citing the older 2026-11-02 no-hash instruction. It did not execute. The remaining probes succeeded without digests and preserve byte/time identities. Hash reproducibility is therefore a specific unresolved artifact gap, not silently claimed complete.
