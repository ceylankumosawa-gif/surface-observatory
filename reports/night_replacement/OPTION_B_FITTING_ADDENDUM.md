# Option B — numerical fitting and acceptance addendum

Addendum ID: **option-b-2026-09-09-fit-v1**. Date: **2026-09-09**. Parent protocol: [option-b-2026-09-09-v1](OPTION_B_PROTOCOL.md). Status: **specified before new fitting; no fitting or deployment performed by this review**. This addendum fixes the choices below before new candidate errors or reserved thermal values are viewed. Its numerical thresholds are proposed project acceptance criteria, not published guarantees of sensor accuracy or sample-size sufficiency.

These are research-design choices, not requirements imposed on the user or a new user-approval step. Authorized acquisition, engineering tests and eligible research fitting can continue. The gates determine which performance claims and deployment scopes the evidence supports. An explicit user-directed change can be recorded in a new protocol version; preserving the blind-test record still requires identifying any temperatures already inspected.

This review read source code, existing model metadata, public product documentation and metadata inventory summaries. It did not access credentials, new protected thermal labels or 2025 temperature values. A run must record this file's SHA-256 and the code/source-registry hashes before fitting. Later changes require a new addendum ID and an explanation; neither a failed result nor sparse coverage licenses moving test data into training.

## 1. What can be learned in this stage

Fit one shared histogram gradient-boosting model for `lst_c − air_temperature_c`, using categorical Köppen class, physical surface descriptors, weather and time. Both observed day and night examples enter this same model. Do not train a separate model for every climate or add pilot identity, coordinates, sensor identity, scene identity or a thermal observation as a predictor. London and Sioux Falls test Cfb/continental conditions; preserving older multi-pilot training rows does not turn their new nighttime evidence into global climate validation.

The useful first fit is roughly **50,000–150,000 quality-screened new rows from at least 128 pilot/day-or-night acquisition dates**, plus the eligible retained fitting rows, under the existing **200,000-row / four-CPU-thread cap**. This is a bounded planning range, not a measured runtime or an accuracy promise. The current model has only 6,606 fitted rows and 40 features. Thousands of extra cells on one night would increase its row count without supplying the missing seasonal, heating/cooling and extreme-weather information. No GPU is required for the specified estimator; acquisition and quality control are likely to dominate the work.

This stage targets **observed clear-sky daytime and nighttime surface temperature**, not through-cloud 100 m truth. Arbitrary supplied-air scenarios remain extrapolations; changing air temperature while keeping observed humidity, radiation and history fixed is not an independently validated weather simulation.

## 2. Freeze labels and source quality before fitting

**Primary teacher:** `ECO_L2T_LSTE.002`, its pinned collection ID from the parent protocol, one processing revision per orbit/scene. V003 and ASTER V004 are separate instrument/product checks, not interchangeable replacements introduced after seeing results. An ECOSTRESS orbit appearing in multiple tiles or processing versions contributes only one acquisition group. Keep instrument and revision in audit fields and report errors by instrument, without using them as predictors.

The following **research choices** define the primary ECOSTRESS label cohort:

1. Verify product/version, actual raster CRS/bounds/transform, acquisition time, fill values and scale from the matching tiled-product documentation and raster metadata. The swath and tiled representations need separate unit contracts; never apply a swath integer scale to an already scaled COG. Preserve raw QC and units in the source manifest.
2. Require dedicated cloud-mask clear status, `(QC & 3) == 0`, `((QC >> 2) & 3) == 0`, `((QC >> 14) & 3) >= 1`, finite positive uncertainty and **LST error estimate ≤2 K** at each contributing native pixel. Reject fill, saturation, failed retrieval and unsupported QC codes. Do not interpret V001 cloud bits in V002. The V002 guide explains that mandatory QC `00` can still be cloudy and that `LST_Err` is a modeled total-uncertainty estimate; it is not a guaranteed Gaussian standard deviation. Consequently, do not use it as an inverse-variance sample weight. [ECOSTRESS L2 V002 user guide, sections 2.2–2.4](https://lpdaac.usgs.gov/documents/1574/ECOL2_User_Guide_V2.pdf)
3. Dilate cloud, uncertain-cloud and nodata masks by **350 m** in the source projected geometry. Retain the distance-to-cloud audit. Run the predeclared **700 m** buffer sensitivity only after the model is frozen; it cannot select the model or a more favorable headline cohort. A cloud-free flag alone does not prove the absence of thin cloud or adjacency effects.
4. Require a matching **Good or Best `GeolocationAccuracyQA`** from the L1B GEO source or an explicit official scene-level record. Missing, Suspect and Poor are quarantined. Do not interpret absence from a bad-scene list as Good. The official metadata warns that uncorrected geolocation can be wrong by kilometres; successful image matching is described as better than 50 m, still substantial at a 100 m boundary. [NASA's L1B GEO product metadata](https://catalog.data.gov/dataset/ecostress-swath-geolocation-instantaneous-l1b-global-70-m-v002)
5. Exclude confirmed ISS obstruction scenes and unresolved obstruction status where the historical field is unreliable. Freeze the obstruction-list hash and scene lookup result. For V002, quarantine observations in **2025-05-15 through 2025-07-01** for the announced instrument-noise issue, and any observation missing `LST_err`, including the reported **2025-12-16 through 2026-06-10** production gap. A later documented reprocessing may be used only through a new source-registry revision made before opening those temperatures. [NASA/LP DAAC V002 known issues](https://data.nasa.gov/dataset/ecostress-tiled-land-surface-temperature-and-emissivity-instantaneous-l2-global-70-m-v002-09540), [NASA processing-issue notice](https://forum.earthdata.nasa.gov/viewtopic.php?t=8015)
6. Require absolute view zenith **≤30°**; require independent land fraction **≥0.95**, and no more than **0.05** source water fraction. Report shoreline/mixed-land exclusions separately. Keep snow-labelled land when its thermal QC passes; snow is a condition to evaluate, not an unconditional discard rule. No active-fire detection/reconstruction claim is in this stage. Do not remove an otherwise valid hot label merely because the model might find it difficult.
7. Aggregate qualified native pixels by their actual area overlap onto the fixed 100 m grid; require **≥90%** valid support of the target cell. Save valid-area fraction and contributing native support. Use an area-weighted arithmetic temperature mean, explicitly identified as the gridded-label approximation; it is not an exact mixed-emissivity radiometric inversion. For the ≤2 K gate, retain the maximum accepted source error estimate, not an uncertainty reduced by `sqrt(pixel_count)`. Do not sharpen or fill thermal labels.
8. Inspect registration on **at least 10 distributed independent surface control features per accepted acquisition** where such features can be identified, using nonthermal coast/water/optical boundaries and the official geolocation information. Do not move labels to minimize model errors. Scenes without a defensible alignment audit cannot support a 100 m spatial-contrast claim; they can be retained only in a separately labelled coarse-support diagnostic. On the final model, shift label geometry by fixed ±50 m east/west/north/south and report metric sensitivity. An improvement that disappears under these plausible offsets is not accepted as established 100 m skill.

**Why this matters for 5–7°C misses:** a quality flag or an attractive image does not distinguish a model error from cloud contamination, wrong emissivity, atmospheric correction, a misplaced road/park boundary, or mixed surface support. A ≤2 K teacher screen reduces some retrieval uncertainty; it does not create exact ground truth. Large-error reduction must survive independent acquisitions, geometry sensitivity and a second reference. Source emissivity, water-vapour and view-angle diagnostics may be used for label QC and error stratification only, never as target-derived predictors.

ASTER's 90 m surface-temperature retrieval uses atmospheric correction and temperature/emissivity separation. Its V004 description does not supply a universal per-pixel uncertainty guarantee. Use ASTER as a separately reported corroboration cohort only after its V004 cloud, radiance-quality, emissivity, geolocation and scaling contracts are verified; absent uncertainty is unknown, not zero. Do not borrow a V001/V002 flag layout or an old nominal accuracy claim to approve V004 pixels. [NASA ASTER V004 product](https://data.nasa.gov/dataset/aster-l2-surface-kinetic-temperature-v004)

### Historical geolocation metadata route

The current LP DAAC `qa_20250423-present.txt` link must not be assumed to cover 2021–2023 just because a catalog description says “all scenes.” The NASA/JPL tutorial repository contains a historical `SP_Geo_Flags.txt` whose visible records are Poor/Suspect; that is an exclusion list, not positive Good/Best evidence. [Official tutorial entry point](https://ecostress.jpl.nasa.gov/tutorials)

NASA's extraction script resolves the matching `ECO_L1B_GEO.002` acquisition, reads its `.h5.dmrpp` metadata, and decodes `L1GEOMetadata/GeolocationAccuracyQA`; its fallback reads only that HDF5 metadata group. Reuse that field contract through the private research adapter, not the tutorial's credential-persistence workflow. Match orbit, scene, timestamp and processing provenance, record the metadata hash, and fail closed on ambiguous matches. [NASA extraction implementation](https://github.com/nasa/ECOSTRESS-Data-Resources/blob/main/python/scripts/extract_geolocation_flag/ECOSTRESS_geolocation.py)

## 3. Feature contract and ablations

The **A feature set is the existing 40-name production registry**, listed in section 10, with the same units and formulas. The new input builder must reproduce those definitions. Surface reflectance features use independent Landsat Collection 2 surface reflectance, not ECOSTRESS/ASTER thermal bands, emissivity or downscaled L3 meteorology. Use a trailing **32-calendar-day** optical composite ending at the acquisition; no contributing observation may occur after the requested time. Require ≥80% clear optical area per contributing 100 m cell; compute a median of each reflectance band across accepted preceding observations, then the existing indices/albedo proxy. Require at least one usable observation and save count, earliest/latest source time and maximum age. No future snapshot or seasonal fallback is allowed in fitting or evaluation. This composite is a static surface-description approximation; source age is audited but is not an added predictor in this experiment.

Same-acquisition Landsat optical reflectance is permitted for a Landsat thermal label only through the independent optical QA path; the thermal label and its QC must never select optical values. For an ECOSTRESS/ASTER observation, the same strict past-only composite rule applies. Cells without a supported optical composite are excluded from **all four matched ablations**. Their coverage is reported. Do not silently substitute Sentinel reflectances without a separate spectral/processing contract.

Terrain remains Copernicus GLO-30 DSM, aggregated as the existing 100 m elevation, Horn slope/aspect and 300 m relief. Climate remains the pinned 1991–2020 Köppen map. Current weather and background-air history remain explicit ERA5 fields; station adjustment uses independently quality-screened reported air, at most **90 minutes old and 100 km away**, with station age/distance recorded. Primary reported-air acceptance is calculated on station-supported examples. ERA5-only examples may be retained as an explicitly reported secondary source subgroup, but cannot satisfy the minimum reported-air evidence counts.

| Fit | Features | Fixed comparison |
|---|---|---|
| S0 — sampling control | Existing 40 | Refit only retained admissible old fitting rows with this estimator and weighting policy |
| S1 — added daytime sampling | Existing 40 | S0 plus new eligible **daytime** rows; separates broader daytime sampling from adding night labels |
| A — shared observed day/night | Existing 40 | S1 plus new eligible night rows; no new feature family |
| B — thermal memory | A + 11 completed-history features below | Same rows and weights as A |
| C — independent surface cover | A + 5 fractions below | Same rows and weights as A |
| D — both additions | A + all 16 new features | Same rows and weights as A |

S0 and S1 are diagnostic controls, not extra candidates selected after seeing test errors. Preserve the frozen production model separately. S0→S1 changes new daytime sampling; S1→A changes the addition of night labels to a shared model; A→B/C/D changes feature families. A comparison only against frozen v1 would confound new data, source processing, weights and feature changes.

All A/B/C/D fits use the intersection of rows with complete required A+D inputs, identical sample IDs, weights, splits and source versions. Also report coverage on the larger A-eligible population, without substituting that population into paired headline comparisons. If a source family is unavailable, mark the affected ablation unavailable rather than inventing a feature or changing the common rows after seeing scores.

### B: eleven thermal-memory predictors

Use the research-only [`thermal_memory.py` contract](THERMAL_MEMORY.md). Cutoff `h = floor(requested UTC hour)`. The last hourly radiation mean ends at `h`; all physical integration support precedes or ends at `h`. Current requested-time solar geometry stays at the original acquisition time. This is retrospective valid-time causality, not a claim that the reanalysis was available as issued at the historical prediction time.

| Exact feature names | Units and calculation |
|---|---|
| `memory_shortwave_energy_{6,12,24}h_j_m2` | Sum the last 6/12/24 hourly mean downward shortwave fluxes ×3,600; J/m² |
| `memory_longwave_mean_{6,24}h_w_m2` | Mean of the last 6/24 hourly mean downward longwave fluxes; W/m² |
| `memory_air_mean_{6,24}h_c` | Mean of 6/24 instantaneous ERA5 background-air samples at hourly endpoints through `h`; °C |
| `memory_air_range_{6,24}h_c` | Maximum minus minimum of those hourly samples; °C, not continuous extrema |
| `memory_air_change_{3,6}h_c` | Background air at `h` minus background air at `h−3/6 hours`; °C |

For a window of length `w`, endpoints are `h−(w−1), …, h`; radiation support is `(h−w, h]`. Integrals/means/ranges require **100% finite, unit-validated history**. Changes require both endpoints. Missing history yields NaN plus explicit coverage/status, never zero, shortened windows or forward filling. These diagnostics are audit fields, not predictors. Do not apply the present station/manual correction retrospectively to the background-air history. No absorbed-shortwave multiplication or hours-since-sunset predictor is in this first 11-feature ablation; those remain later hypotheses, avoiding implementation-dependent feature additions.

### C: five independent surface-class fractions

Use **ESA WorldCover 2020 v100**, fixed for this experiment, to avoid switching classification algorithms across the temporal split. Area-average its 10 m classes into `worldcover_tree_class_fraction` (class 10), `worldcover_grass_class_fraction` (30), `worldcover_crop_class_fraction` (40), `worldcover_built_class_fraction` (50), and `worldcover_bare_class_fraction` (60). Each value is the fraction of the entire supported 100 m cell, in [0,1]; require ≥95% classified support. Other classes remain in the denominator, so the five fractions need not sum to one. These are proportions of mapped classes, **not measured canopy cover, building area, material imperviousness, heights or cast shadows**.

The map's observation period ends before 2021 fitting acquisitions; record its release date separately because this retrospective experiment is not a replay of which published products were available on each historical date. Maximum static-map observation age is **five calendar years**, covering this fixed 2021–2025 experiment. Land-cover change remains a limitation. Do not use WorldCover 2020→2021 differences as a change detector: ESA documents that the two maps use different algorithms. [ESA WorldCover data and version notes](https://esa-worldcover.org/en/data-access)

## 4. Sample units, weights and evidence minimums

Keep the parent's 10 km block grid, `(row + 2×column) % 5 == 0` reserve pattern, and 1 km exclusion buffers for fitting, development and calibration. Raster sampling is deterministic from the parent seed and coordinate hashes. Select at most **200 cells per 10 km block per overpass**, at least **300 m apart** for fitting, and at most **1,000 cells per pilot/acquisition**. Within available independent broad cover groups, sample equally without manufacturing missing groups. Save inclusion probabilities and report both balanced and available-area-weighted scores. Dedicated spatial-contrast samples may be denser but never become extra independent acquisition counts or enter fitting.

A qualifying pilot/day-or-night acquisition needs ≥200 usable cells over ≥2 nonreserved blocks (≥50 cells per block) and at least two mapped surface groups with ≥25 cells each. Count no more than one qualifying acquisition per pilot, UTC date and day/night class. Group all overlapping tiles/revisions and repeated same-date observations together. Also report distinct fixed 72-hour weather-episode bins, anchored at 2021-01-01 UTC, as a dependence diagnostic. Such bins are not asserted to be statistically independent weather events.

Day is solar elevation **≥10°**, night **≤−6°**; intermediate values are twilight and excluded from these fits. They remain a separate unsupported condition. Both London and Sioux Falls use DJF/MAM/JJA/SON as calendar seasons. In other future climates, predefine local wet/dry regimes before fitting rather than extrapolating these four bins as universally meaningful.

| Partition, **for each pilot and each day/night class** | Minimum qualifying acquisition dates | Required spread |
|---|---:|---|
| Fit, 2021–2022 | 32 | ≥12 in each year; ≥8 in each season across the two years; ≥24 distinct 72-hour bins |
| Development, Jan–Jun 2023 | 12 | ≥4 Jan–Feb, ≥4 Mar–May, ≥2 June; ≥8 72-hour bins |
| Calibration, Jul–Dec 2023 | 12 | ≥2 Jul–Aug, ≥4 Sep–Nov, ≥2 December; ≥8 72-hour bins |
| Preserved 2024 test | 16 | ≥4 per season; ≥12 72-hour bins |
| Blind 2025 test | 24 | ≥6 per season; ≥18 72-hour bins |

For final spatial generalization, each pilot/day-or-night class also needs ≥12 test dates, ≥3 seasons and ≥3 reserved blocks with ≥50 usable cells each; score reserved and previously sampled blocks separately. The minimums are deliberately stronger than the current generic trainer's two-day/30-row checks. Even a complete metadata inventory supplies only upper bounds on the usable cohort; its counts do not establish these quality-screened evidence conditions.

Across the A/B/C/D training set, give London-day, London-night, Sioux-day and Sioux-night equal aggregate weight. Within each cell of that grouping, equalize acquisition-date weights, then available broad surface-group weights, then rows. Retained eligible 2021–2022 fitting data from other pilots receive **20% of total fitting weight**, balanced by pilot/date; the four new groups share the remaining **80%**. Preserve all eligible old fitting rows when within the 200,000 cap. If the cap binds, hash-downsample within these fixed strata without changing their aggregate weights. Log effective weights and counts. Unknown climates or a missing essential feature are quarantined; they do not become a newly learned category from test data.

If minimum coverage fails, acquisition/adapter smoke tests and descriptive quality reports may continue, but the dependent model fit or acceptance claim is **unsupported**. Do not fill the deficit with more adjacent pixels. The first full research fit requires the fitting and development minima; calibration or final-test shortfalls prevent the subsequent promotion steps.

## 5. Fixed numerical estimator

Use the existing `model.build_estimators` architecture: numerical passthrough and a training-fitted ordinal categorical climate encoder for HGB; the existing median-imputer/scaler/one-hot pipeline for a ridge residual baseline. The matched main cohort is complete-case, so the imputer cannot quietly solve missing-history cases. Fix HGB settings for S0/S1/A/B/C/D:

```text
loss=squared_error; learning_rate=0.08; max_iter=150
max_leaf_nodes=15; max_depth=6; min_samples_leaf=30
l2_regularization=1.0; max_bins=255; early_stopping=False
random_state=2708; categorical climate only
max_train_rows=200000; OMP/BLAS threads=4
```

These retain current model capacity instead of attributing extra tree depth to a new feature family's success. Explicit acquisition/stratum weights must reach `regressor__sample_weight`; the current generic trainer does not supply them. No hyperparameter sweep, random row validation, target clipping, residual trimming, sensor-specific intercept or target-derived quality predictor is added. Fit the A-feature ridge baseline with `alpha=1.0` and the same weights. Preserve reported air alone as another baseline. A duplicate fit with the same hashes/seed must reproduce predictions within 10⁻⁶ °C on a fixed small fixture.

Night also compares with the withdrawn native-support ERA5 skin–air baseline. ERA5 skin temperature is **only a benchmark target estimator**, not a feature or training teacher. For a ground-radiometer comparison, first audit radiative footprint, time, emissivity and station independence; a coarse flux footprint cannot certify every 100 m urban cell.

## 6. Selection and what counts as deterioration

Choose only among A/B/C/D using **Jan–Jun 2023**. No model-selection decision uses July–December 2023, 2024 or 2025. First calculate acquisition-balanced MAE separately for the four pilot/day-or-night groups. `J` is their equal-weight mean. Also calculate each acquisition's centered spatial contrast error: mean absolute difference between `(prediction − acquisition mean prediction)` and `(label − acquisition mean label)`, using the same eligible locations and sampling weights. This separates whole-scene temperature bias from spatial structure.

Material deterioration, relative to A on a common cohort, is any of:

- Pilot/day-or-night, season, reserved-block or adequately supported surface/air-temperature subgroup MAE increasing by **more than 0.20°C or 10%, whichever is larger**.
- RMSE increasing by **more than 0.30°C or 10%, whichever is larger**; absolute bias increasing by >0.30°C; error-above-5°C frequency increasing by >2 percentage points.
- Acquisition-centered contrast MAE increasing by >0.20°C; any supported acquisition date's MAE increasing by >1.0°C.

Only groups with ≥4 qualifying development dates and ≥100 usable cells per date can pass a subgroup gate; smaller groups are unsupported, not silently pooled away. Require each added-feature candidate to improve `J` over A by **at least 0.15°C and 5%**, while passing all applicable deterioration gates and retaining exactly the same rows. If several qualify, choose the fewest added features within 0.05°C of the best `J`, tie order A, C, B, D. Otherwise retain A as the research candidate if its baseline checks pass; if A fails, stop before calibration. Research model selection alone does not establish production readiness.

Define extreme groups from independent reported air: **cold ≤0°C**, **mild 0–30°C**, **hot ≥30°C**, with exact endpoints counted only once (cold ≤0; mild >0 and <30; hot ≥30). Define snow separately using an independent snow flag or ERA5 snow-water-equivalent ≥0.001 m, auditing the proxy. Require ≥4 development acquisition dates and ≥6 final-test dates per claimed extreme/snow group per pilot/day-or-night class. Insufficient hot London nights, for example, means that claim remains unsupported; no retrospective temperature quantile is substituted to make the count pass.

## 7. Calibration, final tests and promotion gates

After selection, freeze estimator and feature hashes. Do **not** refit on development or calibration labels. Use Jul–Dec 2023 only to obtain day/night-specific empirical absolute-error radii, with equal acquisition-date total weights and no per-climate calibration at this sample size. Choose the smallest radius whose date-balanced empirical coverage is ≥90%. Each day/night pool needs ≥24 calibration dates from both pilots. Describe these as empirical intervals; correlated pixels and seasonal shift preclude an unconditional conformal guarantee. The existing single global `interval_radius_c` bundle cannot faithfully represent this contract without a reviewed bundle/serving update.

Freeze model, all feature/QA policies, sampling IDs, calibration radii and evaluation code before opening the 2025 thermal arrays. Run the preserved 2024 test once as a paired regression check, then run the blind 2025 test once. A failure may stop promotion; it may not select a runner-up or trigger refitting under this same blind-test label. Preserve every failed candidate and its predictions.

Report MAE/RMSE/bias, absolute error >3/>5/>7°C, acquisition mean bias, centered contrast MAE, intervals and counts. Report all figures by pilot, day/night, season, independent hot/cold/snow group, mapped surface group, source instrument, station support and reserved spatial status. Display date-balanced and area-weighted results separately. Confidence intervals use **1,000 paired bootstrap replicates**, seed 2708: resample whole dates and 10 km blocks within pilot/day-or-night strata, applying the product of date and block multiplicities to each row. Keep acquisition tiles together; never bootstrap isolated neighboring cells as independent measurements.

**All relevant gates must pass for a supported pilot/time/condition scope:**

| Gate | Frozen criterion |
|---|---|
| Typical absolute error | Date-balanced MAE ≤2.5°C, RMSE ≤3.5°C and absolute bias ≤0.75°C for each supported pilot/day-or-night group, including reserved blocks |
| Large errors | >5°C frequency ≤10%; >7°C frequency ≤3%, reported with clustered intervals |
| Extreme/snow error | MAE ≤3.0°C, RMSE ≤4.0°C, absolute bias ≤1.0°C, >7°C frequency ≤5%, on each sufficiently supported group |
| Daytime behavior | Versus frozen v1 on the same eligible observations: no material deterioration; plus either MAE improvement ≥0.15°C and 5%, **or** >5°C frequency reduced by ≥25% relative with MAE increase ≤0.10°C |
| Nighttime behavior | MAE improves by ≥0.20°C and 10% versus **both** reported-air-only and coarse ERA5 baseline; also beat the fitted A-feature ridge on MAE |
| Evidence of improvement | For the claimed primary improvement, upper endpoint of the paired 95% bootstrap interval for the MAE difference is ≤0; report intervals even when they prevent a claim |
| Spatial signal | Centered contrast MAE ≤2.0°C, and improvement ≥0.20°C and 10% versus the best applicable air/coarse benchmark; no worse than v1 daytime by >0.10°C |
| Geolocation robustness | Repeat fixed ±50 m label-offset diagnostics. If the claimed contrast improvement reverses sign or overall MAE shifts by >0.5°C, the 100 m spatial claim is unsupported pending independent alignment evidence |
| Interval usefulness | Final date-balanced coverage ≥85% overall in each day/night class and ≥80% in each adequately supported subgroup; mean full width ≤10°C. Wider intervals cannot be used to declare this accuracy problem solved |
| Preserved global daytime behavior | On every other pilot's existing unchanged 2024 cohort, apply the deterioration definition in section 6; sparse dates remain disclosed. No broad global replacement without passing this regression check |

A scene with nearly uniform truth should not require artificial contrast. Report contrast metrics for all scenes, but apply the improvement gate only where acquisition LST interquartile range is ≥1°C and at least two independent surface groups are present. This fixed applicability rule does not remove those scenes from ordinary error metrics. A detailed-looking texture alone never passes a gate.

Passing an observed clear-night gate does not authorize cloudy-night, twilight, unsupported snow/extremes, another climate, arbitrary future weather or general worldwide operation. A missing evidence cell means **no deployment for that scope**; it cannot be labelled passed by averaging with better-supported conditions. There is no automatic deployment under this addendum. The current daytime model and withdrawn-night behavior remain unchanged until the explicit gate report and reviewed integration support a concrete release.

## 8. Exact integration boundaries

The current source is a useful implementation base, but its generic training entry point is **not an Option B runner**:

| Existing interface | Reuse and required boundary |
|---|---|
| `night_inventory.split_for_date`, `pilot_blocks`, `calendar_candidates` artifacts | Reuse frozen dates, block IDs, reserved geometry and ranked acquisition groups. Keep complete-inventory and positive QA flags separate |
| Research NASA adapter | Emit unfilled `lst_c` labels and label QA with acquisition/pixel identity; keep thermal assets apart from predictor construction. Only the private research user may retrieve protected assets |
| `raster.aggregate_optical`, `feature_frame`, `add_raster_context`, terrain helpers | Reuse numerical feature definitions and exact grid identity. Add the reviewed past-only optical composite and fixed WorldCover-2020 fraction layer in a research builder; do not route training through arbitrary-date scenario fallback |
| `weather.enrich_weather`, `assemble.attach_stations`, `radiation.add_radiation` | Join by time/coordinate with `_raster_position`/`sample_id` preserved. Validate full units/history, station age and source times; do not synthesize missing actual weather from a reference year |
| `thermal_memory.py` | Append the eleven agreed predictors and their separate coverage/status manifest, without changing current production features |
| `model.prepare_frame`, `build_estimators`, `target_offset`, `predict_frame` | Reuse these building blocks only after an explicit reviewed allowlist extension for the 16 new feature names; keep all label/identifier/QA diagnostics outside the predictor list |
| `model.assign_splits`, `validate_splits`, `train_and_evaluate` | **Do not call unchanged:** they lack a distinct 2023H1 development split, permanent buffered spatial blocks, acquisition weighting and these minimum counts; they would treat all post-calibration dates as one test and calibrate a single radius |
| `scenario.prepare_scenario`, `night_baseline`, API/renderer | Leave production untouched. Reference-year weather and current coarse night outputs are not training labels. A later accepted serving path must construct the identical feature windows and source-age/missingness behavior |

Minimum parquet schema for each sample:

```text
sample_id                         immutable unique string, no target in its hash
region_id, pixel_id                audit identity, never predictors
pixel_x, pixel_y, pixel_epsg        exact fixed 100 m grid coordinates
latitude, longitude                WGS84 audit/context inputs, not model predictors
datetime_utc                       actual thermal acquisition UTC timestamp
acquisition_group_id, utc_date     orbit/scene grouping and conservative date cluster
block_id, spatial_holdout          frozen 10 km geometry assignment
in_holdout_buffer                 true => exclude from fit/dev/cal
temporal_split                    fit | development | calibration | legacy_2024 | blind_2025
daylight_group, season             deterministic observed-time context
label_product, label_version       audit only; one revision per acquisition
label_scene_id, label_source_hash  provenance, never predictors
lst_c                             target label only
label_valid_fraction              actual qualified native area / target-cell area
label_error_bound_k, geo_qa        QC audit, never predictors
cloud_distance_m, view_zenith_deg  QC audit, never predictors
air_temperature_source            reported/station-adjusted vs background-only audit
station_id, station_age_minutes, station_distance_km
optical_start_utc, optical_end_utc, optical_age_days, optical_count
surface_map_year, surface_map_version, source_manifest_sha256
sample_weight                     frozen acquisition/stratum contribution
the explicit A features plus the selected B/C feature names
```

Before any fit, assert sample-ID uniqueness, no source scene/version duplication, all source times obey the feature contract, no reserved/buffered cells in fit/dev/cal, no 2024/2025 labels in those partitions, no cross-partition acquisition group, and unchanged sample/weight hashes across A/B/C/D. Write counts **after** every mask and after the common-feature intersection. Tests should force weather-group reordering and verify `sample_id`/pixel identity, exact hourly endpoints, missing history, leap days, partial valid-area cells and day/night separation. Keep paired metrics and machine-readable pass/fail/unsupported reasons in `acceptance.json`, alongside the frozen artifacts.

## 9. Immediate decision from existing evidence

The completed **2021–2023 nighttime metadata inventory** has no query gaps for these two pilots: London has **810 orbit groups across 529 UTC dates**, and Sioux Falls **436 across 382**. These replace the earlier partial preflight counts. They are candidate metadata counts, not positive geolocation checks, clear-sky coverage, station-supported labels or complete-feature counts; daytime coverage and the per-year/season/split breakdown must also satisfy the tables above.

Proceed with authorized small nonreserved source/feature QA checks and count the eligible cohort. An engineering output remains `training_eligible=false` until it meets the fitting contract, regardless of whether it successfully reads a TIFF. The completed `sample_2021_seasons_v3` and later adapter-v2 runs use the fitting profile's **350 m cloud buffer, 30° view limit, 90% valid area, zero low four QC bits, LST accuracy codes 1–3, and positive source error estimates no greater than 2 K**. They preserve the maximum contributing source error estimate during 100 m aggregation; this is not a guaranteed error bound. Earlier engineering settings of 70 m, 40° and 80% do not describe these runs; retain their original audit records without retrospectively upgrading them.

The first eight-scene v3 run supplies zero source-screen candidates: its three nonzero scenes all have Suspect/Poor GEO. A second fixed sequence checked 16 further geometry candidates, processed the nine with Good/Best GEO and resolved one transient HTTP failure by retrying that same acquisition. Across **17 distinct thermal scenes**, three Sioux Falls dates (2 January, 15 July and 22 July 2021) have raster-screen cells, Good/Best GEO and no matching entry in the complete parsed official obstruction list. Two April scenes with cells are explicitly obstructed and excluded. NotListed does not prove unobstructed pixels. The three remaining dates total 173,055 provisional cells before independent land/registration checks, spatial exclusions and feature joins; **all remain training_eligible=false** and London has no remaining date. This is far short of the fitting coverage requirements. The sampling audit's generic pending list predates this completed addendum; the addendum itself is now written. See [the engineering report](EARTHDATA_ENGINEERING_2026-09-09.md) for exact stage counts, source hashes and resource measurements.

If qualifying counts fall short, that is a data/evidence limitation, not a request for a larger GPU or for the user to reconfirm already authorized work. Strict QA can legitimately leave too few scenes; publish that shortfall instead of relaxing the teacher to manufacture a model.

## 10. Frozen A feature names

The inspected catalog identifies `pilot_v1_model`, SHA-256 `e5742d24d2126f8a85771a5a3546eaa204a4799209d452165062872bf2e5518b`. Preserve its bundle and source-manifest hashes as the baseline registry, independently of this document's text.

```text
air_temperature_c, ndvi, ndbi, ndwi, albedo_proxy,
elevation, slope, terrain_relief_300m, aspect_sin, aspect_cos, water_fraction,
solar_elevation_deg, solar_azimuth_sin, solar_azimuth_cos, hour_sin, hour_cos,
day_of_year_sin, day_of_year_cos, relative_humidity_pct, dewpoint_c,
wind_speed_m_s, wind_direction_sin, wind_direction_cos, surface_pressure_hpa,
cloud_cover_fraction, shortwave_down_w_m2, direct_shortwave_w_m2,
diffuse_shortwave_w_m2, era5_longwave_down_w_m2,
era5_snow_water_equivalent_m, precipitation_mm_h, rain_mm_24h, rain_mm_72h,
soil_moisture_m3_m3, air_temperature_lag1_c, air_temperature_lag3_c,
air_temperature_lag24_c, shortwave_down_lag1_w_m2,
shortwave_down_mean3_w_m2, climate_class
```
