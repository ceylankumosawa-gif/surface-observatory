# Option B — preflight protocol frozen before thermal labels

Protocol ID: **option-b-2026-09-09-v1**. Date: **2026-09-09**. Status: research preflight; no new thermal labels acquired and no model fitted under this protocol. The current production model, existing heldout sources and prior rejected candidates remain unchanged. This document freezes the temporal/spatial exclusions, metadata inventory and sampling approach. It does **not** constitute a fully specified fitting or deployment protocol.

This protocol supersedes the earlier exploratory split in `DATA_ACCESS.md`. **2024 will not be used for development or calibration.** That source has already been inspected in the previous project and is a preserved retrospective test, not a new blind test. **2025 thermal observations remain reserved for the final assessment.** Metadata discovery and coverage planning are allowed before label access; their temperature values are not.

## Objective and scope

Learn how surface temperature differs from reported air temperature for observed daytime and nighttime cases, using weather, heating/cooling history, surface properties and climate class. Start with London and Sioux Falls to establish a reproducible data and evaluation pipeline. Improvement must be measured against the existing model and simple baselines, including spatial contrasts and large errors; visual texture is not an acceptance criterion.

This two-pilot experiment does not establish worldwide climate transfer. A later stage must include the other pilots with complete site and climate holdouts. A 100 m grid does not imply independent 100 m skill, and infrared labels do not supply through-cloud surface observations. Nighttime, twilight, snow and extreme weather require separately reported evidence. The withdrawn ERA5-only nighttime map is a baseline, not a fine-resolution training label.

## Product and acquisition registry

The metadata CLI verifies these exact product/version/concept-ID pairs against anonymous NASA CMR collection metadata before querying granules:

| Key | Short name | Version | Collection concept ID |
|---|---|---|---|
| `ecostress_v2` | `ECO_L2T_LSTE` | `002` | `C2076090826-LPCLOUD` |
| `ecostress_v3` | `ECO_L2T_LSTE` | `003` | `C3998139651-LPCLOUD` |
| `aster_v4` | `AST_08` | `004` | `C3306885674-LPCLOUD` |

ECOSTRESS V002 is the initial archive-coverage candidate. V003 is inventoried as a separate product; its improved processing and different historical coverage must be reviewed before choosing a version for fitting. Do not pool two versions of the same observation as independent examples. ASTER V004 supplies a separate instrument and sparse complementary acquisitions. ASTER thermal acquisition ended after 2026-01-16, so its role is archived training/validation. [NASA ECOSTRESS products](https://ecostress.jpl.nasa.gov/data/atbds-summary-table), [NASA ASTER V004 tutorial](https://github.com/nasa/ASTER-Data-Resources/blob/main/python/tutorials/Exploring_AST_08_Surface_Kinetic_Temperature.ipynb)

The first inventory covers **2023**, London and Sioux Falls, and night-tagged records. This is a deliberately bounded metadata smoke run. Expand with separate immutable output directories to 2021–2022 fitting years and then 2024–2025 reserved years; include `DAY` to inventory the broader daytime pool. Inventorying reserved years does not authorize opening their thermal arrays.

The CLI counts granules, **unique acquisition groups** and **unique UTC dates** separately. ECOSTRESS tiles and adjacent scenes on one orbit form one overpass group. Versions of the same orbit share a group. ASTER metadata summaries are grouped by acquisition timestamp; they do not expose an independently verified orbit ID, so statistical splits and uncertainty use whole UTC dates conservatively. A group crossing temporal split boundaries is reserved rather than assigned to fitting.

## Geometry, quality and representativeness

CMR bounding-box hits are candidates. The CLI independently checks CMR polygon topology, pilot intersection, plausible area and named ECOSTRESS MGRS zone/band consistency. Antimeridian/spanning polygons and naming/geometry mismatches are quarantined. These metadata checks cannot verify the actual COG footprint or cloud-free pixels. The earlier Gobabeb result naming a distant MGRS tile illustrates why this distinction matters.

After secure access is configured, a separate acquisition step must check actual COG bounds, CRS and pixel support; product-version-specific units/scale/nodata; cloud and quality masks; geolocation quality; thermal uncertainty where provided; and actual solar elevation at the acquisition. Version 002 cloud masking must use the appropriate cloud layer, not assume that its QC layer contains the Version 001 cloud bits. Co-register and aggregate footprint-aware labels to the pilot grid with a predeclared valid-area rule. Inspect alignment using independent optical/surface boundaries, not a desired thermal prediction.

Predeclare analysis groups: daytime solar elevation at least 10°, nighttime at most −6°, and twilight separately. Metadata `DAY`/`NIGHT` is only a discovery filter. Do not fit twilight as ordinary daytime or represent infrared clear-sky samples as an all-weather population.

Sample calendar periods and solar-time bins before inspecting temperatures. The CLI ranks complete overpass groups by SHA-256 of a fixed seed, pilot ID and group ID within year-month, instrument family, six-hour approximate local-solar-time bin and metadata day/night flag. It proposes the first two groups per stratum, retaining a deterministic remainder list. An incomplete inventory produces lower bounds and **cannot freeze final label samples**. Actual coverage failures must be documented and replaced in rank order, without selecting based on errors or target temperatures.

Use independently mapped urban, vegetation, bare, wet and snow surface strata in each accepted acquisition. Put an initial cap of 200 valid 100 m samples per 10 km block per overpass, with a fixed coordinate-based seed. The proposed first fit is capped at 200,000 rows and four CPU threads. Equalize acquisition-date and broad surface-group contribution; the number of independent nights matters more than millions of neighboring pixels. These are operational caps, not a compute benchmark or accuracy promise.

## Fixed temporal and spatial splits

| Role | Dates | Allowed use |
|---|---|---|
| Fitting | 2021-01-01 through 2022-12-31 | Fit parameters only on eligible nonreserved blocks |
| Development | 2023-01-01 through 2023-06-30 | Compare the predeclared ablations |
| Calibration | 2023-07-01 through 2023-12-31 | Calibrate selected-model uncertainty; no model selection |
| Preserved test | All 2024 | Retrospective paired check; previously inspected sources stay excluded from fitting/tuning |
| Final test | All 2025 | Thermal labels blind until model, feature set and calibration are frozen |

No existing 2024 or 2025 source is copied into fitting, development or calibration. Existing source files are never overwritten. The CLI records hashes of known baseline/model/heldout manifests when present; absence is recorded, not silently treated as a verified snapshot. Before acquisition/training, add all retained source manifests and hashes to the run registry, including the previous refinement protocols and their evaluation outputs. Existing 2025 scenario predictions are not independent observed labels.

Each pilot is divided into fixed 10 km blocks anchored at its existing projected lower-left extent. Reserve blocks where `(row + 2 × column) % 5 == 0`, approximately one fifth of the pilot, throughout fitting/development/calibration. Exclude a **1 km buffer around reserved blocks** from those stages. The CLI exports these exact bounds and CRS. Final temporal errors must be shown separately on previously sampled versus reserved spatial blocks. Include heldout-site evaluation when adding other climates; do not claim London/Sioux transfer as global generalization.

If cloud-free acquisition counts cannot support the fixed split, stop the dependent fitting step and publish the coverage shortfall. A revised protocol can widen the archive or change sampling only before opening reserved temperature values, with a new protocol ID and an explicit change log. Do not move viewed test cases into training or repeatedly tune against 2025.

## Broader daytime data and thermal-memory ablations

The first daytime improvement is representative new sampling on eligible 2021–2022 dates, including urban surfaces, heat events, cold/snow conditions and seasons. Use the same spatial/temporal exclusions and serving-compatible feature aggregation. The old London error maps may motivate additional independent surface strata; they do not authorize training on their heldout pixels. Preserve the old global fitting rows and account for acquisition/group weighting.

Retain the simple target `LST − reported air temperature`. Use the same estimator family and a single predeclared capacity configuration across four ablations, with identical training rows and splits:

| Candidate | Feature change | Purpose |
|---|---|---|
| A | Existing available weather/surface/climate inputs | Measure the effect of improved sampling and observed night labels |
| B | A plus thermal-memory features | Test heating/cooling history |
| C | A plus independent surface fractions | Test tree, built, impervious and cover information |
| D | B plus C | Test whether history and surface properties help together |

Proposed memory inputs are past 6/12/24-hour absorbed-shortwave proxies, past 6/24-hour longwave means, past 6/24-hour air-temperature means/ranges, 3/6-hour temperature change, hours since sunset, and existing rain/moisture/snow context. Use only completed, timestamp-aligned history before the prediction. Integrated radiation is the sum of mean hourly W/m² × 3,600 seconds and is expressed in J/m². The absorbed-shortwave proxy multiplies by independent albedo; it remains a proxy, not a surface energy-balance closure or cast-shadow model. Test missing-history handling explicitly.

Surface fractions must come from independent, dated products and be available in deployment. Use a composite ending before the acquisition with a declared maximum age; no future optical scene, target-derived emissivity, observed thermal field or supplied scenario output may become a predictor. Do not use same-acquisition ECOSTRESS L3 downscaled meteorology/energy products without checking their dependency on thermal labels. Feature units, windows and missingness rules are frozen alongside the selected candidate; production allowlists remain unchanged until reviewed implementation and evaluation pass.

**Required before fitting:** publish a protocol addendum that fixes the estimator's numerical capacity/hyperparameters, exact feature sources and availability, units and history windows, optical age/coverage limits, sample/quality rules and numerical criteria for a material deterioration. Define the model-selection rule and deployment thresholds in that addendum. Resolve these details using input availability and coverage audits, without inspecting reserved thermal values or candidate test errors. The current feature table describes the planned ablations; it does not authorize implementation-dependent choices after seeing results.

## Evaluation and deployment gate

Compare each candidate with reported-air-only, a simple fitted linear residual model and the preserved production daytime model where valid. Include the withdrawn coarse ERA5 baseline for nighttime comparison, labelled at its native support. Report MAE, RMSE, signed bias, frequencies of absolute errors above 3/5/7°C, per-acquisition means, and within-scene spatial-contrast errors. Report hot/cold conditions using fixed independent air-temperature thresholds and show sample counts. Bootstrap whole dates and spatial blocks, not random neighboring pixels.

Prefer the simplest candidate whose development results improve the intended behavior without a material deterioration in another predeclared group. Freeze that choice before calibration and the 2025 test. Once a final test has been viewed, it becomes a reported test; subsequent tuning requires new untouched observations. Publish failures and unsupported conditions. Do not ship a detailed-looking raster merely because it is more convincing than coarse blocks.

Ground radiometry at Sioux Falls/Boulder supports continuous nighttime/weather checks only after footprint, emissivity and timing audits. MODIS/SLSTR provide broader temporal checks at their approximately 1 km thermal footprint. Neither validates global 100 m all-weather maps. Weather-only predictions on cloudy nights remain an extrapolation until independently observed under those conditions. No universal error target, compute-time promise or automatic production activation is part of this protocol.

## Practical preflight

The new module performs no asset GETs, training, authentication or package installation. Its HTTP client permits only public CMR collection/granule JSON endpoints, disables environment/netrc authentication, clears cookies and refuses redirects. Requests, bytes, pagination and date range are bounded. It returns exit code 2 for incomplete metadata inventories and writes the reason; this is not a model failure.

On the existing server, using the staged standalone research module:

```sh
/opt/lst-pilot/.venv/bin/python /opt/lst-pilot/src/lst_pilot/night_inventory.py \
  --root /opt/lst-pilot \
  --start 2023-01-01 --end 2023-12-31 \
  --pilots greater_london sioux_falls \
  --products ecostress_v2 ecostress_v3 aster_v4 \
  --max-requests 40 --max-metadata-mib 32 \
  --output runs/option_b_inventory_2023_night_20260909
```

Add `--plan-only` for zero network requests. Add `--day-night NIGHT DAY` in a separate output directory to stage daytime coverage. Use a new output directory for every run: frozen artifacts are not overwritten. Outputs are `plan.json`, `inventory.json`, normalized `granules.json` and `calendar_candidates.json`; the HTTP audit stores response hashes and request parameters without credentials or signed asset URLs. [NASA CMR API documentation](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
