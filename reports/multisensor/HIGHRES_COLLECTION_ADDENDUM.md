# High-resolution collection addendum — 10 September 2026

This extends the root multisensor protocol for acquisition and source screening only. It does not authorize a production model change or establish hourly/all-weather accuracy. All collection runs on the existing Hetzner server.

## Engineering findings and source rules

The initial plan fixed 16 candidates using complete 2021 London/Sioux metadata shards. Three ECOSTRESS candidates failed before thermal access; 13 thermal bundles were read. One ECOSTRESS daytime acquisition passed the source screen: Sioux Falls, 2021-02-02 20:04:12.648 UTC. Its first immutable sampled table contains 360 cells (320 with native fitting support, 40 reserved spatial diagnostics).

ASTER AST_08 V004 is TES kinetic surface temperature. Read unsigned SKT digital numbers with a 0.1 K factor, retain both QA planes, and preserve its rotated affine grid until native masking. First-plane zero is meaningful good QA. All non-good and missing pixels receive the conservative 350 m native buffer before 100 m area aggregation; at least 90% valid area and full source footprint support are required. No ECOSTRESS error threshold is imposed on ASTER. No usable Kelvin uncertainty band has been established; its uncertainty remains null. [NASA ASTER tutorial](https://github.com/nasa/ASTER-Data-Resources/blob/main/python/tutorials/Exploring_AST_08_Surface_Kinetic_Temperature.ipynb), [ASTER QA specification](https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/ASTER%20QA%20Plan%20v2.0.pdf).

The lower four first-plane QA bits, assigned to clouds/adjacency in the specification, were zero for every nonfill SKT pixel in all ten engineering ASTER scenes. Four scenes nevertheless reported critical automatic QA failures. This sample therefore provides no positive evidence that those cloud bits actively detected clouds. Matched AST_L1T metadata exists, but provides no measured registration RMSE in these records; its automatic passed flag describes successful execution bounds. V004's L1T correction must not be attributed to AST_08 L2. ASTER stays outside 100 m fitting until independent cloud and registration checks establish admission. Keep its native observations for explicit aggregate comparisons with matched Terra cloud/LST context. [V004 guide](https://lpdaac.usgs.gov/documents/2265/ASTER_User_Guide_V4_pcP80n5.pdf).

ECOSTRESS retains the previously frozen native cloud/land/QC/view/uncertainty profile, positive matched Good/Best GEO, matching processing build, complete known-obstruction exclusion, full-cell support and 90% valid area. Candidate DAY requires at least 10° solar elevation at the pilot centre and corners; NIGHT requires at most −6°. Final per-cell solar checks remain in feature admission.

## Expanded ECOSTRESS queue, frozen before new 2023 thermal reads

The executable queue is `runs/multisensor_20260910_v1/ecostress_expanded_v2/plan.json`. It binds the completed daytime inventory, prior complete nighttime inventory, source code, root protocol, initial engineering attempts, permanent blocks and prior-inspected-date registry by SHA-256. Registry SHA: `2be509c11960eb23ddc0dd4d9c1e1e63c4b2c77bbfd417270847cab133125453`. The earlier v1 metadata-only plan was superseded before any collection to exclude alternate tiles of initial orbits and every initially qualifying pilot-date.

Ten queues cover London/Sioux fitting DAY, 2023 H1 DAY/NIGHT development and H2 DAY/NIGHT calibration. Targets are 12 additional distinct fitting dates per pilot and three fresh dates per evaluation queue. Targets describe desired source coverage, not a promised yield.

Only 2021–2022 DAY observations enter the new fitting queue. Every previously inspected/attempted 2023 pilot-date is excluded from fresh evaluation across sensors. Whole pilot UTC dates remain in their original half-year; no 2023 date can enter fitting. No new 2024/2025 thermal labels are collected. Cabauw is excluded.

Within month × three-hour local-solar-time strata, retain four deterministic acquisition ranks at most. Choose latest processing per scene/tile before quality screening and then greatest metadata overlap per orbit. Round across queues and calendar/hour strata; do not rank by temperatures, prediction errors or successful residuals. Initial engineering identities are excluded from repeat collection. A failed source check advances only within the frozen queue. One qualifying date counts once per queue; separate day/night observations on one date remain linked by date during calibration.

The extension stops at 200 candidate attempts or 2 GiB additional network transfer, including conservatively charged public metadata, whichever comes first. Each pilot/phase has a cumulative 20-minute soft processing ceiling, checked between candidates; one in-flight bundle may finish. Run in resumable batches with at most two CPU cores and 2 GiB memory per worker. Durable reservations survive interruption, and a completed result is committed before its in-flight marker is cleared. Cached assets require matching identities and hashes and do not create new independent observations.

## Row and support contract

Source screening and fitting support are separate. Every sampled row carries exact source time, product/version, acquisition and scene identity, CMR revision, fixed pilot row/column and coordinates, source and derived-raster hashes, `label_valid_fraction`, `source_screen_pass`, `native_fit_support_pass`, geolocation/cloud proof, and the QA policy. Thermal values remain targets, never source-selection predictors.

`native_fit_support_pass` requires valid source-supported cells outside permanent reserved blocks, their 1 km Euclidean buffers, and an additional conservative margin of two transformed native-pixel diagonals plus 10 m. The margin is documented for each source. This protects source contributions extending beyond a 100 m cell; it does not correct positional uncertainty. Evaluation labels may retain false fitting-support flags.

A fitting date qualifies for acquisition stopping only with at least 200 native-fit-safe cells across at least two nonreserved 10 km blocks containing at least 50 such cells each. Evaluation stopping requires 200 source-QA cells. Feature/station/land/optical admission can still reject these rows. Sample at most four fixed 128-cell tiles, one per quadrant, using valid support to select tiles and the existing fixed random seed to select up to 80 fitting-safe and 20 reserved cells per tile. Do not use thermal magnitudes in sampling.
