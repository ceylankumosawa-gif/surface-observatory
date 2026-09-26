# ERA5-Land delivery correction before model fitting

The user's Copernicus connection was verified on Hetzner. The authenticated ARCO metadata preflight found that its snow store supplies physical snow depth (GRIB3066), not water-equivalent snow (GRIB141). The existing protocol requires water equivalent. No substitution, feature change or outcome-based model adjustment is permitted.

The same hourly ERA5-Land catalogue supplies GRIB141 through its standard CDS service. Retrieve that field at precisely the original requested 0.1-degree cells and UTC floor hours. Keep the other four physical inputs from their verified ARCO stores. A merged native-cell/time table supplies the original five-field contract, with separate source proofs for each delivery route. No weather selection depends on measured LST values.

Original plans, source snapshots, stopped preflights and charged requests are preserved. A new plan records each source-format correction, carried transfer/request charges and a separately timed continuation. The original fitting request contains 44,316 sample rows, mapping to 1,491 unique pilot/cell/hour keys and 213 UTC hours in 2021–22. No evaluation weather extraction or new evaluation-label decoding is authorized before training selection is frozen.

The standard CDS plan groups keys into 131 pilot/calendar-month rectangles. It requests only the necessary days and hours; the API's day-by-hour Cartesian extras and rectangular spatial extras are recorded. All raw GRIB headers must match the full requested identity set. The first request also includes skin and 2m air temperatures for a small cross-delivery comparison declared before any weather values were decoded.

Initial standard-CDS limits are 64MiB transferred, 8MiB per result, 1,500 API requests, at most three queued/running jobs and sequential result downloads. One continuation is limited to 30 minutes of elapsed time; queued jobs and charges survive continuation. This covers exceptionally small output: approximately114,008 bytes of unpacked requested scalar values, before GRIB headers. The separate ARCO limit remains1GiB and its retained ledger accounts for previous metadata requests. These are retrieval limits, not a request for additional hardware or storage.

Preserve GRIB141, instantaneous step type, consolidated expver0001, water-equivalent units, decoded source coordinates and valid times, raw-file hashes, frozen request hashes and decoder source hashes. Zero snow is valid. Missing values remain missing and stop the protocol's complete-cohort arms; there is no nearest-land fallback or zero filling. Historical packaging metadata can differ across delivery services; do not claim identical binary releases without evidence.

The seven comparison arms, training rows, month folds, weights, targets, fitting settings, selection gates and evaluation ordering in `PROTOCOL_2026-09-11.md` remain unchanged. These delivery corrections do not authorize production deployment or opening reserved 2025 thermal labels.

Sources: [ERA5-Land documentation](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation), [standard catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download), [ARCO access](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data).
