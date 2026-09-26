# Exploratory hourly weather baseline on the fixed ground cohort

This is a new reference-support experiment, not another fit of the old satellite labels. The population is the already collected seven SURFRAD sites × four fixed 2021 dates × 24 exact UTC hours: 672 requested observations. Their ground fluxes and some previous comparisons have already been inspected. This cohort cannot provide fresh release confirmation or prove worldwide, urban or 100 m accuracy.

## Prepare sources before fitting

Use the original quality-screened broadband radiometric temperature at the documented facility: the one-minute average ending at each requested hour, with emissivity fixed at 0.97. Preserve the existing 0.95 and 0.99 calculations as sensitivity comparisons. No measured ground flux, ground air temperature, ground humidity, observed LST or residual is an inference feature. All four dates and all weather conditions remain; no error-based source choice, station relocation or hour substitution.

The separate source plan fixes a common weather/static feature contract before extraction. It contains native ERA5-Land air, skin, shallow-soil temperature, soil moisture and true snow water equivalent; historical ERA5 air, humidity/dewpoint, wind, pressure, cloud, sunlight, longwave and precipitation/history; continuous solar/time geometry; and terrain/static land cover. No satellite optical composite, nearby station, climate-category whitelist or daylight gate is required. No site name, date identifier, source ID, longitude or latitude is supplied directly to the estimator. Static terrain and land cover refer to the declared preparation footprint, not a claimed measured radiometer footprint.

Freeze the exact feature order, units, transformations and input bindings in the run plan before fitting. Require a separate recursive source proof binding the feature output and completed preparation to its used raw weather chunks, SWE GRIBs and static caches; a top-level file list alone does not close that provenance. Raw source values must have passed their source contracts. Missing raw predictors are retained in the coverage ledger; this first comparison fits/scores only the common finite feature population and cannot pass coverage while any required row is missing. Predict every feature-supported held row even if its ground target is unavailable; keep prediction coverage separate from reference coverage. No learned imputation. ERA5-Land skin and air controls also retain their own availability separately.

## Two fixed targets, the same small estimator

Use `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=100, max_leaf_nodes=7, max_depth=None, min_samples_leaf=20, l2_regularization=10.0, max_bins=255, early_stopping=False, random_state=20260919)`. Every predictor is numeric. Do not search settings, clip temperatures, add random variation or refit after inspecting held errors.

Two arms share exactly the same features, rows, weights and settings:

1. Air residual: fit ground temperature minus native ERA5-Land air; add that same air back at prediction.
2. Skin residual: fit ground temperature minus native ERA5-Land skin; add that same skin back at prediction.

Raw native air and raw skin are unlearned controls on identical common rows. Previously frozen F is outside this new comparison because it is not available for most station/hours. Compare its smaller fixed-cell population separately.

Within each training split, divide weight equally across sites, then dates within each site, then represented solar phases within each site/date, then observations within a site/date/phase. Multiply normalized weights by the training row count. Recompute using only that split. Solar phases are day (elevation ≥10°), night (≤−6°), and twilight; all three remain eligible, with no threshold on the predicted temperature.

## Hold out whole sites and whole dates

Perform seven leave-one-site-out fits and four leave-one-global-date-out fits for each arm, plus one full research fit per arm: **24 fits total**. Keep these two types of test separate. The latter excludes the entire corresponding global month represented by this cohort. Neither independently establishes joint unseen-site/unseen-date performance. There is no random row split or inner tuning.

Save exact fitting memberships, normalized weights, baselines, residual targets, predictor hashes, model settings and held predictions. Do not fit a fold with fewer than 80 complete training rows, fewer than two sites or fewer than two dates; record the unsupported fold instead. Preserve every requested row in each evaluation mode, including rows without a supported prediction.

For each mode and each source, report coverage, signed bias, MAE, and fractions with absolute difference >3/5/7°C, separately as ordinary and equal-site/date/phase-weighted summaries. Show every site × solar phase and each of the four dates, including empty groups. Also calculate the same predictions' differences under both alternate emissivity assumptions, with explicitly finite paired counts for each assumption. Do not tune emissivity to make a method look better.

No automatic model promotion follows this exploratory comparison. A useful candidate must first improve on raw skin on matched evidence without hiding regional, seasonal or twilight harm, then face broader satellite/ground coverage and fresh independent confirmation. The existing website/model and proof-bound release status remain unchanged.

All work runs on Hetzner. Training is cache-only, with two CPU threads, 2 GiB and a five-minute bound. No new thermal year, no download during fitting, no production edit. Any schema or execution repair preserves the failed attempt and requires a new version with the same scientific population and settings.
