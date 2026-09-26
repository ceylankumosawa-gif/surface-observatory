# F-preserving weather alignment and air-source correction

The user authorized the recommended next experiment after reviewing the completed weather-baseline comparison. Keep F as the reference architecture. This study separates closer weather timing, a small learned air-source correction, and additional winter-night observations. All numerical processing and downloads run on Hetzner. Preserve previous runs, the live website/model, and unopened 2025 thermal labels.

## Stage A: fixed existing observations

Start from the same 44,316 original 2021–22 fitting observations. Freeze a single weather-only inclusion mask requiring valid floor and, when necessary, ceiling endpoints for all five ERA5-Land instantaneous fields: skin temperature, 2 m air temperature, 0–7 cm soil temperature/moisture, and actual GRIB141 snow water equivalent. Retain exact original sample identities and nearest native cells. Preserve raw endpoints, source hashes, normalization flags and interpolation fraction. Apply the prior fixed tiny-negative repair only to moisture/SWE in [-1e-12,0); true missingness and larger negatives remain invalid. No nearest-land replacement.

Interpolate each instantaneous field linearly at the actual satellite UTC timestamp. At an exact hour the fraction is zero and no second endpoint is required. A required ceiling endpoint may be January1 00:00 of the next year, solely for a target in the preceding December31 hour; it does not introduce new thermal target dates. This includes 2023 weather endpoints for 2022 fitting targets and, later, 2025 weather endpoints for 2024 evaluation targets. Do not interpolate accumulated radiation or precipitation. The existing air input A and all 40 original F predictors stay unchanged.

This is retrospective alignment using a later weather valid time. Exact-hour maps collapse to the recorded weather hour. Operational forecasting would require available forecast brackets from the same forecast initialization and separate validation; this experiment does not establish that capability.

All five arms use identical retained rows and the original balancing recipe. Refit F's exact architecture/configuration on that cohort; it is a matched research reference, not a production deployment. If membership equals the previous 44,172-row complete-native cohort, reproduce the previous F and floor-land predictions. If membership changes, disclose the change and refit every control. Freeze exclusions and support losses before calculating any new errors.

## Five fixed arms

1. **F**: original40 predictors, target Y−A, reconstructed as A+f(X).
2. **land_features_floor**: same architecture plus the previous four floor-hour physical predictors; diagnostic control.
3. **land_features_aligned**: same44 predictors/capacity with only those four physical fields aligned to acquisition time.
4. **F_air_shrink_floor**: F−gamma_phase·(A−T_floor).
5. **F_air_shrink_aligned**: F−gamma_phase·(A−T_aligned).

The four physical predictors are skin−native-air, soil−native-air, SWE and soil moisture. Preserve the previous HGB settings, seed, preprocessing and climate feature. No pilot/coordinate/sensor IDs become predictors and no climate-specific model routing is introduced. Raw floor/aligned S and fixed A+(S−T) may be reported separately as source diagnostics; they are not selection candidates.

Each air-shrink arm has two coefficients, one for the existing day phase and one for night. D=A−T contains station, background-model, grid/elevation and timing differences. Gamma is an empirical correction coefficient, not a physical measurement of F's internal sensitivity to air temperature. It can alter contrast across native-cell boundaries, which remains subject to the contrast check.

## Leakage-safe coefficient fitting

Keep the original three global calendar-month outer folds. Within each outer training subset, fit two additional F models: each uses one of the remaining original fold IDs and predicts the other. Thus every coefficient-training row has an F prediction from a model that excluded its entire calendar month, and the outer held months enter neither estimator nor coefficient fitting. Both timing variants use the same inner F predictions.

Within each phase, normalize the unchanged outer-training balance weights. Let N_eff=1/sum(date_mass²), where date_mass is the weight summed by global UTC date. Fit

gamma = clip[N_eff·sum(w·D·(F_innerOOF−Y)) / (N_eff·sum(w·D²)+20), 0, 1].

Require at least six distinct UTC dates in that phase; otherwise use the explicit unsupported fallback gamma=0. Preserve the numerator, denominator, date support, N_eff, unbounded estimate and clipped coefficient. No intercept, temperature clipping or additional learned residual is added. For the final full F model, fit gamma from the stitched original outer-OOF F predictions over the admitted fitting rows. Never estimate one full-data gamma and reuse it in outer-fold predictions.

## Selection and repeated evaluation

Only aligned land features and the two F air-shrink arms are eligible. Use the prior numerical checks against the matched F: balanced MAE gain≥0.10°C; ordinary MAE worsening≤0.10°C; every supported pilot/phase MAE worsening≤0.20°C; overall and supported-group >5°C/>7°C weighted error shares worsening≤0.02; acquisition-centered contrast MAE worsening≤0.10°C. Support means at least six UTC dates, and losing supported groups is not a pass. Within0.05°C of the best qualifying MAE, prefer aligned air shrink, floor air shrink, then aligned land features. Preserve a no-qualifying-remedy result.

Freeze training selection, coefficient evidence and full models before accessing the old/new2023/legacy2024 evaluation feature tables in this experiment. Previously examined dates remain repeated diagnostic checks and cannot choose or tune a candidate. Freeze each family's common endpoint-availability mask before new scoring. Report all five arms, ordinary and balanced errors/tails, dates, missingness, London seasonal cases and Sioux night cases, including adverse outcomes. No automatic production promotion.

## Stage B: targeted winter-night augmentation

Metadata discovery may run alongside Stage A. Select genuinely new acquisition dates using geographic/time metadata and independently available weather, excluding previous acquisitions/attempts before opening new thermal values. Keep 2021–22 fitting candidates separate from unopened2023–24 candidate labels. Use the existing thermal QA, geometry, station and feature rules; weather ranking must not depend on observed LST or residuals. Freeze actual queues, source/request/byte limits and date splits before thermal retrieval.

Before training on new observations, write a separate augmentation addendum using the real weather-only candidate inventory. It must compare unaugmented and augmented fits on identical original held-month evaluation IDs, exclude all outer held months from the augmented training rows, and report new-observation coverage separately. Do not silently change years, migrate known evaluation labels into fitting, or treat more neighboring pixels as independent dates. Failure to obtain eligible new dates must be reported rather than replaced with duplicates.

## Evidence and resource bounds

Reuse cached source bytes first and bind them by hashes. Freeze any additional live weather request inventory before retrieval; review estimates against1GiB additional weather bytes/2000 requests per cohort and the existing3-job CDS concurrency cap. Initial night metadata discovery is capped120 requests/64MiB/20minutes; the first proposed fitting thermal batch is at most24 granules/512MiB, subject to the actual frozen queue. Preserve request/byte charges across any documented continuation.

Independently audit source/time identities, inclusion masks, inner/outer folds, coefficient reconstruction, targets/weights, held predictions and all reported metrics. Update the existing canonical interactive report and executed reproducibility companion after the study is complete. Explicitly distinguish unchanged production state from research findings and retain the previous experiment as a comparison history.
