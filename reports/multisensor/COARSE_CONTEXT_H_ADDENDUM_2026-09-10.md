# Fixed native-context alternative H

Declared before opening the frozen 2023 MODIS/VIIRS context candidates or inspecting their evaluation errors. This extends the root multisensor protocol and the fixed F/G specification. No parameter search, model promotion, new 2024 context or 2025 label access is included.

## Fixed estimator and fitting observations

H is an alternative residual correction to the same F model. It uses **exactly the same complete 2021–2022 fitting rows, F out-of-fold predictions, residual targets and balanced weights as G**. The three folds remain whole global UTC calendar-month groups: `((year−2021)×12+month−1) mod 3`. No held-out geography or 2023/2024/2025 fine label enters any correction fit.

H adds exactly three predictors to G's fixed core design:

1. `coarse_lst_c − air_temperature_c`, using the **current target-time station air input** already supplied to F. This is a prior coarse surface temperature relative to current air, not a simultaneous or source-time surface–air anomaly.
2. `coarse_age_hours`, the target time minus the granule start: the oldest possible observation age within the source interval.
3. An explicit binary `context_missing` indicator.

Sensor/product names, native cell identifiers, acquisition identities, coordinates, uncertainty estimates and QA flags remain provenance and eligibility fields; they are not predictors. Native coarse pixels are never treated as independent 100 m labels.

Numeric core means/scales and effective global UTC-date weight normalization are unchanged from G. Standardize the two additional numeric context fields using only eligible observed 2021–2022 context and its fitting weights; fill missing standardized values with zero and set the missingness indicator to one. Retain every fitting row, including missing-context rows. Keep the intercept and all coefficients penalized. Ridge alpha remains **20**, and weights sum to the same effective global UTC-date count as G. Thus this is a feature addition with the same residual-fitting cohort, rather than a comparison confounded by changing the fitting observations.

The resulting H adjustment to F is `clip(0.5 × fitted_residual, −3°C, +3°C)`. It replaces G's adjustment where eligible native context exists and the relevant broad-climate/target-phase group has at least **six covered global UTC dates, three UTC year-months and two OOF groups** in 2021–2022. Otherwise **H equals G exactly**. No second correction is added on top of G. The ±3°C limit constrains the adjustment, not prediction error.

## Native context admission and immutable joins

Only the frozen MOD21.061/VNP21.002 native swath readers are admitted. Current screening requires good mandatory/data/cloud QA, uncertainty class 2 or 3, explicit positive LST error no greater than 1.5 K, view angle at most 30°, native land support, full native geometry and excluded scan seams. These product-specific gates are recorded with source and native-table hashes; they are not guarantees of accuracy.

The **entire granule interval** must precede or end at the target: `end ≤ target`, with `start ≥ target−24 hours` and `start ≤ end`. Validate the reported oldest/youngest age against both exact timestamps. Preserve the native footprint and the uncertainty-expanded support geometry. The sample location must lie within the native footprint in the fixed pilot CRS. Fitting context must clear the permanent held-out geography and 1 km buffers using its whole expanded native support. Evaluation may use admissible context in reserved geography without changing fitting support.

The join must preserve sample identities one-to-one, bind the exact target table and audited native source tables by SHA-256, and record whether it was prepared for fitting or evaluation. A missing observation remains missing; an invalid claimed match is rejected. No source-time ERA5 air feature is presently available or invented. Causality describes observation valid time; it is not a claim that these archived products were published in time for an operational forecast.

## Evaluation and reporting

Freeze H with F/G before evaluating any new 2023 context-assisted predictions. Report E/F/G/H on the same earlier evaluation rows and unchanged weights, keeping fresh 2023 fine-source observations and repeated-date diagnostics separate. Compare H with G both on all matched rows and on the subset where H's contextual correction is supported. Record that subset's actual dates and rows; do not infer an improvement from changed coverage.

Report context availability by product, target phase and source age, covered independent dates, native-cell/acquisition counts, mean and maximum correction, fallback rates, balanced and ordinary pixel MAE, signed bias, weighted and raw errors exceeding 5/7°C, and centered spatial contrast. New 2023 observations do not tune H or its eligibility thresholds. Empirical error radii continue to use only the earlier fixed 2023 H2 calibration set and are explicitly nonguaranteed.

No new 2024 context is collected or joined. In the repeated 2024 reference comparison, **H equals G by design through missing-context fallback**; this is not a contextual accuracy test. Preserve the 2025 blind boundary. The model remains research-only even if some averages improve.
