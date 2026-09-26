# Global release accuracy and one bounded remedy experiment

Frozen before the new coefficient fit on 18 September 2026. No existing model or study is changed. All computation runs on the existing Hetzner VM with at most four CPU threads and a 6 GiB process memory ceiling. No new download and no 2025 thermal label access is required.

## Acceptance means coverage and accuracy together

The user target is mean absolute error no greater than 3°C consistently across regions and day/night. An overall average does not satisfy it. Freeze the source-screened reference identities, all intended region × solar-phase × output-resolution groups and actual fitting memberships before scoring a candidate. Evaluate ordinary pixel MAE and an equal-UTC-date MAE separately within every required group. Both must be ≤3°C. Also require both MAEs ≤3°C in each calendar quarter; quarters distinguish seasonal performance, rather than claiming each individual prediction is within 3°C.

Each required group needs at least 100 reference observations, 12 independent UTC dates, and at least three dates in every calendar quarter. These are minimum research coverage criteria, not a theorem establishing worldwide accuracy. All reference rows must have predictions. Missing groups, insufficient dates, unavailable feature rows and omitted predictions remain visible failures; a candidate cannot pass by masking difficult cells or dropping a region. Finite label/source QA is fixed upstream, independently of errors. Report >5°C and >7°C error frequencies and signed bias alongside MAE.

Date tests exclude entire global calendar months from each estimator and correction fitting population. Geographic tests exclude the entire evaluated region from all fitting. Verify both against actual sample/date/region membership, not a caller's split name. Spatial-only checks are useful diagnostics but cannot substitute for these independent tests. Existing 2021–22 OOF and previously inspected 2023–24 data are repeated diagnostics. A release qualification needs a subsequently locked candidate, new independent region/date coverage, and fresh confirmation whose source and novelty are audited. A finite twelve-pilot panel cannot establish all-region global coverage, even if every observed group passes. The evaluator therefore never changes `global_target_met` to true on the basis of that panel.

The initial machine-readable status will list all twelve existing pilots × day/night at 100 m, including missing groups, and a separate whole-region Cabauw check. Other resolutions require their own spatially matched reference and coverage registry; resampling an image does not inherit a 100 m accuracy claim. Historic clear-sky satellite labels also do not establish all-weather performance or present-time forecast accuracy.

## Why another experiment is warranted

The completed winter study shows F original OOF balanced MAE 3.696°C, earlier six-night error 6.63°C and newer twelve-night error 8.09°C. Extra dates, reweighting, generic/physical residuals, aligned feature HGBs and station adjustment have not qualified. Raw ERA5-Land skin sometimes helps difficult winter nights but is much too cold in hot desert daytime. A source-aware convex blend tests that complementary information directly while preserving F where support is absent. No prior reviewed protocol implements this exact phase × snow-state convex F/skin correction.

## Exactly one new candidate: phase/snow skin blend

Use the unchanged Stage A 44,172-row cohort and its three global calendar-month folds. F is the saved `full/F.joblib` reference (SHA `c955f3a69e393eef29dd95e291d751b73e845a43e7ab2067289c6f1cc394f447`). No HGB is refitted. Reuse its saved outer and six inner held-month predictions, verifying their frozen hashes, sample alignment and excluded months.

Define four fixed strata from existing solar phase and independently sourced time-aligned ERA5-Land SWE: day/night × snow/non-snow, with snow = SWE ≥0.001 m water equivalent. Never use observed LST, city ID, residual or source-sensor identity to define routing. Let S be aligned ERA5-Land skin temperature, F the saved out-of-fit prediction and D=S−F. In each stratum, normalize the original study's date/acquisition/surface-balanced training weights. Define N_eff = 1/Σ(date weight mass²), using global UTC dates. Fit

`lambda = clip[N_eff * Σ(w D (Y−F)) / (N_eff * Σ(w D²) + 20), 0, 1]`.

At least six distinct training UTC dates and two calendar months must support a stratum; otherwise lambda=0 exactly. There is no intercept, no city-specific adjustment, no temperature clipping and no hyperparameter search. The candidate prediction is `F + lambda * (S−F)`. The correction lies between F and S, but that bound does not guarantee a prediction-error bound.

For each outer held fold, estimate coefficients only on the matching saved inner held-month F predictions from its other two folds. Neither the outer held months nor their labels enter coefficient fitting. Estimate the final coefficients from all original outer-OOF F predictions, then freeze the coefficients and source/model hashes before applying them to repeated 2023–24 checks. Preserve all coefficients, support, numerator/denominator, clipped and raw estimates, fit weights and row hashes. Keep every original eligible row and unchanged F prediction. Known source-unavailable rows are retained in the broader acceptance reference with missing predictions.

Selection remains exploratory: compare F and the single candidate on exactly matched original OOF rows using the previous balanced metrics plus the new equal-date/pixel acceptance view. Qualification requires overall balanced MAE improvement ≥0.10°C; ordinary MAE worsening ≤0.10°C; every ≥6-date region/phase group MAE worsening ≤0.20°C; overall and those groups' weighted >5/>7 tails worsening ≤0.02; and acquisition-centred contrast MAE worsening ≤0.10°C. Show sparse groups too. No qualification if any of these fail. The 3°C release evaluator is stricter and separate; satisfying relative-improvement checks does not establish the release target.

All repeated external cohorts are reported regardless of selection outcome, after coefficient freeze. No second candidate is chosen from their results and no reserved newer thermal cohort is opened this turn. The prior six and additional twelve nights remain out-of-fit stress checks where compatible saved weather already exists; absence of compatible weather is reported rather than causing a source fetch. All model artifacts remain research-only and cannot alter the website's frozen F.

Aligned weather uses a future valid-time endpoint for non-exact satellite hours and is a retrospective diagnostic. Exact-hour application uses that hour's field. A present-time forecasting adapter requires a separate source and validation contract; this experiment supplies neither.
