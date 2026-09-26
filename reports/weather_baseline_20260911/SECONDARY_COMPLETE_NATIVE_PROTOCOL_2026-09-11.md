# Secondary comparison conditional on complete native weather

This is a new, explicitly secondary experiment. The original full-cohort primary in `PROTOCOL_2026-09-11.md` is stopped before fitting: the four-field source audit found actual NaN values for144 of44,316 training samples (74CapeTown and70Utqiaġvik), across five coastal coarse cells. Full-cohort inference cannot be claimed from this secondary experiment.

The same audit also found1,115Gobabeb topsoil-moisture values at approximately−1.588e−20m³/m³. Fix a numerical-domain rule before any model outcomes: preserve the raw values, map only finite negative moisture in[-1e−12,0) tozero, and record per-row repair flags and counts. Retain positive values unchanged. Larger negative values remain invalid. Do not fill any true missing value or substitute a neighboring land cell. This is a rounding repair far below useful moisture precision, not target-dependent correction.

## Cohort freeze

Join all five official physical weather fields to the unchanged sample identities and exact native cell/hour. The inclusion mask is determined only by finite, valid weather availability, the fixed numerical-domain rule, and exact source alignment. Require skin/air/topsoil temperatures, true GRIB141 water-equivalent snow and topsoil volumetric moisture. Freeze ordered included/excluded IDs, row hashes, source hashes and support inventory before fitting or examining any resulting errors. The four-field audit suggests44,172 retained samples, but final membership must wait for the fifth field. No measured LST value, model residual or extreme-temperature outcome may determine membership.

Before scoring, export inclusion counts/rates by pilot, phase, UTC date, acquisition, climate and native weather cell. List original pilot/phase groups with at least six UTC dates that lose that support after masking. Disappearance is not a regional guard pass. Retain a geographic map/table of excluded native cells.

## Seven fixed arms on identical rows

Refit the existing F architecture and the existing ExtraTrees reference on exactly the secondary cohort. Recompute the original balancing recipe independently within each training fold and full fit, identically for every learned arm. Preserve the original three global calendar-month folds and all frozen hyperparameters and seeds. The references must be labeled as refitted on the complete-native cohort; saved full-cohort F and the prior saved blend cannot substitute for them.

The seven arms are:

1. Refit F on the complete-native cohort.
2. Fixed0.75refitF+0.25refitExtraTrees reference.
3. Raw ERA5-Land skin temperature.
4. Station-adjusted land baseline `B=A+(S−T)` using the unchanged stored air-temperature inputA.
5. F architecture with the four added land predictors; target`Y−A`, reconstruct`A+f(X)`.
6. The identical architecture/predictors over land baselineB; target`Y−B`, reconstruct`B+g(X)`.
7. Fixed0.75refitF+0.25land-residual prediction.

The four new predictors and the distinction between targets follow the original protocol. Every OOF blend uses only held-fold predictions. There is no climate-specific model routing, new hyperparameter search, recalibration or model promotion.

Apply all original numerical selection gates unchanged, against the refitted F reference: balanced MAE improvement≥0.10°C, ordinary MAE worsening≤0.10°C, each supported pilot/phase MAE worsening≤0.20°C, overall and supported-group>5°C and>7°C tail-share worsening≤0.02, and overall acquisition-centered contrast MAE worsening≤0.10°C. Supported means≥6distinctUTCdates. Preserve the original0.05°C tie tolerance and fixed arm preference. None qualifying is an acceptable result. Do not weaken gates after scores.

## Evaluation ordering and reporting

Freeze training-period selection and full fits before decoding evaluation labels. Only then extract evaluation weather. Within each original evaluation family, independently freeze the same five-field availability mask and its source/support inventory before calculating any new errors. Evaluate all seven arms on identical retained rows and weights. Publish denominators and missing/excluded counts for every family. Previously inspected2023/2024data remain exploratory regression checks, not untouched confirmation or a new tuning set. Reserved2025thermal labels remain unopened.

Saved original full-cohort F predictions may be reported separately for retained and excluded rows to describe context. They must not be pooled with secondary model errors, used in place of the refitted control or used to revise the mask. New secondary performance estimates are conditional on native weather availability. They do not establish worldwide coverage, operational forecast availability, all-weather100m truth, or accuracy at the excluded coastal cells.

Report all original metrics and key London/Sioux acquisition, season and day/night breakdowns, plus the independently verified ICON diagnostic. Preserve the original stopped-primary record, secondary source/selection/full-fit freezes, software hashes and independent audit. Production stays unchanged.
