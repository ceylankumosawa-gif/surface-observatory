# Local solar season on the expanded training population

This prospective comparison begins only after the paired geography comparison
passes its independent audit. It uses that experiment's expanded arm as the
fixed control. Replace only the two day-of-year fields with the independently
checked footprint means of daily potential sunlight and its signed daily change.
Keep the other 29 predictors, air-residual target, weights, seed, model capacity,
all source masks and every fitting/held identity unchanged. Do not combine this
test with the four thermal/weather contrasts or terrain corrections.

The checked solar geometry sidecars cover 180,253 prepared footprints across
the original, first-eight and second-twenty populations. Join by exact native
identity and region, checking footprint area and interval midpoint. Every
previously complete row must retain complete support. Solar geometry cannot
make a previously missing weather row eligible. The approximate continuous
365.2425-day Spencer recipe remains exactly as documented in
`SOLAR_SEASON_PREPARATION.md`; no parameter is chosen using observed errors.

At most 45 fitting recipes are inherited from the expanded arm, including
whole-area, whole-month, Cabauw reference and full-data recipes. Reuse is allowed
only for identical ordered memberships, weights, targets, transformed values
and model settings within this candidate. Baseline predictions are reused from
the audited geography experiment. Reserved thermal observations and 2025 remain
unopened; all 171 reserved physical passes stay excluded from fitting.

Report paired ordinary/balanced MAE, bias and errors above 3/5/7°C, preserving
all 96 regional day/night outcomes and all 960 requested cases. Keep original
and new areas separate. Also retain fitting error to distinguish easier fitting
from geographic transfer. No source downloads or model deployment occur here.
Execution is offline, two CPU threads, six GiB and at most 1,800 seconds. An
independent replay precedes interpretation. Native-footprint, primarily clear-sky
validation cannot establish 100 m or all-weather accuracy.
