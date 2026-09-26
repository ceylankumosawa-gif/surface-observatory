# Native satellite baseline: prospective comparison

This is an exploratory model comparison at the actual VIIRS footprint. It is not a release qualification or a recipe for treating coarse observations as 100 m truth. The experiment will be frozen only after the feature join is checked. No fits are authorized by this draft alone.

## Question and controls

Does broader geographic, seasonal and nighttime supervision improve an hourly weather baseline at native satellite support? Use the same 31 weather/solar/static predictor names and the same fixed seven-leaf, 100-iteration HGB settings as `ground_native_trial.py`. Compare two residual targets, LST minus footprint-average ERA5-Land air and LST minus footprint-average skin. Reconstruct each with its own baseline. Keep unlearned air and skin controls on identical evaluation rows. Do not choose features, tree capacity, quality rules or dates from resulting errors.

The feature definitions change spatial support: weather is a piecewise-constant native-grid approximation, aggregated by projected footprint intersections; static information is an area-weighted approximation over intersecting 100 m cells. Neither is a true continuous surface integral. Midpoint time is an acquisition-interval proxy. Retain start/end feature evaluations separately, with instantaneous versus preceding-hour radiation conventions explicit. This support change means that equality of column names does not establish direct equivalence to point-ground training.

## Rows and admission

Use the fixed 384 requested pilot/date/phase groups, including missing sources, astronomical absences, failed joins and zero quality support. Native source rows remain immutable. A separate trial field admits only rows passing the unchanged native QA, view angle, footprint and expanded-support gates plus the declared feature-valid-area requirements. Preserve the original `training_eligible=false` fields as collection provenance rather than rewriting them.

Fit only `context_fit_eligible` rows, respecting the existing London/Sioux reserved-geography buffers. Cabauw remains reference-only. Evaluate all quality-admitted predictor-complete native rows in each held group, and disclose fitting eligibility separately. Do not let the target, prediction error or an uncertainty estimate select a source, replacement date or feature request. Feature incompleteness must remain in requested-versus-scored counts.

Require unique physical IDs (source stem plus absolute native row/column) before any fit; unexpected duplicated IDs stop assembly for a declared resolution, without silently dropping a pilot's record. A repeated granule cannot enter training through another pilot while it is held out for evaluation. Distinct source records retain their source intervals; a granule is indivisible. Measure geometric overlap within acquisitions and disclose it. Legitimate distinct native observations may overlap; no arbitrary overlap-area cutoff selects labels or changes their weights.

## Fixed tests and weighting

1. Leave one whole pilot out. Exclude every granule occurring in that held pilot from the other pilots' fitting rows. Cabauw is never part of the fitting population.
2. Leave one whole global calendar month out: January, April, July and October 2021. Exclude a complete granule from training when either interval endpoint belongs to the held month. These four months do not use the older modulo-three fold rule.
3. An explicit Cabauw reference-only test. Exclude Cabauw and every granule occurring there from fitting, including that granule's London or other pilot rows. This is distinct from applying a full model that may have seen the same source.
4. One full research fit per arm, for saved exploratory use only. Full-fit predictions are never reported as held-out accuracy.

For training and balanced scoring, allocate equal weight to each supported pilot/phase, then equally to UTC dates within that group, then equally to distinct acquisitions within that date. Within an acquisition normalize each observation's native footprint area by the sum of those observation areas. This is observation-support weighting, not a geographic union-area integral; overlapping and adjacent observations remain correlated. Do not sum native areas into a unique-coverage claim or treat cell counts as independent replication. Recompute all weights within each fitting split, scaling their sum to the number of fit rows for the fixed HGB regularization. Also report ordinary native-cell MAE, signed bias, and fractions above 3, 5 and 7 °C.

Report every requested pilot/phase and month, including empty groups. A pooled mean below 3 °C cannot override a regional failure or missing season. Minimum date/season counts are release evidence requirements; insufficient counts do not prevent an explicitly exploratory comparison on the observations that actually exist. Whole-pilot and whole-month tests are separate, weaker than a joint unseen-region-and-time test.

## Interpretation and stopping

Do not promote either arm automatically. Require the source collection and independent values/QA audit, the feature join and its missingness/support report, and a frozen plan binding those artifacts before fitting. Bound the first comparison to at most 34 fits, two CPU threads, 4 GiB and 15 minutes on Hetzner, with network disabled. Record any incomplete fits without restarting a budget or changing the recipe.

Any subsequent ground comparison is an external-support diagnostic with its own point-versus-footprint and emissivity qualifications. Clear-sky infrared evaluation does not demonstrate cloudy-surface, twilight or every-hour accuracy. Fine spatial contrast and complete worldwide serving remain separate work.
