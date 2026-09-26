# Research status

**Closeout: 26 September 2026.** The current website is retained as an
experimental release. The live global **F40** model remains unchanged. Research
comparisons below are completed evidence, not deployed replacements or a claim
that the ≤3°C regional day/night target has been met. This release does not
schedule further collection or fitting.

## Reading the results

The following experiments compare predictions with **native satellite
footprints** from January, April, July and October 2021. These footprints are
not 100 m ground truth, may overlap, and mainly represent conditions admitted
by the satellite quality rules. Missing observations remain missing.

Balanced mean absolute error (MAE) gives equal weight to area/day–night groups,
then dates and acquisitions, with footprint-area weights inside each
acquisition. This prevents a large image from dominating the result; it does
not make adjacent pixels or dates independent weather events. Lower is better.
Whole-area and whole-month holdouts test different forms of transfer. Cabauw
is always a separate reference, excluded from fitting.

## Fixed changes on the original cohort

These six comparisons use API-delivered weather and start from a 31-input
research model. The illumination variant has 34 inputs. They are separate from
both F40 and the later direct-weather geographic comparison.

| Research variant | Whole-area balanced MAE | Whole-month balanced MAE |
|---|---:|---:|
| Small air-residual baseline | 4.68°C | 4.32°C |
| Terrain-height representation | 4.40°C | 4.26°C |
| Larger model | 4.92°C | 4.63°C |
| Terrain illumination | 4.85°C | 4.26°C |
| Absolute-error training objective | 4.58°C | 4.40°C |
| Four thermal/weather contrasts | 4.58°C | 4.51°C |

None establishes the target across regions. Nineteen of 24 original
area/day–night groups have observations; five remain unavailable. The
[public evidence snapshot](https://degenerate.energy/research/evidence-v2.json)
and its linked CSVs retain the individual groups, large-error fractions and
independent audit identifiers.

## Learning from additional places

A deterministic source-only catalogue selected 28 new development areas and
eight reserved areas. The paired trial changed only training geography, using
the same small 31-input air-residual recipe for both arms. Both used the same
direct ERA5 weather preparation, chosen before fitting; their air baseline is
ERA5-Land air temperature, not reported station air.

| Held test | Evaluation population | Original-only training | Expanded training |
|---|---|---:|---:|
| Whole area | Original development areas | 4.672°C | 3.885°C |
| Whole area | New development areas | 3.623°C | 3.395°C |
| Whole month | Original development areas | 4.289°C | 3.917°C |
| Whole month | New development areas | 3.632°C | 3.379°C |
| Separate reference | Cabauw | 2.236°C | 2.073°C |

All values are balanced MAE on the same paired observations. Expansion helps
these aggregates, but regional changes are mixed: 12 of 17 observed original
area/phase groups and 23 of 44 new groups improve. New-area daytime ordinary
MAE worsens from 3.681 to 3.830°C. The target remains unmet.

The checked cohort retains 1,819,067 source identities and all 960 requested
cases. It has 177,226 complete observations and 141,517 fitting-eligible rows
before split-specific exclusions. All 96 area/day–night outcomes remain:
63 have observations, 17 development groups lack paired support and 16 belong
to the unopened reserved areas. All 171 reserved physical acquisitions are
excluded from fitting across revisions and shared geographic coverage.

## Solar-season representation

This subsequent paired test replaces only the two calendar inputs with
footprint-mean daily potential sunlight at the top of the atmosphere and its
daily change. The other 29 inputs, expanded training population, targets,
weights, model settings and held observations stay fixed. These analytic solar
inputs are not measured surface radiation or a cloud correction.

| Held test | Evaluation population | Expanded baseline | Solar-season candidate |
|---|---|---:|---:|
| Whole area | Original development areas | 3.885°C | 3.829°C |
| Whole area | New development areas | 3.395°C | 3.374°C |
| Whole month | Original development areas | 3.917°C | 3.956°C |
| Whole month | New development areas | 3.379°C | 3.414°C |
| Separate reference | Cabauw | 2.073°C | 2.003°C |

The independent audit passed for all 45 recipes, 41 distinct fitted models,
all 3,530,763 prediction-ledger rows and 25,560 metric rows. Area holdout
balanced errors decrease slightly, but month holdout errors worsen. New-area
whole-area ordinary MAE also increases, from 3.232 to 3.247°C. This mixed result
does not establish a generally better model or justify an accuracy claim.
The live model was not changed.

## Remaining limits

- Global accuracy ≤3°C, cloudy-condition performance and genuine 100 m
  accuracy are unvalidated. Satellite quality flags do not independently prove
  that every admitted observation is cloud-free.
- The selected new catalogue has 23 named climate classes but lacks
  built-dominated areas, snow/ice and Antarctica; twelve proposed slots remain
  unfilled. Its eight reserved areas and the reserved 2025 thermal targets
  remain unopened.
- Polar/dateline input coverage and complete worldwide hourly production are
  unfinished. Current-date forecast/reanalysis/station mixtures have different
  semantics from the historical research inputs.
- The 31-input research model lacks time-varying albedo and vegetation-state
  inputs. A follow-up source study was prepared but was not completed; it is
  not part of this release. F40's optical albedo proxy is a separate input.

These are release limitations, not promises of additional experiments.

## Audit identifiers

The scientific raw data, fitted research models and full replay artifacts are
held separately from the source checkout. The following SHA-256 identifiers
pin the checked aggregate evidence used here; they are not publisher-provided
dataset checksums.

| Receipt | SHA-256 |
|---|---|
| Geographic paired-trial audit | `b2ebe0aac1bd7686dc119c14c8e4c7a87832a4aff5ed3a39f78000205e3536a1` |
| Solar-season paired-trial audit | `58924ef0aa94583f84751701c80863b31e827c2067ddfea7bcbbc05af3bc5794` |
| Independently replayed solar-season metrics | `bcc7df8bd01728d2101446717b7ac60f38833d3ae67da178fa6f0bcd5da3c13e` |
