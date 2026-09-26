# Geographic data comparison

This experiment compares the original training areas with the same areas plus
the 28 new development areas. Both arms retain the original 31 inputs, air
temperature residual target, small squared-error model, seed and weighting
hierarchy. No terrain-height correction, added illumination inputs, larger
model, alternate loss or label-quality relaxation is bundled into this test.

Both arms predict identical held observations. Whole-area tests exclude every
selected physical satellite pass for the held area, including acquisitions
with no retained geometry. Month tests exclude passes touching either endpoint
of the held month. Cabauw remains reference-only. All 171 physical acquisitions
associated with the eight reserved areas are excluded from every fitting
population across processing revisions. The reserved thermal observations stay
unopened. Every one of the 960 requested groups remains in the coverage record.

There are 45 split pairs and at most 90 fitting recipes. Only recipes with the
same ordered identities, target residuals, weights, feature values and model
parameters can reuse a fitted artifact within this experiment. Previous model
artifacts are not reused. A separately reviewed cohort and independent check
must precede the frozen fit plan. Missing input rows remain visible and cannot
be filled using observed error. No acquisition or source replacement occurs in
this runner.

Reports include ordinary MAE, equal-area/phase/date/acquisition-balanced MAE,
signed bias and fractions above 3, 5 and 7 degrees. Comparisons use the same
paired support, with separate own-available figures for transparency. Every
region/phase and requested date/phase, including unobserved reserved regions,
is retained; missing error is not zero error. Original and new geographic
groups are reported separately so a change in group composition cannot be
mistaken for improvement.

The proposed execution is network-denied, two CPU threads, six GiB and 1,800
seconds. Preparation has a separate bounded unit. Prediction ledgers are saved
before scoring. Failed runs preserve completed fits and receipts, with no
automatic restart. Completion requires independent recipe, exclusion,
prediction and metric replay before interpretation. This is an exploratory
native-footprint comparison, not proof of 100 m or cloudy-condition accuracy
and not permission to replace the public model.
