# Selected research source

This directory preserves the training and evaluation code behind the published
research, together with the protocols that define sample admission, held-out
evaluation and source provenance. Start with [the research status](../docs/RESEARCH_STATUS.md)
for the completed findings and [the main README](../README.md) for the application.

- `weather_alignment_20260911/models.py` and `run_study.py` define the frozen
  40-feature reference and weather-alignment comparisons.
- `global_release_20260918/accuracy/expanded_native_trial_v1.py` compares the
  original and geographically expanded native-scale training sets.
- `global_release_20260918/accuracy/expanded_solar_season_trial_v1.py` tests a
  physical seasonal representation; its independent audit is included.
- Earlier source helpers and protocol documents are retained where these
  experiments or the package tests depend on them.

These are archived experiment runners, not one-command downloads or a production
training service. They deliberately require the original source manifests,
checksums and prepared cohorts. Those data, model binaries, run logs and machine
configuration are not distributed here. Paths under `/opt/lst-pilot` identify
the original deployment layout. Reproducing an experiment requires obtaining its
upstream data and rebuilding the documented inputs; removing a checksum check
does not reproduce the experiment.

The small public evidence tables in `web/public/research/` contain aggregate
results for inspection. They are not the withheld evaluation observations.
Historical protocol instructions describe those experiments; current setup
instructions live in the root README.
