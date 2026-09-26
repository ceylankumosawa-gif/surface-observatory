"""Offline aggregate-only solar-season report; never publishes or loads models."""
from pathlib import Path
import argparse
import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd
import summarize_expanded_native_trial_v2 as summary

BASE_EXPORT_SHA = '73c1fa80ca9f376ec5e97124a75efa9f7285c51eb83f93150a68a8c39b3ca1bd'
BASE_EVIDENCE_SHA = '40c6a01420a2749be08e2e297041227d565dffa7606bd5a852d667421e528d4d'
BASE_TRIAL_SHA = '54e70333622b9c7c511cee5b5fe6425832e849e1f96e34d66262134d5346e5a2'
BASE_AUDIT_SHA = 'b2ebe0aac1bd7686dc119c14c8e4c7a87832a4aff5ed3a39f78000205e3536a1'
ROLES = {'original_development', 'new_development', 'original_reference', 'reserved_new_geography'}
LABELS = {'original_development': 'Original development areas', 'new_development': 'New development areas',
          'original_reference': 'Cabauw reference', 'reserved_new_geography': 'Reserved · unopened'}
bind, check, read, save = summary.bind, summary.check, summary.read, summary.save


def need(condition, message):
    if not condition:
        raise ValueError(message)


def explicit_columns(frame):
    """Generic pairing labels are internal only; every written arm is explicit."""
    return frame.rename(columns={c: ('baseline_' + c[4:] if c.startswith('old_') else
                                    'candidate_' + c[9:] if c.startswith('expanded_') else c)
                                 for c in frame.columns})


def paired_tables(metrics, registry):
    keys = ['test_mode', 'segment_type', 'segment', 'view', 'model']
    need(not metrics.duplicated(keys).any(), 'Duplicate metric population')
    need(set(metrics.model) == {'expanded', 'solar_season', 'raw_air', 'raw_skin'}, 'Unexpected arms')
    need(len(registry) == 960 and not registry.duplicated(['region_id', 'date', 'phase']).any(), 'Wrong case registry')
    need(set(registry.geography_role) == ROLES, 'Unknown geography roles')
    renamed = metrics.copy()
    renamed['model'] = renamed.model.replace({'expanded': 'old_only', 'solar_season': 'expanded'})
    regional = summary.regional_rows(renamed, registry)
    comparisons = summary.pair(renamed, 'geography_role').rename(columns={'segment': 'geography_role'})
    cases = summary.pair(renamed, 'pilot_date_phase')
    cases[['region_id', 'date', 'phase']] = cases.segment.str.split('|', expand=True)
    cases = cases.loc[(cases.test_mode.eq('reference') & cases.region_id.eq('cabauw')) |
                      (cases.test_mode.eq('geography') & cases.region_id.ne('cabauw'))]
    cases = registry[['region_id', 'date', 'phase', 'geography_role']].merge(
        cases, on=['region_id', 'date', 'phase'], how='left', validate='one_to_one')
    need(len(cases) == 960 and cases.test_mode.notna().all(), 'Missing case metric rows')
    outputs = [explicit_columns(f) for f in (regional, cases, comparisons)]
    for frame in outputs:
        score_fields = [c for c in frame if c.startswith(('baseline_', 'candidate_', 'delta_'))
                        and not c.endswith('numerically_le3')]
        need(frame.loc[frame.paired.eq(0), score_fields].isna().all().all(), 'Missing error became a score')
        need(np.isfinite(frame.loc[frame.paired.gt(0), score_fields].to_numpy(float)).all(), 'Observed nonfinite score')
    need(outputs[0].loc[outputs[0].geography_role.eq('reserved_new_geography'), 'paired'].eq(0).all(),
         'Reserved observations unexpectedly opened')
    return outputs


def records(frame):
    # pandas JSON emits proper null, not NaN, for absent metrics.
    return json.loads(frame.to_json(orient='records', double_precision=15))


def append_section(base, section):
    need('solar_season' not in base and base['expansion']['status'] == 'complete', 'Unexpected base snapshot')
    result = dict(base, solar_season=section)
    need({k: v for k, v in result.items() if k != 'solar_season'} == base, 'Existing evidence changed')
    return result


def run(trial, audit_path, audit_sha, base_export_path, out):
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('Offline export'))
                    if event == 'socket.connect' else None)
    started = time.monotonic()
    need(not out.exists(), 'Output already exists')
    audit_binding = {'path': str(audit_path.resolve()), 'sha256': audit_sha}
    audit = read(check(audit_binding))
    done_binding = bind(trial / 'completion.json')
    need(audit['status'] == 'passed' and audit['trial_completion'] == done_binding, 'Wrong or incomplete candidate audit')
    need(audit['baseline_completion']['sha256'] == BASE_TRIAL_SHA and audit['baseline_audit']['sha256'] == BASE_AUDIT_SHA,
         'Different geography baseline')
    need(audit['fitting_recipes'] == 45 and audit['requested_groups'] == 960 and audit['regional_phase_groups'] == 96
         and audit['reserved_physical_keys'] == 171 and audit['unchanged_original_fields'] == 29
         and audit['paired_baseline_models_replayed'] and audit['independent_memberships_weights_metrics'], 'Audit scope mismatch')
    need(audit['fits'] == 0 and audit['requests'] == 0 and not audit['reserved_thermal_opened'], 'Audit scope changed')
    done = read(check(done_binding))
    need(done['status'] == 'complete_exploratory_solar_season_comparison', 'Wrong completed experiment')
    plan = read(check(done['plan']))
    baseline_done = read(check(audit['baseline_completion']))
    need(plan['baseline_plan'] == baseline_done['plan'], 'Baseline plan differs')
    cohort = read(check(plan['cohort_receipt']))
    need(cohort['format'] == 'expanded_native_direct_weather_cohort_v1', 'Expected common direct weather cohort')
    need(bind(base_export_path)['sha256'] == BASE_EXPORT_SHA, 'Base export was replaced')
    base_export = read(base_export_path)
    base_evidence = base_export['public_files']['evidence.json']
    need(base_evidence['sha256'] == BASE_EVIDENCE_SHA, 'Different published geography snapshot')
    for binding in base_export['public_files'].values():
        check(binding)
    metrics_binding = audit['artifacts']['independent_metrics.csv']
    registry_binding = cohort['registry']
    bindings = [bind(__file__), bind(summary.__file__), audit_binding, done_binding, done['plan'], audit['plan'],
                audit['baseline_completion'], audit['baseline_audit'], plan['cohort_receipt'],
                metrics_binding, registry_binding, bind(base_export_path), *base_export['public_files'].values()]
    for binding in bindings:
        check(binding)
    regional, cases, comparisons = paired_tables(pd.read_csv(check(metrics_binding)),
                                                 pd.read_csv(check(registry_binding), keep_default_na=False))
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    section = dict(status='complete', generated_at_utc=timestamp, baseline_label='Expanded baseline',
        candidate_label='Solar-season candidate', target_mae_c=3, qualified=False,
        description='The same expanded training population, held observations and 31-input model recipe. '
                    'Only two calendar inputs become footprint-mean daily sunlight at the top of the atmosphere '
                    'and its daily change. Both research models start from ERA5-Land air temperature and '
                    'use reanalysis weather, checked against native satellite footprints. '
                    'The live F40 station/weather map is separate. This is not 100 m or all-weather validation.',
        comparisons=records(comparisons), regions=records(regional),
        download_url='/research/solar-season-region-phase.csv', cases_download_url='/research/solar-season-date-phase.csv',
        source=dict(audit_sha256=audit_sha, metrics_sha256=metrics_binding['sha256'], trial_sha256=done_binding['sha256']))
    evidence = append_section(read(check(base_evidence)), section)
    out.mkdir(parents=True)
    save(out / 'plan.json', dict(bindings=bindings, status='frozen_before_aggregate_export',
         scope='Independent audit metrics only; 96 area/phase and 960 requested cases; old snapshot unchanged',
         baseline_label=section['baseline_label'], candidate_label=section['candidate_label'], requests=0, fits=0))
    regional.to_csv(out / 'solar-season-region-phase.csv', index=False)
    cases.to_csv(out / 'solar-season-date-phase.csv', index=False)
    comparisons.to_csv(out / 'role_comparisons.csv', index=False)
    save(out / 'evidence.json', evidence)
    lines = ['# Solar-season representation: checked comparison', '', section['description'], '',
             '| Held test | Population | Expanded baseline balanced MAE | Solar-season candidate balanced MAE | Candidate − baseline | Paired |',
             '|---|---|---:|---:|---:|---:|']
    for r in comparisons.itertuples(index=False):
        if (r.test_mode == 'reference') != (r.geography_role == 'original_reference'):
            continue
        lines.append(f'| {r.test_mode} | {LABELS[r.geography_role]} | {summary.number(r.baseline_balanced_mae)} | '
                     f'{summary.number(r.candidate_balanced_mae)} | {summary.number(r.delta_balanced_mae)} | {r.paired:,} |')
    lines += ['', 'Errors are in °C. Negative change means lower error. All 96 area/day–night groups and 960 '
              'requested cases remain in the CSVs; missing is not zero. Reserved areas remain unopened, and '
              'all 171 reserved physical passes remain excluded from fitting. A low aggregate is not global qualification. '
              'The live F40 model and the six earlier API-weather studies are unchanged.', '']
    (out / 'COMPARISON.md').write_text('\n'.join(lines))
    for binding in bindings:
        check(binding)
    public = ['evidence.json', 'solar-season-region-phase.csv', 'solar-season-date-phase.csv']
    save(out / 'completion.json', dict(status='complete_audited_solar_season_export', plan=bind(out / 'plan.json'),
        trial_completion=done_binding, audit=audit_binding, base_export=bind(base_export_path),
        region_phase_groups=96, requested_cases=960, existing_evidence_preserved=True,
        global_qualification=False, production_changed=False, requests=0, fits=0,
        wall_seconds=time.monotonic() - started,
        public_files={name: bind(out / name) for name in public},
        artifacts={p.name: bind(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(bind(out / 'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['trial', 'audit', 'base-export', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    args = parser.parse_args()
    run(args.trial, args.audit, args.audit_sha256, args.base_export, args.output)
