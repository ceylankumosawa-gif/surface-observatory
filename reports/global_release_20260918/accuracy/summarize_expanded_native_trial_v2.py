"""Compact, coverage-preserving report from independently checked trial metrics.

No source acquisition, fitting, prediction, model selection or deployment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']
    return Path(binding['path'])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pair(metrics, segment_type):
    q = metrics.loc[metrics.view.eq('all_four_matched') &
                    metrics.segment_type.eq(segment_type) &
                    metrics.model.isin(['old_only', 'expanded'])].copy()
    keys = ['test_mode', 'segment_type', 'segment']
    old = q.loc[q.model.eq('old_only')].drop(columns='model').set_index(keys)
    new = q.loc[q.model.eq('expanded')].drop(columns='model').set_index(keys)
    assert old.index.equals(new.index)
    assert old.paired.equals(new.paired)
    result = old[['paired', 'dates', 'sites', 'acquisitions', 'collected_rows',
                  'qa_admitted', 'feature_complete', 'unscored']].copy()
    for name in ['mae', 'balanced_mae', 'bias', 'balanced_bias',
                 'above_3_fraction', 'above_5_fraction', 'above_7_fraction',
                 'balanced_above_3_fraction', 'balanced_above_5_fraction', 'balanced_above_7_fraction']:
        result['old_' + name] = old[name]
        result['expanded_' + name] = new[name]
        result['delta_' + name] = new[name] - old[name]
    result['observed'] = result.paired.gt(0)
    for arm in ['old', 'expanded']:
        result[arm + '_numerically_le3'] = (result.observed & result[arm + '_mae'].le(3) &
                                          result[arm + '_balanced_mae'].le(3))
    result['outcome'] = np.select(
        [~result.observed, result.delta_balanced_mae.lt(-1e-10), result.delta_balanced_mae.gt(1e-10)],
        ['missing', 'improved', 'worsened'], default='unchanged')
    return result.reset_index()


def regional_rows(metrics, registry):
    q = pair(metrics, 'pilot_phase')
    q[['region_id', 'phase']] = q.segment.str.rsplit('|', n=1, expand=True)
    q = q.loc[(q.test_mode.eq('reference') & q.region_id.eq('cabauw')) |
              (q.test_mode.eq('geography') & q.region_id.ne('cabauw'))].copy()
    required = registry[['region_id', 'phase', 'geography_role']].drop_duplicates()
    result = required.merge(q, on=['region_id', 'phase'], how='left', validate='one_to_one')
    assert len(result) == 96 and result.test_mode.notna().all()
    return result


def counts(regional):
    rows = []
    for role, q in regional.groupby('geography_role', sort=True):
        rows.append({'geography_role': role, 'requested_phase_groups': len(q),
            'observed': int(q.observed.sum()), 'missing': int((~q.observed).sum()),
            'improved': int(q.outcome.eq('improved').sum()),
            'worsened': int(q.outcome.eq('worsened').sum()),
            'unchanged': int(q.outcome.eq('unchanged').sum()),
            'old_numerically_le3': int(q.old_numerically_le3.sum()),
            'expanded_numerically_le3': int(q.expanded_numerically_le3.sum())})
    return rows


def number(value):
    return 'missing' if pd.isna(value) else f'{value:.3f}'


def run(trial, audit_path, out):
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('Report is offline'))
                    if event == 'socket.connect' else None)
    started = time.monotonic()
    assert not out.exists()
    audit = read(audit_path)
    assert audit['status'] == 'passed' and audit['trial_completion'] == bind(trial / 'completion.json')
    done = read(trial / 'completion.json')
    assert done['status'] == 'complete_exploratory_data_comparison'
    plan = read(check(done['plan']))
    cohort = read(check(plan['inputs']['cohort_receipt']))
    assert cohort['format'] == 'expanded_native_direct_weather_cohort_v1'
    metrics_path = check(done['artifacts']['metrics.csv'])
    registry_path = check(cohort['registry'])
    admission_path = check(cohort['admission'])
    bindings = [bind(__file__), bind(audit_path), bind(trial / 'completion.json'),
                done['plan'], plan['inputs']['cohort_receipt'], bind(metrics_path),
                bind(registry_path), bind(admission_path)]
    metrics = pd.read_csv(metrics_path)
    registry = pd.read_csv(registry_path, keep_default_na=False)
    assert len(registry) == 960 and registry.region_id.nunique() == 48
    regional = regional_rows(metrics, registry)
    comparison = pair(metrics, 'geography_role')
    group_counts = counts(regional)
    date_cases = pair(metrics, 'pilot_date_phase')
    date_cases[['region_id', 'date', 'phase']] = date_cases.segment.str.split('|', expand=True)
    date_cases = date_cases.loc[(date_cases.test_mode.eq('reference') & date_cases.region_id.eq('cabauw')) |
                                (date_cases.test_mode.eq('geography') & date_cases.region_id.ne('cabauw'))]
    date_cases = registry[['region_id', 'date', 'phase', 'geography_role']].merge(
        date_cases, on=['region_id', 'date', 'phase'], how='left', validate='one_to_one')
    assert len(date_cases) == 960 and date_cases.test_mode.notna().all()
    out.mkdir(parents=True)
    save(out / 'plan.json', {'inputs': bindings, 'scope': 'Audited paired metrics only; unchanged denominators',
                            'requests': 0, 'fits': 0, 'automatic_promotion': False})
    regional.to_csv(out / 'all96_region_phase.csv', index=False)
    date_cases.to_csv(out / 'all960_date_phase.csv', index=False)
    comparison.to_csv(out / 'role_comparisons.csv', index=False)
    pd.DataFrame(group_counts).to_csv(out / 'coverage_counts.csv', index=False)
    (out / 'all960_admission.csv').write_bytes(admission_path.read_bytes())
    lines = ['# Geographic training expansion: paired comparison', '',
        'The two arms use the same small model and predict the same held observations. '
        'Only the fitting geography changes. Both arms use the same direct ERA5 weather recipe. '
        'That source was selected for complete delivery before either arm was fitted; the earlier '
        'six original-cohort studies used API-delivered weather and are separate comparisons. '
        'Errors below are satellite-footprint differences in °C; '
        'they do not establish 100 m or all-weather accuracy.', '',
        '| Held test | Evaluation population | Original-only balanced MAE | Expanded balanced MAE | Change | Paired observations |',
        '|---|---|---:|---:|---:|---:|']
    labels = {'original_development': 'Original development areas', 'new_development': 'New development areas',
              'original_reference': 'Cabauw reference', 'reserved_new_geography': 'Reserved areas — unopened'}
    for row in comparison.itertuples(index=False):
        if row.test_mode == 'reference' and row.segment != 'original_reference':
            continue
        if row.test_mode != 'reference' and row.segment == 'original_reference':
            continue
        lines.append(f'| {row.test_mode} | {labels[row.segment]} | {number(row.old_balanced_mae)} | '
                     f'{number(row.expanded_balanced_mae)} | {number(row.delta_balanced_mae)} | {row.paired:,} |')
    lines += ['', 'Negative change means lower error. The original and new populations are separate '
              'because pooling them can hide regression in the original areas.', '',
              '| Population | Requested day/night groups | Observed | Missing | Improved / worsened | ≤3°C: original / expanded |',
              '|---|---:|---:|---:|---:|---:|']
    for row in group_counts:
        lines.append(f"| {labels[row['geography_role']]} | {row['requested_phase_groups']} | {row['observed']} | "
                     f"{row['missing']} | {row['improved']} / {row['worsened']} | "
                     f"{row['old_numerically_le3']} / {row['expanded_numerically_le3']} |")
    lines += ['', 'The ≤3°C count requires both ordinary and balanced MAE ≤3°C on available observations. '
              'It is a numerical diagnostic, not a qualification: missing seasons, dates, cloudy conditions '
              'and fine-resolution reference evidence remain limitations. Reserved areas are unopened, not failed temperature estimates.', '',
              'All 96 region/phase outcomes and all 960 requested date/phase cases are retained in the accompanying CSVs. '
              'Missing error stays missing. All 171 reserved physical passes were excluded from fitting. '
              'No public model change follows automatically from this report.', '']
    (out / 'COMPARISON.md').write_text('\n'.join(lines))
    for binding in bindings:
        check(binding)
    save(out / 'completion.json', {'status': 'complete_audited_comparison_report', 'plan': bind(out / 'plan.json'),
        'trial_completion': bind(trial / 'completion.json'), 'audit': bind(audit_path),
        'region_phase_groups': 96, 'requested_cases': 960, 'counts': group_counts,
        'common_weather_source': 'direct_era5_physical_nonnegative_v3',
        'source_decision_before_model_errors': True,
        'global_qualification': False, 'production_changed': False, 'requests': 0, 'fits': 0,
        'wall_seconds': time.monotonic() - started,
        'artifacts': {p.name: bind(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(bind(out / 'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.trial, args.audit, args.output)
