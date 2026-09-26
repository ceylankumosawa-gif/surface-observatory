"""Freeze year-separated context joins after the complete source collection closes.

This is orchestration over the unchanged coarse_join.join_frame implementation.
It reads only six fine-sample metadata columns; native LST remains coarse context.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import pandas as pd
from . import coarse_join
from .coarse_inventory import identity, save, sha
from .coarse_join_manifest import prepare

TARGET_COLUMNS = ['sample_id', 'region_id', 'datetime_utc', 'latitude', 'longitude', 'epsg']
REFERENCE_HASHES = {
    'E_fit': '53572203ce571075fed009b31898f286e9aac6af7e56eb6ef2257400cbf830e3',
    'old_evaluation': '038c31f1db8fe543fea0830988e5e36d8f3972decd0efe2bd2236df33a54ebf2',
}


def check_targets(frame, *, fitting):
    if list(frame.columns) != TARGET_COLUMNS or frame.sample_id.duplicated().any():
        raise ValueError('Exactly six metadata columns and unique sample IDs required.')
    if any(pd.Timestamp(t).tzinfo is None for t in frame.datetime_utc.unique()):
        raise ValueError('Target time must retain its explicit UTC offset.')
    allowed = [2021, 2022] if fitting else [2021, 2022, 2023]
    if not pd.to_datetime(frame.datetime_utc, utc=True).dt.year.isin(allowed).all():
        raise ValueError('Context target table crosses its predeclared year boundary.')


def split_targets(frame):
    check_targets(frame, fitting=False)
    years = pd.to_datetime(frame.datetime_utc, utc=True).dt.year
    return frame.loc[years.le(2022)].copy(), frame.loc[years.eq(2023)].copy()


def count_support(targets, joined):
    if list(targets.sample_id) != list(joined.sample_id):
        raise ValueError('Join changed sample identity/order.')
    audit = targets[['sample_id', 'region_id', 'datetime_utc']].merge(
        joined[['sample_id', 'coarse_context_eligible', 'coarse_product', 'coarse_native_id']],
        on='sample_id', how='left', validate='one_to_one')
    audit['utc_date'] = pd.to_datetime(audit.datetime_utc, utc=True).dt.strftime('%Y-%m-%d')
    matched = audit.loc[audit.coarse_context_eligible]
    return {'rows': len(joined), 'matched_rows': int(joined.coarse_context_eligible.sum()),
            'pilot_dates': len(audit[['region_id', 'utc_date']].drop_duplicates()),
            'matched_pilot_dates': len(matched[['region_id', 'utc_date']].drop_duplicates()),
            'unique_native_cells': int(joined.coarse_native_id.nunique()),
            'matched_by_product': matched.coarse_product.value_counts().to_dict(),
            'by_pilot_date': audit.groupby(['region_id', 'utc_date'], sort=True)
                .agg(rows=('sample_id', 'size'), matched_rows=('coarse_context_eligible', 'sum'))
                .reset_index().to_dict('records')}


def source_manifests(root):
    phases = [root/'context_engineering_v2/manifest.json',
              root/'context_2021_2022_v1/context_manifest.json',
              root/'context_2023_and_eco1_v1/context_manifest.json']
    phases += [root/f'context_eco{n}_v1/context_manifest.json' for n in range(2, 11)]
    phases += [root/f'context_night{n}_v1/context_manifest.json' for n in range(1, 5)]
    phases += [root/'context_recovery_v1/context_manifest.json']
    for path in phases:
        if not path.is_file() or json.loads(path.read_text()).get('complete') is False:
            raise ValueError(f'Final source phase is incomplete: {path.parent.name}')
    return phases


def prepare_targets(run_root, output):
    tasks = []; provenance = []
    def add(name, section, frame, source, source_sha):
        fitting = section == 'fit'
        check_targets(frame, fitting=fitting)
        if frame.empty:
            return
        path = output/'targets'/f'{name}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        tasks.append({'name': name, 'section': section, 'target_path': str(path.resolve()),
                      'target_sha256': sha(path), 'rows': len(frame),
                      'source_path': str(source.resolve()), 'source_sha256': source_sha})
    for name, expected in REFERENCE_HASHES.items():
        path = run_root/'context_reference_metadata_v1'/f'{name}.parquet'
        if sha(path) != expected:
            raise ValueError('Parent-frozen reference metadata changed.')
        frame = pd.read_parquet(path, columns=TARGET_COLUMNS)
        add(name, 'fit' if name == 'E_fit' else 'old_evaluation', frame, path, expected)
    for kind, count in [('eco', 10), ('night', 4)]:
        for n in range(1, count+1):
            source = run_root/f'{kind}_source_batch{n}'
            sampling_path = source/'sampling_manifest.json'
            sampling = json.loads(sampling_path.read_text())
            frozen = sampling['outputs']['source_screened_samples']
            if sha(frozen['path']) != frozen['sha256']:
                raise ValueError('Immutable fine-source sample hash changed.')
            metadata_dir = run_root/'coarse'/f'{kind}_batch{n}_metadata'
            reference = json.loads((metadata_dir/'reference_targets.json').read_text())
            metadata_path = metadata_dir/'samples.parquet'
            if (reference['input_sha256'] != frozen['sha256'] or
                    sha(metadata_path) != reference['sample_metadata_sha256']):
                raise ValueError('Fine-source metadata extraction is not hash-bound.')
            frame = pd.read_parquet(metadata_path, columns=TARGET_COLUMNS)
            fitting, evaluation = split_targets(frame)
            add(f'{kind}{n}_fit', 'fit', fitting, metadata_path, sha(metadata_path))
            add(f'{kind}{n}_evaluation', 'new_evaluation', evaluation, metadata_path, sha(metadata_path))
            provenance.append({'kind': kind, 'batch': n, 'sampling_manifest_path': str(sampling_path.resolve()),
                'sampling_manifest_sha256': sha(sampling_path), 'original_sample_sha256': frozen['sha256'],
                'metadata_path': str(metadata_path.resolve()), 'metadata_sha256': sha(metadata_path),
                'source_rows': len(frame), 'fitting_rows': len(fitting), 'evaluation_rows': len(evaluation)})
    return tasks, provenance


def load_native(manifest_path, *, fitting):
    manifest = json.loads(Path(manifest_path).read_text()); parts = []
    for record in manifest['records']:
        # Inspect source identity before numeric reads. Fitting never decodes a 2023 table.
        year = pd.Timestamp(identity(record['stem'])['start']).year
        if year not in ([2021, 2022] if fitting else [2021, 2022, 2023]):
            raise ValueError('Native pool crosses its predeclared year boundary.')
        if sha(record['table_path']) != record['table_sha256']:
            raise ValueError('Native source table changed.')
        frame = pd.read_parquet(record['table_path'], columns=coarse_join.CONTEXT_COLUMNS)
        gate = 'context_fit_eligible' if fitting else 'context_eligible'
        frame = frame.loc[frame[gate]].copy()
        frame['context_table_sha256'] = record['table_sha256']
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def write_join(task, native, context_path, output, recovery_only_hashes):
    input_path = Path(task['target_path']); fitting = task['section'] == 'fit'
    targets = pd.read_parquet(input_path, columns=TARGET_COLUMNS)
    check_targets(targets, fitting=fitting)
    joined = coarse_join.join_frame(targets, native, fitting)
    directory = output/'joins'/task['name']; directory.mkdir(parents=True, exist_ok=True)
    table = directory/'coarse_context.parquet'; joined.to_parquet(table, index=False)
    audit = {'input_sha256': sha(input_path), 'input_columns_read': TARGET_COLUMNS,
        'context_manifest_sha256': sha(context_path), 'join_source_sha256': sha(coarse_join.__file__),
        'orchestration_source_sha256': sha(__file__), 'for_fitting': fitting,
        'maximum_oldest_possible_age_hours': 24,
        'time_rule': 'Entire granule interval completed at target; start at or after target minus 24 hours',
        'age_semantics': 'coarse_age_hours = target minus source start (oldest possible age); coarse_age_min_hours = target minus source end',
        'native_label_warning': 'Shared native context is not an independent fine-resolution label',
        'output_path': str(table.resolve()), 'output_sha256': sha(table), **count_support(targets, joined)}
    save(directory/'manifest.json', audit)
    # Coverage-only diagnostic. Never supplied as a second conflicting H join.
    before_native = native.loc[~native.context_table_sha256.isin(recovery_only_hashes)]
    before = coarse_join.join_frame(targets, before_native, fitting)
    baseline = count_support(targets, before)
    changed = ~before.coarse_context_eligible & joined.coarse_context_eligible
    audit_recovery = {'before': baseline, 'after': count_support(targets, joined),
                     'previously_missing_rows_recovered': int(changed.sum())}
    specification = {'join_dir': str(directory.resolve()), 'target_path': str(input_path.resolve()),
                     'context_manifest_path': str(context_path.resolve())}
    return specification, audit, audit_recovery


def run(run_root, output):
    run_root = Path(run_root).resolve(); output = Path(output).resolve(); root = run_root/'coarse'
    if output.exists():
        raise FileExistsError('Final context output is immutable; choose a fresh version.')
    sources = source_manifests(root)  # All final phases must exist before anything is emitted.
    output.mkdir(parents=True)
    combined = output/'native_context_manifest.json'; prepare(sources, combined)
    all_manifest = json.loads(combined.read_text())
    fit_manifest = {**all_manifest, 'original_manifest_path': str(combined), 'original_manifest_sha256': sha(combined),
                    'year_scope': [2021, 2022], 'records': [r for r in all_manifest['records']
                    if pd.Timestamp(identity(r['stem'])['start']).year in (2021, 2022)]}
    fit_path = output/'native_context_fit_manifest.json'; save(fit_path, fit_manifest)
    previous_hashes = {r['table_sha256'] for path in sources[:-1]
                       for r in json.loads(path.read_text())['records'] if r.get('table_sha256')}
    recovery_hashes = {r['table_sha256'] for r in all_manifest['records']} - previous_hashes
    tasks, provenance = prepare_targets(run_root, output)
    save(output/'target_manifest.json', {'tasks': tasks, 'source_batches': provenance,
        'fine_input_columns_read': TARGET_COLUMNS, 'source_sha256': sha(__file__)})
    specification = {'fit': [], 'old_evaluation': [], 'new_evaluation': []}; audits = []; recovery = {}
    for fitting in (True, False):
        context_path = fit_path if fitting else combined
        native = load_native(context_path, fitting=fitting)
        for task in tasks:
            if (task['section'] == 'fit') != fitting:
                continue
            spec, audit, comparison = write_join(task, native, context_path, output, recovery_hashes)
            specification[task['section']].append(spec)
            audits.append({'name': task['name'], 'section': task['section'], **audit})
            recovery[task['name']] = comparison
            print(json.dumps({'name': task['name'], 'rows': audit['rows'], 'matched_rows': audit['matched_rows'],
                              'matched_pilot_dates': audit['matched_pilot_dates']}), flush=True)
        del native
        gc.collect()
    save(output/'context_specification.json', specification)
    save(output/'recovery_coverage_comparison.json', recovery)
    save(output/'final_manifest.json', {'complete': True, 'source_code_sha256': sha(__file__),
        'context_specification_path': str(output/'context_specification.json'),
        'context_specification_sha256': sha(output/'context_specification.json'),
        'native_context_manifest_sha256': sha(combined), 'native_context_fit_manifest_sha256': sha(fit_path),
        'target_manifest_sha256': sha(output/'target_manifest.json'),
        'recovery_coverage_comparison_sha256': sha(output/'recovery_coverage_comparison.json'),
        'transfer_budget_snapshot': json.loads((root/'transfer_budget.json').read_text()), 'joins': audits})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.run_root, a.output)
