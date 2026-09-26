"""Versioned, read-only validation for ARCO four-field + CDS true-SWE delivery.

Only weather is decoded; all requested-file years are checked first. No fitting,
new source acquisition, interpolation, row dropping, or physical-value changes.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

FORMAT = 'arco4_cds_swe_merge_v1'
IDENTITY = ['sample_id', 'region_id', 'datetime_utc', 'latitude', 'longitude']
PARAMS = {'skin': 235, 'air': 167, 'soil_temperature': 139, 'soil_water': 39}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_years(path, column, years):
    stamps = pd.read_parquet(path, columns=[column])[column]
    require(isinstance(stamps.dtype, pd.DatetimeTZDtype), 'Native/request timestamps must be timezone-aware')
    dates = pd.to_datetime(stamps, utc=True)
    require(dates.notna().all() and dates.dt.year.isin(years).all(), f'Merged weather whole-file years outside scope: {path}')


def load(directory, period):
    directory = Path(directory).resolve()
    require(period in ('fit', 'evaluation'), 'Unknown weather period')
    years = [2021, 2022] if period == 'fit' else [2021, 2022, 2023, 2024]
    files = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, f'Merged weather source changed: {path.name}')
        files[str(path)] = actual
        return path
    cp = bind(directory/'completion.json')
    done = read(cp)
    require(done.get('delivery_format') == FORMAT and done.get('status') == 'complete', 'Merged weather delivery incomplete/version unsupported')
    require(done.get('no_new_thermal_labels') is True and done.get('no_training') is True, 'Merged weather source scope missing')
    pp = bind(directory/'plan.json', done['plan_sha256'])
    output = bind(directory/'values.parquet', done['output_sha256'])
    plan = read(pp)
    require(plan.get('delivery_format') == FORMAT, 'Merged plan delivery version mismatch')
    for field in ('all_five_fields_required', 'same_cell_hour_required', 'no_interpolation', 'no_land_substitution', 'retrospective_reanalysis', 'no_new_thermal_labels'):
        require(plan.get(field) is True, f'Merged source policy missing: {field}')
    source_dir = Path(__file__).resolve().parent
    merger_path = bind(source_dir/'merge_era5_land.py', plan['source_sha256'])
    require(done['source_sha256'] == plan['source_sha256'], 'Mixed merger revisions')
    expected = {'arco_plan', 'arco_preflight', 'arco_completion', 'arco_values', 'native_cells', 'swe_completion', 'swe_values'}
    require(set(plan['inputs']) == expected and set(done['input_hashes']) == expected, 'Incomplete merged source bindings')
    paths = {}
    for key, item in plan['inputs'].items():
        require(item['sha256'] == done['input_hashes'][key], 'Merged child binding mismatch')
        paths[key] = bind(item['path'], item['sha256'])
    require(done['arco_completion_sha256'] == files[str(paths['arco_completion'])]
            and done['swe_completion_sha256'] == files[str(paths['swe_completion'])], 'Child completion mismatch')
    ap, ac, pre = [read(paths[k]) for k in ('arco_plan', 'arco_completion', 'arco_preflight')]
    require(ap.get('version') == 3 and ap['period'] == period and ap['columns_decoded'] == IDENTITY, 'Wrong ARCO request version/period/schema')
    require(ap.get('retrospective_reanalysis_not_realtime_availability') is True, 'ARCO temporal availability caveat missing')
    require(ac.get('status') == 'four_fields_complete_SWE_pending' and ac.get('no_new_thermal_labels') is True
            and ac.get('no_training') is True and ac.get('no_2025_requested_rows') is True, 'ARCO source scope/completion failed')
    for key, field in [('arco_plan', 'plan_sha256'), ('arco_preflight', 'preflight_sha256'), ('arco_values', 'output_sha256'), ('native_cells', 'native_cells_sha256')]:
        require(files[str(paths[key])] == ac[field], 'ARCO completion binding failed')
    bind(source_dir/'era5_land_v3.py', ap['source_sha256'])
    bind(source_dir/'era5_land.py', ap['base_source_sha256'])
    require(ac['source_sha256'] == ap['source_sha256'] and ac['base_source_sha256'] == ap['base_source_sha256'], 'ARCO source versions differ')
    arco_dir = paths['arco_plan'].parent
    points_path = bind(arco_dir/'points.parquet', ap['points_sha256'])
    input_path = bind(ap['input_path'], ap['input_sha256'])
    bind(arco_dir/'transfer_ledger.json', ac['ledger_sha256'])
    require(pre['plan_sha256'] == ac['plan_sha256'] and pre['native_cells_sha256'] == ac['native_cells_sha256'], 'ARCO preflight binding failed')
    require(pre.get('all_four_variable_native_spatial_axes_identical') is True
            and pre.get('all_selected_valid_times_identical') is True, 'ARCO identical native grid/hour proof missing')
    require(set(pre['stores']) == set(PARAMS), 'Unexpected ARCO variable set')
    coordinate_proofs = []
    for name, param in PARAMS.items():
        store = pre['stores'][name]
        attrs = store['attributes']
        require(attrs.get('GRIB_paramId') == param and attrs.get('GRIB_stepType') == 'instant', 'ARCO parameter/instantaneous proof failed')
        units = attrs.get('units')
        valid_units = ('m**3 m**-3', 'm3 m-3', 'm^3 m^-3', 'm3/m3') if name == 'soil_water' else ('K',)
        require(units in valid_units, 'ARCO physical units failed')
        proof = store['coordinate_decoded_sha256']
        coordinate_proofs.append(tuple(proof[k] for k in ('latitude', 'longitude', 'selected_valid_time')))
    require(len(set(coordinate_proofs)) == 1, 'ARCO decoded grid/hour hashes differ')
    sc = read(paths['swe_completion'])
    require(sc.get('status') == 'complete' and sc.get('reserved_2025_opened') is False
            and sc.get('thermal_labels_decoded') is False, 'SWE extraction scope/completion failed')
    require(sc['output_sha256'] == files[str(paths['swe_values'])], 'SWE output binding failed')
    swe_dir = paths['swe_completion'].parent
    sp = read(bind(swe_dir/'plan.json', sc['request_plan_sha256']))
    require(sp.get('dataset') == 'reanalysis-era5-land' and sp.get('period') == period
            and sp.get('years') == years and sp.get('reserved_2025_opened') is False
            and sp.get('new_thermal_labels_read') is False, 'SWE request scope differs')
    bind(source_dir/'cds_swe.py', sc['source_sha256'])
    require(sp['source_sha256'] == sc['source_sha256'], 'SWE source revisions differ')
    bind(sp['native_input_path'], sp['native_input_sha256'])
    swe_cells = bind(swe_dir/'native_cells.parquet', sc['native_cells_sha256'])
    require(sc['native_cells_sha256'] == sp['native_cells_sha256'], 'SWE requested cells changed')
    bind(swe_dir/'sources.json', sc['sources_sha256'])
    bind(swe_dir/'state.json', sc['state_sha256'])
    require(sc.get('source_files'), 'SWE raw-response hashes missing')
    raw_hashes = set()
    for item in sc['source_files']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = swe_dir/path
        raw_hashes.add(files[str(bind(path, item['sha256']))])
    require(raw_hashes == set(plan['swe_verified_source_hashes']), 'Merged SWE raw-source set differs')
    # Guard every supplied table, including extras, before any temperature/SWE decoding.
    for path in (points_path, input_path, output, paths['arco_values']):
        check_years(path, 'datetime_utc', years)
    for path in (paths['native_cells'], paths['swe_values'], swe_cells):
        check_years(path, 'era5_land_valid_time_utc', years)
    four = pd.read_parquet(paths['arco_values'])
    snow = pd.read_parquet(paths['swe_values'])
    requests = pd.read_parquet(paths['native_cells'])
    pd.testing.assert_frame_equal(requests, pd.read_parquet(swe_cells), check_exact=True, check_dtype=False)
    spec = importlib.util.spec_from_file_location('verified_land_merger', merger_path)
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    reconstructed = merger.merge_frames(four, snow, requests, raw_hashes)
    values = pd.read_parquet(output)
    pd.testing.assert_frame_equal(values, reconstructed, check_exact=True)
    require(done['rows'] == len(values) == ac['rows'], 'Merged row count differs')
    available = values.era5_land_complete.eq(True).sum()
    require(done['available_rows'] == available and done['all_five_fields_available'] == bool(available == len(values)), 'Merged completeness audit disagrees')
    require(values.era5_land_preflight_sha256.eq(ac['preflight_sha256']).all(), 'Merged ARCO preflight provenance differs')
    # Hash-bound audit includes upstream sources for start/end checks in fitting and evaluation.
    for path, checksum in files.items():
        require(sha(path) == checksum, 'Source changed during attachment')
    return pd.read_parquet(points_path), values, ac['preflight_sha256'], {path: Path(path) for path in files}
