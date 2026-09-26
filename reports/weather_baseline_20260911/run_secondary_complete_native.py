"""Secondary complete-native land-baseline research comparison; never deploys.

Run `fit` only after protocol/source approval. It freezes the training-month
selection and full fits without decoding evaluation labels. A separate
`evaluate` stage consumes completed evaluation weather extracts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from lst_pilot import option_b_train as old
from lst_pilot import multisensor_train as multi
from lst_pilot import physics_correction as physics

PRIOR_CODE = Path(__file__).resolve().parents[1] / 'shared_model_remedies'
EXTRA = ('land_skin_minus_air_c', 'land_soil_minus_air_c',
         'land_snow_water_equivalent_m', 'land_soil_moisture_m3_m3')
FEATURES = (*multi.BASE, *EXTRA)
LAND_VALUES = ('era5_land_skin_temperature_c', 'era5_land_air_temperature_c',
               'era5_land_soil_temperature_0_7cm_c', 'era5_land_snow_water_equivalent_m',
               'era5_land_soil_moisture_0_7cm_m3_m3')
IDENTITY = ('sample_id', 'region_id', 'datetime_utc', 'latitude', 'longitude')
LABELS = {'F': 'F complete-native refit', 'blend_et25': '75% F / 25% Extra Trees complete-native refits',
          'raw_land_skin': 'Raw ERA5-Land skin', 'station_adjusted_land': 'Station-adjusted land skin',
          'land_features': 'Shared model with land inputs',
          'land_residual': 'Shared residual over land baseline',
          'land_blend25': '75% F / 25% land-residual model'}
FITTED = ('land_features', 'land_residual')
ALL_FITTED = ('F', 'extra_trees', *FITTED)
TIE_ORDER = ('land_features', 'land_residual', 'land_blend25')
FAMILIES = ('old', 'newly_collected_2023', 'legacy_2024')
GATES = {'overall_mae_improvement_c': .10, 'ordinary_mae_worsening_c': .10,
         'supported_group_mae_worsening_c': .20, 'tail_share_worsening': .02,
         'overall_contrast_worsening_c': .10, 'supported_group_utc_dates': 6,
         'tie_mae_c': .05}
META = ['sample_id', 'region_id', 'phase', 'climate_class', 'datetime_utc', 'utc_day', 'split',
        'season', 'air_group', 'label_product', 'acquisition_id', 'weight_surface_group',
        'lst_c', 'air_temperature_c', 'solar_elevation_deg']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def frame_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def log(**values):
    print(json.dumps(old.json_ready(values)), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def previous_runner():
    # Saved Extra Trees wrappers use the original `candidates` module name.
    if str(PRIOR_CODE) not in sys.path:
        sys.path.insert(0, str(PRIOR_CODE))
    import candidates
    require(Path(candidates.__file__).resolve() == PRIOR_CODE / 'candidates.py', 'Wrong saved-model module')
    spec = importlib.util.spec_from_file_location('frozen_shared_reference_runner', PRIOR_CODE / 'run_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_years(path, years):
    stamps = pd.to_datetime(pd.read_parquet(path, columns=['datetime_utc']).datetime_utc, utc=True)
    require(stamps.notna().all() and stamps.dt.year.isin(years).all(), f'Unexpected whole-file years: {path}')


def verify_completed(path, required):
    path = Path(path).resolve()
    completed = read_json(path / 'completion.json')
    require(completed.get('status') == 'complete' and completed.get('reserved_2025_opened') is False
            and completed.get('production_unchanged') is True, 'Reference run is incomplete or out of scope')
    for name in required:
        require(old.sha(path / name) == completed['artifacts'].get(name), f'Frozen reference changed: {name}')
    return completed


def reference_bindings(reference, blends):
    needed = ['manifest.json', 'fit_frame.parquet', 'oof_predictions.parquet', 'selection.json',
              'selection_freeze.json', 'full_fit_freeze.json', 'full/fits.json',
              *[f'{family}_predictions.parquet' for family in FAMILIES]]
    for stage in ('fold_0', 'fold_1', 'fold_2', 'full'):
        needed += [f'{stage}/{model}.joblib' for model in ('F', 'extra_trees')]
    reference_complete = verify_completed(reference, needed)
    blend_names = ['manifest.json', 'selection.json', 'selection_freeze.json',
                   'oof_predictions.parquet', *[f'{family}_predictions.parquet' for family in FAMILIES]]
    blend_complete = verify_completed(blends, blend_names)
    bm = read_json(blends / 'manifest.json')
    require(bm['reference_completion_sha256'] == old.sha(reference / 'completion.json'), 'Blend parent changed')
    require(bm.get('fractions', {}).get('blend_et25') == .25, 'Unexpected prior blend')
    for name, checksum in bm['input_sha256'].items():
        require(reference_complete['artifacts'].get(name) == checksum, 'Prior blend input hash mismatch')
    source_manifest = read_json(reference / 'manifest.json')
    protected_path = Path(source_manifest['input_paths']['baseline'])
    protected_sha = source_manifest['protected_baseline_sha256']
    require(old.sha(protected_path) == protected_sha, 'Protected production baseline changed')
    for path, checksum in source_manifest['source_hashes'].items():
        require(old.sha(path) == checksum, f'Frozen reference dependency changed: {path}')
    rf = read_json(reference / 'selection_freeze.json')
    require(rf.get('evaluation_labels_decoded_before_selection') is False, 'Prior selection temporal proof missing')
    for name, key in [('manifest.json', 'manifest_sha256'), ('selection.json', 'selection_sha256'),
                      ('oof_predictions.parquet', 'oof_predictions_sha256')]:
        require(reference_complete['artifacts'][name] == rf[key], 'Prior training freeze changed')
    bf = read_json(blends / 'selection_freeze.json')
    for name in ('manifest.json', 'selection.json', 'oof_predictions.parquet'):
        require(blend_complete['artifacts'][name] == bf[name], 'Prior blend training freeze changed')
    return {'reference': str(reference), 'blends_reference': str(blends),
            'reference_completion_sha256': old.sha(reference / 'completion.json'),
            'blend_completion_sha256': old.sha(blends / 'completion.json'),
            'protected_baseline_path': str(protected_path), 'protected_baseline_sha256': protected_sha,
            'reference_files': {name: reference_complete['artifacts'][name] for name in needed},
            'blend_files': {name: blend_complete['artifacts'][name] for name in blend_names}}


def aligned_reference(frame, reference, blends, family):
    paths = [reference / (family + '_predictions.parquet'), blends / (family + '_predictions.parquet')]
    years = (2021, 2022) if family == 'oof' else (2021, 2022, 2023, 2024)
    for path in paths:
        check_years(path, years)
    left = pd.read_parquet(paths[0])
    right = pd.read_parquet(paths[1])
    for saved in (left, right):
        require(saved.sample_id.notna().all() and not saved.sample_id.duplicated().any()
                and set(saved.sample_id) == set(frame.sample_id), 'Reference sample identities differ')
    left = left.set_index('sample_id').loc[frame.sample_id].reset_index()
    right = right.set_index('sample_id').loc[frame.sample_id].reset_index()
    for column in META:
        pd.testing.assert_series_equal(left[column], frame[column].reset_index(drop=True),
                                      check_names=False, check_dtype=False, check_categorical=False, check_exact=True)
        pd.testing.assert_series_equal(right[column], left[column], check_names=False,
                                      check_dtype=False, check_categorical=False, check_exact=True)
    require(np.array_equal(left.F_lst_c, right.F_lst_c), 'Prior F predictions differ')
    expected = .75 * left.F_lst_c.to_numpy(float) + .25 * left.extra_trees_lst_c.to_numpy(float)
    require(np.array_equal(expected, right.blend_et25_lst_c), 'Prior blend is not the exact fixed mixture')
    if family == 'oof':
        folds = multi.month_folds(frame)
        require(np.array_equal(left.fold, folds) and np.array_equal(right.fold, folds), 'Reference folds changed')
    return {'F': left.F_lst_c.to_numpy(float), 'blend_et25': right.blend_et25_lst_c.to_numpy(float)}, left


def derive_land_features(frame):
    """Preserve A; the two learned targets differ only by their declared baseline."""
    data = frame.copy()
    require(np.isfinite(data[list(LAND_VALUES)].to_numpy(float)).all(), 'Incomplete physical inputs')
    data['land_skin_minus_air_c'] = data[LAND_VALUES[0]] - data[LAND_VALUES[1]]
    data['land_soil_minus_air_c'] = data[LAND_VALUES[2]] - data[LAND_VALUES[1]]
    data['land_snow_water_equivalent_m'] = data[LAND_VALUES[3]]
    data['land_soil_moisture_m3_m3'] = data[LAND_VALUES[4]]
    data['land_station_adjusted_baseline_c'] = data.air_temperature_c + data.land_skin_minus_air_c
    require(np.isfinite(data[[*EXTRA, 'land_station_adjusted_baseline_c']].to_numpy(float)).all(), 'Invalid land transform')
    require(np.array_equal(data.air_temperature_c, frame.air_temperature_c), 'Original air values changed')
    return data


def attach_land_values(frame, points, values, preflight_sha256, paths, period, directory):
    for source in (points, values):
        require(source.sample_id.notna().all() and not source.sample_id.duplicated().any()
                and len(source) == len(frame) and set(source.sample_id) == set(frame.sample_id), 'Land sample identities differ')
        indexed = source.set_index('sample_id').loc[frame.sample_id].reset_index()
        for column in IDENTITY:
            pd.testing.assert_series_equal(indexed[column], frame[column].reset_index(drop=True),
                                          check_names=False, check_dtype=False, check_categorical=False, check_exact=True)
    values = values.set_index('sample_id').loc[frame.sample_id].reset_index()
    stamp = pd.to_datetime(values.datetime_utc, utc=True)
    valid = pd.to_datetime(values.era5_land_valid_time_utc, utc=True)
    require(np.array_equal(valid, stamp.dt.floor('h')), 'Land valid time is not the exact UTC floor hour')
    age = (stamp-valid).dt.total_seconds().to_numpy() / 60
    require(np.array_equal(age, values.era5_land_time_age_minutes) and np.all((age >= 0) & (age < 60)), 'Invalid land age')
    require(values.era5_land_preflight_sha256.eq(preflight_sha256).all(), 'Mixed land metadata revisions')
    latitude_delta = values.era5_land_grid_latitude.to_numpy(float) - values.latitude.to_numpy(float)
    longitude_delta = (values.era5_land_grid_longitude.to_numpy(float)-values.longitude.to_numpy(float)+180)%360-180
    require(np.isfinite(latitude_delta).all() and np.isfinite(longitude_delta).all()
            and np.all(np.abs(latitude_delta) <= .050001) and np.all(np.abs(longitude_delta) <= .050001),
            'Land cell is not the nearest native support')
    finite = np.isfinite(values[list(LAND_VALUES)].to_numpy(float)).all(axis=1)
    physical = finite.copy()
    for column in LAND_VALUES[:3]:
        physical &= values[column].between(-173.15, 126.85).to_numpy()
    physical &= values[LAND_VALUES[3]].between(0, 10).to_numpy()
    physical &= values[LAND_VALUES[4]].between(0, 1).to_numpy()
    flag = old.strict_bool(values.era5_land_complete, 'era5_land_complete').to_numpy()
    require(not np.any(flag & ~physical), 'Land availability flag includes invalid physical values')
    finite = flag
    audit = {'directory': str(directory), 'files': {name: old.sha(path) for name, path in paths.items()},
             'rows': len(frame), 'available_rows': int(finite.sum()), 'missing_rows': int((~finite).sum()),
             'maximum_age_minutes': float(age.max()), 'period': period,
             'available_by_region': values.assign(available=finite).groupby('region_id', observed=True).available.agg(['size', 'sum']).reset_index().to_dict('records'),
             'retrospective_not_operational_forecast': True, 'row_drops': 0, 'imputation': False}
    require(values.loc[finite, 'era5_land_status'].eq('native_cell_available').all(), 'Unexpected land support status')
    require(values.loc[~finite, 'era5_land_status'].eq('native_cell_missing_or_invalid_no_land_substitution').all(), 'Unexpected missing support status')
    original = frame.copy()
    appended = [c for c in values if c not in IDENTITY]
    require(not set(appended).intersection(frame), 'Land fields already exist in input')
    data = pd.concat([frame.reset_index(drop=True), values[appended].reset_index(drop=True)], axis=1)
    pd.testing.assert_frame_equal(data[list(original)], original.reset_index(drop=True), check_exact=True)
    return data, audit


def attach_land(frame, directory, period):
    directory = Path(directory).resolve()
    plan = read_json(directory/'plan.json')
    if plan.get('delivery_format') == 'arco4_cds_swe_merge_v2':
        path = Path(__file__).with_name('merged_land_contract_v2.py')
        spec = importlib.util.spec_from_file_location('merged_land_contract', path)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        points, values, preflight_sha256, paths = contract.load(directory, period)
        data, audit = attach_land_values(frame, points, values, preflight_sha256, paths, period, directory)
        audit['delivery_format'] = contract.FORMAT
        return data, audit
    raise ValueError('Secondary experiment requires the approved merged native v2 delivery')


class CoverageError(ValueError):
    def __init__(self, audit):
        super().__init__(f"Physical input coverage incomplete: {audit['missing_rows']} of {audit['rows']} rows")
        self.audit = audit


def training_guard(frame):
    stamps = pd.to_datetime(frame.datetime_utc, utc=True)
    require(stamps.notna().all() and stamps.dt.year.isin([2021, 2022]).all(), 'Training rows outside 2021–22')
    require(frame.sample_id.notna().all() and not frame.sample_id.duplicated().any(), 'Duplicate fitting identities')
    require(frame.split.eq('fit').all() and not frame.region_id.eq('cabauw').any(), 'Reserved training rows')
    for name in ('spatial_holdout', 'in_holdout_buffer'):
        require(not old.strict_bool(frame[name], name).any(), 'Reserved spatial rows')
    require(old.complete_rows(frame, FEATURES).all(), 'Nonfinite learned inputs')


def targets(frame, name):
    require(name in FITTED, 'Unknown learned target')
    baseline = frame.air_temperature_c if name == 'land_features' else frame.land_station_adjusted_baseline_c
    y = frame.lst_c.to_numpy(float) - baseline.to_numpy(float)
    require(np.isfinite(y).all(), 'Nonfinite fitting target')
    return y


def freeze_availability(frame, output, family, land_audit):
    """Write the weather-only mask before any prediction/error computation."""
    output.mkdir(parents=True, exist_ok=True)
    include = old.strict_bool(frame.era5_land_complete, 'era5_land_complete').to_numpy()
    names = list(dict.fromkeys([*IDENTITY, 'phase', 'climate_class', 'split', 'utc_day', 'acquisition_id',
        'era5_land_valid_time_utc', 'era5_land_grid_latitude', 'era5_land_grid_longitude',
        'era5_land_complete', 'era5_land_status', 'era5_land_soil_moisture_zero_normalized',
        'era5_land_snow_water_equivalent_zero_normalized']))
    table = frame[names].copy()
    table['included'] = include
    reasons = []
    for position in range(len(frame)):
        if include[position]:
            reasons.append('')
        else:
            absent = [name for name in LAND_VALUES if not np.isfinite(frame[name].iloc[position])]
            reasons.append('missing:' + '|'.join(absent) if absent else 'invalid_native_physical_value_or_source_flag')
    table['availability_reason'] = reasons
    inventory = []
    dimensions = {
        'pilot_phase': ['region_id', 'phase'],
        'pilot_phase_date': ['region_id', 'phase', 'utc_day'],
        'acquisition': ['region_id', 'acquisition_id'],
        'climate': ['climate_class'],
        'native_cell': ['region_id', 'era5_land_grid_latitude', 'era5_land_grid_longitude'],
    }
    for kind, keys in dimensions.items():
        for key, group in table.groupby(keys, observed=True, dropna=False, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            kept = group.loc[group.included]
            inventory.append({'dimension': kind, **dict(zip(keys, key)), 'original_rows': len(group),
                'retained_rows': len(kept), 'excluded_rows': len(group)-len(kept),
                'retained_fraction': len(kept)/len(group), 'original_utc_dates': group.utc_day.nunique(),
                'retained_utc_dates': kept.utc_day.nunique(),
                'original_acquisitions': group.acquisition_id.nunique(), 'retained_acquisitions': kept.acquisition_id.nunique()})
    losses = [r for r in inventory if r['dimension'] == 'pilot_phase'
              and r['original_utc_dates'] >= GATES['supported_group_utc_dates']
              and r['retained_utc_dates'] < GATES['supported_group_utc_dates']]
    mask_path = output/(family+'_mask.parquet')
    inventory_path = output/(family+'_support.csv')
    table.to_parquet(mask_path, index=False)
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    frozen = {'family': family, 'original_rows': len(frame), 'retained_rows': int(include.sum()),
        'excluded_rows': int((~include).sum()), 'full_sample_id_sha256': old.row_hash(frame),
        'retained_sample_id_sha256': old.row_hash(frame.loc[include]),
        'excluded_sample_id_sha256': old.row_hash(frame.loc[~include]),
        'mask_path': str(mask_path), 'mask_sha256': old.sha(mask_path),
        'support_path': str(inventory_path), 'support_sha256': old.sha(inventory_path),
        'land_audit': land_audit, 'normalization_rows': int(table.era5_land_soil_moisture_zero_normalized.sum()),
        'snow_normalization_rows': int(table.era5_land_snow_water_equivalent_zero_normalized.sum()),
        'lost_supported_groups': losses, 'lost_support_is_not_a_guard_pass': True,
        'mask_uses_labels_or_predictions': False, 'created_before_new_error_scoring': True,
        'availability_only_fields': list(LAND_VALUES), 'source_alignment_required': True,
        'conditional_on_native_weather_coverage': True, 'no_imputation': True}
    old.save_json(output/(family+'_mask_freeze.json'), frozen)
    return include, frozen


def original_control_diagnostics(frame, saved, included, family):
    """Separate context only; never fed to selection or pooled into seven-arm scores."""
    rows = []
    for split, positions in frame.groupby('split', observed=True).indices.items():
        for status, keep in [('retained', included), ('excluded', ~included)]:
            ids = np.asarray([i for i in positions if keep[i]], dtype=int)
            if not len(ids):
                continue
            group = frame.iloc[ids].reset_index(drop=True)
            for name, prediction in saved.items():
                rows.append({'family': family, 'split': split, 'coverage': status,
                    'model': 'original_full_cohort_'+name, 'diagnostic_only': True,
                    'not_a_secondary_control': True, **old.metrics(group, prediction[ids])})
    return rows


def fit_models(frame, output, stage):
    training_guard(frame)
    output.mkdir(parents=True)
    import candidates
    weights = old.balanced_weights(frame)
    models, details = {}, {}
    for name in ALL_FITTED:
        started = time.monotonic()
        names = multi.BASE if name in ('F', 'extra_trees') else FEATURES
        y = old.target_offset(frame) if name in ('F', 'extra_trees') else targets(frame, name)
        if name == 'extra_trees':
            model = candidates.fit_candidate(name, frame, weights * len(frame))
        else:
            model, _ = old.build_estimators(names, old.CONFIG)
            model.fit(frame[list(names)], y, regressor__sample_weight=weights * len(frame))
        path = output/(name+'.joblib')
        joblib.dump(model, path, compress=3)
        models[name] = model
        details[name] = {'stage': stage, 'rows': len(frame), 'sample_id_sha256': old.row_hash(frame),
            'normalized_weight_sha256': digest(weights), 'target_sha256': digest(y),
            'input_feature_sha256': frame_digest(frame[list(names)]), 'features': list(names),
            'target_definition': 'Y-B' if name == 'land_residual' else 'Y-A',
            'reconstruction': 'B+prediction' if name == 'land_residual' else 'A+prediction',
            'total_fit_weight': len(frame), 'model_sha256': old.sha(path),
            'conditional_complete_native_refit': True, 'fit_seconds': time.monotonic()-started}
        old.save_json(output/'fits.json', details)
        log(stage=stage, model=name, seconds=details[name]['fit_seconds'])
    frame[['sample_id', 'region_id', 'phase', 'utc_day', 'datetime_utc']].assign(
        normalized_weight=weights, reference_target_c=old.target_offset(frame),
        land_features_target_c=targets(frame, 'land_features'),
        land_residual_target_c=targets(frame, 'land_residual')).to_parquet(output/'fitting_rows.parquet', index=False)
    return models


def predict(frame, models, references=None, *, components=None):
    require(references is None, 'Saved full-cohort predictions cannot be substituted for secondary controls')
    if frame.empty:
        if components is not None: components['extra_trees_lst_c'] = np.empty(0)
        return {name: np.empty(0) for name in LABELS}
    air = frame.air_temperature_c.to_numpy(float)
    refit_f = air + models['F'].predict(frame[list(multi.BASE)])
    refit_et = air + models['extra_trees'].predict(frame)
    if components is not None: components['extra_trees_lst_c'] = refit_et
    values = {'F': refit_f, 'blend_et25': .75*refit_f+.25*refit_et,
        'raw_land_skin': frame.era5_land_skin_temperature_c.to_numpy(float),
        'station_adjusted_land': frame.land_station_adjusted_baseline_c.to_numpy(float),
        'land_features': air + models['land_features'].predict(frame[list(FEATURES)]),
        'land_residual': frame.land_station_adjusted_baseline_c.to_numpy(float) + models['land_residual'].predict(frame[list(FEATURES)])}
    values['land_blend25'] = .75*values['F']+.25*values['land_residual']
    require(set(values) == set(LABELS) and all(v.shape == (len(frame),) and np.isfinite(v).all() for v in values.values()), 'Invalid secondary predictions')
    return values


def reproduce_reference_models(frame, source, reference, stage):
    result = []
    for name in ('F', 'extra_trees'):
        model = joblib.load(reference / stage / (name + '.joblib'))
        predicted = frame.air_temperature_c.to_numpy(float) + model.predict(frame)
        delta = float(np.max(np.abs(predicted-source[name + '_lst_c'].to_numpy(float))))
        require(delta <= 1e-9, f'Saved {stage}/{name} does not reproduce its predictions')
        result.append({'stage': stage, 'model': name, 'rows': len(frame), 'max_difference_c': delta,
                       'model_sha256': old.sha(reference / stage / (name + '.joblib'))})
    return result


def score(frame, predictions, cohort, prior):
    # Reuse precisely the old slicing/metric implementation without mutating it.
    original = prior.LABELS
    try:
        prior.LABELS = LABELS
        return prior.score(frame, predictions, cohort)
    finally:
        prior.LABELS = original


def select(metrics):
    require(set(metrics.cohort) == {'training_2021_22_oof'}, 'Selection accepts training OOF only')
    require(not metrics.duplicated(['segment_type', 'segment', 'model']).any(), 'Duplicate selection comparison')
    table = metrics.set_index(['segment_type', 'segment', 'model'])
    for key, subset in metrics.groupby(['segment_type', 'segment'], observed=True):
        require(set(subset.model) == set(LABELS), f'Missing selection arm: {key}')
        for field in ('n', 'date_count', 'utc_date_count', 'acquisition_count', 'sample_id_sha256'):
            require(subset[field].nunique(dropna=False) == 1, f'Unmatched selection support: {field}')
    numeric = ['mae_c', 'unweighted_pixel_mae_c', 'fraction_abs_error_gt_5c',
               'fraction_abs_error_gt_7c', 'centered_contrast_mae_c']
    require(np.isfinite(metrics[numeric].to_numpy(float)).all(), 'Nonfinite selection metrics')
    baseline = table.loc[('overall', 'overall', 'F')]
    groups = metrics.loc[(metrics.segment_type == 'region_phase') & (metrics.model == 'F')
                         & (metrics.utc_date_count >= GATES['supported_group_utc_dates'])].segment.tolist()
    decisions = []
    for name in TIE_ORDER:
        candidate = table.loc[('overall', 'overall', name)]
        failures = []
        if candidate.mae_c > baseline.mae_c - .10 + 1e-12:
            failures.append('overall balanced MAE improves less than 0.10 C')
        if candidate.unweighted_pixel_mae_c > baseline.unweighted_pixel_mae_c + .10 + 1e-12:
            failures.append('ordinary-observation MAE worsens more than 0.10 C')
        if candidate.centered_contrast_mae_c > baseline.centered_contrast_mae_c + .10 + 1e-12:
            failures.append('overall acquisition-centered contrast worsens more than 0.10 C')
        for kind, group in [('overall', 'overall'), *[('region_phase', g) for g in groups]]:
            a, b = table.loc[(kind, group, name)], table.loc[(kind, group, 'F')]
            if kind == 'region_phase' and a.mae_c > b.mae_c + .20 + 1e-12:
                failures.append(group + ': MAE worsens more than 0.20 C')
            for threshold in (5, 7):
                field = f'fraction_abs_error_gt_{threshold}c'
                if a[field] > b[field] + .02 + 1e-12:
                    failures.append(group + f': over-{threshold} C frequency increases more than 2 percentage points')
        decisions.append({'model': name, 'eligible': not failures, 'mae_c': candidate.mae_c,
                          'unweighted_pixel_mae_c': candidate.unweighted_pixel_mae_c,
                          'centered_contrast_mae_c': candidate.centered_contrast_mae_c, 'failures': failures})
    eligible = [d for d in decisions if d['eligible']]
    selected = None
    if eligible:
        best = min(d['mae_c'] for d in eligible)
        selected = next(d['model'] for d in eligible if d['mae_c'] <= best + .05 + 1e-12)
    return {'selected_candidate': selected, 'status': 'qualified_in_training_month_cv' if selected else 'no_qualifying_remedy',
            'selection_data': '2021–22 stitched held-calendar-month predictions only',
            'eligible_candidate_names': list(TIE_ORDER), 'diagnostic_reference_names': [n for n in LABELS if n not in TIE_ORDER],
            'supported_groups': groups, 'gates': GATES, 'tie_order': list(TIE_ORDER), 'decisions': decisions,
            'auto_promotion': False, 'reserved_2025_opened': False}


def coverage_summary(frame):
    rows = []
    for (region, phase), group in frame.groupby(['region_id', 'phase'], observed=True):
        record = {'region_id': region, 'phase': phase, 'rows': len(group), 'utc_dates': int(group.utc_day.nunique()),
                  'land_complete_rows': int(group.era5_land_complete.sum()),
                  'maximum_land_age_minutes': float(group.era5_land_time_age_minutes.max())}
        if 'station_pair_available' in group:
            record['station_pair_available_rows'] = int(old.strict_bool(group.station_pair_available, 'station_pair_available').sum())
        if 'air_temperature_source' in group:
            record['air_source_counts'] = group.air_temperature_source.fillna('unknown').value_counts().to_dict()
        if 'station_id' in group:
            record['station_id_present_rows'] = int(group.station_id.fillna('').astype(str).ne('').sum())
        record['air_definition'] = 'Unchanged stored air_temperature_c; raw reports and background fallbacks are not substituted'
        rows.append(record)
    return rows


def export_predictions(frame, values, path, *, fold=None, extra_trees=None):
    extra = [*IDENTITY, *LAND_VALUES, *EXTRA, 'land_station_adjusted_baseline_c',
             'era5_land_valid_time_utc', 'era5_land_time_age_minutes', 'era5_land_grid_latitude',
             'era5_land_grid_longitude', 'era5_land_preflight_sha256', 'era5_land_complete']
    output = frame[list(dict.fromkeys([*META, *extra]))].copy()
    if fold is not None:
        output['fold'] = fold
    for name, prediction in values.items():
        output[name + '_lst_c'] = prediction
    if extra_trees is not None:
        require(np.asarray(extra_trees).shape == (len(frame),) and np.isfinite(extra_trees).all(), 'Invalid refit ET component')
        output['extra_trees_lst_c'] = extra_trees
    output.to_parquet(path, index=False)


def source_paths(prior):
    return [Path(__file__), Path(old.__file__), Path(multi.__file__), Path(physics.__file__),
            Path(old.__file__).with_name('model.py'), Path(prior.__file__), PRIOR_CODE/'candidates.py',
            Path(__file__).with_name('era5_land.py'),
            *[Path(__file__).with_name(name) for name in ('merged_land_contract_v2.py',
              'merge_era5_land.py', 'era5_land_v3.py', 'cds_swe.py',
              'PROTOCOL_2026-09-11.md', 'SOURCE_CONTINUATION_2026-09-11.md',
              'SWE_NUMERICAL_DOMAIN_2026-09-11.md')]]


def fit_run(args):
    started = time.monotonic()
    reference, blends, out = args.reference.resolve(), args.blends_reference.resolve(), args.output.resolve()
    require(not out.exists(), 'Use a new immutable secondary run directory')
    require(args.protocol.name == 'SECONDARY_COMPLETE_NATIVE_PROTOCOL_2026-09-11.md', 'Secondary protocol required')
    stopped_path = Path('/opt/lst-pilot/runs/weather_baseline_20260911_v1/primary_full_cohort_stopped_v1.json')
    require(old.sha(stopped_path) == 'c468f9490e9a4987b2968d7827c587715252e1c0b0eceda077ff22e69e355031', 'Stopped primary evidence changed')
    stopped = read_json(stopped_path)
    require(stopped['status'] == 'stopped_before_model_fitting' and stopped['new_models_trained'] is False, 'Original primary is not preserved')
    for item in stopped['inputs'].values():
        require(old.sha(item['path']) == item['sha256'], 'Stopped-primary source evidence changed')
    prior = previous_runner()
    bindings = reference_bindings(reference, blends)
    path = reference/'fit_frame.parquet'
    check_years(path, (2021, 2022))
    original = pd.read_parquet(path)
    require(len(original) == 44316 and not original.sample_id.duplicated().any(), 'Expected all 44,316 source fitting rows')
    require(old.row_hash(original) == read_json(reference/'manifest.json')['fit_id_sha256'], 'Source fitting identity changed')
    out.mkdir(parents=True)
    attached, land_audit = attach_land(original, args.land_fit, 'fit')
    include, mask = freeze_availability(attached, out/'coverage', 'fit', land_audit)
    require(include.any(), 'No complete native fitting rows')
    frame = derive_land_features(attached.loc[include].reset_index(drop=True))
    training_guard(frame)
    fold = multi.month_folds(frame)
    require(set(fold) == {0, 1, 2}, 'Secondary cohort loses a training month fold')
    # Old full-cohort values are loaded only after the weather-only mask is immutable.
    saved, saved_frame = aligned_reference(original, reference, blends, 'oof')
    manifest = {'study': 'secondary_complete_native', 'research_only': True, 'auto_promotion': False,
        'reserved_2025_opened': False, 'protocol_path': str(args.protocol.resolve()), 'protocol_sha256': old.sha(args.protocol),
        'primary_stopped_path': str(stopped_path), 'primary_stopped_sha256': old.sha(stopped_path),
        'source_hashes': {str(p): old.sha(p) for p in source_paths(prior)}, 'bindings': bindings,
        'protected_baseline_path': bindings['protected_baseline_path'], 'protected_baseline_sha256': bindings['protected_baseline_sha256'],
        'labels': LABELS, 'learned_arms': list(FITTED), 'fitted_models': list(ALL_FITTED),
        'extra_features': list(EXTRA), 'features': list(FEATURES), 'base_config': asdict(old.CONFIG),
        'extra_trees_specification': __import__('candidates').SPECS['extra_trees'], 'gates': GATES,
        'source_fitting_rows': len(original), 'source_fit_id_sha256': old.row_hash(original),
        'fitting_rows': len(frame), 'fit_id_sha256': old.row_hash(frame),
        'fit_mask_freeze_path': str(out/'coverage/fit_mask_freeze.json'), 'fit_mask_freeze_sha256': old.sha(out/'coverage/fit_mask_freeze.json'),
        'fit_coverage': mask, 'original_air_sha256': digest(frame.air_temperature_c),
        'feature_sha256': frame_digest(frame[list(FEATURES)]), 'target_sha256': {n: digest(targets(frame, n)) for n in FITTED},
        'land_input_audit': land_audit, 'air_weather_support': coverage_summary(frame),
        'complete_cohort_required': False, 'all_seven_arms_same_native_complete_rows': True,
        'F_and_ET_refitted_on_each_same_fold': True, 'conditional_on_native_weather_availability': True,
        'retrospective_reanalysis': True, 'python': sys.version, 'sklearn': sklearn.__version__}
    old.save_json(out/'manifest.json', manifest)
    frame.to_parquet(out/'fit_frame.parquet', index=False)
    values = {name: np.full(len(frame), np.nan) for name in LABELS}
    et_oof = np.full(len(frame), np.nan)
    with threadpool_limits(limits=4):
        for number in range(3):
            held_mask = fold == number
            train, held = frame.loc[~held_mask].reset_index(drop=True), frame.loc[held_mask].reset_index(drop=True)
            require(set(train.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(held.datetime_utc.dt.strftime('%Y-%m')), 'Month crosses folds')
            models = fit_models(train, out/f'fold_{number}', f'fold_{number}')
            held[['sample_id', 'datetime_utc', 'utc_day', 'region_id', 'phase']].to_parquet(out/f'fold_{number}'/'held_rows.parquet', index=False)
            components = {}
            for name, prediction in predict(held, models, components=components).items():
                values[name][held_mask] = prediction
            et_oof[held_mask] = components['extra_trees_lst_c']
            del models
    require(all(np.isfinite(v).all() for v in values.values()), 'Missing secondary OOF predictions')
    export_predictions(frame, values, out/'oof_predictions.parquet', fold=fold, extra_trees=et_oof)
    metrics = pd.DataFrame(score(frame, values, 'training_2021_22_oof', prior))
    metrics.to_csv(out/'oof_metrics.csv', index=False)
    selection = select(metrics)
    selection.update({'conditional_on_native_weather_availability': True,
        'reference_model': 'F_complete_native_refit', 'lost_supported_groups': mask['lost_supported_groups'],
        'support_loss_is_not_a_pass': True, 'original_primary_stopped': True})
    old.save_json(out/'selection.json', selection)
    frozen_files = {str(p.relative_to(out)): old.sha(p) for p in out.rglob('*') if p.is_file()}
    old.save_json(out/'selection_freeze.json', {'files': frozen_files,
        'evaluation_labels_decoded_before_selection': False, 'selected_candidate': selection['selected_candidate']})
    log(stage='selection_frozen', **selection)
    pd.DataFrame(original_control_diagnostics(original, saved, include, 'fit_oof')).to_csv(out/'original_full_cohort_control_diagnostics.csv', index=False)
    with threadpool_limits(limits=4):
        fit_models(frame, out/'full', 'full')
    full_files = {str(p.relative_to(out)): old.sha(p) for p in (out/'full').iterdir() if p.is_file()}
    old.save_json(out/'full_fit_freeze.json', {'files': full_files,
        'selection_freeze_sha256': old.sha(out/'selection_freeze.json'), 'frozen_before_evaluation': True})
    verify_new_freeze(out)
    require(reference_bindings(reference, blends) == bindings, 'Original references changed during fitting')
    for name, checksum in land_audit['files'].items():
        require(old.sha(Path(land_audit['directory'])/name) == checksum, 'Fitting weather extraction changed')
    old.save_json(out/'fitting_complete.json', {'status': 'complete', 'evaluation_labels_decoded': False,
        'study': 'secondary_complete_native', 'fit_mask_freeze_sha256': old.sha(out/'coverage/fit_mask_freeze.json'),
        'selection_freeze_sha256': old.sha(out/'selection_freeze.json'), 'full_fit_freeze_sha256': old.sha(out/'full_fit_freeze.json'),
        'seconds': time.monotonic()-started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'protected_baseline_sha256': old.sha(Path(bindings['protected_baseline_path'])), 'production_unchanged': True})
    log(stage='fitting_complete', rows=len(frame), seconds=time.monotonic()-started)


def verify_new_freeze(run):
    manifest = read_json(run/'manifest.json')
    require(old.sha(manifest['primary_stopped_path']) == manifest['primary_stopped_sha256'], 'Stopped primary evidence changed')
    for item in read_json(manifest['primary_stopped_path'])['inputs'].values():
        require(old.sha(item['path']) == item['sha256'], 'Stopped-primary source evidence changed')
    require(old.sha(manifest['fit_mask_freeze_path']) == manifest['fit_mask_freeze_sha256'], 'Training availability mask changed')
    require(old.sha(Path(manifest['protected_baseline_path'])) == manifest['protected_baseline_sha256'],
            'Protected production baseline changed')
    require(old.sha(manifest['protocol_path']) == manifest['protocol_sha256'], 'New protocol changed')
    for path, checksum in manifest['source_hashes'].items():
        require(old.sha(path) == checksum, f'New dependency changed: {path}')
    frozen = read_json(run/'selection_freeze.json')
    require(frozen.get('evaluation_labels_decoded_before_selection') is False, 'Training freeze unavailable')
    for name, checksum in frozen['files'].items():
        require(old.sha(run/name) == checksum, f'Selection/fitting evidence changed: {name}')
    full = read_json(run/'full_fit_freeze.json')
    require(full.get('frozen_before_evaluation') is True and full['selection_freeze_sha256'] == old.sha(run/'selection_freeze.json'),
            'Full fits not bound to selection freeze')
    for name, checksum in full['files'].items():
        require(old.sha(run/name) == checksum, f'Full learned fit changed: {name}')
    return manifest


def evaluation_frames(prior, frame):
    """Exactly the prior comparison's evaluation assembly, after the new freeze."""
    reference = multi.verify_run(prior.BASE_RUN)
    physics.verify_inputs(reference)
    reconstructed, _, _, original_eval, known, areas = physics.reconstruct_fit(reference, prior.BASE_RUN)
    pd.testing.assert_frame_equal(reconstructed[list(multi.BASE)], frame[list(multi.BASE)], check_dtype=False, check_exact=True)
    require(np.array_equal(old.target_offset(reconstructed), old.target_offset(frame)), 'Prior fit reconstruction differs')
    paths = reference['paths']
    new_frame, registry = multi.load_new_evaluation(paths['new_evaluation'], paths['freshness_audit'], reference['input_hashes']['freshness_audit'], known)
    new_eval = multi.validate_new(new_frame, known, areas, evaluation=True,
                                 freshness_sha=reference['input_hashes']['freshness_audit'], registry=registry)
    require(old.sha(prior.LEGACY) == reference['reference_hashes']['legacy_2024_input'], 'Legacy 2024 source changed')
    legacy = old.prepare_input(old.load_paired_input(prior.LEGACY, evaluation_2024=True), evaluation_2024=True)
    baseline = joblib.load(paths['baseline'])
    needed = tuple(dict.fromkeys([*multi.BASE, *baseline['features']]))
    legacy = legacy.loc[old.complete_rows(legacy, needed) & legacy.phase.isin(multi.PHASES)].sort_values('sample_id').reset_index(drop=True)
    legacy['split'] = 'legacy_2024'
    return {'old': original_eval, 'newly_collected_2023': new_eval, 'legacy_2024': legacy}


def evaluate_run(args):
    started, run = time.monotonic(), args.run.resolve()
    require(not (run/'completion.json').exists(), 'Completed evaluation is immutable')
    ready = read_json(run/'fitting_complete.json')
    require(ready['status'] == 'complete' and ready['evaluation_labels_decoded'] is False, 'Fitting is incomplete')
    manifest = verify_new_freeze(run)
    require(ready['selection_freeze_sha256'] == old.sha(run/'selection_freeze.json')
            and ready['full_fit_freeze_sha256'] == old.sha(run/'full_fit_freeze.json'), 'Fitting completion changed')
    prior = previous_runner()
    reference, blends = (Path(manifest['bindings'][k]) for k in ('reference', 'blends_reference'))
    require(reference_bindings(reference, blends) == manifest['bindings'], 'Prior references changed')
    fit = pd.read_parquet(run/'fit_frame.parquet')
    source_fit = pd.read_parquet(reference/'fit_frame.parquet')
    indexed_source = source_fit.set_index('sample_id').loc[fit.sample_id].reset_index()
    pd.testing.assert_frame_equal(indexed_source[list(multi.BASE)], fit[list(multi.BASE)], check_dtype=False, check_exact=True)
    require(np.array_equal(old.target_offset(indexed_source), old.target_offset(fit)), 'Retained source targets changed')
    frames = evaluation_frames(prior, source_fit)
    weather = dict(zip(FAMILIES, (args.land_old, args.land_new, args.land_legacy)))
    joined, audits, masks, saved_references = {}, {}, {}, {}
    for family, frame in frames.items():
        require(pd.to_datetime(frame.datetime_utc, utc=True).dt.year.isin([2021, 2022, 2023, 2024]).all(), 'Evaluation exceeds 2024')
        attached, audits[family] = attach_land(frame, weather[family], 'evaluation')
        included, coverage = freeze_availability(attached, run/'coverage', family, audits[family])
        masks[family] = included
        joined[family] = derive_land_features(attached.loc[included].reset_index(drop=True))
        saved_references[family], _ = aligned_reference(frame, reference, blends, family)
    # All three complete weather-only masks are frozen before any new model/error computation.
    old.save_json(run/'evaluation_mask_freeze.json', {
        'files': {str(p.relative_to(run)): old.sha(p) for p in (run/'coverage').iterdir() if p.is_file()},
        'created_before_new_error_scoring': True, 'evaluation_labels_follow_training_selection': True})
    models = {name: joblib.load(run/'full'/(name+'.joblib')) for name in ALL_FITTED}
    rows, acquisitions, reproduction = [], [], []
    for family, frame in joined.items():
        components = {}
        with threadpool_limits(limits=4):
            values = predict(frame, models, components=components)
        export_predictions(frame, values, run/(family+'_predictions.parquet'), extra_trees=components['extra_trees_lst_c'])
        reproduction.append({'family': family, 'rows': len(frame), 'F_and_ET': 'complete-native refits',
            'all_seven_arms_same_rows': True, 'original_full_cohort_controls_not_substituted': True})
        for split, ids in frame.groupby('split', observed=True).indices.items():
            cohort = 'legacy_2024' if family == 'legacy_2024' else family+'_'+split
            group = frame.iloc[ids].reset_index(drop=True)
            px = {name: value[ids] for name, value in values.items()}
            rows.extend(score(group, px, cohort, prior))
            focal = group.loc[group.region_id.isin(['greater_london', 'sioux_falls'])]
            for acq, positions in focal.groupby('acquisition_id', observed=True).groups.items():
                data = group.loc[positions].reset_index(drop=True)
                weight = old.balanced_weights(data)
                for name, value in px.items():
                    prediction = value[positions]
                    error = np.abs(prediction-data.lst_c.to_numpy(float))
                    acquisitions.append({'cohort': cohort, 'region_id': data.region_id.iloc[0], 'phase': data.phase.iloc[0],
                        'acquisition_id': acq, 'date': data.utc_day.iloc[0].strftime('%Y-%m-%d'),
                        'timestamp_utc': data.datetime_utc.iloc[0].isoformat(), 'model': name, 'model_label': LABELS[name],
                        **old.metrics(data, prediction), 'raw_gt5_count': int((error > 5).sum()),
                        'raw_gt7_count': int((error > 7).sum()), 'raw_gt5_fraction': float((error > 5).mean()),
                        'raw_gt7_fraction': float((error > 7).mean()),
                        'observed_mean_c': float(weight @ data.lst_c.to_numpy(float)),
                        'predicted_mean_c': float(weight @ prediction),
                        'air_mean_c': float(weight @ data.air_temperature_c.to_numpy(float)),
                        'land_skin_mean_c': float(weight @ data.era5_land_skin_temperature_c.to_numpy(float)),
                        'land_air_mean_c': float(weight @ data.era5_land_air_temperature_c.to_numpy(float)),
                        'land_snow_swe_mean_m': float(weight @ data.land_snow_water_equivalent_m.to_numpy(float))})
        audits[family]['air_weather_support'] = coverage_summary(frame)
        log(stage='scored', family=family, rows=len(frame))
    diagnostics = []
    for family, source_frame in frames.items():
        diagnostics.extend(original_control_diagnostics(source_frame, saved_references[family], masks[family], family))
    pd.DataFrame(diagnostics).to_csv(run/'original_full_cohort_evaluation_diagnostics.csv', index=False)
    pd.DataFrame(rows).to_csv(run/'metrics.csv', index=False)
    pd.DataFrame(acquisitions).to_csv(run/'acquisition_metrics.csv', index=False)
    old.save_json(run/'evaluation_land_audits.json', audits)
    old.save_json(run/'reproduction.json', reproduction)
    verify_new_freeze(run)
    require(reference_bindings(reference, blends) == manifest['bindings'], 'References changed during evaluation')
    for audit in [manifest['land_input_audit'], *audits.values()]:
        for name, checksum in audit['files'].items():
            require(old.sha(Path(audit['directory'])/name) == checksum, 'Land extraction changed during run')
    old.save_json(run/'completion.json', {'status': 'complete', 'study': 'secondary_complete_native',
        'conditional_on_native_weather_availability': True,
        'evaluation_mask_freeze_sha256': old.sha(run/'evaluation_mask_freeze.json'), 'evaluation_seconds': time.monotonic()-started,
        'fitting_seconds': ready['seconds'], 'fitting_peak_rss_bytes': ready['peak_rss_bytes'],
        'evaluation_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'peak_rss_bytes': max(ready['peak_rss_bytes'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        'protected_baseline_sha256': old.sha(Path(manifest['protected_baseline_path'])),
        'research_only': True, 'reserved_2025_opened': False, 'production_unchanged': True,
        'artifacts': {str(path.relative_to(run)): old.sha(path) for path in run.rglob('*') if path.is_file()}})
    log(stage='complete', seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--reference', type=Path, default=Path('/opt/lst-pilot/runs/shared_model_remedies_20260910_v1'))
    fit.add_argument('--blends-reference', type=Path, default=Path('/opt/lst-pilot/runs/shared_model_blends_20260910_v1'))
    fit.add_argument('--land-fit', type=Path, required=True)
    fit.add_argument('--protocol', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--run', type=Path, required=True)
    for family in ('old', 'new', 'legacy'):
        evaluate.add_argument('--land-'+family, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        fit_run(args)
    else:
        evaluate_run(args)


if __name__ == '__main__':
    main()
