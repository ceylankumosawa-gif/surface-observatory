"""One prospective seasonal representation on the audited expanded cohort.

Preparation is impossible before the geographic comparison passes audit. Only
the two calendar fields change; targets, support, fitting recipes and all other
29 fields are inherited. No source retrieval or deployment is provided.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import expanded_native_trial_v1 as baseline
import native_solar_season_v2 as season
from native_model_trial import bind, read, save, frame_sha

KEY = ['native_cell_id', 'region_id']
FEATURES = list(season.OUTPUT_FEATURES)
MODES = ['geography', 'month', 'reference']
PREDICTORS = ['expanded', 'solar_season', 'raw_air', 'raw_skin']
PROTOCOL = Path(__file__).with_name('EXPANDED_SOLAR_SEASON_PROTOCOL.md')


def same_recipe(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def sidecars(proof_path):
    proof = read(proof_path)
    baseline.need(proof['status'] == 'passed', 'Passed independent geometry check required')
    frames, inputs = [], [bind(proof_path)]
    for binding in proof['sidecar_completions']:
        done = read(baseline.verify(binding))
        baseline.need(done['status'] == 'complete_source_only_solar_season', 'Incomplete solar sidecar')
        baseline.need(done['fields'] == list(season.FIELDS), 'Different seasonal fields')
        artifact = done['artifacts']['features_midpoint.parquet']
        f = pd.read_parquet(baseline.verify(artifact), columns=KEY + ['native_footprint_area_m2',
            'datetime_utc'] + list(season.FIELDS) + [x + '__valid_area_fraction' for x in season.FIELDS])
        baseline.need(len(f) == done['rows'], 'Changed sidecar row count')
        for field in season.FIELDS:
            baseline.need(np.isfinite(f[field]).all(), 'Nonfinite seasonal field')
            baseline.need(f[field + '__valid_area_fraction'].ge(1 - 1e-8).all(), 'Partial solar support')
        frames.append(f)
        inputs.extend([binding, done['plan'], artifact])
    f = pd.concat(frames, ignore_index=True)
    baseline.need(len(f) == proof['total_rows'] == 180253, 'Changed geometry population')
    baseline.need(not f.duplicated(KEY).any(), 'Duplicate solar identity')
    return f.set_index(KEY), inputs


def matrix(frame, solar):
    """Identity join with exact midpoint/area checks; never alter admission."""
    index = pd.MultiIndex.from_frame(frame[KEY])
    baseline.need(index.is_unique and solar.index.is_unique, 'Duplicate join identity')
    positions = solar.index.get_indexer(index)
    baseline.need((positions >= 0).all(), 'A previously complete row lacks solar geometry')
    f = solar.iloc[positions]
    np.testing.assert_array_equal(f.native_footprint_area_m2.to_numpy(),
                                  frame.native_footprint_area_m2.to_numpy())
    start = pd.to_datetime(frame.granule_start_utc, utc=True, format='mixed')
    end = pd.to_datetime(frame.granule_end_utc, utc=True, format='mixed')
    expected_time = start + (end - start) / 2
    actual_time = pd.to_datetime(f.datetime_utc, utc=True, format='mixed')
    np.testing.assert_array_equal(actual_time.to_numpy(), expected_time.to_numpy())
    result = season.replace_calendar(frame[baseline.FEATURES], f[list(season.FIELDS)].to_numpy())
    baseline.need(np.isfinite(result.to_numpy()).all(), 'Complete fitting/prediction rows required')
    return result


def fingerprint(membership, values):
    return hashlib.sha256(json.dumps({'recipe': frame_sha(membership),
        'features': frame_sha(values), 'parameters': baseline.PARAMS, 'fields': FEATURES},
        sort_keys=True).encode()).hexdigest()


def specifications(d, g, associations, reserved, solar, expected):
    for split, arm, positions, train, membership, original in baseline.specifications(d, g, associations, reserved):
        if arm != 'expanded':
            continue
        baseline.need(same_recipe(original, expected[split.mode, split.key]), 'Expanded baseline recipe changed')
        transformed = matrix(train, solar)
        spec = dict(original, baseline_fingerprint=original['fingerprint'],
                    transformed_feature_sha256=frame_sha(transformed),
                    fingerprint=fingerprint(membership, transformed))
        yield split, train, membership, transformed, spec


def prepare(trial, audit_path, geometry_check, out):
    baseline.offline()
    started = time.monotonic()
    baseline.need(not out.exists(), 'New trial directory required')
    audit = read(audit_path)
    done = read(trial / 'completion.json')
    baseline.need(audit['status'] == 'passed' and audit['trial_completion'] == bind(trial / 'completion.json'),
                  'Geographic comparison must pass independent audit first')
    baseline.need(done['status'] == 'complete_exploratory_data_comparison', 'Complete geography trial required')
    bp = read(baseline.verify(done['plan']))
    baseline.need(bp['features'] == baseline.FEATURES and bp['parameters'] == baseline.PARAMS,
                  'Different baseline recipe')
    for binding in bp['inputs'].values():
        baseline.verify(binding)
    d, g, a, reserved, _ = baseline.load(bp['inputs']['cohort_receipt']['path'])
    baseline.need(frame_sha(d) == bp['cohort_frame_sha256'], 'Baseline cohort changed')
    solar, inputs = sidecars(geometry_check)
    # Check all supported held rows, including reference and fit-excluded rows.
    matrix(d.loc[d.feature_complete], solar)
    expected = {(x['mode'], x['key']): x for x in bp['fits'] if x['arm'] == 'expanded'}
    fits = [spec for *_, spec in specifications(d, g, a, reserved, solar, expected)]
    baseline.need(len(fits) == len(expected) == 45, 'Exactly 45 expanded recipes required')
    controls = {mode: done['artifacts'][mode + '_predictions.parquet'] for mode in MODES}
    inputs.extend([bind(__file__), bind(PROTOCOL), bind(season.__file__), bind(baseline.__file__),
                   bind(trial / 'completion.json'), bind(audit_path), done['plan']])
    inputs.extend(bp['inputs'].values())
    inputs.extend(controls.values())
    for binding in inputs:
        baseline.verify(binding)
    out.mkdir(parents=True)
    save(out / 'plan.json', {'status': 'frozen_before_fits', 'bindings': inputs,
        'baseline_plan': done['plan'], 'geometry_check': bind(geometry_check),
        'cohort_receipt': bp['inputs']['cohort_receipt'], 'cohort_frame_sha256': frame_sha(d),
        'baseline_predictions': controls, 'fits': fits, 'features': FEATURES,
        'parameters': baseline.PARAMS, 'baseline': baseline.AIR, 'row_count': len(d),
        'requested_groups': 960, 'preparation_wall_seconds': time.monotonic() - started,
        'limits': {'maximum_fit_recipes': 45, 'cpu_threads': 2, 'memory_bytes': 6 * 2**30, 'seconds': 1800},
        'network_allowed': False, 'automatic_promotion': False, 'production_changed': False})
    print(json.dumps(bind(out / 'plan.json')), flush=True)


def score(predictions, registry):
    # Reuse the already-checked metric definition without renaming stored data.
    renamed = predictions.rename(columns={'expanded': 'old_only', 'solar_season': 'expanded'})
    rows = baseline.score(renamed, registry)
    names = {'old_only': 'expanded', 'expanded': 'solar_season', 'raw_air': 'raw_air', 'raw_skin': 'raw_skin'}
    return [dict(row, model=names[row['model']]) for row in rows]


def run(out):
    baseline.offline()
    started, cpu = time.monotonic(), time.process_time()
    plan = read(out / 'plan.json')
    baseline.need(not (out / 'models').exists() and not (out / 'fits.json').exists(), 'No automatic fitting restart')
    for binding in plan['bindings']:
        baseline.verify(binding)
    baseline.need(plan['features'] == FEATURES and plan['parameters'] == baseline.PARAMS, 'Changed candidate')
    bp = read(baseline.verify(plan['baseline_plan']))
    d, g, a, reserved, _ = baseline.load(plan['cohort_receipt']['path'])
    baseline.need(frame_sha(d) == plan['cohort_frame_sha256'], 'Changed cohort')
    solar, _ = sidecars(plan['geometry_check']['path'])
    expected = {(x['mode'], x['key']): x for x in bp['fits'] if x['arm'] == 'expanded'}
    frozen = {(x['mode'], x['key']): x for x in plan['fits']}
    predictions = {m: np.full(len(d), np.nan) for m in MODES}
    fit_indices = {m: np.full(len(d), -1, dtype=np.int16) for m in MODES}
    completed, models, training = [], {}, []
    complete = d.feature_complete.to_numpy()
    for split, train, membership, transformed, spec in specifications(d, g, a, reserved, solar, expected):
        baseline.need(time.monotonic() - started < plan['limits']['seconds'] - 30, 'Fitting deadline reached')
        baseline.need(same_recipe(spec, frozen[split.mode, split.key]), 'Prepared recipe changed')
        if not spec['eligible']:
            completed.append(dict(spec, status='insufficient_fitting_support'))
            save(out / 'fits.json', completed)
            continue
        key = spec['fingerprint']
        if key in models:
            binding = models[key]
            model = joblib.load(baseline.verify(binding['model']))
            result = dict(spec, status='reused_exact_recipe', **binding)
        else:
            target = out / 'models' / split.mode / split.key
            target.mkdir(parents=True)
            membership.to_parquet(target / 'fitting_rows.parquet', index=False)
            model = HistGradientBoostingRegressor(**baseline.PARAMS)
            model.fit(transformed, membership.target_residual_c,
                      sample_weight=membership.weight.to_numpy() * len(train))
            joblib.dump(model, target / 'model.joblib', compress=3)
            binding = {'model': bind(target / 'model.joblib'), 'fitting_rows': bind(target / 'fitting_rows.parquet'),
                       'origin': '/'.join((split.mode, split.key))}
            models[key] = binding
            result = dict(spec, status='fitted', **binding)
        if split.mode != 'full':
            mode = 'geography' if split.mode in ['original_pilot', 'new_pilot'] else split.mode
            held = split.held_positions[complete[split.held_positions]]
            if len(held):
                baseline.need((fit_indices[mode][held] == -1).all(), 'Held prediction overwritten')
                predictions[mode][held] = d.iloc[held][baseline.AIR].to_numpy() + model.predict(matrix(d.iloc[held], solar))
                fit_indices[mode][held] = len(completed)
        delta = model.predict(transformed) - membership.target_residual_c.to_numpy()
        training.append({'mode': split.mode, 'key': split.key, 'rows': len(train),
                         'mae': float(np.abs(delta).mean()),
                         'balanced_mae': float(membership.weight.to_numpy() @ np.abs(delta))})
        completed.append(result)
        save(out / 'fits.json', completed)
        print(json.dumps({'mode': split.mode, 'key': split.key, 'status': result['status'],
                          'elapsed_seconds': time.monotonic() - started}), flush=True)
        del model, transformed, train, membership
    identities = d[baseline.IDENTITY + baseline.EXTRA + baseline.MASKS + ['lst_c']].copy()
    del d, solar
    gc.collect()
    # Store candidate predictions before reading baseline prediction values.
    for mode in MODES:
        keep = identities.region_id.eq('cabauw').to_numpy() if mode == 'reference' else identities.region_id.ne('cabauw').to_numpy()
        p = identities.loc[keep].reset_index(drop=True)
        p['solar_season'] = predictions[mode][keep]
        p['solar_season_fit_index'] = fit_indices[mode][keep]
        p.to_parquet(out / (mode + '_candidate_predictions.parquet'), index=False)
    del identities, predictions, fit_indices, p
    gc.collect()
    metrics = []
    for mode in MODES:
        candidate = pd.read_parquet(out / (mode + '_candidate_predictions.parquet'))
        control = pd.read_parquet(baseline.verify(plan['baseline_predictions'][mode]))
        columns = baseline.IDENTITY + baseline.EXTRA + baseline.MASKS + ['lst_c']
        pd.testing.assert_frame_equal(candidate[columns], control[columns], check_exact=True)
        np.testing.assert_array_equal(np.isfinite(candidate.solar_season), np.isfinite(control.expanded))
        p = control.drop(columns=['old_only', 'old_only_fit_index']).copy()
        p['solar_season'] = candidate.solar_season
        p['solar_season_fit_index'] = candidate.solar_season_fit_index
        p.to_parquet(out / (mode + '_predictions.parquet'), index=False)
        metrics.extend(dict(test_mode=mode, **row) for row in score(p, g))
        del p, candidate, control
        gc.collect()
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(training).to_csv(out / 'training_metrics.csv', index=False)
    for binding in plan['bindings']:
        baseline.verify(binding)
    save(out / 'completion.json', {'status': 'complete_exploratory_solar_season_comparison',
        'plan': bind(out / 'plan.json'), 'fit_recipes': len(completed),
        'unique_fits': sum(x['status'] == 'fitted' for x in completed),
        'exact_recipe_reuses': sum(x['status'] == 'reused_exact_recipe' for x in completed),
        'requested_groups': 960, 'row_count': plan['row_count'],
        'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'requests': 0, 'production_changed': False, 'globally_qualified': False,
        'artifacts': {p.name: bind(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(bind(out / 'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--baseline-trial', type=Path)
    parser.add_argument('--baseline-audit', type=Path)
    parser.add_argument('--geometry-check', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.mode == 'prepare':
            prepare(args.baseline_trial, args.baseline_audit, args.geometry_check, args.output)
        else:
            run(args.output)
