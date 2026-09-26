"""Paired geographic-data experiment; unchanged small 31-feature AIR model.

The only intended difference is old-only versus old-plus-development training.
All new/reserved source and feature checks precede preparation. No retrieval or
automatic promotion is provided by this runner.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import expanded_native_design_v1 as design
import ground_native_trial as common
import native_trial_design as weighting
import native_model_trial as native_io
from ground_native_trial import FEATURES, PARAMS, AIR, SKIN
from native_trial_design import balanced_weights
from native_model_trial import IDENTITY, MASKS, sha, bind, read, save, frame_sha

ARMS = ('old_only', 'expanded')
PREDICTORS = [*ARMS, 'raw_air', 'raw_skin']
EXTRA = ['native_row', 'native_col', 'physical_acquisition_key']
GROUP = ['region_id', 'date', 'phase']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def verify(binding):
    need(sha(binding['path']) == binding['sha256'], 'Changed input: ' + binding['path'])
    return Path(binding['path'])


def offline():
    def deny(event, args):
        if event == 'socket.connect':
            raise RuntimeError('Model trial prohibits network')
    sys.addaudithook(deny)


def load(receipt_path):
    receipt = read(receipt_path)
    need(receipt['status'] == 'ready_for_exploratory_native_trial', 'Checked cohort required')
    columns = list(dict.fromkeys(IDENTITY + EXTRA + MASKS + ['lst_c'] + FEATURES))
    d = pd.read_parquet(verify(receipt['cohort']), columns=columns).reset_index(drop=True)
    g = pd.read_csv(verify(receipt['registry']), dtype=str, keep_default_na=False)
    a = pd.read_csv(verify(receipt['source_associations']), dtype=str, keep_default_na=False)
    reserved = pd.read_csv(verify(receipt['reserved_physical_acquisitions']), dtype=str)
    keys = set(reserved.physical_acquisition_key)
    need(len(g) == 960 and g.region_id.nunique() == 48, 'Full 960-case, 48-area registry required')
    need(len(reserved) == len(keys) == 171, 'Reserved physical-pass denominator changed')
    design.validate_metadata(d, g, a, keys)
    need(np.isfinite(d.loc[d.native_label_admitted, 'lst_c']).all(), 'Nonfinite admitted label')
    need(np.isfinite(d.loc[d.feature_complete, FEATURES].to_numpy(float)).all(), 'Invalid complete feature row')
    need((np.isfinite(d.native_footprint_area_m2) & d.native_footprint_area_m2.gt(0)).all(), 'Invalid native area')
    return d, g, a, keys, receipt


def recipe(d, positions):
    train = d.iloc[positions]
    membership = train[IDENTITY + EXTRA].copy()
    membership['weight'] = balanced_weights(train)
    membership['baseline_c'] = train[AIR].to_numpy()
    membership['target_residual_c'] = (train.lst_c - train[AIR]).to_numpy()
    eligible = len(train) >= 80 and train.region_id.nunique() >= 2 and train.utc_date.nunique() >= 2
    return train, membership, eligible


def fingerprint(membership, features):
    return hashlib.sha256(json.dumps({'recipe': frame_sha(membership),
        'features': frame_sha(features), 'parameters': PARAMS, 'fields': FEATURES},
        sort_keys=True).encode()).hexdigest()


def specifications(d, g, associations, reserved):
    for split in design.paired_split_definitions(d, g, associations, reserved):
        for arm, positions in [('old_only', split.old_only_train_positions),
                               ('expanded', split.expanded_train_positions)]:
            train, membership, eligible = recipe(d, positions)
            yield split, arm, positions, train, membership, {
                'mode': split.mode, 'key': split.key, 'arm': arm,
                'rows': len(train), 'eligible': eligible,
                'recipe_sha256': frame_sha(membership),
                'feature_sha256': frame_sha(train[FEATURES]),
                'fingerprint': fingerprint(membership, train[FEATURES]),
                'held_positions_sha256': hashlib.sha256(split.held_positions.astype('<i8').tobytes()).hexdigest(),
                'held_rows': len(split.held_positions),
                'held_group_ids': split.held_group_ids,
                'excluded_physical_keys': split.excluded_physical_keys,
            }


def prepare(receipt, independent_check, out):
    offline()
    started = time.monotonic()
    need(not out.exists(), 'New immutable trial directory required')
    proof = read(independent_check)
    need(proof['status'] == 'passed', 'Independent cohort check required')
    need(proof['cohort_receipt'] == bind(receipt), 'Independent check binds another cohort')
    d, g, a, keys, done = load(receipt)
    specs = [spec for *_, spec in specifications(d, g, a, keys)]
    need(len(specs) == 90, 'Expected 45 split pairs, retaining zero-support places')
    out.mkdir(parents=True)
    inputs = {'cohort_receipt': bind(receipt), 'cohort_check': bind(independent_check),
              'runner': bind(__file__), 'design': bind(design.__file__),
              'common_model': bind(common.__file__),
              'weighting': bind(weighting.__file__), 'native_io': bind(native_io.__file__),
              'protocol': bind(Path(__file__).with_name('EXPANDED_NATIVE_TRIAL_PROTOCOL.md'))}
    for name in ['cohort', 'registry', 'source_associations', 'reserved_physical_acquisitions']:
        inputs[name] = done[name]
    save(out / 'plan.json', {
        'status': 'frozen_before_fits', 'inputs': inputs, 'features': FEATURES,
        'parameters': PARAMS, 'baseline': AIR, 'fits': specs, 'row_count': len(d),
        'requested_groups': len(g), 'cohort_frame_sha256': frame_sha(d),
        'limits': {'maximum_fit_recipes': 90, 'cpu_threads': 2,
                   'memory_bytes': 6 * 2**30, 'seconds': 1800},
        'identical_recipe_reuse': 'within this experiment only; exact ordered identity, targets, weights, features and parameters',
        'saved_previous_model_reuse': False, 'network_allowed': False,
        'automatic_promotion': False, 'production_changed': False,
        'preparation_seconds': time.monotonic() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    })
    print(json.dumps({'plan': bind(out / 'plan.json'), 'rows': len(d),
                      'fit_recipes': len(specs),
                      'unique_fingerprints': len({s['fingerprint'] for s in specs if s['eligible']})}), flush=True)


def segments(p, registry):
    """Build indices once; retain every requested empty date/phase segment."""
    yield 'overall', 'all', np.arange(len(p))
    for family in ['original_development', 'new_development', 'original_reference', 'reserved_new_geography']:
        yield 'geography_role', family, np.flatnonzero(p.geography_role.eq(family))
    by_phase = p.groupby(['region_id', 'phase'], observed=True).indices
    for region, phase in sorted(set(zip(registry.region_id, registry.phase))):
        yield 'pilot_phase', region + '|' + phase, by_phase.get((region, phase), np.empty(0, int))
    by_date = p.groupby(['region_id', 'utc_date', 'phase'], observed=True).indices
    for region, date, phase in registry[GROUP].itertuples(index=False, name=None):
        yield 'pilot_date_phase', region + '|' + date + '|' + phase, by_date.get((region, date, phase), np.empty(0, int))
    for month in design.MONTHS:
        yield 'month', month, np.flatnonzero(p.utc_date.str.startswith(month))


def score(predictions, registry):
    rows = []
    for kind, key, positions in segments(predictions, registry):
        part = predictions.iloc[positions]
        common_mask = part.native_label_admitted.to_numpy() & np.isfinite(part[PREDICTORS].to_numpy()).all(axis=1)
        for view in ['all_four_matched', 'own_available']:
            for name in PREDICTORS:
                mask = common_mask if view == 'all_four_matched' else part.native_label_admitted.to_numpy() & np.isfinite(part[name].to_numpy())
                values = part.loc[mask]
                weight = balanced_weights(values)
                delta = (values[name] - values.lst_c).to_numpy(float)
                absolute = np.abs(delta)
                record = {'segment_type': kind, 'segment': key, 'view': view, 'model': name,
                    'collected_rows': len(part), 'qa_admitted': int(part.native_label_admitted.sum()),
                    'feature_complete': int(part.feature_complete.sum()),
                    'predicted': int(np.isfinite(part[name]).sum()), 'paired': len(values),
                    'unscored': len(part) - len(values), 'sites': values.region_id.nunique(),
                    'dates': values.utc_date.nunique(), 'acquisitions': values.acquisition_id.nunique(),
                    'reserved_physical_rows': int(values.reserved_acquisition_excluded.sum()),
                    'mae': float(absolute.mean()) if len(values) else None,
                    'balanced_mae': float(weight @ absolute) if len(values) else None,
                    'bias': float(delta.mean()) if len(values) else None,
                    'balanced_bias': float(weight @ delta) if len(values) else None}
                for threshold in [3, 5, 7]:
                    record[f'above_{threshold}_fraction'] = float(np.mean(absolute > threshold)) if len(values) else None
                    record[f'balanced_above_{threshold}_fraction'] = float(weight @ (absolute > threshold)) if len(values) else None
                rows.append(record)
    return rows


def run(out):
    offline()
    started, cpu_started = time.monotonic(), time.process_time()
    plan = read(out / 'plan.json')
    need(not (out / 'models').exists() and not (out / 'fits.json').exists(), 'No automatic fitting restart')
    for b in plan['inputs'].values():
        verify(b)
    need(plan['features'] == FEATURES and plan['parameters'] == PARAMS, 'Model recipe changed')
    d, g, a, reserved, _ = load(plan['inputs']['cohort_receipt']['path'])
    need(len(d) == plan['row_count'] and frame_sha(d) == plan['cohort_frame_sha256'], 'Prepared cohort changed')
    expected = {(s['mode'], s['key'], s['arm']): s for s in plan['fits']}
    predictions = {m: np.full((len(d), 2), np.nan) for m in ['geography', 'month', 'reference']}
    fit_ids = {m: np.full((len(d), 2), -1, dtype=np.int16) for m in predictions}
    available = d.feature_complete.to_numpy()
    completed, models = [], {}
    for split, arm, positions, train, membership, spec in specifications(d, g, a, reserved):
        need(time.monotonic() - started < plan['limits']['seconds'] - 30, 'Fitting deadline reached')
        want = expected[split.mode, split.key, arm]
        need(json.dumps(spec, sort_keys=True) == json.dumps(want, sort_keys=True), 'Prepared fit recipe changed')
        if not spec['eligible']:
            completed.append(dict(spec, status='insufficient_fitting_support'))
            save(out / 'fits.json', completed)
            continue
        fingerprint_key = spec['fingerprint']
        if fingerprint_key in models:
            binding = models[fingerprint_key]
            model = joblib.load(verify(binding['model']))
            result = dict(spec, status='reused_exact_recipe', **binding)
        else:
            target = out / 'models' / split.mode / split.key / arm
            target.mkdir(parents=True)
            membership.to_parquet(target / 'fitting_rows.parquet', index=False)
            model = HistGradientBoostingRegressor(**PARAMS)
            model.fit(train[FEATURES], membership.target_residual_c,
                      sample_weight=membership.weight.to_numpy() * len(train))
            joblib.dump(model, target / 'model.joblib', compress=3)
            binding = {'model': bind(target / 'model.joblib'),
                       'fitting_rows': bind(target / 'fitting_rows.parquet'),
                       'origin': '/'.join((split.mode, split.key, arm))}
            models[fingerprint_key] = binding
            result = dict(spec, status='fitted', **binding)
        if split.mode != 'full':
            mode = 'geography' if split.mode in ['original_pilot', 'new_pilot'] else split.mode
            held = split.held_positions[available[split.held_positions]]
            column = ARMS.index(arm)
            if len(held):
                need((fit_ids[mode][held, column] == -1).all(), 'Held prediction written twice')
                predictions[mode][held, column] = d.iloc[held][AIR].to_numpy() + model.predict(d.iloc[held][FEATURES])
                fit_ids[mode][held, column] = len(completed)
        completed.append(result)
        save(out / 'fits.json', completed)
        print(json.dumps({'mode': split.mode, 'key': split.key, 'arm': arm,
                          'rows': len(train), 'status': result['status']}), flush=True)
        del model, train, membership
    roles = g.drop_duplicates('region_id').set_index('region_id').geography_role
    base = d[IDENTITY + EXTRA + MASKS + ['lst_c']].copy()
    base['raw_air'], base['raw_skin'] = d[AIR].to_numpy(), d[SKIN].to_numpy()
    base['geography_role'] = base.region_id.map(roles)
    base['reserved_acquisition_excluded'] = base.physical_acquisition_key.isin(reserved)
    del d
    gc.collect()
    for mode in predictions:
        keep = base.region_id.eq('cabauw').to_numpy() if mode == 'reference' else base.region_id.ne('cabauw').to_numpy()
        p = base.loc[keep].reset_index(drop=True)
        for column, arm in enumerate(ARMS):
            p[arm] = predictions[mode][keep, column]
            p[arm + '_fit_index'] = fit_ids[mode][keep, column]
        p.to_parquet(out / (mode + '_predictions.parquet'), index=False)
        del p
    del base, predictions, fit_ids
    gc.collect()
    metrics = []
    for mode in ['geography', 'month', 'reference']:
        p = pd.read_parquet(out / (mode + '_predictions.parquet'))
        metrics.extend(dict(test_mode=mode, **row) for row in score(p, g))
        print(json.dumps({'scored': mode, 'wall_seconds': time.monotonic() - started}), flush=True)
        del p
        gc.collect()
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    for b in plan['inputs'].values():
        verify(b)
    save(out / 'completion.json', {
        'status': 'complete_exploratory_data_comparison', 'plan': bind(out / 'plan.json'),
        'fit_recipes': len(completed), 'unique_fits': sum(s['status'] == 'fitted' for s in completed),
        'exact_recipe_reuses': sum(s['status'] == 'reused_exact_recipe' for s in completed),
        'requested_groups': len(g), 'rows': plan['row_count'],
        'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'reserved_geographies_opened': False, 'production_changed': False,
        'globally_qualified': False,
        'artifacts': {p.name: bind(p) for p in out.iterdir() if p.is_file()},
    })
    print(json.dumps(bind(out / 'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--cohort-receipt', type=Path)
    parser.add_argument('--cohort-check', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.mode == 'prepare':
            prepare(args.cohort_receipt, args.cohort_check, args.output)
        else:
            run(args.output)
