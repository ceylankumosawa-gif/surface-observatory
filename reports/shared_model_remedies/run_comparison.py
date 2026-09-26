"""Bounded shared-model remedies; Hetzner research only, no promotion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from lst_pilot.physics_features import add_physics_features, FEATURES as PHYSICS
import candidates

BASE_RUN = Path('/opt/lst-pilot/runs/multisensor_20260910_v1/model_experiment_v1')
PHYSICS_RUN = Path('/opt/lst-pilot/runs/physics_correction_20260910_v1')
LEGACY = Path('/opt/lst-pilot/runs/option_b_retrain_20260909/features_legacy_2024_v1/station_audit/features_station_audited.parquet')
CACHE = Path('/opt/lst-pilot/cache')
LABELS = {'E': 'Earlier shared model E', 'F': 'Current research model F',
          'larger_hgb': 'More flexible shared model', 'robust_hgb': 'Robust shared model',
          'larger_robust_hgb': 'Flexible robust shared model', 'physics_hgb': 'Shared model with radiation proxies',
          'extra_trees': 'Extra Trees', 'random_forest': 'Random forest',
          'conservative_new_weights': 'Half weight on new observations',
          'half_blend': 'Equal blend of E and F', 'phase_models': 'Separate day and night models',
          'climate_models': 'Separate climate models'}
TIE_ORDER = ['conservative_new_weights', 'robust_hgb', 'larger_hgb', 'larger_robust_hgb',
             'physics_hgb', 'extra_trees', 'random_forest', 'half_blend', 'phase_models', 'climate_models', 'E']
META = ['sample_id', 'region_id', 'phase', 'climate_class', 'datetime_utc', 'utc_day', 'split',
        'season', 'air_group', 'label_product', 'acquisition_id', 'weight_surface_group',
        'lst_c', 'air_temperature_c', 'solar_elevation_deg']


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def log(**kwargs):
    print(json.dumps(old.json_ready(kwargs)), flush=True)


def fit_only(manifest):
    """Decode thermal values only for immutable saved fitting identities."""
    records = pd.read_parquet(BASE_RUN / 'fitting_rows.parquet')
    ids = records.sample_id.tolist()
    parts = []
    for name in ('original_input', 'e_additions', 'new_fit'):
        path = manifest['paths'][name]
        filters = [('sample_id', 'in', ids)]
        timestamps = pd.to_datetime(pd.read_parquet(path, columns=['datetime_utc'], filters=filters).datetime_utc, utc=True)
        assert timestamps.notna().all() and timestamps.dt.year.isin([2021, 2022]).all()
        frame = pd.read_parquet(path, filters=filters)
        if len(frame):
            parts.append(old.prepare_input(frame))
    fit = pd.concat(parts, ignore_index=True).sort_values('sample_id').reset_index(drop=True)
    assert not fit.sample_id.duplicated().any() and len(fit) == 44316
    assert np.array_equal(fit.sample_id, records.sample_id)
    assert old.row_hash(fit) == manifest['fit_row_sha256']
    assert fit.split.eq('fit').all() and not fit.spatial_holdout.any() and not fit.in_holdout_buffer.any()
    assert not fit.region_id.eq('cabauw').any() and old.complete_rows(fit, multi.BASE).all()
    weights = old.balanced_weights(fit)
    assert np.array_equal(weights, records.weight.to_numpy(float))
    saved_oof = pd.read_parquet(BASE_RUN / 'oof_predictions.parquet')
    assert np.array_equal(saved_oof.sample_id, fit.sample_id)
    assert np.allclose(old.target_offset(fit), saved_oof.oof_offset_c + saved_oof.residual_target_c, rtol=0, atol=1e-12)
    e_records = pd.read_parquet(Path(manifest['paths']['e_run']) / 'fitting_rows.parquet')
    fit['is_e'] = fit.sample_id.isin(e_records.sample_id)
    assert fit.is_e.sum() == 39254
    efit = fit.loc[fit.is_e].reset_index(drop=True)
    assert np.array_equal(old.balanced_weights(efit), e_records.set_index('sample_id').loc[efit.sample_id].weight)
    return fit, weights, saved_oof


def conservative_weights(frame, weights):
    adjusted = np.asarray(weights, float) * np.where(frame.is_e, 1., .5)
    for ids in frame.groupby(['region_id', 'phase'], observed=True).indices.values():
        adjusted[ids] *= weights[ids].sum() / adjusted[ids].sum()
    assert np.isclose(adjusted.sum(), 1.)
    return adjusted


def base_fit(frame, weights):
    model, _ = old.build_estimators(multi.BASE, old.CONFIG)
    model.fit(frame[list(multi.BASE)], old.target_offset(frame), regressor__sample_weight=weights * len(frame))
    return model


def fit_all(frame, out, stage):
    out.mkdir(parents=True)
    weights = old.balanced_weights(frame)
    models, details = {}, {}
    for name in LABELS:
        start = time.monotonic()
        subset = frame.loc[frame.is_e].reset_index(drop=True) if name == 'E' else frame
        w = old.balanced_weights(subset) if name == 'E' else weights.copy()
        if name == 'conservative_new_weights':
            w = conservative_weights(frame, weights)
        if name in ('E', 'F', 'conservative_new_weights'):
            model = base_fit(subset, w)
        elif name == 'half_blend':
            # Prediction is calculated from the two same-stage saved models.
            details[name] = {'components': ['E', 'F'], 'fractions': [.5, .5], 'stage': stage}
            continue
        elif name in ('phase_models', 'climate_models'):
            model = candidates.fit_grouped(frame, w * len(frame), models['F'], 'phase' if name == 'phase_models' else 'climate_class')
        else:
            model = candidates.fit_candidate(name, frame, w * len(frame))
        models[name] = model
        path = out / (name + '.joblib')
        joblib.dump(model, path, compress=3)
        subset[['sample_id', 'region_id', 'phase', 'utc_day', 'datetime_utc', 'is_e']].assign(
            normalized_weight=w).to_parquet(out / (name + '_fitting_rows.parquet'), index=False)
        detail = {'rows': len(subset), 'sample_id_sha256': old.row_hash(subset),
                  'normalized_weight_sha256': digest(w), 'target_sha256': digest(old.target_offset(subset)),
                  'total_fit_weight': len(subset), 'model_sha256': old.sha(path),
                  'fit_seconds': time.monotonic() - start,
                  'features': list(multi.BASE) + (list(PHYSICS) if name == 'physics_hgb' else [])}
        if hasattr(model, 'support'):
            detail['group_support'] = model.support
        details[name] = detail
        old.save_json(out / 'fits.json', details)
        log(stage=stage, model=name, seconds=detail['fit_seconds'])
    return models, details


def predict_all(frame, models):
    air = frame.air_temperature_c.to_numpy(float)
    values = {name: air + model.predict(frame) for name, model in models.items()}
    values['half_blend'] = .5 * values['E'] + .5 * values['F']
    assert set(values) == set(LABELS)
    assert all(np.isfinite(v).all() and v.shape == (len(frame),) for v in values.values())
    return values


def slices(frame):
    yield 'overall', 'overall', np.arange(len(frame))
    for kind, keys in [('region_phase', ['region_id', 'phase']),
                       ('region_phase_season', ['region_id', 'phase', 'season']),
                       ('source', ['label_product']), ('region_phase_air', ['region_id', 'phase', 'air_group']),
                       ('region_phase_surface', ['region_id', 'phase', 'surface_temperature_group'])]:
        for key, ids in frame.groupby(keys, observed=True).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            yield kind, '|'.join(map(str, key)), ids


def score(frame, predictions, cohort):
    rows = []
    frame = frame.copy()
    frame['surface_temperature_group'] = np.select([frame.lst_c.le(0), frame.lst_c.ge(35)], ['cold', 'hot'], default='middle')
    for kind, group, ids in slices(frame):
        selected = frame.iloc[ids].reset_index(drop=True)
        for name, px in predictions.items():
            errors = np.abs(px[ids] - selected.lst_c.to_numpy())
            rows.append({'cohort': cohort, 'segment_type': kind, 'segment': group, 'model': name,
                         'model_label': LABELS[name], **old.metrics(selected, px[ids]),
                         'raw_gt5_count': int((errors > 5).sum()), 'raw_gt7_count': int((errors > 7).sum()),
                         'raw_gt5_fraction': float((errors > 5).mean()), 'raw_gt7_fraction': float((errors > 7).mean())})
    return rows


def select(metrics):
    table = metrics.set_index(['segment_type', 'segment', 'model'])
    baseline = table.loc[('overall', 'overall', 'F')]
    groups = metrics.loc[(metrics.segment_type == 'region_phase') & (metrics.model == 'F') & (metrics.utc_date_count >= 6)].segment.tolist()
    decisions = []
    for name in TIE_ORDER:
        record = table.loc[('overall', 'overall', name)]
        failures = []
        if record.mae_c > baseline.mae_c - .10 + 1e-12:
            failures.append('overall balanced MAE improves less than 0.10 C')
        if record.unweighted_pixel_mae_c > baseline.unweighted_pixel_mae_c + .10 + 1e-12:
            failures.append('ordinary pixel MAE worsens more than 0.10 C')
        if record.fraction_abs_error_gt_5c > baseline.fraction_abs_error_gt_5c + .02 + 1e-12:
            failures.append('overall over-5 C frequency increases more than 2 percentage points')
        for group in groups:
            a, b = table.loc[('region_phase', group, name)], table.loc[('region_phase', group, 'F')]
            if a.mae_c > b.mae_c + .20 + 1e-12:
                failures.append(group + ': MAE worsens more than 0.20 C')
            if a.fraction_abs_error_gt_5c > b.fraction_abs_error_gt_5c + .02 + 1e-12:
                failures.append(group + ': over-5 C frequency increases more than 2 percentage points')
        decisions.append({'model': name, 'eligible': not failures, 'mae_c': record.mae_c,
                          'unweighted_pixel_mae_c': record.unweighted_pixel_mae_c, 'failures': failures})
    eligible = [d for d in decisions if d['eligible']]
    selected = None
    if eligible:
        best = min(d['mae_c'] for d in eligible)
        selected = next(d['model'] for d in eligible if d['mae_c'] <= best + .05 + 1e-12)
    return {'selected_candidate': selected, 'status': 'qualified_in_training_month_cv' if selected else 'no_qualifying_remedy',
            'selection_data': '2021–22 stitched held-calendar-month predictions only',
            'supported_groups': groups, 'tie_order': TIE_ORDER, 'decisions': decisions,
            'auto_promotion': False, 'reserved_2025_opened': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    out = args.output
    if out.exists():
        raise FileExistsError('Use a new immutable run directory')
    reference = multi.verify_run(BASE_RUN)
    physics.verify_inputs(reference)
    fit, weights, original_oof = fit_only(reference)
    fit, feature_audit = add_physics_features(fit, cache_root=CACHE)
    assert fit.physics_complete.all() and old.complete_rows(fit, (*multi.BASE, *PHYSICS)).all()
    fold = multi.month_folds(fit)
    assert np.array_equal(fold, original_oof.fold)
    out.mkdir(parents=True)
    source_paths = [Path(__file__), Path(candidates.__file__), Path(old.__file__), Path(multi.__file__),
                    Path(physics.__file__), Path(old.__file__).with_name('model.py'), Path(old.__file__).with_name('physics_features.py')]
    manifest = {'research_only': True, 'auto_promotion': False, 'reserved_2025_opened': False,
                'protocol_sha256': old.sha(args.protocol), 'source_hashes': {str(p): old.sha(p) for p in source_paths},
                'reference_freeze_sha256': old.sha(BASE_RUN / 'post_evaluation_freeze.json'),
                'protected_baseline_sha256': old.sha(reference['paths']['baseline']),
                'input_paths': reference['paths'], 'input_hashes': reference['input_hashes'],
                'fit_rows': len(fit), 'fit_id_sha256': old.row_hash(fit), 'target_sha256': digest(old.target_offset(fit)),
                'feature_audit': feature_audit, 'candidate_specs': candidates.SPECS,
                'base_config': asdict(old.CONFIG), 'labels': LABELS, 'python': sys.version, 'sklearn': sklearn.__version__}
    old.save_json(out / 'manifest.json', manifest)
    fit.to_parquet(out / 'fit_frame.parquet', index=False)
    oof_values = {name: np.full(len(fit), np.nan) for name in LABELS}
    with threadpool_limits(limits=4):
        for f in range(3):
            train, held = fit.loc[fold != f].reset_index(drop=True), fit.loc[fold == f].reset_index(drop=True)
            assert set(train.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(held.datetime_utc.dt.strftime('%Y-%m'))
            models, details = fit_all(train, out / f'fold_{f}', f'fold_{f}')
            train[['sample_id', 'utc_day', 'region_id', 'phase', 'is_e']].assign(weight=old.balanced_weights(train)).to_parquet(out / f'fold_{f}' / 'fitting_rows.parquet', index=False)
            held[META].to_parquet(out / f'fold_{f}' / 'held_rows.parquet', index=False)
            values = predict_all(held, models)
            for name in LABELS:
                oof_values[name][fold == f] = values[name]
            del models
    assert all(np.isfinite(v).all() for v in oof_values.values())
    max_oof_diff = float(np.max(np.abs(oof_values['F'] - fit.air_temperature_c.to_numpy() - original_oof.oof_offset_c.to_numpy())))
    assert max_oof_diff <= 1e-9, max_oof_diff
    predictions = fit[META].assign(fold=fold, is_e=fit.is_e)
    for name, values in oof_values.items():
        predictions[name + '_lst_c'] = values
    predictions.to_parquet(out / 'oof_predictions.parquet', index=False)
    oof_metrics = pd.DataFrame(score(fit, oof_values, 'training_2021_22_oof'))
    oof_metrics.to_csv(out / 'oof_metrics.csv', index=False)
    selection = select(oof_metrics)
    selection['F_reference_oof_max_difference_c'] = max_oof_diff
    old.save_json(out / 'selection.json', selection)
    selection_freeze = {'selection_sha256': old.sha(out / 'selection.json'),
                        'oof_predictions_sha256': old.sha(out / 'oof_predictions.parquet'),
                        'oof_metrics_sha256': old.sha(out / 'oof_metrics.csv'),
                        'manifest_sha256': old.sha(out / 'manifest.json'),
                        'evaluation_labels_decoded_before_selection': False}
    old.save_json(out / 'selection_freeze.json', selection_freeze)
    log(stage='selection_frozen', **selection)
    with threadpool_limits(limits=4):
        full_models, _ = fit_all(fit, out / 'full', 'full')
    old.save_json(out / 'full_fit_freeze.json', {'files': {str(p.relative_to(out)): old.sha(p) for p in (out / 'full').iterdir() if p.is_file()},
                 'selection_freeze_sha256': old.sha(out / 'selection_freeze.json'), 'frozen_before_evaluation': True})

    # Only now decode the repeatedly inspected evaluation labels.
    reconstructed, fw, _, original_eval, known, areas = physics.reconstruct_fit(reference, BASE_RUN)
    pd.testing.assert_frame_equal(reconstructed[list(multi.BASE)], fit[list(multi.BASE)], check_dtype=False)
    assert np.array_equal(old.target_offset(reconstructed), old.target_offset(fit))
    p = reference['paths']
    new_frame, registry = multi.load_new_evaluation(p['new_evaluation'], p['freshness_audit'], reference['input_hashes']['freshness_audit'], known)
    new_eval = multi.validate_new(new_frame, known, areas, evaluation=True, freshness_sha=reference['input_hashes']['freshness_audit'], registry=registry)
    assert old.sha(LEGACY) == reference['reference_hashes']['legacy_2024_input']
    legacy = old.prepare_input(old.load_paired_input(LEGACY, evaluation_2024=True), evaluation_2024=True)
    baseline = joblib.load(p['baseline'])
    needed = tuple(dict.fromkeys([*multi.BASE, *baseline['features']]))
    legacy = legacy.loc[old.complete_rows(legacy, needed) & legacy.phase.isin(multi.PHASES)].sort_values('sample_id').reset_index(drop=True)
    legacy['split'] = 'legacy_2024'
    rows, acquisitions, reproduction, audits = [], [], [], {}
    for family, frame in [('old', original_eval), ('newly_collected_2023', new_eval), ('legacy_2024', legacy)]:
        frame, audits[family] = add_physics_features(frame, cache_root=CACHE)
        assert frame.physics_complete.all() and old.complete_rows(frame, (*multi.BASE, *PHYSICS)).all()
        with threadpool_limits(limits=4):
            values = predict_all(frame, full_models)
        saved = pd.read_parquet(PHYSICS_RUN / (family + '_predictions.parquet'))
        assert set(saved.sample_id) == set(frame.sample_id) and not saved.sample_id.duplicated().any()
        saved = saved.set_index('sample_id').loc[frame.sample_id]
        assert np.array_equal(saved.lst_c, frame.lst_c)
        for name in ['E', 'F']:
            delta = float(np.max(np.abs(values[name] - saved[name + '_lst_c'].to_numpy())))
            reproduction.append({'family': family, 'model': name, 'max_difference_c': delta})
            assert delta <= 1e-9, reproduction
        exported = frame[META].copy()
        for name, predicted in values.items():
            exported[name + '_lst_c'] = predicted
        exported.to_parquet(out / (family + '_predictions.parquet'), index=False)
        for split, ids in frame.groupby('split', observed=True).indices.items():
            cohort = 'legacy_2024' if family == 'legacy_2024' else family + '_' + split
            selected = frame.iloc[ids].reset_index(drop=True)
            px = {name: v[ids] for name, v in values.items()}
            rows.extend(score(selected, px, cohort))
            focal = selected.loc[selected.region_id.isin(['greater_london', 'sioux_falls'])]
            for acq, positions in focal.groupby('acquisition_id', observed=True).groups.items():
                group = selected.loc[positions].reset_index(drop=True)
                w = old.balanced_weights(group)
                for name, predicted in px.items():
                    acquisitions.append({'cohort': cohort, 'region_id': group.region_id.iloc[0], 'phase': group.phase.iloc[0],
                        'acquisition_id': acq, 'date': group.utc_day.iloc[0].strftime('%Y-%m-%d'),
                        'timestamp_utc': group.datetime_utc.iloc[0].isoformat(), 'model': name, 'model_label': LABELS[name],
                        **old.metrics(group, predicted[positions]), 'observed_mean_c': float(w @ group.lst_c.to_numpy()),
                        'predicted_mean_c': float(w @ predicted[positions])})
        log(stage='scored', family=family, rows=len(frame))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(acquisitions).to_csv(out / 'acquisition_metrics.csv', index=False)
    old.save_json(out / 'evaluation_feature_audits.json', audits)
    old.save_json(out / 'reproduction.json', reproduction)
    for filename, key in [('selection.json', 'selection_sha256'), ('oof_predictions.parquet', 'oof_predictions_sha256'),
                          ('oof_metrics.csv', 'oof_metrics_sha256'), ('manifest.json', 'manifest_sha256')]:
        assert old.sha(out / filename) == selection_freeze[key]
    for path, checksum in manifest['source_hashes'].items():
        assert old.sha(path) == checksum, path
    assert old.sha(args.protocol) == manifest['protocol_sha256']
    assert old.sha(p['baseline']) == manifest['protected_baseline_sha256']
    multi.verify_run(BASE_RUN)
    old.save_json(out / 'completion.json', {'status': 'complete', 'elapsed_seconds': time.monotonic()-start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, 'research_only': True,
        'reserved_2025_opened': False, 'production_unchanged': True,
        'artifacts': {str(path.relative_to(out)): old.sha(path) for path in out.rglob('*') if path.is_file()}})
    log(stage='complete', seconds=time.monotonic()-start)


if __name__ == '__main__':
    main()
