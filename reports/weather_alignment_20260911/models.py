"""Small F-preserving research candidates; no retrieval or deployment code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from lst_pilot import option_b_train as old
from lst_pilot import multisensor_train as multi

LABELS = {
    'F': 'F matched reference',
    'land_features_floor': 'F with floor-hour land inputs',
    'land_features_aligned': 'F with time-aligned land inputs',
    'F_air_shrink_floor': 'F with bounded floor-hour air correction',
    'F_air_shrink_aligned': 'F with bounded aligned-air correction',
}
ELIGIBLE = ('F_air_shrink_aligned', 'F_air_shrink_floor', 'land_features_aligned')
PHASES = ('day', 'night')
TIMINGS = ('floor', 'aligned')
REGULARIZATION_C2 = 20.0
MIN_DATES = 6
LAND_QUANTITIES = ('skin_c', 'air_c', 'soil_c', 'swe_m', 'moisture_m3_m3')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def frame_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(old.json_ready(value), indent=2, allow_nan=False) + '\n')


def extra_columns(timing):
    require(timing in TIMINGS, 'Unknown land timing')
    return tuple(f'{timing}_{x}' for x in ('skin_minus_air_c', 'soil_minus_air_c', 'swe_m', 'moisture_m3_m3'))


def prepare_features(frame):
    """A remains exactly unchanged; endpoint-derived fields have separate names."""
    data = frame.copy()
    for timing in TIMINGS:
        columns = [f'{timing}_{x}' for x in LAND_QUANTITIES]
        require(np.isfinite(data[columns].to_numpy(float)).all(), 'Incomplete land features')
        data[f'{timing}_skin_minus_air_c'] = data[f'{timing}_skin_c'] - data[f'{timing}_air_c']
        data[f'{timing}_soil_minus_air_c'] = data[f'{timing}_soil_c'] - data[f'{timing}_air_c']
    pd.testing.assert_frame_equal(data[list(frame)], frame, check_exact=True)
    return data


def features(name):
    require(name in ('F', 'land_features_floor', 'land_features_aligned'), 'Unknown fitted model')
    return tuple(multi.BASE) if name == 'F' else (*multi.BASE, *extra_columns(name.rsplit('_', 1)[-1]))


def fit_hgb(frame, name, output, stage):
    names = features(name)
    weights = old.balanced_weights(frame)
    target = old.target_offset(frame)
    model, _ = old.build_estimators(names, old.CONFIG)
    started = time.monotonic()
    model.fit(frame[list(names)], target, regressor__sample_weight=weights * len(frame))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / (name + '.joblib')
    require(not path.exists(), 'Fitted model already exists')
    joblib.dump(model, path, compress=3)
    detail = {'stage': stage, 'model': name, 'rows': len(frame),
              'features': list(names), 'sample_id_sha256': old.row_hash(frame),
              'weight_sha256': digest(weights), 'target_sha256': digest(target),
              'feature_sha256': frame_digest(frame[list(names)]),
              'target_definition': 'Y-A', 'prediction_definition': 'A+f(X)',
              'model_sha256': old.sha(path), 'seconds': time.monotonic() - started}
    save(output / (name + '_fit.json'), detail)
    frame[['sample_id', 'datetime_utc', 'utc_day', 'region_id', 'phase']].assign(
        normalized_weight=weights, target_offset_c=target
    ).to_parquet(output / (name + '_training_rows.parquet'), index=False)
    return model


def predict_hgb(frame, model, name):
    return frame.air_temperature_c.to_numpy(float) + model.predict(frame[list(features(name))])


def fit_air_correction(frame, held_f_predictions, timing, weights=None):
    """Caller supplies strictly held-month predictions, never in-sample F values."""
    require(timing in TIMINGS, 'Unknown correction timing')
    require(set(frame.phase).issubset(PHASES), 'Unknown phase')
    f = np.asarray(held_f_predictions, dtype=float)
    require(f.shape == (len(frame),) and np.isfinite(f).all(), 'Incomplete calibration predictions')
    weights = old.balanced_weights(frame) if weights is None else np.asarray(weights, dtype=float)
    require(weights.shape == f.shape and np.isfinite(weights).all() and (weights > 0).all()
            and weights.sum() > 0, 'Invalid calibration weights')
    weights = weights / weights.sum()
    d = frame.air_temperature_c.to_numpy(float) - frame[f'{timing}_air_c'].to_numpy(float)
    error = f - frame.lst_c.to_numpy(float)
    require(np.isfinite(d).all() and np.isfinite(error).all(), 'Nonfinite calibration inputs')
    phases = {}
    for phase in PHASES:
        keep = frame.phase.eq(phase).to_numpy()
        n = int(keep.sum())
        dates = int(frame.loc[keep, 'utc_day'].nunique())
        record = {'phase': phase, 'rows': n, 'utc_dates': dates,
                  'regularization_c2': REGULARIZATION_C2, 'minimum_utc_dates': MIN_DATES,
                  'gamma': 0.0, 'status': 'unsupported_identity_fallback'}
        if n and weights[keep].sum() > 0:
            w = weights[keep] / weights[keep].sum()
            date_mass = pd.DataFrame({'date': frame.loc[keep, 'utc_day'].to_numpy(), 'weight': w}).groupby('date').weight.sum()
            effective_dates = 1.0 / float(np.square(date_mass.to_numpy()).sum())
            numerator = effective_dates * float(np.dot(w, d[keep] * error[keep]))
            denominator = effective_dates * float(np.dot(w, np.square(d[keep]))) + REGULARIZATION_C2
            unbounded = numerator / denominator
            record.update({'effective_utc_dates': effective_dates, 'numerator': numerator,
                           'denominator': denominator, 'unbounded_gamma': unbounded,
                           'normalized_phase_weight_sha256': digest(w)})
            if dates >= MIN_DATES:
                record.update(gamma=float(np.clip(unbounded, 0, 1)), status='fitted')
        phases[phase] = record
    return {'timing': timing, 'definition': 'F-gamma_phase*(A-T)', 'phases': phases,
            'calibration_sample_id_sha256': old.row_hash(frame),
            'calibration_weight_sha256': digest(weights), 'held_F_sha256': digest(f),
            'air_difference_sha256': digest(d), 'F_error_sha256': digest(error),
            'not_a_physical_station_reliability_coefficient': True}


def apply_air_correction(frame, f_predictions, fit):
    timing = fit['timing']
    require(timing in TIMINGS and set(frame.phase).issubset(PHASES), 'Unknown correction inputs')
    gamma = np.asarray([fit['phases'][phase]['gamma'] for phase in frame.phase], float)
    require(np.isfinite(gamma).all() and ((0 <= gamma) & (gamma <= 1)).all(), 'Unbounded coefficient')
    f = np.asarray(f_predictions, float)
    require(f.shape == (len(frame),) and np.isfinite(f).all(), 'Invalid reference predictions')
    d = frame.air_temperature_c.to_numpy(float) - frame[f'{timing}_air_c'].to_numpy(float)
    require(np.isfinite(d).all(), 'Invalid air difference')
    return f - gamma * d


def predict_all(frame, estimators, corrections):
    result = {name: predict_hgb(frame, estimators[name], name)
              for name in ('F', 'land_features_floor', 'land_features_aligned')}
    for timing in TIMINGS:
        result['F_air_shrink_' + timing] = apply_air_correction(frame, result['F'], corrections[timing])
    require(set(result) == set(LABELS) and all(np.isfinite(x).all() for x in result.values()), 'Incomplete five-arm predictions')
    return result
