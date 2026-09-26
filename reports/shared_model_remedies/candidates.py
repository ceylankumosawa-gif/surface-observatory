"""Fixed research estimator alternatives; no file access, acquisition or serving.

Every estimator predicts LST minus air temperature. The caller owns frozen
cohorts, out-of-fold construction, source verification and numerical thread
limits. Shared candidates use the supplied weights unchanged. Group experts
use only their group's supplied weights, rescaled to mean one.

Import this module under a stable, importable name when serializing estimators.
Physics features must already have been derived and validated by the caller.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lst_pilot import option_b_train as old
from lst_pilot.model import build_estimators, target_offset
from lst_pilot.physics_features import FEATURES as PHYSICS

VERSION = "shared-model-remedy-candidates-v1"
BASE = tuple(old.BASE_FEATURES)
SEED = old.SEED
LARGER = dict(max_iter=300, max_leaf_nodes=31, max_depth=8,
              min_samples_leaf=30, l2_regularization=5.0, learning_rate=.08)
EXPERT_MINIMUM = dict(rows=300, utc_dates=6, year_months=3)
SPECS = {
    "larger_hgb": {"family": "hgb", "features": list(BASE),
                   "loss": "squared_error", **LARGER},
    "robust_hgb": {"family": "hgb", "features": list(BASE),
                   "loss": "absolute_error", "capacity": "original_F"},
    "larger_robust_hgb": {"family": "hgb", "features": list(BASE),
                          "loss": "absolute_error", **LARGER},
    "physics_hgb": {"family": "hgb", "features": [*BASE, *PHYSICS],
                    "loss": "squared_error", "capacity": "original_F",
                    "physics_inputs": "precomputed, finite and explicitly physics_complete"},
    "extra_trees": {"family": "extra_trees", "features": list(BASE),
                    "n_estimators": 256, "min_samples_leaf": 20,
                    "max_features": 1.0, "n_jobs": 4, "bootstrap": False,
                    "climate_encoding": "training-fitted dense one-hot; unknown ignored"},
    "random_forest": {"family": "random_forest", "features": list(BASE),
                      "n_estimators": 256, "min_samples_leaf": 20,
                      "max_features": .75, "n_jobs": 4, "bootstrap": True,
                      "climate_encoding": "training-fitted dense one-hot; unknown ignored"},
    "phase_models": {"family": "grouped_original_F", "group": "phase",
                     "minimum": dict(EXPERT_MINIMUM), "expert_weight_mean": 1.0,
                     "fallback": "caller-supplied SAME-FOLD shared estimator"},
    "climate_models": {"family": "grouped_original_F", "group": "climate_class",
                       "minimum": dict(EXPERT_MINIMUM), "expert_weight_mean": 1.0,
                       "fallback": "caller-supplied SAME-FOLD shared estimator"},
}
for _spec in SPECS.values():
    _spec.update(random_state=SEED, target="lst_c - air_temperature_c",
                 acquisition_or_sensor_or_coordinates_as_predictors=False,
                 model_deployment=False)


def _features(frame, names, *, fitting=False, physics=False):
    missing = set(names) - set(frame)
    if missing:
        raise ValueError(f"Missing fixed candidate predictors: {sorted(missing)}")
    values = frame.loc[:, list(names)]
    numeric = [n for n in names if n != "climate_class"]
    if not np.isfinite(values[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Fixed candidate predictors must be finite; no rows are dropped or imputed.")
    if fitting:
        climate = values.climate_class
        if (climate.isna().any() or climate.astype(str).str.strip().str.lower().isin(
                ["", "unknown", "__unknown__", "none", "nan", "0"]).any()):
            raise ValueError("Fitting requires known climate classifications.")
    if physics:
        if "physics_complete" not in frame:
            raise ValueError("physics_hgb requires explicit physics_complete input proof.")
        if not old.strict_bool(frame.physics_complete, "physics_complete").all():
            raise ValueError("physics_hgb needs complete physical inputs; caller must use its declared same-fold fallback for unavailable rows.")
    return values


def _fit_inputs(frame, weights, names=BASE, *, physics=False):
    if frame.empty or len(frame) > old.MAX_ROWS:
        raise ValueError("Candidate fitting needs 1 to 200,000 preselected rows.")
    required = {"datetime_utc", "region_id", "split", "spatial_holdout", "in_holdout_buffer", "lst_c"}
    if not required.issubset(frame):
        raise ValueError(f"Missing frozen fitting metadata: {sorted(required-set(frame))}")
    stamps = pd.to_datetime(frame.datetime_utc, utc=True, errors="raise")
    if stamps.isna().any() or not stamps.dt.year.isin([2021, 2022]).all():
        raise ValueError("Only 2021–2022 fitting timestamps are allowed.")
    if (not frame.split.eq("fit").all() or frame.region_id.eq("cabauw").any()
            or old.strict_bool(frame.spatial_holdout, "spatial_holdout").any()
            or old.strict_bool(frame.in_holdout_buffer, "in_holdout_buffer").any()):
        raise ValueError("Candidate fitting includes reserved temporal/geographic rows.")
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(frame),) or not np.isfinite(w).all() or (w <= 0).any() or not np.isfinite(w.sum()):
        raise ValueError("Supplied fitting weights must be finite, positive and positionally aligned.")
    values = _features(frame, names, fitting=True, physics=physics)
    target = target_offset(frame)
    if not np.isfinite(target).all():
        raise ValueError("Fitting target offsets must be finite.")
    return values, target, w, stamps


def _hgb(names, *, larger=False, loss="squared_error"):
    config = replace(old.CONFIG, **LARGER) if larger else old.CONFIG
    estimator, _ = build_estimators(names, config)
    estimator.set_params(regressor__loss=loss)
    return estimator


def _forest(name):
    spec = SPECS[name]
    numeric = [n for n in BASE if n != "climate_class"]
    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("climate", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["climate_class"]),
    ], remainder="drop", sparse_threshold=0)
    cls = ExtraTreesRegressor if name == "extra_trees" else RandomForestRegressor
    estimator = cls(n_estimators=spec["n_estimators"], min_samples_leaf=spec["min_samples_leaf"],
                    max_features=spec["max_features"], n_jobs=spec["n_jobs"],
                    bootstrap=spec["bootstrap"], criterion="squared_error",
                    random_state=SEED)
    return Pipeline([("features", preprocessor), ("regressor", estimator)])


@dataclass
class OffsetEstimator:
    name: str
    estimator: object
    feature_names: tuple[str, ...]
    specification: dict
    fitting_audit: dict

    def predict(self, frame):
        values = _features(frame, self.feature_names, physics=self.name == "physics_hgb")
        if frame.empty:
            return np.empty(0, dtype=float)
        result = np.asarray(self.estimator.predict(values), dtype=float)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Candidate returned invalid or misaligned surface–air offsets.")
        return result


def fit_candidate(name, frame, weights):
    """Fit one declared shared alternative using caller weights unchanged."""
    if name not in SPECS or SPECS[name]["family"] == "grouped_original_F":
        raise ValueError("Use a fixed shared candidate name; grouped candidates use fit_grouped.")
    names = tuple(SPECS[name]["features"])
    values, target, w, stamps = _fit_inputs(frame, weights, names, physics=name == "physics_hgb")
    if name in ("extra_trees", "random_forest"):
        estimator = _forest(name)
    else:
        estimator = _hgb(names, larger=name in ("larger_hgb", "larger_robust_hgb"), loss=SPECS[name]["loss"])
    estimator.fit(values, target, regressor__sample_weight=w)
    return OffsetEstimator(name, estimator, names, deepcopy(SPECS[name]),
                           {"rows": len(frame), "utc_dates": int(stamps.dt.floor("D").nunique()),
                            "year_months": int(stamps.dt.strftime("%Y-%m").nunique()),
                            "weight_sum": float(w.sum()), "weight_mean": float(w.mean()),
                            "caller_weights_unchanged": True})


def _group_key(frame, group, *, fitting=False):
    if group == "phase":
        solar = frame.solar_elevation_deg.to_numpy(dtype=float)
        values = pd.Series(np.where(solar >= 10, "day", np.where(solar <= -6, "night", "twilight")),
                           index=frame.index)
        if fitting and ("phase" not in frame or not frame.phase.eq(values).all() or values.eq("twilight").any()):
            raise ValueError("Expert fitting phase must agree with the declared solar geometry.")
        return values
    if group == "climate_class":
        return frame.climate_class.astype("string")
    raise ValueError("Expert group must be phase or full climate_class.")


@dataclass
class GroupedOffsetEstimator:
    group: str
    shared: object
    experts: dict
    support: dict
    specification: dict

    def predict(self, frame):
        # Only the fixed allowlist is passed to shared F. There is no access to
        # labels, city IDs or source IDs during either routing or prediction.
        values = _features(frame, BASE)
        if frame.empty:
            return np.empty(0, dtype=float)
        result = np.asarray(self.shared.predict(values), dtype=float).copy()
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Shared fallback returned invalid surface–air offsets.")
        labels = _group_key(frame, self.group)
        for key, expert in self.experts.items():
            positions = np.flatnonzero(labels.eq(key).fillna(False).to_numpy(dtype=bool))
            if len(positions):
                result[positions] = expert.predict(frame.iloc[positions])
        return result

    def expert_support(self, frame):
        """Return positional audit flags without evaluating labels or models."""
        return _group_key(frame, self.group).isin(self.experts).to_numpy(dtype=bool)


def fit_grouped(frame, weights, shared, group):
    """Fit eligible original-F experts; retain the supplied SAME-FOLD fallback.

    The caller must pass its shared estimator trained on exactly the current
    fold's fitting rows. This function never loads or retrains that fallback.
    Eligibility counts rows, distinct UTC dates and distinct UTC year-months;
    these support thresholds are design choices, not generalization claims.
    """
    aliases = {"phase_models": "phase", "climate_models": "climate_class", "climate": "climate_class"}
    group = aliases.get(group, group)
    if group not in ("phase", "climate_class") or shared is None or not callable(getattr(shared, "predict", None)):
        raise ValueError("A phase/full-climate group and caller-supplied same-fold shared model are required.")
    _, _, w, stamps = _fit_inputs(frame, weights)
    labels = _group_key(frame, group, fitting=True)
    experts, support = {}, {}
    for key in sorted(labels.dropna().unique().tolist()):
        positions = np.flatnonzero(labels.eq(key).fillna(False).to_numpy(dtype=bool))
        times = stamps.iloc[positions]
        counts = {"rows": len(positions), "utc_dates": int(times.dt.floor("D").nunique()),
                  "year_months": int(times.dt.strftime("%Y-%m").nunique())}
        eligible = all(counts[k] >= EXPERT_MINIMUM[k] for k in EXPERT_MINIMUM)
        support[str(key)] = {**counts, "eligible": eligible,
                             "weight_sum_before_rescaling": float(w[positions].sum())}
        if not eligible:
            continue
        subset = frame.iloc[positions]
        # Mean one keeps expert L2 regularization from depending on the group's
        # share of the global weight budget. Relative within-group weights stay.
        group_weights = w[positions] / w[positions].mean()
        values, target, group_weights, _ = _fit_inputs(subset, group_weights)
        estimator = _hgb(BASE)
        estimator.fit(values, target, regressor__sample_weight=group_weights)
        support[str(key)]["expert_weight_mean"] = float(group_weights.mean())
        experts[str(key)] = OffsetEstimator("group_original_F", estimator, BASE,
                                           {"capacity": "original_F", "loss": "squared_error"},
                                           {**counts, "weight_mean": float(group_weights.mean()),
                                            "weight_sum": float(group_weights.sum())})
    name = "phase_models" if group == "phase" else "climate_models"
    return GroupedOffsetEstimator(group, shared, experts, support, deepcopy(SPECS[name]))
