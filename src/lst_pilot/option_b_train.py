"""Research-only Option B fitting. No acquisition, web serving or promotion.

Input is an already paired, quality-screened 2021--2023 table. This runner
rejects later dates and never imports a scenario fallback or thermal predictor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .model import ModelConfig, build_estimators, target_offset

VERSION = "option-b-exploratory-trainer-v2"
SEED = 2708
MAX_ROWS = 200_000
BASE_FEATURES = (
    "air_temperature_c", "ndvi", "ndbi", "ndwi", "albedo_proxy", "elevation", "slope",
    "terrain_relief_300m", "aspect_sin", "aspect_cos", "water_fraction",
    "solar_elevation_deg", "solar_azimuth_sin", "solar_azimuth_cos", "hour_sin", "hour_cos",
    "day_of_year_sin", "day_of_year_cos", "relative_humidity_pct", "dewpoint_c",
    "wind_speed_m_s", "wind_direction_sin", "wind_direction_cos", "surface_pressure_hpa",
    "cloud_cover_fraction", "shortwave_down_w_m2", "direct_shortwave_w_m2",
    "diffuse_shortwave_w_m2", "era5_longwave_down_w_m2", "era5_snow_water_equivalent_m",
    "precipitation_mm_h", "rain_mm_24h", "rain_mm_72h", "soil_moisture_m3_m3",
    "air_temperature_lag1_c", "air_temperature_lag3_c", "air_temperature_lag24_c",
    "shortwave_down_lag1_w_m2", "shortwave_down_mean3_w_m2", "climate_class",
)
MEMORY_FEATURES = (
    "memory_shortwave_energy_6h_j_m2", "memory_shortwave_energy_12h_j_m2",
    "memory_shortwave_energy_24h_j_m2", "memory_longwave_mean_6h_w_m2",
    "memory_longwave_mean_24h_w_m2", "memory_air_mean_6h_c", "memory_air_mean_24h_c",
    "memory_air_range_6h_c", "memory_air_range_24h_c", "memory_air_change_3h_c",
    "memory_air_change_6h_c",
)
SURFACE_FEATURES = tuple(f"worldcover_{name}_class_fraction" for name in ("tree", "grass", "crop", "built", "bare"))
FEATURE_SETS = {"S0": BASE_FEATURES, "A": BASE_FEATURES,
                "B": BASE_FEATURES + MEMORY_FEATURES,
                "C": BASE_FEATURES + SURFACE_FEATURES,
                "D": BASE_FEATURES + MEMORY_FEATURES + SURFACE_FEATURES}
REQUIRED = ("sample_id", "region_id", "datetime_utc", "latitude", "longitude", "lst_c",
            "label_product", "acquisition_id", "block_id", "spatial_holdout",
            "in_holdout_buffer", "cohort_origin") + BASE_FEATURES
CONFIG = ModelConfig(heldout_regions=("cabauw",), train_end="2022-12-31",
                     calibration_end="2023-12-31", random_state=SEED)
SELECTION_RULE = {
    "development_period": "2023-01-01/2023-07-01 (end exclusive)",
    "score": "mean region-by-phase date/acquisition/surface-balanced MAE",
    "minimum_overall_development_pilot_date_clusters": 2,
    "minimum_development_dates_per_fitted_expanded_London_Sioux_phase": 2,
    "sparse_legacy_groups": "retain observed scores, report limited support; no-regression screen still applies",
    "added_features_min_mae_improvement_c": 0.15,
    "added_features_min_relative_mae_improvement": 0.05,
    "group_regression_max_absolute_c": 0.20,
    "group_regression_max_relative": 0.10,
    "fewest_features_within_best_mae_c": 0.05,
    "tie_order": ["A", "C", "B", "D"],
    "fallback": "A; unsupported development never establishes improvement",
    "auto_promotion": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")


def strict_bool(values, name):
    allowed = values.map(lambda v: isinstance(v, (bool, np.bool_, int, np.integer)) and v in (0, 1))
    if not allowed.all():
        raise ValueError(f"{name} must contain explicit nonmissing booleans, not strings.")
    return values.astype(bool)


def load_paired_input(path, *, evaluation_2024=False):
    """Check only timestamps before decoding any thermal columns from a file."""
    dates = pd.read_parquet(path, columns=["datetime_utc"])
    timestamps = pd.to_datetime(dates.datetime_utc, utc=True, errors="raise")
    start, stop = (("2024-01-01", "2025-01-01") if evaluation_2024 else ("2021-01-01", "2024-01-01"))
    if timestamps.isna().any() or not timestamps.between(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC"), inclusive="left"
    ).all():
        period = "2024 legacy evaluation" if evaluation_2024 else "2021--2023"
        raise ValueError(f"Only {period} input is permitted; thermal columns were not loaded.")
    return pd.read_parquet(path)


def prepare_input(frame, *, evaluation_2024=False):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Required paired input columns missing: {missing}")
    data = frame.copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("Paired input is empty.")
    data["datetime_utc"] = pd.to_datetime(data.datetime_utc, utc=True, errors="raise")
    start, stop = (("2024-01-01", "2025-01-01") if evaluation_2024 else ("2021-01-01", "2024-01-01"))
    if data.datetime_utc.isna().any() or not data.datetime_utc.between(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC"), inclusive="left"
    ).all():
        raise ValueError("Date is outside the explicit 2024 legacy window." if evaluation_2024 else
                         "This runner accepts only 2021--2023; 2024 legacy tests and 2025 blind labels stay separate.")
    for name in ("sample_id", "region_id", "label_product", "acquisition_id", "block_id", "cohort_origin"):
        if data[name].isna().any() or data[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name} identity.")
        data[name] = data[name].astype(str)
    if data.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id; never duplicate labels across candidate cohorts.")
    if not data.cohort_origin.isin(["legacy", "expanded"]).all():
        raise ValueError("cohort_origin must be legacy or expanded.")
    for name in ("spatial_holdout", "in_holdout_buffer"):
        data[name] = strict_bool(data[name], name)
    for name in ("latitude", "longitude", "lst_c"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite {name}; the paired builder must screen labels first.")
    if not data.latitude.between(-90, 90).all() or not data.longitude.between(-180, 180).all():
        raise ValueError("Coordinates must be WGS84.")
    for name in set(BASE_FEATURES + MEMORY_FEATURES + SURFACE_FEATURES) - {"climate_class"}:
        if name in data:
            data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
            if np.isinf(data[name]).any():
                raise ValueError(f"Infinite predictor {name}.")
    data["climate_class"] = data.climate_class.astype("string").str.strip()
    if data.climate_class.isna().any() or data.climate_class.str.lower().isin(["", "unknown", "nan", "none", "0", "__unknown__"]).any():
        raise ValueError("A known climate class is required for every paired observation.")
    data["climate_class"] = data.climate_class.astype(str)
    data["utc_day"] = data.datetime_utc.dt.floor("D")
    data["phase"] = np.select([data.solar_elevation_deg.ge(10), data.solar_elevation_deg.le(-6)],
                              ["day", "night"], default="twilight")
    data["air_group"] = np.select([data.air_temperature_c.le(0), data.air_temperature_c.ge(30)],
                                  ["cold", "hot"], default="mild")
    data["snow_group"] = np.where(data.era5_snow_water_equivalent_m.ge(0.001), "snow", "no_snow")
    data["season"] = data.datetime_utc.dt.month.map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
    surface = "weight_surface_group" if "weight_surface_group" in data else "surface_group"
    data["weight_surface_group"] = (data[surface].fillna("unclassified").astype(str)
                                     if surface in data else "unclassified")
    data["temporal_partition"] = np.select(
        [data.datetime_utc.lt(pd.Timestamp("2023-01-01", tz="UTC")),
         data.datetime_utc.lt(pd.Timestamp("2023-07-01", tz="UTC"))],
        ["fit", "development"], default="calibration")
    data["split"] = np.select(
        [data.region_id.eq("cabauw"), data.spatial_holdout, data.in_holdout_buffer, data.phase.eq("twilight")],
        ["heldout_region", "heldout_spatial", "excluded_buffer", "excluded_twilight"],
        default=data.temporal_partition)
    # A spatial test can share an acquisition with a fitting cell. Its temporal
    # period must still be unique; whole acquisitions cannot straddle date splits.
    if data.groupby("acquisition_id").temporal_partition.nunique().gt(1).any():
        raise ValueError("An acquisition crosses temporal partitions.")
    if data.groupby(["region_id", "block_id"]).spatial_holdout.nunique().gt(1).any():
        raise ValueError("A spatial block changes its reserved status across rows/dates.")
    if data.duplicated(["region_id", "acquisition_id", "latitude", "longitude"]).any():
        raise ValueError("Duplicate physical sample for one acquisition, including alternate product versions.")
    assert not data.loc[data.split.isin(["fit", "development", "calibration"]), "region_id"].eq("cabauw").any()
    return data


def balanced_weights(data):
    """Equal region/phase, date, acquisition, surface, then pixel contribution."""
    if data.empty:
        return np.array([], dtype=float)
    group = ["region_id", "phase"]
    day = group + ["utc_day"]
    acq = day + ["acquisition_id"]
    leaf = acq + ["weight_surface_group"]
    n_group = data[group].drop_duplicates().shape[0]
    n_days = data.groupby(group, observed=True).utc_day.transform("nunique").to_numpy()
    n_acq = data.groupby(day, observed=True).acquisition_id.transform("nunique").to_numpy()
    n_surface = data.groupby(acq, observed=True).weight_surface_group.transform("nunique").to_numpy()
    n_pixels = data.groupby(leaf, observed=True).sample_id.transform("size").to_numpy()
    weights = 1 / (n_group * n_days * n_acq * n_surface * n_pixels)
    return weights / weights.sum()


def complete_rows(data, features):
    numeric = [f for f in features if f != "climate_class"]
    return np.isfinite(data[numeric].to_numpy(dtype=float)).all(axis=1)


def cap_rows(data, maximum=MAX_ROWS):
    if len(data) <= maximum:
        return data.sort_values("sample_id").copy()
    ranked = data.copy()
    ranked["_hash"] = ranked.sample_id.map(lambda s: hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest())
    ranked = ranked.sort_values("_hash")
    strata = ["region_id", "phase", "utc_day", "acquisition_id", "weight_surface_group"]
    ranked["_rank"] = ranked.groupby(strata, observed=True).cumcount()
    return ranked.sort_values(["_rank", "_hash"]).head(maximum).drop(columns=["_rank", "_hash"]).sort_values("sample_id")


def support_counts(data):
    return [{"region_id": str(r), "phase": str(p), "rows": len(g),
             "dates": g.utc_day.nunique(), "acquisitions": g.acquisition_id.nunique(),
             "expanded_dates": g.loc[g.cohort_origin.eq("expanded"), "utc_day"].nunique(),
             "blocks": g.block_id.nunique(), "seasons": sorted(g.season.unique().tolist())}
            for (r, p), g in data.groupby(["region_id", "phase"], observed=True)]


def candidate_cohorts(data):
    available = ["A"]
    if all(f in data for f in MEMORY_FEATURES):
        available.append("B")
    if all(f in data for f in SURFACE_FEATURES):
        available.append("C")
        if "B" in available:
            available.append("D")
    union = tuple(dict.fromkeys(f for c in available for f in FEATURE_SETS[c]))
    usable_phase = data.phase.isin(["day", "night"])
    common = data.loc[complete_rows(data, union) & usable_phase].copy()
    legacy = data.loc[data.cohort_origin.eq("legacy") & complete_rows(data, BASE_FEATURES) & usable_phase].copy()
    dropped_groups = []
    for (region, phase), group in common.loc[common.split.eq("fit") & common.cohort_origin.eq("expanded") & common.phase.eq("night")].groupby(["region_id", "phase"], observed=True):
        if group.utc_day.nunique() < 2:
            mask = common.split.eq("fit") & common.region_id.eq(region) & common.phase.eq(phase) & common.cohort_origin.eq("expanded")
            common.loc[mask, "split"] = "excluded_sparse_fit"
            dropped_groups.append({"region_id": region, "phase": phase, "fit_dates": group.utc_day.nunique(), "reason": "fewer_than_two_expanded_night_dates"})
    fit = cap_rows(common.loc[common.split.eq("fit")])
    eval_data = common.loc[common.split.isin(["development", "calibration", "heldout_region", "heldout_spatial"])].sort_values("sample_id")
    result = {c: fit for c in available}
    result["S0"] = cap_rows(legacy.loc[legacy.split.eq("fit")])
    audit = {"input_rows": len(data), "common_complete_rows": len(common),
             "common_required_features": list(union), "available_candidates": available,
             "incomplete_or_twilight_rows": len(data) - int((complete_rows(data, union) & usable_phase).sum()),
             "surface_weighting_fallback_rows": int(common.weight_surface_group.eq("unclassified").sum()),
             "excluded_sparse_fit_groups": dropped_groups,
             "support_by_split": {s: support_counts(g) for s, g in common.groupby("split", observed=True)}}
    return result, eval_data, audit


def row_hash(data):
    return hashlib.sha256("\n".join(data.sample_id.astype(str)).encode()).hexdigest()


def metrics(data, predicted):
    if data.empty:
        return {"status": "no_observations", "n": 0, "date_count": 0}
    p = np.asarray(predicted, dtype=float)
    if p.shape != (len(data),) or not np.isfinite(p).all():
        raise ValueError("Predictions are nonfinite or misaligned.")
    y = data.lst_c.to_numpy(dtype=float)
    w = balanced_weights(data)
    e = p - y
    centred = data[["region_id", "acquisition_id"]].copy()
    centred["_weighted_error"] = e * w
    centred["_weight"] = w
    keys = ["region_id", "acquisition_id"]
    mean_error = (centred.groupby(keys, observed=True)._weighted_error.transform("sum") /
                  centred.groupby(keys, observed=True)._weight.transform("sum")).to_numpy()
    date_count = data[["region_id", "utc_day"]].drop_duplicates().shape[0]
    return {"status": "exploratory" if date_count >= 2 else "unsupported_single_date",
            "n": len(data), "date_count": date_count, "utc_date_count": data.utc_day.nunique(),
            "acquisition_count": data.acquisition_id.nunique(),
            "mae_c": float(np.sum(w * np.abs(e))), "rmse_c": float(np.sqrt(np.sum(w * e**2))),
            "bias_c": float(np.sum(w * e)),
            **{f"fraction_abs_error_gt_{t}c": float(np.sum(w * (np.abs(e) > t))) for t in (3, 5, 7)},
            "centered_contrast_mae_c": float(np.sum(w * np.abs(e - mean_error))),
            "unweighted_pixel_mae_c": float(np.mean(np.abs(e))),
            "sample_id_sha256": row_hash(data)}


def group_metrics(data, predicted):
    output = {"overall": metrics(data, predicted)}
    if data.empty:
        return output
    for name, keys in (("by_region", ["region_id"]), ("by_phase", ["phase"]),
                       ("by_region_phase", ["region_id", "phase"]),
                       ("by_air_group", ["air_group"]), ("by_snow", ["snow_group"]),
                       ("by_season", ["season"]), ("by_label_product", ["label_product"]),
                       ("by_climate_class", ["climate_class"]),
                       ("by_region_phase_air", ["region_id", "phase", "air_group"]),
                       ("by_region_phase_snow", ["region_id", "phase", "snow_group"]),
                       ("by_temporal_period", ["temporal_partition"])):
        values = {}
        for key, indexes in data.groupby(keys, observed=True).indices.items():
            label = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            values[label] = metrics(data.iloc[indexes], np.asarray(predicted)[indexes])
        output[name] = values
    return output


def choose_candidate(dev_scores, fitted_support):
    decision = {"selected_candidate": "A", "status": "unsupported_development", "rule": SELECTION_RULE}
    if "A" not in dev_scores or not dev_scores["A"].get("by_region_phase"):
        return decision
    groups = dev_scores["A"]["by_region_phase"]
    decision["sparse_observed_groups"] = sorted(k for k, g in groups.items() if g["date_count"] < 2)
    required = {f"{r['region_id']}|{r['phase']}" for r in fitted_support
                if r["region_id"] in ("greater_london", "sioux_falls") and r.get("expanded_dates", 0) > 0}
    if dev_scores["A"]["overall"]["date_count"] < 2 or any(k not in groups or groups[k]["date_count"] < 2 for k in required):
        decision["missing_or_sparse_groups"] = sorted(k for k in required if k not in groups or groups[k]["date_count"] < 2)
        return decision
    base = dev_scores["A"]["overall"]["mae_c"]
    eligible = {"A": base}
    checks = {}
    for name in ("B", "C", "D"):
        if name not in dev_scores:
            continue
        candidate = dev_scores[name]
        score = candidate["overall"]["mae_c"]
        improvement = base - score
        regressions = [k for k, g in groups.items() if
                       candidate["by_region_phase"][k]["mae_c"] - g["mae_c"] > max(0.20, 0.10 * g["mae_c"])]
        passed = improvement >= 0.15 and improvement >= 0.05 * base and not regressions
        checks[name] = {"score_mae_c": score, "improvement_vs_A_c": improvement,
                        "regressing_groups": regressions, "eligible": passed}
        if passed:
            eligible[name] = score
    best = min(eligible.values())
    near = [k for k, v in eligible.items() if v <= best + 0.05]
    chosen = min(near, key=lambda k: (len(FEATURE_SETS[k]), SELECTION_RULE["tie_order"].index(k)))
    return {**decision, "selected_candidate": chosen, "status": "exploratory_development_selection",
            "scores": eligible, "checks": checks, "improvement_claim": False}


def phase_calibration(data, predicted):
    result = {}
    for phase in ("day", "night"):
        mask = data.phase.eq(phase).to_numpy()
        g = data.loc[mask]
        if g.empty:
            result[phase] = {"status": "no_calibration_observations", "radius_c": None, "dates": 0}
            continue
        e = np.abs(np.asarray(predicted)[mask] - g.lst_c.to_numpy())
        w = balanced_weights(g)
        order = np.argsort(e, kind="stable")
        radius = float(e[order[min(np.searchsorted(np.cumsum(w[order]), 0.9), len(e) - 1)]])
        dates = g[["region_id", "utc_day"]].drop_duplicates().shape[0]
        result[phase] = {"status": "exploratory_empirical" if dates >= 2 else "unsupported_single_date",
                         "radius_c": radius if dates >= 2 else None, "dates": dates,
                         "nominal_coverage": 0.9, "guaranteed": False}
    return result


def interval_coverage(data, predicted, intervals):
    result = {}
    for phase in ("day", "night"):
        mask = data.phase.eq(phase).to_numpy()
        g = data.loc[mask]
        radius = intervals[phase]["radius_c"]
        if g.empty or radius is None:
            result[phase] = {"status": "unsupported", "rows": len(g), "coverage": None}
        else:
            inside = np.abs(np.asarray(predicted)[mask] - g.lst_c.to_numpy()) <= radius
            result[phase] = {"status": "exploratory_empirical", "rows": len(g),
                             "coverage": float(np.sum(balanced_weights(g) * inside)),
                             "interval_width_c": 2 * radius, "guaranteed": False}
    return result


def run_experiment(frame, output_dir, *, baseline_model_path=None, protocol_path=None, input_path=None):
    started = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new research run directory; results are immutable.")
    data = prepare_input(frame)
    cohorts, evaluation, cohort_audit = candidate_cohorts(data)
    if len(cohorts["A"]) < 2:
        raise ValueError("Fewer than two complete paired fitting rows after exclusions; no estimator can be fitted.")
    output.mkdir(parents=True)
    baseline = joblib.load(baseline_model_path) if baseline_model_path else None
    manifest = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "research_only": True, "auto_promotion": False, "estimator": asdict(CONFIG),
                "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "thread_limit": 4, "max_fit_rows": MAX_ROWS, "selection_rule": SELECTION_RULE,
                "fit_weight_scale": "balanced_weights normalized to mean one (multiply by fitting row count)",
                "protocol_sha256": sha(protocol_path) if protocol_path else None,
                "input_sha256": sha(input_path) if input_path else None,
                "baseline_sha256": sha(baseline_model_path) if baseline_model_path else None,
                "source_sha256": sha(__file__), "cohorts": cohort_audit,
                "model_dependency_sha256": sha(Path(__file__).with_name("model.py")),
                "candidates": {k: {"features": FEATURE_SETS[k], "fit_rows": len(v),
                                    "fit_sample_id_sha256": row_hash(v),
                                    "weight_sha256": hashlib.sha256(balanced_weights(v).tobytes()).hexdigest(),
                                    "support": support_counts(v)} for k, v in cohorts.items()}}
    save_json(output / "frozen_manifest.json", manifest)
    pd.concat([v[["sample_id"]].assign(candidate=k, weight=balanced_weights(v)) for k, v in cohorts.items()]).to_parquet(output / "fitting_rows.parquet", index=False)
    evaluation[["sample_id", "region_id", "datetime_utc", "split", "temporal_partition", "phase", "acquisition_id", "block_id"]].to_parquet(output / "evaluation_manifest.parquet", index=False)
    models, linear_models, dev_scores = {}, {}, {}
    dev = evaluation.loc[evaluation.split.eq("development")].copy()
    with threadpool_limits(limits=4):
        for name, training in cohorts.items():
            if len(training) < 2:
                continue
            features = FEATURE_SETS[name]
            model, linear = build_estimators(features, CONFIG)
            # Normalize to mean one so HGB's L2 penalty retains its v1 scale.
            weights = balanced_weights(training) * len(training)
            model.fit(training[list(features)], target_offset(training), regressor__sample_weight=weights)
            linear.fit(training[list(features)], target_offset(training), regressor__sample_weight=weights)
            models[name], linear_models[name] = model, linear
            if not dev.empty:
                predicted = dev.air_temperature_c.to_numpy() + model.predict(dev[list(features)])
                dev_scores[name] = group_metrics(dev, predicted)
        selection = choose_candidate(dev_scores, support_counts(cohorts["A"]))
        save_json(output / "selection.json", selection)
        # Selection is now frozen. Calibration and geographic tests do not choose
        # a feature family, even when their later scores look more attractive.
        results = {"research_only": True, "auto_promotion": False,
                   "selection": selection, "cohorts": cohort_audit, "candidates": {}}
        predictions = []
        for name, model in models.items():
            features = FEATURE_SETS[name]
            predicted = evaluation.air_temperature_c.to_numpy() + model.predict(evaluation[list(features)]) if len(evaluation) else np.array([])
            linear_prediction = evaluation.air_temperature_c.to_numpy() + linear_models[name].predict(evaluation[list(features)]) if len(evaluation) else np.array([])
            reference = None
            if baseline is not None and len(evaluation):
                reference = evaluation.air_temperature_c.to_numpy() + baseline["model"].predict(evaluation[baseline["features"]])
            calmask = evaluation.split.eq("calibration").to_numpy()
            intervals = phase_calibration(evaluation.loc[calmask], predicted[calmask])
            bundle = {"research_only": True, "schema_version": "option-b-v2", "model": model,
                      "features": list(features), "target": "lst_c - air_temperature_c",
                      "candidate": name, "phase_intervals": intervals, "config": asdict(CONFIG),
                      "training_climate_classes": sorted(cohorts[name].climate_class.unique().tolist()),
                      "selection": selection, "auto_promotion": False}
            directory = output / name
            directory.mkdir()
            joblib.dump(bundle, directory / "model.joblib", compress=3)
            candidate_result = {"fit_rows": len(cohorts[name]), "features": list(features),
                                "model_sha256": sha(directory / "model.joblib"), "phase_intervals": intervals,
                                "evaluation": {}}
            for split in ("development", "calibration", "heldout_region", "heldout_spatial"):
                mask = evaluation.split.eq(split).to_numpy()
                g = evaluation.loc[mask]
                stats = {"model": group_metrics(g, predicted[mask]),
                         "air_only": group_metrics(g, g.air_temperature_c.to_numpy()),
                         "ridge": group_metrics(g, linear_prediction[mask])}
                if reference is not None:
                    stats["v1"] = group_metrics(g, reference[mask])
                paired = {}
                for label in ("air_only", "ridge", "v1"):
                    if label in stats and len(g):
                        paired[label] = {"same_sample_id_sha256": row_hash(g),
                                         "delta_mae_c": stats["model"]["overall"]["mae_c"] - stats[label]["overall"]["mae_c"],
                                         "delta_centered_contrast_mae_c": stats["model"]["overall"]["centered_contrast_mae_c"] - stats[label]["overall"]["centered_contrast_mae_c"],
                                         "interpretation": "negative favors candidate; descriptive paired comparison, no significance claim"}
                stats["paired_comparisons"] = paired
                stats["empirical_interval_coverage"] = interval_coverage(g, predicted[mask], intervals)
                candidate_result["evaluation"][split] = stats
            rows = evaluation[["sample_id", "region_id", "datetime_utc", "split", "temporal_partition", "phase", "air_group", "snow_group", "acquisition_id", "block_id", "label_product", "lst_c", "air_temperature_c"]].copy()
            rows["candidate"] = name
            rows["predicted_lst_c"] = predicted
            rows["ridge_lst_c"] = linear_prediction
            if reference is not None:
                rows["v1_lst_c"] = reference
            predictions.append(rows)
            results["candidates"][name] = candidate_result
        pd.concat(predictions, ignore_index=True).to_parquet(output / "evaluation_predictions.parquet", index=False)
    results["elapsed_seconds"] = time.monotonic() - started
    results["status"] = "exploratory_fitted_no_promotion"
    results["selected_model_path"] = str(output / selection["selected_candidate"] / "model.joblib")
    results["warnings"] = ["No 2024 or 2025 labels were accepted by this runner.",
                           "Cabauw and spatial holdouts never selected the model or calibrated intervals.",
                           "Scores describe available clear-sky source observations, not all-weather truth.",
                           "Small acquisition groups do not establish seasonal or global skill.",
                           "v1 was trained on some legacy locations/dates; matched error comparisons alone are not independent geographic evidence for v1."]
    save_json(output / "results.json", results)
    save_json(output / "post_selection_freeze.json", {
        "version": VERSION, "selected_candidate": selection["selected_candidate"],
        "selection_sha256": sha(output / "selection.json"),
        "results_sha256": sha(output / "results.json"),
        "manifest_sha256": sha(output / "frozen_manifest.json"),
        "selected_model_sha256": sha(results["selected_model_path"]),
        "baseline_sha256": manifest["baseline_sha256"],
        "source_sha256": manifest["source_sha256"],
        "phase_intervals": results["candidates"][selection["selected_candidate"]]["phase_intervals"],
        "calibration_complete_before_legacy_evaluation": True,
        "legacy_2024_may_select_or_refit": False,
        "blind_2025_opened": False,
    })
    lines = ["# Option B exploratory fit", "", f"Status: {results['status']}. Production was not updated.", "",
             f"Development selection: {selection['selected_candidate']} ({selection['status']}).", "",
             "| Candidate | Fitting rows | Features | Development MAE °C |", "|---|---:|---:|---:|"]
    for name, result in results["candidates"].items():
        score = result["evaluation"]["development"]["model"]["overall"].get("mae_c")
        lines.append(f"| {name} | {result['fit_rows']} | {len(result['features'])} | {score:.3f} |" if score is not None else f"| {name} | {result['fit_rows']} | {len(result['features'])} | Unsupported: no observations |")
    lines += ["", "See results.json for date counts, geographic and temporal groups, hot/cold/snow errors, error-tail frequencies, centered spatial contrast and paired baselines.", "",
              "All reported errors are exploratory. Missing development support does not establish improvement; no model is promoted automatically."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return results


def verify_frozen_run(run_dir, baseline_model_path):
    """Verify saved selection/calibration/model before reading legacy labels."""
    root = Path(run_dir).resolve()
    freeze = json.loads((root / "post_selection_freeze.json").read_text())
    candidate = freeze["selected_candidate"]
    if candidate not in ("A", "B", "C", "D"):
        raise ValueError("Invalid frozen candidate name.")
    expected = {"selection.json": "selection_sha256", "results.json": "results_sha256",
                "frozen_manifest.json": "manifest_sha256", f"{candidate}/model.joblib": "selected_model_sha256"}
    for filename, field in expected.items():
        if sha(root / filename) != freeze[field]:
            raise ValueError(f"Frozen artifact hash mismatch: {filename}")
    if not freeze.get("baseline_sha256") or sha(baseline_model_path) != freeze["baseline_sha256"]:
        raise ValueError("v1 baseline hash differs from the frozen comparison baseline.")
    result = json.loads((root / "results.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    if result["selection"] != selection or selection["selected_candidate"] != candidate:
        raise ValueError("Frozen selection and results disagree.")
    if result["candidates"][candidate]["phase_intervals"] != freeze["phase_intervals"]:
        raise ValueError("Frozen calibration intervals disagree.")
    if not freeze.get("calibration_complete_before_legacy_evaluation") or freeze.get("legacy_2024_may_select_or_refit"):
        raise ValueError("Legacy evaluation requires completed frozen calibration and no refitting.")
    return freeze, root / candidate / "model.joblib"


def evaluate_legacy_2024(input_path, run_dir, output_dir, *, baseline_model_path):
    """Read only 2024, after verified selection; never fit or choose a runner-up."""
    freeze, model_path = verify_frozen_run(run_dir, baseline_model_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new legacy evaluation directory; results are immutable.")
    # Metadata-only date validation precedes decoding thermal columns.
    data = prepare_input(load_paired_input(input_path, evaluation_2024=True), evaluation_2024=True)
    bundle = joblib.load(model_path)
    baseline = joblib.load(baseline_model_path)
    if bundle["phase_intervals"] != freeze["phase_intervals"] or bundle["candidate"] != freeze["selected_candidate"]:
        raise ValueError("Loaded bundle does not match frozen candidate/calibration.")
    features = bundle["features"]
    if tuple(features) != FEATURE_SETS[freeze["selected_candidate"]]:
        raise ValueError("Frozen candidate feature allowlist mismatch.")
    required = tuple(dict.fromkeys(features + baseline["features"]))
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Legacy evaluator needs paired frozen predictors: {missing}")
    common = data.loc[complete_rows(data, required) & data.phase.isin(["day", "night"])].sort_values("sample_id").copy()
    common["temporal_partition"] = "legacy_test_2024"
    common["split"] = "legacy_test_2024"
    if common.empty:
        raise ValueError("No complete common 2024 observations; no evaluation is possible.")
    with threadpool_limits(limits=4):
        prediction = common.air_temperature_c.to_numpy() + bundle["model"].predict(common[features])
        reference = common.air_temperature_c.to_numpy() + baseline["model"].predict(common[baseline["features"]])
    output.mkdir(parents=True)
    stats = {"candidate": group_metrics(common, prediction),
             "v1": group_metrics(common, reference),
             "air_only": group_metrics(common, common.air_temperature_c.to_numpy())}
    result = {"status": "legacy_2024_evaluated_no_refit_no_promotion", "research_only": True,
              "selected_candidate": freeze["selected_candidate"], "input_sha256": sha(input_path),
              "post_selection_freeze_sha256": sha(Path(run_dir) / "post_selection_freeze.json"),
              "selected_model_sha256": sha(model_path), "baseline_sha256": sha(baseline_model_path),
              "input_rows": len(data), "common_complete_rows": len(common),
              "dropped_incomplete_or_twilight_rows": len(data) - len(common),
              "unseen_candidate_training_climate_rows": int((~common.climate_class.isin(bundle["training_climate_classes"])).sum()),
              "frozen_phase_intervals": bundle["phase_intervals"],
              "metrics": stats, "support": support_counts(common),
              "empirical_interval_coverage": interval_coverage(common, prediction, bundle["phase_intervals"]),
              "paired_comparisons": {label: {"same_sample_id_sha256": row_hash(common),
                    "delta_mae_c": stats["candidate"]["overall"]["mae_c"] - stats[label]["overall"]["mae_c"],
                    "delta_centered_contrast_mae_c": stats["candidate"]["overall"]["centered_contrast_mae_c"] - stats[label]["overall"]["centered_contrast_mae_c"]}
                    for label in ("v1", "air_only")},
              "warnings": ["Previously inspected 2024 legacy cohort, not a blind test.",
                           "This result cannot select a runner-up, refit, recalibrate or promote a model.",
                           "Unknown candidate training climates and missing night/extreme groups remain unsupported.",
                           "2025 labels were not loaded."]}
    rows = common[["sample_id", "region_id", "datetime_utc", "phase", "air_group", "snow_group", "acquisition_id", "block_id", "lst_c", "air_temperature_c"]].copy()
    rows["candidate_lst_c"] = prediction
    rows["v1_lst_c"] = reference
    rows.to_parquet(output / "predictions.parquet", index=False)
    save_json(output / "results.json", result)
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "legacy-2024":
        parser = argparse.ArgumentParser(description="Evaluate frozen research selection on the original 2024 cohort; never fit.")
        parser.add_argument("--input", required=True, type=Path)
        parser.add_argument("--training-run", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--baseline-model", required=True, type=Path)
        args = parser.parse_args(sys.argv[2:])
        result = evaluate_legacy_2024(args.input, args.training_run, args.output, baseline_model_path=args.baseline_model)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    result = run_experiment(load_paired_input(args.input), args.output, baseline_model_path=args.baseline_model,
                            protocol_path=args.protocol, input_path=args.input)
    print(json.dumps({"status": result["status"], "selection": result["selection"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
