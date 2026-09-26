"""Auditable CPU model for the pilot: LST = air temperature + learned offset.

This module deliberately does not derive features from thermal labels. Run it on
the remote pilot server. Numerical thread limits belong in the launcher.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


REQUIRED_COLUMNS = (
    "region_id", "datetime_utc", "latitude", "longitude", "lst_c",
    "air_temperature_c", "climate_class",
)
NUMERIC_FEATURE_ALLOWLIST = (
    "air_temperature_c", "ndvi", "ndbi", "ndwi", "albedo", "albedo_proxy", "elevation", "slope", "terrain_relief_300m",
    "aspect_sin", "aspect_cos", "tree_cover_fraction", "canopy_height_m",
    "built_fraction", "building_height_m", "impervious_fraction", "water_fraction",
    "sky_view_factor", "shadow_fraction", "solar_elevation_deg",
    "solar_azimuth_sin", "solar_azimuth_cos", "hour_sin", "hour_cos",
    "day_of_year_sin", "day_of_year_cos", "relative_humidity_pct", "dewpoint_c",
    "wind_speed_m_s", "wind_direction_sin", "wind_direction_cos",
    "surface_pressure_hpa", "cloud_cover_fraction", "shortwave_down_w_m2",
    "direct_shortwave_w_m2", "diffuse_shortwave_w_m2", "longwave_down_w_m2",
    "era5_longwave_down_w_m2", "era5_snow_water_equivalent_m",
    "precipitation_mm_h", "rain_mm_24h", "rain_mm_72h", "soil_moisture_m3_m3",
    "snow_cover_fraction", "snow_depth_m", "air_temperature_lag1_c",
    "air_temperature_lag3_c", "air_temperature_lag24_c",
    "shortwave_down_lag1_w_m2", "shortwave_down_mean3_w_m2", "days_since_rain",
    "wetness_index",
)
CATEGORICAL_FEATURE = "climate_class"
UNKNOWN_CLIMATE = "__unknown__"
EVALUATION_SPLITS = ("test_temporal", "test_region")


@dataclass(frozen=True)
class ModelConfig:
    heldout_regions: tuple[str, ...]
    train_end: str
    calibration_end: str
    features: tuple[str, ...] | None = None
    max_iter: int = 150
    max_leaf_nodes: int = 15
    max_depth: int = 6
    learning_rate: float = 0.08
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    max_train_rows: int = 200_000
    interval_coverage: float = 0.90
    random_state: int = 42
    min_train_rows: int = 100
    min_calibration_rows: int = 30
    min_evaluation_rows: int = 30
    min_train_days: int = 2
    min_calibration_days: int = 2
    allow_unknown_climate_smoke: bool = False


def utc_day(value: str) -> pd.Timestamp:
    """Treat a boundary as an inclusive UTC calendar date."""
    return pd.Timestamp(value, tz="UTC").normalize() if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC").normalize()


def validate_config(config: ModelConfig) -> None:
    if not config.heldout_regions:
        raise ValueError("At least one explicit heldout region is required.")
    if utc_day(config.train_end) >= utc_day(config.calibration_end):
        raise ValueError("train_end must precede calibration_end by at least one UTC day.")
    if not 0 < config.interval_coverage < 1:
        raise ValueError("interval_coverage must be strictly between 0 and 1.")
    for name in (
        "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf",
        "max_train_rows", "min_train_rows", "min_calibration_rows",
        "min_evaluation_rows", "min_train_days", "min_calibration_days",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if config.max_train_rows < config.min_train_rows:
        raise ValueError("max_train_rows cannot be smaller than min_train_rows.")


def select_features(
    columns: Sequence[str], requested: Sequence[str] | None = None,
) -> list[str]:
    """An allowlist prevents coordinates, station IDs and label products leaking in."""
    available = set(columns)
    if requested is not None:
        forbidden = sorted(set(requested) - set(NUMERIC_FEATURE_ALLOWLIST) - {CATEGORICAL_FEATURE})
        if forbidden:
            raise ValueError(f"Unsafe or unsupported requested features: {forbidden}. Only the feature allowlist is accepted.")
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Requested features are absent from the input: {missing}.")
        chosen = set(requested)
    else:
        chosen = available
    numeric = [name for name in NUMERIC_FEATURE_ALLOWLIST if name in chosen and name in available]
    if "air_temperature_c" not in available:
        raise ValueError("air_temperature_c is required to predict the surface–air difference.")
    if "air_temperature_c" not in numeric:
        numeric.insert(0, "air_temperature_c")
    return numeric + [CATEGORICAL_FEATURE]


def prepare_frame(frame: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, list[str]]:
    validate_config(config)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Required input columns missing: {missing}.")
    if frame.empty:
        raise ValueError("The input dataset is empty.")
    data = frame.copy().reset_index(drop=True)
    if data["region_id"].isna().any() or data["region_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row needs a nonempty region_id.")
    data["region_id"] = data["region_id"].astype(str)
    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"], utc=True, errors="raise")
    if data["datetime_utc"].isna().any():
        raise ValueError("datetime_utc cannot contain missing dates.")
    data["utc_day"] = data["datetime_utc"].dt.floor("D")
    for name in ("lst_c", "air_temperature_c", "latitude", "longitude"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain only finite values; quality-filter labels and coordinates before training.")
    if not data["latitude"].between(-90, 90).all() or not data["longitude"].between(-180, 180).all():
        raise ValueError("Coordinates must be WGS84 latitude/longitude in degrees.")
    climate = data[CATEGORICAL_FEATURE].astype("string").str.strip()
    climate = climate.mask(climate.isna() | climate.eq("") | climate.str.lower().isin(["unknown", "nan", "none", "null", "0"]), UNKNOWN_CLIMATE)
    data[CATEGORICAL_FEATURE] = climate.astype(str)
    if data[CATEGORICAL_FEATURE].eq(UNKNOWN_CLIMATE).all() and not config.allow_unknown_climate_smoke:
        raise ValueError("All climate classifications are unknown. Supply climate labels or explicitly use --allow-unknown-climate-smoke for an engineering smoke run only.")
    features = select_features(data.columns, config.features)
    for name in features:
        if name != CATEGORICAL_FEATURE:
            data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
            if np.isinf(data[name].to_numpy()).any():
                raise ValueError(f"Feature {name} contains infinity; missing values must be NaN.")
    return data, features


def assign_splits(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Split whole UTC days; heldout regions never enter training or calibration."""
    result = data.copy()
    if "utc_day" not in result:
        result["utc_day"] = pd.to_datetime(result["datetime_utc"], utc=True, errors="raise").dt.floor("D")
    missing_regions = sorted(set(config.heldout_regions) - set(result["region_id"].astype(str)))
    if missing_regions:
        raise ValueError(f"Requested heldout regions have no records: {missing_regions}.")
    train_end, calibration_end = utc_day(config.train_end), utc_day(config.calibration_end)
    heldout = result["region_id"].isin(config.heldout_regions)
    result["split"] = np.select(
        [heldout, result["utc_day"].le(train_end), result["utc_day"].le(calibration_end)],
        ["test_region", "train", "calibration"], default="test_temporal",
    )
    result["time_period"] = np.select(
        [result["utc_day"].le(train_end), result["utc_day"].le(calibration_end)],
        ["training_dates", "calibration_dates"], default="future_dates",
    )
    return result


def validate_splits(data: pd.DataFrame, config: ModelConfig) -> None:
    """Fail before fitting when partitions cannot support the declared evaluation."""
    required = {"train": config.min_train_rows, "calibration": config.min_calibration_rows, "test_region": config.min_evaluation_rows}
    for split, minimum in required.items():
        rows = int(data["split"].eq(split).sum())
        if rows < minimum:
            raise ValueError(f"Split {split!r} has {rows} records; at least {minimum} are required. Adjust boundaries or provide more independent observations.")
    temporal_count = int(data["split"].eq("test_temporal").sum())
    if 0 < temporal_count < config.min_evaluation_rows:
        raise ValueError(f"Split 'test_temporal' has only {temporal_count} records; at least {config.min_evaluation_rows} are required if this split is present.")
    for split, minimum in (("train", config.min_train_days), ("calibration", config.min_calibration_days)):
        days = data.loc[data["split"].eq(split), "utc_day"].nunique()
        if days < minimum:
            raise ValueError(f"Split {split!r} has {days} UTC days; at least {minimum} are required.")
    for region in config.heldout_regions:
        count = int((data["region_id"].eq(region) & data["split"].eq("test_region")).sum())
        if count < config.min_evaluation_rows:
            raise ValueError(f"Heldout region {region!r} has {count} records; at least {config.min_evaluation_rows} are required.")
    if data.groupby(["region_id", "utc_day"])["split"].nunique().gt(1).any():
        raise AssertionError("A region/day crossed partitions.")


def target_offset(data: pd.DataFrame) -> np.ndarray:
    return data["lst_c"].to_numpy(dtype=float) - data["air_temperature_c"].to_numpy(dtype=float)


def temperature_from_offset(air_temperature_c: Sequence[float], offset_c: Sequence[float]) -> np.ndarray:
    return np.asarray(air_temperature_c, dtype=float) + np.asarray(offset_c, dtype=float)


def build_estimators(features: Sequence[str], config: ModelConfig) -> tuple[Pipeline, Pipeline]:
    numeric = [name for name in features if name != CATEGORICAL_FEATURE]
    nonlinear_preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("climate", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), [CATEGORICAL_FEATURE]),
    ], remainder="drop", sparse_threshold=0)
    nonlinear = Pipeline([
        ("features", nonlinear_preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=config.learning_rate,
            max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes,
            max_depth=config.max_depth, min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            categorical_features=[False] * len(numeric) + [True],
            early_stopping=False, random_state=config.random_state,
        )),
    ])
    linear_numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    linear_preprocessor = ColumnTransformer([
        ("numeric", linear_numeric, numeric),
        ("climate", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CATEGORICAL_FEATURE]),
    ], remainder="drop", sparse_threshold=0)
    linear = Pipeline([("features", linear_preprocessor), ("regressor", Ridge(alpha=1.0))])
    return nonlinear, linear


def residual_interval_radius(residuals: Sequence[float], coverage: float) -> float:
    """Finite-sample order statistic; dependence/shift still limit coverage claims."""
    residuals = np.abs(np.asarray(residuals, dtype=float))
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Interval calibration requires finite heldout residuals.")
    rank = math.ceil((len(residuals) + 1) * coverage)
    if rank > len(residuals):
        raise ValueError("Too few calibration records for the requested interval coverage.")
    return float(np.partition(residuals, rank - 1)[rank - 1])


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(predicted) - np.asarray(observed)
    return {"n": len(errors), "mae_c": float(np.mean(np.abs(errors))), "rmse_c": float(np.sqrt(np.mean(errors ** 2))), "bias_c": float(np.mean(errors))}


def prediction_metrics(data: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, column in (("ml", "predicted_lst_c"), ("linear", "linear_lst_c"), ("air_only", "air_temperature_c")):
        result[label] = error_metrics(data["lst_c"].to_numpy(), data[column].to_numpy())
    inside = data["lst_c"].between(data["lower_lst_c"], data["upper_lst_c"])
    result["empirical_interval_coverage"] = float(inside.mean())
    result["mean_interval_width_c"] = float((data["upper_lst_c"] - data["lower_lst_c"]).mean())
    result["region_count"] = int(data["region_id"].nunique())
    result["utc_day_count"] = int(data["utc_day"].nunique())
    return result


def grouped_metrics(data: pd.DataFrame, column: str) -> dict[str, Any]:
    return {str(key): prediction_metrics(group) for key, group in data.groupby(column, observed=True)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def train_and_evaluate(frame: pd.DataFrame, output_dir: str | Path, config: ModelConfig) -> dict[str, Any]:
    """Fit only on training rows and write reviewable models and heldout results."""
    output = Path(output_dir)
    artifact_names = ("model.joblib", "metrics.json", "model_card.md", "heldout_predictions.parquet")
    if any((output / name).exists() for name in artifact_names):
        raise FileExistsError(f"{output} already contains model artifacts. Choose a new run directory.")
    data, features = prepare_frame(frame, config)
    data = assign_splits(data, config)
    validate_splits(data, config)
    training = data.loc[data["split"].eq("train")]
    if len(training) > config.max_train_rows:
        training = training.sample(n=config.max_train_rows, random_state=config.random_state)
    training = training.sort_values(["utc_day", "region_id"]).copy()
    if training[CATEGORICAL_FEATURE].eq(UNKNOWN_CLIMATE).all() and not config.allow_unknown_climate_smoke:
        raise ValueError("All training climate classifications are unknown. Heldout climate labels cannot supply training categories; provide climate data or explicitly request an engineering smoke run.")
    # Categorical vocabularies and the linear imputer/scaler are fitted here only.
    if training[CATEGORICAL_FEATURE].nunique() > 255:
        raise ValueError("climate_class has more than 255 training categories; expected climate classes, not station IDs.")
    model, linear = build_estimators(features, config)
    model.fit(training[features], target_offset(training))
    linear.fit(training[features], target_offset(training))
    calibration = data.loc[data["split"].eq("calibration")].copy()
    calibration_prediction = temperature_from_offset(calibration["air_temperature_c"], model.predict(calibration[features]))
    radius = residual_interval_radius(calibration["lst_c"].to_numpy() - calibration_prediction, config.interval_coverage)
    heldout = data.loc[data["split"].isin(EVALUATION_SPLITS)].copy()
    heldout["predicted_offset_c"] = model.predict(heldout[features])
    heldout["predicted_lst_c"] = temperature_from_offset(heldout["air_temperature_c"], heldout["predicted_offset_c"])
    heldout["linear_lst_c"] = temperature_from_offset(heldout["air_temperature_c"], linear.predict(heldout[features]))
    heldout["lower_lst_c"] = heldout["predicted_lst_c"] - radius
    heldout["upper_lst_c"] = heldout["predicted_lst_c"] + radius
    heldout["error_c"] = heldout["predicted_lst_c"] - heldout["lst_c"]
    training_classes = sorted(training[CATEGORICAL_FEATURE].unique().tolist())
    heldout["climate_unseen_in_training"] = ~heldout[CATEGORICAL_FEATURE].isin(training_classes)
    climate_unknown_fraction = float(data[CATEGORICAL_FEATURE].eq(UNKNOWN_CLIMATE).mean())
    smoke_only = bool(training[CATEGORICAL_FEATURE].eq(UNKNOWN_CLIMATE).all())
    warnings = [
        "This pilot does not establish worldwide 100 m accuracy or all-weather/nighttime accuracy; claims must match the independently observed labels.",
        "Intervals use a dedicated later-date calibration split. Spatial/temporal dependence and distribution shift prevent guaranteed nominal coverage, particularly in unseen regions or climate classes.",
        "A pixel count is not an independent observation count. Region/day metrics are reported alongside pooled errors.",
        "Features must use information available at prediction time; upstream lag construction and station/reanalysis joins require separate auditing.",
        "No early-stopping validation or random row train/test split is used; estimator settings are fixed before heldout evaluation.",
    ]
    if smoke_only:
        warnings.insert(0, "ENGINEERING SMOKE RUN ONLY: all climate labels are unknown; these artifacts cannot validate learning by climate classification.")
    if not data["split"].eq("test_temporal").any():
        warnings.append("No known-region future-date test records exist. Region holdout metrics do not establish future-date generalization.")
    all_missing_features = [name for name in features if name != CATEGORICAL_FEATURE and training[name].isna().all()]
    if all_missing_features:
        warnings.append(f"All training values are missing for these selected numeric features: {all_missing_features}.")
    metrics: dict[str, Any] = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "engineering_smoke_only" if smoke_only else "pilot_evaluation",
        "target": "lst_c - air_temperature_c", "features": features,
        "config": asdict(config), "n_input_rows": len(data), "n_fitted_rows": len(training),
        "split_counts": {str(key): int(value) for key, value in data["split"].value_counts().items()},
        "split_days": {str(key): int(value) for key, value in data.groupby("split")["utc_day"].nunique().items()},
        "training_regions": sorted(training["region_id"].unique().tolist()),
        "training_climate_classes": training_classes, "unknown_climate_fraction": climate_unknown_fraction,
        "all_missing_training_features": all_missing_features,
        "interval": {"method": "heldout_absolute_residual_order_statistic", "nominal_coverage": config.interval_coverage, "radius_c": radius, "calibration_rows": len(calibration), "calibration_days": int(calibration["utc_day"].nunique()), "guaranteed_coverage": False},
        "by_split": grouped_metrics(heldout, "split"),
        "by_region": grouped_metrics(heldout, "region_id"),
        "by_climate": grouped_metrics(heldout, CATEGORICAL_FEATURE),
        "by_time_period": grouped_metrics(heldout, "time_period"),
        "by_split_and_time_period": {
            f"{split}/{period}": prediction_metrics(group)
            for (split, period), group in heldout.groupby(["split", "time_period"])
        },
        "warnings": warnings,
    }
    # Summaries across whole region/day blocks avoid hiding difficult dates in pooled RMSE.
    block_rmse = heldout.groupby(["region_id", "utc_day"])["error_c"].agg(lambda values: float(np.sqrt(np.mean(values.to_numpy() ** 2))))
    metrics["region_day_rmse_c"] = {"n_blocks": len(block_rmse), "mean": float(block_rmse.mean()), "median": float(block_rmse.median()), "p90": float(block_rmse.quantile(0.9))}
    if "solar_elevation_deg" in heldout and heldout["solar_elevation_deg"].notna().any():
        heldout["daylight_group"] = np.where(heldout["solar_elevation_deg"].isna(), "unknown", np.where(heldout["solar_elevation_deg"].gt(0), "day", "night"))
        metrics["by_daylight"] = grouped_metrics(heldout, "daylight_group")
    if "cloud_cover_fraction" in heldout and heldout["cloud_cover_fraction"].notna().any():
        heldout["cloud_group"] = pd.cut(heldout["cloud_cover_fraction"], [-np.inf, 0.2, 0.8, np.inf], labels=["mostly_clear", "mixed", "mostly_cloudy"]).astype("string").fillna("unknown")
        metrics["by_cloud"] = grouped_metrics(heldout, "cloud_group")
        warnings.append("Cloud-group errors describe the available labels only; cloud context alone does not prove cloudy-sky surface-temperature labels are observed.")
    if "air_temperature_source" in heldout:
        metrics["by_air_temperature_source"] = grouped_metrics(heldout, "air_temperature_source")
    if "station_distance_km" in heldout:
        heldout["station_distance_group"] = pd.cut(heldout["station_distance_km"], [-np.inf, 25, 50, 100, np.inf], labels=["within_25km", "25_to_50km", "50_to_100km", "over_100km"]).astype("string").fillna("no_station")
        metrics["by_station_distance"] = grouped_metrics(heldout, "station_distance_group")
    output.mkdir(parents=True, exist_ok=True)
    prediction_columns = list(dict.fromkeys(list(REQUIRED_COLUMNS) + ["utc_day", "split", "time_period", "predicted_offset_c", "predicted_lst_c", "linear_lst_c", "lower_lst_c", "upper_lst_c", "error_c", "climate_unseen_in_training"] + [name for name in ("daylight_group", "cloud_group", "label_source", "label_quality", "source_scene_id", "station_id", "air_temperature_source", "station_distance_km", "station_age_minutes") if name in heldout]))
    heldout[prediction_columns].to_parquet(output / "heldout_predictions.parquet", index=False)
    joblib.dump({"schema_version": 1, "model": model, "linear_baseline": linear, "features": features, "target": metrics["target"], "interval_radius_c": radius, "config": asdict(config), "training_climate_classes": training_classes, "smoke_only": smoke_only}, output / "model.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    feature_text = ", ".join(f"`{name}`" for name in features)
    card = f"""# LST pilot model card

Status: **{metrics['status']}**. Created {metrics['created_utc']}.

The histogram gradient boosting model learns `lst_c - air_temperature_c` and adds the predicted difference to supplied air temperature. A ridge regression trained on the same rows/features and air temperature alone are the two baselines. Climate class is categorical. Coordinates, station identifiers, surface-temperature labels, thermal brightness temperatures and target-derived emissivity are excluded from predictors by an explicit allowlist.

## Inputs and fitted model

Features: {feature_text}.

Training rows actually fitted: {len(training):,} of {int(data['split'].eq('train').sum()):,}. The deterministic cap, if needed, samples only after partitioning training rows. Fixed maximum iterations: {config.max_iter}; maximum leaves: {config.max_leaf_nodes}; maximum depth: {config.max_depth}. Early stopping is disabled. No parameter search used heldout data.

Training climate classes: {', '.join(training_classes)}. Unknown climate fraction across inputs: {climate_unknown_fraction:.2%}. Model preprocessors, category vocabularies, the linear imputer and scaler are fitted on training rows only. Unseen climate classes become an unknown category/missing value at prediction time.

## Evaluation design

Heldout regions: {', '.join(config.heldout_regions)}. These regions supply no fitting or calibration observations. For other regions, whole UTC days through {config.train_end} are training, subsequent days through {config.calibration_end} are calibration, and later days are a temporal test when present. Heldout-region results retain a time-period label so geographic and future-date transfer can be distinguished.

`metrics.json` contains split/region/climate errors for this model and both baselines, interval coverage, and region/day block summaries. `heldout_predictions.parquet` contains actual heldout observations and predictions. No training or calibration rows appear in that file. Repeated inspection for tuning would turn these tests into development data; reserve new regions/dates for a final blind assessment.

## Uncertainty and limitations

The dedicated calibration period sets a shared interval radius of ±{radius:.3f} °C at nominal {config.interval_coverage:.0%} coverage. The interval is empirical and is not guaranteed to achieve that coverage on new dates, surfaces or regions. The output is not a validated global, hourly, all-weather product.

""" + "\n".join(f"- {warning}" for warning in warnings) + "\n\n## Saved artifacts\n\n`model.joblib` stores the pipelines, feature list, training climate classes, target definition, interval radius and configuration. Only load trusted joblib files. `metrics.json` and this card describe the exact run.\n"
    (output / "model_card.md").write_text(card, encoding="utf-8")
    return metrics


def predict_frame(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Apply a trusted saved bundle; callers retain responsibility for data provenance."""
    data = frame.copy()
    features = bundle["features"]
    select_features(data.columns, features)
    if CATEGORICAL_FEATURE not in data:
        raise ValueError("Prediction input requires climate_class.")
    data[CATEGORICAL_FEATURE] = data[CATEGORICAL_FEATURE].astype("string").fillna(UNKNOWN_CLIMATE).replace("", UNKNOWN_CLIMATE).astype(str)
    for name in features:
        if name != CATEGORICAL_FEATURE:
            data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
    if not np.isfinite(data["air_temperature_c"]).all():
        raise ValueError("Prediction input requires finite air_temperature_c.")
    offset = bundle["model"].predict(data[features])
    prediction = temperature_from_offset(data["air_temperature_c"], offset)
    return pd.DataFrame({"predicted_lst_c": prediction, "predicted_offset_c": offset, "lower_lst_c": prediction - bundle["interval_radius_c"], "upper_lst_c": prediction + bundle["interval_radius_c"], "climate_unseen_in_training": ~data[CATEGORICAL_FEATURE].isin(bundle["training_climate_classes"])}, index=data.index)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heldout-regions", required=True, nargs="+")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--calibration-end", required=True)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--interval-coverage", type=float, default=0.90)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-calibration-rows", type=int, default=30)
    parser.add_argument("--min-evaluation-rows", type=int, default=30)
    parser.add_argument("--allow-unknown-climate-smoke", action="store_true")
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    config = ModelConfig(
        heldout_regions=tuple(args.heldout_regions), train_end=args.train_end,
        calibration_end=args.calibration_end, features=tuple(args.features) if args.features else None,
        max_iter=args.max_iter, max_train_rows=args.max_train_rows,
        interval_coverage=args.interval_coverage, min_train_rows=args.min_train_rows,
        min_calibration_rows=args.min_calibration_rows, min_evaluation_rows=args.min_evaluation_rows,
        allow_unknown_climate_smoke=args.allow_unknown_climate_smoke,
    )
    metrics = train_and_evaluate(frame, args.output_dir, config)
    print(json.dumps({"output_dir": str(args.output_dir), "status": metrics["status"], "n_fitted_rows": metrics["n_fitted_rows"], "by_split": metrics["by_split"]}, indent=2))


if __name__ == "__main__":
    main()
