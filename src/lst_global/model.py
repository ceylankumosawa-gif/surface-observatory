"""Strict inference for the frozen, 44,172-row research F reference.

This module neither prepares features nor establishes worldwide accuracy. Its
only prediction is A + f(X), using the untouched fitted pipeline. In particular,
the website's optical snapshot workflow is not an equivalent feature builder.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


ADAPTER_VERSION = "frozen-f-global-v1"
MODEL_ID = "F_complete_native_44172"
MODEL_PATH = Path("/opt/lst-pilot/runs/weather_alignment_20260911_v1/stage_a_model_v1/full/F.joblib")
MODEL_SHA256 = "c955f3a69e393eef29dd95e291d751b73e845a43e7ab2067289c6f1cc394f447"
MODEL_BYTES = 148_557
SKLEARN_VERSION = "1.9.0"
SUPPORT_STATUS = "global_extrapolation"
# Intentionally frozen here, rather than imported from a mutable research adapter.
FEATURES = (
    "air_temperature_c", "ndvi", "ndbi", "ndwi", "albedo_proxy", "elevation", "slope",
    "terrain_relief_300m", "aspect_sin", "aspect_cos", "water_fraction", "solar_elevation_deg",
    "solar_azimuth_sin", "solar_azimuth_cos", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
    "relative_humidity_pct", "dewpoint_c", "wind_speed_m_s", "wind_direction_sin", "wind_direction_cos",
    "surface_pressure_hpa", "cloud_cover_fraction", "shortwave_down_w_m2", "direct_shortwave_w_m2",
    "diffuse_shortwave_w_m2", "era5_longwave_down_w_m2", "era5_snow_water_equivalent_m",
    "precipitation_mm_h", "rain_mm_24h", "rain_mm_72h", "soil_moisture_m3_m3", "air_temperature_lag1_c",
    "air_temperature_lag3_c", "air_temperature_lag24_c", "shortwave_down_lag1_w_m2",
    "shortwave_down_mean3_w_m2", "climate_class",
)
NUMERIC_FEATURES = FEATURES[:-1]
CLIMATE_CLASSES = ("Af", "Aw", "BSk", "BWh", "Cfb", "Csa", "Csb", "Dfa", "Dfb", "Dfc", "Dwb", "Dwc", "ET")
FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(FEATURES, separators=(",", ":")).encode()).hexdigest()


def validate_features(frame: pd.DataFrame) -> None:
    """No reordering, implicit numeric conversion, imputation or unknown climate."""
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != FEATURES:
        raise ValueError("Provide only the exact 40 feature columns in frozen FEATURES order.")
    for name in NUMERIC_FEATURES:
        series = frame[name]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
            raise ValueError(f"Feature {name} must be real numeric values, not strings or booleans.")
        if series.isna().any() or not np.isfinite(series.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Feature {name} must be complete and finite; no prediction was emitted.")
    if frame.climate_class.isna().any() or not frame.climate_class.isin(CLIMATE_CLASSES).all():
        raise ValueError("climate_class must be one of the 13 frozen trained climate classes.")


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Nonempty source_provenance is required for every prediction batch.")
    # Copy, reject non-JSON / nonfinite metadata, and avoid caller mutation.
    return json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class PredictionResult:
    values: pd.DataFrame
    provenance: dict[str, Any]


class FrozenF:
    """Load only the pinned artifact; an alternate path may hold identical bytes."""

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH, *, backend: str = "sklearn") -> "FrozenF":
        if backend not in ("sklearn", "native"):
            raise ValueError("Choose the sklearn or verified native F backend.")
        path = Path(path)
        # Deserialize exactly the bytes that were hashed, without a path race.
        with path.open("rb") as stream:
            payload = stream.read(MODEL_BYTES + 1)
        if len(payload) != MODEL_BYTES or hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
            raise ValueError("Frozen F model byte count or trusted SHA-256 does not match.")
        if sklearn.__version__ != SKLEARN_VERSION:
            raise ValueError(f"Frozen F requires scikit-learn {SKLEARN_VERSION}; found {sklearn.__version__}.")
        estimator = joblib.load(io.BytesIO(payload))
        if not isinstance(estimator, Pipeline) or tuple(estimator.named_steps) != ("features", "regressor"):
            raise ValueError("Frozen F artifact is not the expected raw offset Pipeline.")
        if tuple(estimator.feature_names_in_) != FEATURES:
            raise ValueError("Frozen F fitted feature order differs from its pinned schema.")
        preprocessor = estimator.named_steps["features"]
        if tuple(preprocessor.transformers_[0][2]) != NUMERIC_FEATURES:
            raise ValueError("Frozen F numeric transformation order differs.")
        if tuple(preprocessor.named_transformers_["climate"].categories_[0]) != CLIMATE_CLASSES:
            raise ValueError("Frozen F fitted climate vocabulary differs.")
        if not isinstance(estimator.named_steps["regressor"], HistGradientBoostingRegressor):
            raise ValueError("Frozen F regressor differs.")
        result = cls()
        result._estimator = estimator
        result._model_path = str(path.resolve())
        result._native = None
        if backend == "native":
            from .compiled import NativeF
            result._native = NativeF(estimator, model_sha256=MODEL_SHA256)
        return result

    def predict(
        self, features: pd.DataFrame, *, source_provenance: Mapping[str, Any],
        chunk_size: int = 65_536,
    ) -> PredictionResult:
        """Preserve row positions/index; reject the whole batch if any row is invalid.

        Chunking bounds temporary prediction memory. For a worldwide job the
        caller must also stream feature tiles, rather than allocate a global
        DataFrame. Source provenance must identify those prepared inputs.
        """
        if not hasattr(self, "_estimator"):
            raise ValueError("Use FrozenF.load() to verify and load the trusted artifact.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 1_048_576:
            raise ValueError("chunk_size must be an integer from 1 to 1048576.")
        source = _json_copy(source_provenance)
        validate_features(features)
        n = len(features)
        offset = np.empty(n, dtype=np.float64)
        input_digest = hashlib.sha256()
        for start in range(0, n, chunk_size):
            chunk = features.iloc[start:start + chunk_size]
            engine = self._native if getattr(self, '_native', None) is not None else self._estimator
            predicted = np.asarray(engine.predict(chunk), dtype=np.float64)
            if predicted.shape != (len(chunk),) or not np.isfinite(predicted).all():
                raise ValueError("Frozen F returned invalid offsets; no prediction was emitted.")
            offset[start:start + len(chunk)] = predicted
            # Same digest regardless of chunk boundaries; includes index identity.
            hashed_rows = pd.util.hash_pandas_object(chunk, index=True, categorize=False)
            input_digest.update(hashed_rows.to_numpy(dtype="<u8").tobytes())
        with np.errstate(over="ignore", invalid="ignore"):
            lst = features.air_temperature_c.to_numpy(dtype=np.float64) + offset
        if not np.isfinite(lst).all():
            raise ValueError("Air plus offset is nonfinite; no prediction was emitted.")
        values = pd.DataFrame({"predicted_lst_c": lst, "surface_air_offset_c": offset,
                               "support_status": SUPPORT_STATUS}, index=features.index.copy())
        provenance = {
            "adapter_version": ADAPTER_VERSION, "model_id": MODEL_ID,
            "model_path": self._model_path, "model_sha256": MODEL_SHA256,
            "feature_order": list(FEATURES), "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "input_rows": n, "input_feature_and_index_sha256": input_digest.hexdigest(),
            "input_hash_definition": "SHA256 of ordered pandas row hashes including index; little-endian uint64",
            "prediction_definition": "air_temperature_c + frozen_F_offset_c",
            "units": "degC", "support_status": SUPPORT_STATUS,
            "uncertainty_available": False, "global_accuracy_validated": False,
            "training_rows": 44_172, "training_years": [2021, 2022],
            "night_training_pilots": ["greater_london", "sioux_falls"],
            "air_basis": "Unchanged supplied model air input; station availability and adjustment belong in source provenance.",
            "feature_preparation_verified_by_adapter": False,
            "source": source, "chunk_size": chunk_size,
            "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__, "joblib": joblib.__version__},
            "inference_backend": (self._native.provenance if getattr(self, '_native', None) is not None
                                  else {"name": "sklearn", "model_changed": False}),
        }
        return PredictionResult(values, provenance)
