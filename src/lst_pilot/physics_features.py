"""Three research proxies from existing inputs; no acquisition or predictions.

The six-hour proxy uses a single optical snapshot available at the target time,
held fixed over six completed radiation intervals. It is neither a reconstruction
of historical albedo nor a measurement of stored heat. Longwave is a blackbody
reference at reported air temperature, not actual surface net radiation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

VERSION = "physics-proxies-v1"
SIGMA_W_M2_K4 = 5.670374419e-8
HISTORY_HOURS = 6
HISTORY_SECONDS = HISTORY_HOURS * 3600
FEATURES = (
    "physics_absorbed_shortwave_w_m2",
    "physics_longwave_cooling_potential_w_m2",
    "physics_absorbed_shortwave_mean6h_w_m2",
)
RAW_HISTORY_FEATURE = "physics_raw_shortwave_mean6h_w_m2"
INPUT_UNITS = {
    "albedo_proxy": "1",
    "air_temperature_c": "°C",
    "shortwave_down_w_m2": "W/m²",
    "era5_longwave_down_w_m2": "W/m²",
    "memory_shortwave_energy_6h_j_m2": "J/m²",
}
FEATURE_UNITS = {name: "W/m²" for name in (*FEATURES, RAW_HISTORY_FEATURE)}
MAX_CACHE_FILE_BYTES = 4 * 1024**2
MAX_CACHE_BYTES = 128 * 1024**2
MAX_CACHE_FILES = 512
WEATHER_SOURCE = "https://archive-api.open-meteo.com/v1/archive"


def _unit(value):
    return str(value).strip().replace(" ", "").replace("²", "2").lower()


def _numbers(frame, name):
    values = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce").astype(float)


def _times(frame, name):
    """Missing times remain missing; naive or malformed times are schema errors."""
    values = frame[name] if name in frame else [pd.NaT] * len(frame)
    result = []
    for value in values:
        if pd.isna(value):
            result.append(pd.NaT)
            continue
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError(f"{name} requires explicitly timezone-aware timestamps.")
        result.append(stamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(result, tz="UTC"), index=frame.index)


def _finite(values):
    return pd.Series(np.isfinite(values.to_numpy(float)), index=values.index)


class _CachedShortwave:
    """Bounded exact-key reader; it cannot make network requests or write files."""

    def __init__(self, root):
        self.root = (Path(root) / "weather").resolve() if root is not None else None
        self.records = {}
        self.bytes_read = 0
        self.attempts = 0

    def load(self, key):
        key = str(key)
        if key in self.records:
            return self.records[key]
        record = {"key": key, "status": "cache_not_configured"}
        self.records[key] = record
        if self.root is None:
            return record
        if not re.fullmatch(r"[0-9a-f]{24}", key):
            record["status"] = "invalid_cache_key"
            return record
        if self.attempts >= MAX_CACHE_FILES:
            record["status"] = "cache_file_budget_exceeded"
            return record
        self.attempts += 1
        path = self.root / f"era5_{key}.json"
        if path.resolve().parent != self.root:
            record["status"] = "cache_path_outside_root"
            return record
        try:
            size = path.stat().st_size
            if size > MAX_CACHE_FILE_BYTES or self.bytes_read + size > MAX_CACHE_BYTES:
                record["status"] = "cache_byte_budget_exceeded"
                return record
            # Read one extra byte to detect a concurrent size increase.
            with path.open("rb") as stream:
                raw = stream.read(size + 1)
            self.bytes_read += len(raw)
            if len(raw) != size:
                raise ValueError("Cache file changed while being read.")
            record.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            payload = json.loads(raw)
            request, response = payload["request"], payload["response"]
            actual_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
            if actual_key != key:
                raise ValueError("Request hash does not match the referenced cache key.")
            if (payload.get("source") != WEATHER_SOURCE or payload.get("model") != "ERA5"
                    or request.get("models") != "era5" or request.get("timezone") != "UTC"
                    or request.get("elevation") != "nan" or request.get("cell_selection") != "nearest"
                    or response.get("utc_offset_seconds") != 0):
                raise ValueError("Cache source/model/timezone/grid-selection contract differs.")
            hourly = response["hourly"]
            if _unit(response["hourly_units"]["shortwave_radiation"]) != _unit("W/m²"):
                raise ValueError("Cached shortwave does not have W/m² units.")
            # Open-Meteo's UTC response strings are timezone-naive by contract;
            # explicit request+response timezone validation above authorizes UTC.
            times = pd.to_datetime(hourly["time"], utc=True, errors="raise")
            if times.isna().any() or times.duplicated().any() or not times.equals(times.floor("h")):
                raise ValueError("Cached timestamps are not unique complete UTC hours.")
            values = pd.to_numeric(pd.Series(hourly["shortwave_radiation"]), errors="raise").to_numpy(float)
            if len(values) != len(times):
                raise ValueError("Cached times and shortwave have different lengths.")
            grid = tuple(float(response[name]) for name in ("latitude", "longitude"))
            if not np.isfinite(grid).all():
                raise ValueError("Cached source grid is not finite.")
            record.update(status="validated", table=pd.Series(values, index=times), grid=grid)
        except FileNotFoundError:
            record["status"] = "cache_missing"
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            record.update(status="cache_invalid", error=str(exc))
        return record

    def mean6(self, row, cutoff):
        record = self.load(row.get("weather_cache_key", ""))
        if record["status"] != "validated":
            return np.nan, record["status"], "", 0
        try:
            supplied_grid = tuple(float(row.get(name, np.nan)) for name in (
                "weather_grid_latitude", "weather_grid_longitude"))
            if not np.isfinite(supplied_grid).all() or not np.allclose(
                    supplied_grid, record["grid"], rtol=0, atol=1e-8):
                return np.nan, "cache_grid_mismatch", record["sha256"], 0
            # Validate that the same cache also generated the current shortwave.
            current = float(row.get("shortwave_down_w_m2", np.nan))
            cached_current = record["table"].get(cutoff, np.nan)
            if not np.isfinite(current) or not np.isclose(current, cached_current, rtol=0, atol=1e-8):
                return np.nan, "cache_current_shortwave_mismatch", record["sha256"], 0
            wanted = pd.date_range(cutoff - pd.Timedelta(hours=5), cutoff, freq="h")
            found = record["table"].reindex(wanted).to_numpy(float)
            good = np.isfinite(found) & (found >= 0)
            count = int(good.sum())
            if count != 6:
                return np.nan, "cache_incomplete_six_hours", record["sha256"], count
            return float(found.mean()), "existing_weather_cache", record["sha256"], 6
        except (TypeError, ValueError, OverflowError):
            return np.nan, "cache_row_contract_invalid", record["sha256"], 0


def add_physics_features(frame: pd.DataFrame, *, units=INPUT_UNITS,
                         cache_root=None) -> tuple[pd.DataFrame, dict]:
    """Append proxies and audit flags without filtering or modifying any row.

    Both radiation inputs must end at floor(target UTC hour). Existing six-hour
    energy additionally needs its exact endpoint, complete status, six valid
    hours and coverage=1. A missing/unproven energy may be independently rebuilt
    from the exact referenced existing weather-cache record, never downloaded.

    Optical causality uses the latest of all present actual optical timestamps
    (optical_latest_source_utc and the older optical_source_datetime_utc field).
    Request-window endpoints are not treated as observation timestamps. Missing
    or invalid inputs disable the relevant proxy; physics_complete requires all
    three. The caller must leave the base ML prediction unchanged when false.

    Only mathematical domain checks are imposed: finite numbers, albedo in
    [0,1], nonnegative solar forcing/energy, positive downward longwave and
    positive Kelvin air temperature. No empirical hot/cold value trimming,
    physical-value clipping or seasonal/phase temperature ordering is applied.
    """
    if "datetime_utc" not in frame:
        raise ValueError("Physics proxies require datetime_utc.")
    if any(str(name).startswith("physics_") for name in frame):
        raise ValueError("Refusing to overwrite existing physics fields.")
    for name, expected in INPUT_UNITS.items():
        if not isinstance(units, dict) or _unit(units.get(name)) != _unit(expected):
            raise ValueError(f"Physics input unit mismatch for {name}: expected {expected}.")
    source = frame.reset_index(drop=True)
    target = _times(source, "datetime_utc")
    cutoff = target.dt.floor("h")
    optical_times = [_times(source, name) for name in (
        "optical_latest_source_utc", "optical_source_datetime_utc")]
    optical = pd.concat(optical_times, axis=1).max(axis=1)
    optical_ok = target.notna() & optical.notna() & optical.le(target)
    weather_time = _times(source, "weather_datetime_utc")
    radiation_time = _times(source, "radiation_era5_time_utc")
    weather_ok = target.notna() & weather_time.eq(cutoff)
    radiation_ok = target.notna() & radiation_time.eq(cutoff)
    albedo = _numbers(source, "albedo_proxy")
    albedo_ok = _finite(albedo) & albedo.between(0, 1)
    solar = _numbers(source, "shortwave_down_w_m2")
    solar_ok = _finite(solar) & solar.ge(0)
    longwave = _numbers(source, "era5_longwave_down_w_m2")
    longwave_ok = _finite(longwave) & longwave.gt(0)
    air_k = _numbers(source, "air_temperature_c") + 273.15
    air_ok = _finite(air_k) & air_k.gt(0)
    energy = _numbers(source, "memory_shortwave_energy_6h_j_m2")
    history_end = _times(source, "memory_history_end_utc")
    energy_ok = _finite(energy) & energy.ge(0)
    status = source.get("memory_shortwave_6h_status", pd.Series("", index=source.index))
    memory_ok = (energy_ok & target.notna() & history_end.eq(cutoff)
                 & status.eq("complete").fillna(False)
                 & _numbers(source, "memory_shortwave_6h_valid_hours").eq(6)
                 & _numbers(source, "memory_shortwave_6h_coverage_fraction").eq(1))
    history_mean = (energy / HISTORY_SECONDS).where(memory_ok)
    history_source = pd.Series(np.where(memory_ok, "existing_six_hour_energy", "unavailable"))
    history_failure = pd.Series(np.where(memory_ok, "", "missing_or_invalid_memory_proof"))
    history_sha = pd.Series("", index=source.index)
    history_count = pd.Series(np.where(memory_ok, 6, 0))
    reader = _CachedShortwave(cache_root)
    if cache_root is not None:
        for position in source.index[~memory_ok & target.notna() & weather_ok]:
            value, origin, checksum, count = reader.mean6(source.iloc[position], cutoff.iloc[position])
            history_count.iloc[position] = count
            history_sha.iloc[position] = checksum
            if np.isfinite(value):
                history_mean.iloc[position] = value
                history_source.iloc[position] = origin
                history_failure.iloc[position] = ""
            else:
                history_failure.iloc[position] = origin
    history_ok = _finite(history_mean) & history_mean.ge(0)
    with np.errstate(over="ignore", invalid="ignore"):
        absorbed = ((1 - albedo) * solar).where(albedo_ok & optical_ok & solar_ok & weather_ok)
        cooling = (longwave - SIGMA_W_M2_K4 * air_k**4).where(longwave_ok & air_ok & radiation_ok)
        recent = ((1 - albedo) * history_mean).where(albedo_ok & optical_ok & history_ok)
    additions = pd.DataFrame({
        FEATURES[0]: absorbed.where(_finite(absorbed)),
        FEATURES[1]: cooling.where(_finite(cooling)),
        FEATURES[2]: recent.where(_finite(recent)),
        RAW_HISTORY_FEATURE: history_mean.where(history_ok),
        "physics_optical_source_utc": optical,
        "physics_optical_causal": optical_ok,
        "physics_albedo_valid": albedo_ok,
        "physics_shortwave_valid": solar_ok,
        "physics_longwave_valid": longwave_ok,
        "physics_air_valid": air_ok,
        "physics_shortwave_time_valid": weather_ok,
        "physics_longwave_time_valid": radiation_ok,
        "physics_history_start_utc": cutoff - pd.Timedelta(hours=6),
        "physics_history_end_utc": cutoff,
        "physics_history_valid_hours": history_count,
        "physics_history_source": history_source,
        "physics_history_unavailable_reason": history_failure,
        "physics_history_cache_sha256": history_sha,
        "physics_static_albedo_over_history": True,
    })
    for name in FEATURES:
        additions[name + "_available"] = additions[name].notna()
    additions["physics_complete"] = additions[list(FEATURES)].notna().all(axis=1)
    reasons = pd.Series("", index=source.index)
    for name, mask in (("missing_target_time", target.notna()),
                       ("missing_or_future_optical_source", optical_ok),
                       ("invalid_albedo", albedo_ok), ("invalid_shortwave", solar_ok),
                       ("invalid_longwave", longwave_ok), ("invalid_air_temperature", air_ok),
                       ("shortwave_hour_mismatch", weather_ok),
                       ("longwave_hour_mismatch", radiation_ok), ("incomplete_six_hour_history", history_ok)):
        missing = ~mask
        reasons.loc[missing] = reasons.loc[missing] + name + ";"
    reasons = reasons.str.rstrip(";")
    reasons.loc[~additions.physics_complete & reasons.eq("")] = "nonfinite_derived_proxy"
    additions["physics_unavailable_reason"] = reasons
    result = frame.copy()
    for name in additions:
        result[name] = additions[name].array
    audit = {
        "version": VERSION, "research_only": True, "input_rows": len(frame),
        "output_rows": len(result), "complete_rows": int(additions.physics_complete.sum()),
        "features": list(FEATURES), "raw_control_feature": RAW_HISTORY_FEATURE,
        "feature_units": FEATURE_UNITS, "input_units": dict(units),
        "sigma_w_m2_k4": SIGMA_W_M2_K4,
        "formula": {FEATURES[0]: "(1-albedo_proxy)*shortwave_down_w_m2",
                    FEATURES[1]: "era5_longwave_down_w_m2-sigma*(air_temperature_c+273.15)^4",
                    FEATURES[2]: "(1-albedo_proxy)*completed_six_hour_mean_shortwave_w_m2",
                    RAW_HISTORY_FEATURE: "six_hour_energy_j_m2/(6*3600), or six exact saved hourly means"},
        "history_rule": "Six complete intervals (cutoff-6h,cutoff], cutoff=floor(target UTC hour); no gaps, interpolation or future intervals.",
        "optical_rule": "All present actual optical-source timestamps must be <= target. The same available snapshot is held fixed over history, not reconstructed at past times.",
        "valid_time_not_publication_latency": True,
        "no_flux_closure_or_stored_heat_claim": True,
        "base_predictions_or_input_rows_changed": False,
        "downloaded_bytes": 0, "network_access": False,
        "cache_read_bytes": reader.bytes_read,
        "cache_read_limits": {"files": MAX_CACHE_FILES, "total_bytes": MAX_CACHE_BYTES, "file_bytes": MAX_CACHE_FILE_BYTES},
        "cache_records": [{k: v for k, v in record.items() if k not in ("table", "grid")}
                          for record in reader.records.values()],
        "history_source_counts": history_source.value_counts(dropna=False).to_dict(),
        "unavailable_reason_counts": additions.physics_unavailable_reason.value_counts().to_dict(),
    }
    return result, audit
