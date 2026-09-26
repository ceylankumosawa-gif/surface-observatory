"""Daytime reported-air scenarios with explicit weather provenance.

Requested solar geometry always belongs to the requested timestamp. When the
complete ERA5 context is unavailable, a single same-season/hour day in 2023 is
used and identified as reference weather, never as observed weather or a
climatological mean. The supplied air value remains the user's constraint.

The former coarse ERA5 night baseline is withdrawn: it cannot represent learned
fine-scale surface behaviour. Legacy numerical helpers remain for audit only.
"""
from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import radiation
from .weather import enrich_weather, WeatherResponseError

SCENARIO_START = "1900-01-01"
SCENARIO_END = "2100-12-31"
REFERENCE_YEAR = 2023
NIGHT_WITHDRAWAL_REASON = (
    "Nighttime surface prediction is unavailable. The coarse weather-grid baseline "
    "has been withdrawn because it does not learn how different surfaces cool. "
    "Choose a fully sunlit daytime hour (the time selector uses UTC)."
)


class UnsupportedNighttime(ValueError):
    """A known scientific support boundary, safe to explain to API users."""


def require_daylight_area(target, polygon):
    """Bound whole-area daylight without source reads or a large feature table.

    Elevation can change by at most angular distance on the sphere. The sum
    of latitude and longitude half-spans bounds that distance anywhere in
    this small non-dateline polygon's bounding box. Requests near the horizon
    are conservatively unsupported rather than switching to an untrained model.
    """
    import pvlib
    from shapely.geometry import shape
    geometry = shape(polygon) if isinstance(polygon, dict) else polygon
    west, south, east, north = geometry.bounds
    latitude, longitude = (north+south)/2, (east+west)/2
    margin = (north-south+east-west)/2 + .001
    stamp = pd.Timestamp(target).tz_convert("UTC")
    elevation = float(pvlib.solarposition.spa_python(pd.DatetimeIndex([stamp]),
        latitude=latitude, longitude=longitude).elevation.iloc[0])
    if not np.isfinite(elevation) or elevation <= margin:
        raise UnsupportedNighttime(NIGHT_WITHDRAWAL_REASON)


def require_daytime_rows(frame):
    if not np.isfinite(frame.solar_elevation_deg).all() or frame.solar_elevation_deg.le(0).any():
        raise UnsupportedNighttime(NIGHT_WITHDRAWAL_REASON)


class ContextUnavailable(ValueError):
    """The requested ERA5 hour lacks complete context, not a unit-contract error."""


def reference_timestamp(target):
    target = pd.Timestamp(target).tz_convert("UTC")
    day = min(target.day, calendar.monthrange(REFERENCE_YEAR, target.month)[1])
    return target.replace(year=REFERENCE_YEAR, day=day)


def select_surface_scene(scenes, target, scene_id=None):
    """Select a fixed land-property snapshot, with no claim it is contemporaneous.

    Prefer the closest season. Within the same seasonal distance choose a
    preceding snapshot if one exists, then the closest year. No target LST is
    inspected, and this choice is never used to select evaluation examples.
    """
    if scene_id is not None:
        selected = next((s for s in scenes if s.get("scene_id", s.get("id")) == scene_id), None)
        if selected is None:
            raise ValueError("Choose a surface snapshot listed for this pilot.")
        return selected
    if not scenes:
        raise ValueError("This pilot has no surface-property snapshot.")
    target = pd.Timestamp(target).tz_convert("UTC")
    def rank(scene):
        stamp = pd.Timestamp(scene["datetime_utc"]).tz_convert("UTC")
        a = reference_timestamp(stamp).dayofyear
        b = reference_timestamp(target).dayofyear
        distance = abs(a-b)
        return min(distance, 365-distance), stamp > target, abs((target-stamp).total_seconds())
    return min(scenes, key=rank)


def sample_skin_temperature(frame, cache):
    """Read one validated global hour (~4 MiB decoded), then sample coarse cells."""
    import fsspec
    times = pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("h")
    if times.nunique() != 1:
        raise ValueError("A night baseline requires one requested weather hour.")
    target = times.iloc[0]
    root = Path(cache) / "arco-era5-v3"
    fs = fsspec.filesystem("gcs", token="anon", timeout=30)
    objects = []
    metadata = json.loads(radiation._cached_object(fs, ".zmetadata", root, radiation.MAX_METADATA_BYTES, objects))["metadata"]
    radiation._validate_metadata(metadata)
    name = "skin_temperature"
    desc, attrs = metadata[name+"/.zarray"], metadata[name+"/.zattrs"]
    if desc["chunks"] != [1, 721, 1440] or desc["shape"][1:] != [721, 1440] or attrs.get("units") != "K":
        raise WeatherResponseError("ERA5 skin-temperature grid or Kelvin units changed.")
    if attrs.get("_ARRAY_DIMENSIONS") != ["time", "latitude", "longitude"] or np.dtype(desc["dtype"]) != np.dtype("float32"):
        raise WeatherResponseError("ERA5 skin-temperature array contract changed.")
    start = pd.Timestamp(metadata[".zattrs"]["valid_time_start"], tz="UTC")
    stop = pd.Timestamp(metadata[".zattrs"].get("valid_time_stop_era5t", metadata[".zattrs"]["valid_time_stop"]), tz="UTC")+pd.Timedelta(hours=23)
    if target < start or target > stop:
        raise ContextUnavailable("Skin temperature is outside the published ERA5/ERA5T coverage.")
    # Check coordinates and the actual stored hour index independently of the
    # previously read radiation variables, including in stand-alone tests/jobs.
    for coord, expected in [("latitude", np.linspace(90, -90, 721)), ("longitude", np.arange(1440)/4)]:
        values = radiation._decode(radiation._cached_object(fs, coord+"/0", root, 65536, objects), metadata[coord+"/.zarray"])
        if not np.allclose(values, expected):
            raise WeatherResponseError("ERA5 night-baseline coordinates changed.")
    index = int((target-radiation.EPOCH)/pd.Timedelta(hours=1))
    td = metadata["time/.zarray"]
    n = td["chunks"][0]
    tv = radiation._decode(radiation._cached_object(fs, f"time/{index//n}", root, radiation.MAX_METADATA_BYTES, objects), td)
    if tv[index % n] != index:
        raise WeatherResponseError("ERA5 night-baseline time index differs from the request.")
    key = f"{name}/{index}.0.0"
    values = radiation._decode(radiation._cached_object(fs, key, root, radiation.MAX_OBJECT_BYTES, objects), desc)
    rows, cols = radiation._grid_indices(frame.latitude, frame.longitude)
    selected = values[0, rows, cols].astype(float)
    if not np.isfinite(selected).all() or (selected <= 0).any():
        raise ContextUnavailable("ERA5 skin temperature is missing for part of the selected area.")
    return selected-273.15, {"dataset": "ERA5 modelled skin temperature", "status": "used_for_night_baseline",
        "url": radiation.SOURCE_URL, "documentation": "https://codes.ecmwf.int/grib/param-db/235",
        "time_utc": target.isoformat(), "native_grid_degrees": .25, "units": "Kelvin converted to degrees Celsius",
        "method": "Nearest coarse cell at the last completed hour; this is modelled surface temperature, not satellite or ground truth.",
        "objects": objects}


def _weather_at(frame, target, cache, need_skin):
    work = frame.copy()
    work["datetime_utc"] = target
    work, weather_audit = enrich_weather(work, cache, max_requests=64)
    radiation_audit = None
    if frame.solar_elevation_deg.gt(0).any():
        work = radiation.add_radiation(work, Path(cache)/"radiation", max_hours=1, allow_provisional=True)
        radiation_audit = work.attrs["radiation_context"]
        if any(not np.isfinite(work[field]).all() for field in radiation.VARIABLES.values()):
            statuses = {field: sorted(work[field+"_status"].unique().tolist()) for field in radiation.VARIABLES.values()}
            raise ContextUnavailable("Required hourly radiation/snow context is unavailable: "+json.dumps(statuses))
    skin_audit = None
    if need_skin:
        work["era5_skin_temperature_c"], skin_audit = sample_skin_temperature(work, Path(cache)/"radiation")
    return work, weather_audit, radiation_audit, skin_audit


def prepare_scenario(frame, grid, target, cache, air_override, now=None):
    """Resolve source weather and apply the provided air temperature at the centre."""
    if air_override is None or isinstance(air_override, bool) or not np.isfinite(float(air_override)) or not -90 <= float(air_override) <= 65:
        raise ValueError("Provide an air temperature between -90 and 65 °C for this date and time.")
    require_daytime_rows(frame)
    target = pd.Timestamp(target).tz_convert("UTC")
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    need_skin = bool((frame.solar_elevation_deg <= 0).any())
    reference = reference_timestamp(target)
    reasons = []
    work = None
    if pd.Timestamp("1940-01-05", tz="UTC") <= target <= now-pd.Timedelta(days=6):
        try:
            work, weather_audit, radiation_audit, skin_audit = _weather_at(frame, target, cache, need_skin)
            source_time, basis = target, "actual_reanalysis"
        except WeatherResponseError:
            # A schema/unit change is not evidence of merely missing weather.
            raise
        except Exception as exc:
            reasons.append(f"Requested-hour context unavailable ({type(exc).__name__}); used the documented reference weather instead.")
    else:
        reasons.append("The request is outside the complete historical-weather window; no actual weather is asserted for this timestamp.")
    if work is None:
        work, weather_audit, radiation_audit, skin_audit = _weather_at(frame, reference, cache, need_skin)
        source_time, basis = reference, "seasonal_reference"
    lon, lat = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform(grid.polygon.centroid.x, grid.polygon.centroid.y)
    anchor = pd.DataFrame({"datetime_utc": [source_time], "longitude": [lon], "latitude": [lat]})
    anchor_weather, anchor_audit = enrich_weather(anchor, cache, max_requests=1)
    correction = float(air_override)-float(anchor_weather.background_air_temperature_c.iloc[0])
    work["air_temperature_c"] = work.background_air_temperature_c+correction
    work["air_temperature_source"] = "reported_air_at_centroid_plus_ERA5_spatial_background"
    work["station_id"] = ""
    work["station_air_correction_c"] = correction
    sampled_hour = source_time.floor("h")
    work["weather_reference_datetime_utc"] = sampled_hour
    work["datetime_utc"] = target
    context = {"weather_basis": basis, "weather_reference_datetime_utc": sampled_hour.isoformat(),
        "requested_datetime_utc": target.isoformat(), "reported_air_temperature_c": float(air_override),
        "reference_is_climatology": False, "reference_year": REFERENCE_YEAR if basis == "seasonal_reference" else None,
        "anchor_longitude": lon, "anchor_latitude": lat, "warnings": reasons,
        "method": "Reported air temperature minus ERA5 air at the area centre, plus ERA5 air at each coarse grid cell. All remaining weather and its history retain the identified weather-source time.",
        "surface_interpretation": "Fixed land-property snapshot under requested conditions; not a reconstruction of historical or future land cover.",
        "anchor_weather_audit": anchor_audit}
    return work, context, weather_audit, radiation_audit, skin_audit


def night_baseline(frame):
    """Air-adjusted coarse energy-balance baseline, without daytime ML intervals."""
    fields = ["air_temperature_c", "background_air_temperature_c", "era5_skin_temperature_c"]
    if not np.isfinite(frame[fields].to_numpy(dtype=float)).all():
        raise ValueError("The night baseline requires finite reported air, background air and ERA5 skin temperature.")
    return frame.air_temperature_c + frame.era5_skin_temperature_c - frame.background_air_temperature_c


def predict_with_night_baseline(frame, bundle, day_predict):
    """Legacy entry point; the withdrawn baseline can no longer run."""
    require_daytime_rows(frame)
    daylight = frame.solar_elevation_deg.gt(0)
    predicted = pd.DataFrame(index=frame.index)
    for field in ["predicted_lst_c", "predicted_offset_c", "lower_lst_c", "upper_lst_c"]:
        predicted[field] = np.nan
    predicted["climate_unseen_in_training"] = False
    if daylight.any():
        part = day_predict(frame.loc[daylight], bundle)
        for field in predicted.columns:
            predicted.loc[daylight, field] = part[field].to_numpy()
    if (~daylight).any():
        values = night_baseline(frame.loc[~daylight])
        predicted.loc[~daylight, "predicted_lst_c"] = values
        predicted.loc[~daylight, "predicted_offset_c"] = values-frame.loc[~daylight, "air_temperature_c"]
    return predicted
