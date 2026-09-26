"""Bounded, cached ERA5 weather retrieval for a retrospective research pilot.

Open-Meteo is a delivery service; the explicitly selected model is ERA5.
Its free endpoint is for noncommercial use. No paid API is configured here.
Weather lags are ERA5 background-air history, not station-temperature history.
Nearest grid selection can return coastal/ocean cells. Grid distance and missing
soil context are recorded; neither is a definitive land/sea classification.
https://open-meteo.com/en/docs/historical-weather-api
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = {
    "temperature_2m": "background_air_temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "dew_point_2m": "dewpoint_c",
    "wind_speed_10m": "wind_speed_m_s",
    "wind_direction_10m": "wind_direction_deg",
    "surface_pressure": "surface_pressure_hpa",
    "cloud_cover": "cloud_cover_pct",
    "shortwave_radiation": "shortwave_down_w_m2",
    "direct_radiation": "direct_shortwave_w_m2",
    "diffuse_radiation": "diffuse_shortwave_w_m2",
    "precipitation": "precipitation_mm_h",
    "rain": "rain_mm_h",
    "soil_moisture_0_to_7cm": "soil_moisture_m3_m3",
}
EXPECTED_UNITS = {
    "temperature_2m": "°C", "relative_humidity_2m": "%", "dew_point_2m": "°C",
    "wind_speed_10m": "m/s", "wind_direction_10m": "°", "surface_pressure": "hPa",
    "cloud_cover": "%", "shortwave_radiation": "W/m²", "direct_radiation": "W/m²",
    "diffuse_radiation": "W/m²", "precipitation": "mm", "rain": "mm",
    "soil_moisture_0_to_7cm": "m³/m³",
}


class WeatherResponseError(ValueError):
    """A response contract failure that assembly must not turn into a fallback."""


def _normalise_unit(unit):
    return str(unit).strip().replace(" ", "").replace("²", "2").replace("³", "3").lower()


def validate_weather_response(data):
    """Reject wrong units/missing temperature before a response enters the cache."""
    if data.get("error"):
        raise WeatherResponseError(data.get("reason", "Weather API returned an error"))
    if data.get("utc_offset_seconds", 0) != 0:
        raise WeatherResponseError("Weather response was not UTC")
    hourly, units = data.get("hourly", {}), data.get("hourly_units", {})
    if not hourly.get("time"):
        raise WeatherResponseError("Weather response has no hourly timestamps.")
    times = pd.to_datetime(hourly["time"], utc=True, errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise WeatherResponseError("Weather response has missing or duplicate hourly timestamps.")
    missing = sorted(set(VARIABLES) - set(hourly))
    if missing:
        raise WeatherResponseError(f"Weather response is missing requested variables: {missing}.")
    all_missing = []
    for variable, expected in EXPECTED_UNITS.items():
        actual = units.get(variable)
        if _normalise_unit(actual) != _normalise_unit(expected):
            raise WeatherResponseError(f"Weather unit mismatch for {variable}: expected {expected!r}, received {actual!r}.")
        if len(hourly[variable]) != len(times):
            raise WeatherResponseError(f"Weather variable {variable} does not match the timestamp count.")
        values = np.asarray(pd.to_numeric(pd.Series(hourly[variable]), errors="raise"), dtype=float)
        if np.isinf(values).any():
            raise WeatherResponseError(f"Weather variable {variable} contains infinity.")
        if variable == "temperature_2m" and not np.isfinite(values).all():
            raise WeatherResponseError("Weather response contains missing air temperature; no temperature gaps are silently accepted.")
        if np.isnan(values).all():
            all_missing.append(variable)
    for coordinate in ("latitude", "longitude"):
        if coordinate not in data or not np.isfinite(float(data[coordinate])):
            raise WeatherResponseError(f"Weather response lacks a finite grid {coordinate}.")
    return {"validated_units": units, "all_missing_variables": all_missing,
            "lag_source": "ERA5 background air temperature, not weather-station history",
            "coastal_grid_caution": "Nearest selection can return an ocean grid cell; missing soil moisture is a diagnostic, not a land/sea classification."}


def grid_distance_km(latitude, longitude, grid_latitude, grid_longitude):
    """Great-circle separation, including longitudes across the antimeridian."""
    lat, lon, grid_lat, grid_lon = [np.deg2rad(np.asarray(value, dtype=float)) for value in (latitude, longitude, grid_latitude, grid_longitude)]
    haversine = np.sin((grid_lat-lat)/2)**2 + np.cos(lat)*np.cos(grid_lat)*np.sin((grid_lon-lon)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def fetch_archive(latitude, longitude, start, end, cache_dir):
    """One short time series at a native 0.25-degree grid query point."""
    params = dict(latitude=float(latitude), longitude=float(longitude),
                  start_date=str(start)[:10], end_date=str(end)[:10],
                  hourly=",".join(VARIABLES), models="era5", timezone="UTC",
                  wind_speed_unit="ms", temperature_unit="celsius",
                  precipitation_unit="mm", elevation="nan", cell_selection="nearest")
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(cache_dir) / "weather"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"era5_{key}.json"
    if not path.exists():
        for attempt in range(3):
            response = requests.get(URL, params=params, timeout=(15, 120))
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if attempt == 2:
                response.raise_for_status()
            time.sleep(min(30, 3 * 2 ** attempt))
        response.raise_for_status()
        data = response.json()
        validation = validate_weather_response(data)
        record = {"retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "source": URL, "model": "ERA5", "request": params, "response": data,
                  "validation": validation}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=path.stem + "_", suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                # This contains public ERA5 weather, not credentials. Share it
                # with the cache's existing group so research refreshes do not
                # make an otherwise identical response unreadable to the site.
                os.fchown(stream.fileno(), -1, root.stat().st_gid)
                os.fchmod(stream.fileno(), 0o664)
                json.dump(record, stream)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        time.sleep(.2)
    record = json.loads(path.read_text())
    data = record["response"]
    record["validation"] = validate_weather_response(data)
    frame = pd.DataFrame(data["hourly"]).rename(columns={"time": "weather_datetime_utc", **VARIABLES})
    frame["weather_datetime_utc"] = pd.to_datetime(frame["weather_datetime_utc"], utc=True)
    frame["cloud_cover_fraction"] = frame.pop("cloud_cover_pct") / 100
    direction = np.deg2rad(frame.pop("wind_direction_deg"))
    frame["wind_direction_sin"] = np.sin(direction)
    frame["wind_direction_cos"] = np.cos(direction)
    frame["weather_grid_latitude"] = data["latitude"]
    frame["weather_grid_longitude"] = data["longitude"]
    frame["weather_grid_elevation_m"] = data.get("elevation", np.nan)
    frame["weather_source"] = "ERA5 via Open-Meteo; native coarse grid"
    frame["weather_cache_key"] = key
    frame["weather_lag_source"] = "ERA5 background air history, not station history"
    frame["weather_land_context_missing"] = frame["soil_moisture_m3_m3"].isna()
    frame = frame.sort_values("weather_datetime_utc").drop_duplicates("weather_datetime_utc")
    frame = frame.set_index("weather_datetime_utc").asfreq("h")
    for lag in (1, 3, 24):
        frame[f"air_temperature_lag{lag}_c"] = frame["background_air_temperature_c"].shift(lag)
    frame["shortwave_down_lag1_w_m2"] = frame["shortwave_down_w_m2"].shift(1)
    frame["shortwave_down_mean3_w_m2"] = frame["shortwave_down_w_m2"].rolling(3, min_periods=3).mean()
    frame["rain_mm_24h"] = frame["rain_mm_h"].rolling(24, min_periods=24).sum()
    frame["rain_mm_72h"] = frame["rain_mm_h"].rolling(72, min_periods=72).sum()
    return frame.reset_index(), record


def enrich_weather(samples, cache_dir, max_requests=200):
    """Backward hourly joins; radiation/precipitation describe preceding hour.

    Weather is queried separately for each rounded 0.25-degree cell/month.
    No contemporaneous or future surface-temperature field is requested.
    """
    data = samples.copy()
    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"], utc=True)
    data["_weather_lat"] = np.round(data.latitude / .25) * .25
    data["_weather_lon"] = np.round(data.longitude / .25) * .25
    data["_month"] = data.datetime_utc.dt.strftime("%Y-%m")
    groups = data.groupby(["_weather_lat", "_weather_lon", "_month"], sort=True)
    if groups.ngroups > max_requests:
        raise ValueError(f"Weather requires {groups.ngroups} groups; explicit cap is {max_requests}. Subset or increase cap deliberately.")
    output, audit = [], []
    for number, ((lat, lon, month), group) in enumerate(groups, 1):
        print(f"weather {number}/{groups.ngroups}: {lat},{lon} {month}", flush=True)
        start = group.datetime_utc.min().floor("D") - pd.Timedelta(days=4)
        end = group.datetime_utc.max().floor("D")
        weather, record = fetch_archive(lat, lon, start, end, cache_dir)
        joined = pd.merge_asof(group.sort_values("datetime_utc"), weather,
                               left_on="datetime_utc", right_on="weather_datetime_utc",
                               direction="backward", tolerance=pd.Timedelta("1h"))
        joined["weather_age_minutes"] = (joined.datetime_utc - joined.weather_datetime_utc).dt.total_seconds() / 60
        joined["weather_grid_distance_km"] = grid_distance_km(joined.latitude, joined.longitude, joined.weather_grid_latitude, joined.weather_grid_longitude)
        output.append(joined)
        audit.append({"requested_latitude":lat, "requested_longitude":lon, "month":month,
                      "grid_latitude":record["response"]["latitude"],
                      "grid_longitude":record["response"]["longitude"],
                      "rows":len(group), "hourly_units":record["response"].get("hourly_units"),
                      "response_validation":record["validation"],
                      "maximum_sample_grid_distance_km":float(joined["weather_grid_distance_km"].max()),
                      "unmatched_sample_count":int(joined.weather_datetime_utc.isna().sum())})
    result = pd.concat(output, ignore_index=True).drop(columns=["_weather_lat", "_weather_lon", "_month"])
    return result, audit
