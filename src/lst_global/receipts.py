"""Cache-only source receipts for one historical global patch hour.

No fetch helper is called. Existing predictors are preserved exactly. This
checks the cached delivery used by the feature adapter, not satellite truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lst_pilot import stations, weather


VERSION = "global-cache-receipts-v1"
STATION_OK = "exact_cached_report_and_background_verified"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path, root, limit):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Receipt source lies outside the supplied public cache.")
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise ValueError("Receipt source exceeds its bounded size or is empty.")
    return {"path": str(path), "sha256": _sha(path), "bytes": size}


def _weather(cache, key):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{24}", key):
        raise ValueError("Malformed weather cache identity.")
    path = cache / "weather" / f"era5_{key}.json"
    binding = _file(path, cache, 20_000_000)
    record = json.loads(path.read_text())
    request = record["request"]
    actual = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    if (actual != key or record.get("source") != weather.URL or request.get("models") != "era5"
            or request.get("timezone") != "UTC" or request.get("elevation") != "nan"
            or request.get("cell_selection") != "nearest"
            or request.get("hourly") != ",".join(weather.VARIABLES)):
        raise ValueError("Weather cache request differs from its F40 source identity.")
    response = record["response"]
    weather.validate_weather_response(response)
    result = pd.DataFrame(response["hourly"]).rename(columns={"time": "weather_datetime_utc", **weather.VARIABLES})
    result["weather_datetime_utc"] = pd.to_datetime(result.weather_datetime_utc, utc=True)
    result["cloud_cover_fraction"] = result.pop("cloud_cover_pct") / 100
    angle = np.deg2rad(result.pop("wind_direction_deg"))
    result["wind_direction_sin"], result["wind_direction_cos"] = np.sin(angle), np.cos(angle)
    result["weather_grid_latitude"], result["weather_grid_longitude"] = response["latitude"], response["longitude"]
    result = result.set_index("weather_datetime_utc").sort_index().asfreq("h")
    for lag in (1, 3, 24):
        result[f"air_temperature_lag{lag}_c"] = result.background_air_temperature_c.shift(lag)
    result["shortwave_down_lag1_w_m2"] = result.shortwave_down_w_m2.shift(1)
    result["shortwave_down_mean3_w_m2"] = result.shortwave_down_w_m2.rolling(3, min_periods=3).mean()
    result["rain_mm_24h"] = result.rain_mm_h.rolling(24, min_periods=24).sum()
    result["rain_mm_72h"] = result.rain_mm_h.rolling(72, min_periods=72).sum()
    binding.update(kind="Open-Meteo ERA5 response", key=key, request=request,
                   native_latitude=response["latitude"], native_longitude=response["longitude"],
                   hourly_units=response["hourly_units"])
    return result, binding


def _equal(actual, expected, description):
    if not np.array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), equal_nan=True):
        raise ValueError(f"Cached source does not reproduce {description} exactly.")


def _background_key(latitude, longitude, report_time):
    # This mirrors the exact single-time request in assemble.attach_stations
    # -> weather.enrich_weather -> fetch_archive; no fallback cache search.
    params = dict(latitude=float(np.round(latitude / .25) * .25),
                  longitude=float(np.round(longitude / .25) * .25),
                  start_date=str(report_time.floor("D") - pd.Timedelta(days=4))[:10],
                  end_date=str(report_time.floor("D"))[:10], hourly=",".join(weather.VARIABLES),
                  models="era5", timezone="UTC", wind_speed_unit="ms", temperature_unit="celsius",
                  precipitation_unit="mm", elevation="nan", cell_selection="nearest")
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]


def bind_cached_sources(frame, cache):
    """Return unchanged predictors plus per-row verification and a file receipt.

    Supports one exact target hour, at most262,144 pixels,64 weather cache keys
    and four station/year groups. Missing station evidence masks that source;
    inconsistent/corrupted existing evidence raises. No source is downloaded,
    and no missing parsed station cache is generated.
    """
    cache = Path(cache)
    required = {"sample_id", "datetime_utc", "weather_cache_key", "weather_datetime_utc",
                "weather_grid_latitude", "weather_grid_longitude", "background_air_temperature_c",
                "station_id", "station_age_minutes", "station_distance_km", "station_air_correction_c",
                "air_temperature_c", "observed_station_air_temperature_c", "air_temperature_source"}
    if not required.issubset(frame) or not 0 < len(frame) <= 262_144 or frame.sample_id.duplicated().any():
        raise ValueError("Incomplete or unbounded label-free input for cache receipt.")
    target = pd.to_datetime(frame.datetime_utc, utc=True)
    if target.nunique() != 1 or not target.eq(target.dt.floor("h")).all():
        raise ValueError("Cache receipt supports exactly one requested UTC hour.")
    if frame.weather_cache_key.nunique(dropna=False) > 64:
        raise ValueError("Too many weather cache identities for one patch.")
    original, data = frame, frame.copy()
    for name in ("verified_weather_response_sha256", "verified_station_raw_sha256",
                 "verified_station_parsed_sha256", "verified_station_background_weather_key"):
        if name in data:
            raise ValueError("Input already contains source receipt columns.")
        data[name] = ""
    data["verified_station_report_status"] = "no_station_pair"
    data["verified_station_observation_datetime_utc"] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    files, weather_tables, station_records = {}, {}, []

    def load_weather(key):
        if key not in weather_tables:
            table, binding = _weather(cache, key)
            files[binding["path"]] = binding
            weather_tables[key] = table
        return weather_tables[key]

    for key, part in data.groupby("weather_cache_key", dropna=False, sort=True):
        table = load_weather(key)
        times = pd.to_datetime(part.weather_datetime_utc, utc=True)
        if not times.eq(target.loc[part.index]).all():
            raise ValueError("Whole-hour patch weather is not at the requested UTC hour.")
        matched = table.reindex(pd.DatetimeIndex(times))
        columns = [name for name in table.columns if name in part]
        for name in columns:
            _equal(part[name], matched[name], name)
        binding = files[str((cache / "weather" / f"era5_{key}.json").resolve())]
        data.loc[part.index, "verified_weather_response_sha256"] = binding["sha256"]

    usable = data.station_id.fillna("").ne("")
    times = (target - pd.to_timedelta(data.station_age_minutes, unit="m")).dt.round("us")
    if (usable & times.isna()).any():
        raise ValueError("Named station lacks a finite observation timestamp.")
    groups = list(data.loc[usable].groupby([data.loc[usable, "station_id"], times.loc[usable].dt.year], sort=True))
    if len(groups) > 4:
        raise ValueError("Too many station/year identities for one patch.")
    inventory_path = cache / "stations" / "ghcnh-station-list.csv"
    inventory = None
    if groups and inventory_path.exists():
        binding = _file(inventory_path, cache, 10_000_000)
        binding["kind"] = "NOAA GHCNh station inventory"
        files[binding["path"]] = binding
        inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False).set_index("GHCN_ID")

    for (station_id, year), part in groups:
        url = stations.station_year_url(station_id, int(year))
        raw = cache / "stations" / str(int(year)) / url.rsplit("/", 1)[-1]
        record = {"station_id": station_id, "year": int(year), "rows": len(part)}
        station_records.append(record)
        if not raw.exists() or inventory is None or station_id not in inventory.index:
            status = "raw_or_inventory_cache_missing"
            data.loc[part.index, "verified_station_report_status"] = status
            record["status"] = status
            continue
        raw_binding = _file(raw, cache, 250_000_000)
        raw_binding.update(kind="NOAA GHCNh raw PSV", source_url=url)
        files[raw_binding["path"]] = raw_binding
        identity = {"schema_version": 1, "raw_sha256": raw_binding["sha256"],
                    "parser_source_sha256": _sha(stations.__file__), "keep_flags": True}
        parser_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        parsed_path = raw.parent / f"{station_id}_parsed_{parser_key}.parquet"
        sidecar = parsed_path.with_suffix(".provenance.json")
        if not parsed_path.exists() or not sidecar.exists():
            status = "matching_parser_cache_missing"
            data.loc[part.index, "verified_station_report_status"] = status
            record["status"] = status
            continue
        sidecar_binding = _file(sidecar, cache, 1_000_000)
        if json.loads(sidecar.read_text()) != identity:
            raise ValueError("Station parser provenance does not match raw source and current parser.")
        parsed_binding = _file(parsed_path, cache, 250_000_000)
        files[sidecar_binding["path"]], files[parsed_binding["path"]] = sidecar_binding, parsed_binding
        parsed = pd.read_parquet(parsed_path)
        if not parsed.station_id.eq(station_id).all():
            raise ValueError("Parsed station cache contains a different station identity.")
        # Do not expand dozens of unneeded raw-report columns to 262k pixels.
        observations = parsed[["timestamp_utc", "air_temperature_c"]].dropna(subset=["air_temperature_c"]).sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
        obs_times = pd.DatetimeIndex(times.loc[part.index])
        matched = observations.set_index("timestamp_utc").reindex(obs_times)
        _equal(part.observed_station_air_temperature_c, matched.air_temperature_c, "station observation")
        if (not np.isfinite(part.observed_station_air_temperature_c).all()
                or not np.isfinite(part.station_air_correction_c).all()
                or not part.station_air_correction_c.abs().le(20).all()):
            raise ValueError("Station report or residual is missing or outside the original correction bound.")
        if (not part.station_age_minutes.between(0, 90).all()
                or not part.station_distance_km.between(0, 100).all()
                or not part.air_temperature_source.eq("observed_station_residual_plus_ERA5_spatial_background").all()):
            raise ValueError("Named station violates source, distance or backward-age policy.")
        station = inventory.loc[station_id]
        if not isinstance(station, pd.Series):
            raise ValueError("Station inventory identity is duplicated.")
        for report_time, rows in part.groupby(times.loc[part.index], sort=True):
            key = _background_key(float(station.LATITUDE), float(station.LONGITUDE), report_time)
            path = cache / "weather" / f"era5_{key}.json"
            if not path.exists():
                data.loc[rows.index, "verified_station_report_status"] = "station_background_cache_missing"
                continue
            background = load_weather(key)
            hour = report_time.floor("h")
            if hour not in background.index or not np.isfinite(background.loc[hour, "background_air_temperature_c"]):
                raise ValueError("Station background has no valid requested report-hour air.")
            correction = rows.observed_station_air_temperature_c - float(background.loc[hour, "background_air_temperature_c"])
            _equal(rows.station_air_correction_c, correction, "station residual")
            _equal(rows.air_temperature_c, rows.background_air_temperature_c + correction, "model air input")
            data.loc[rows.index, "verified_station_report_status"] = STATION_OK
            data.loc[rows.index, "verified_station_background_weather_key"] = key
            data.loc[rows.index, "verified_station_observation_datetime_utc"] = report_time
        data.loc[part.index, "verified_station_raw_sha256"] = raw_binding["sha256"]
        data.loc[part.index, "verified_station_parsed_sha256"] = parsed_binding["sha256"]
        record.update(status="checked", statuses=data.loc[part.index, "verified_station_report_status"].value_counts().to_dict(),
                      raw=raw_binding, parsed=parsed_binding, parser=identity)
    pd.testing.assert_frame_equal(original, data[original.columns], check_exact=True)
    receipt = {"version": VERSION, "network_requests": 0, "original_columns_exactly_preserved": True,
               "rows": len(data), "files": list(files.values()), "stations": station_records,
               "station_status_counts": data.verified_station_report_status.value_counts().to_dict(),
               "source_code_sha256": {"receipts": _sha(__file__), "stations": _sha(stations.__file__), "weather": _sha(weather.__file__)},
               "verification_scope": "Exact cached weather and parsed report values, raw/parser identity and station residual; no independent reparse of full raw PSV or new downloads."}
    return data, receipt
