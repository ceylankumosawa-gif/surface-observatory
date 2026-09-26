"""Explicit ERA5/ERA5T availability and bounded experimental current GFS inputs.

No requested timestamp is replaced by an older available day. This adapter does
not produce station observations, establish forecast/model compatibility, or
authorize a prediction. Consumers must preserve its source/compatibility receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import requests

from lst_pilot import weather as legacy_weather
from lst_pilot import radiation as legacy_radiation

VERSION = "global_weather_access_v1"
METADATA_URL = "https://storage.googleapis.com/" + legacy_radiation.STORE + "/.zmetadata"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GFS_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_VARIABLES = {**legacy_weather.VARIABLES}
GFS_VARIABLES["soil_moisture_0_to_10cm"] = GFS_VARIABLES.pop("soil_moisture_0_to_7cm")
GFS_UNITS = {**legacy_weather.EXPECTED_UNITS}
GFS_UNITS["soil_moisture_0_to_10cm"] = GFS_UNITS.pop("soil_moisture_0_to_7cm")
NUMERIC_OUTPUTS = sorted(set(GFS_VARIABLES.values()) - {"cloud_cover_pct", "wind_direction_deg"})
DERIVED_OUTPUTS = ["cloud_cover_fraction", "wind_direction_sin", "wind_direction_cos",
                   "air_temperature_lag1_c", "air_temperature_lag3_c", "air_temperature_lag24_c",
                   "shortwave_down_lag1_w_m2", "shortwave_down_mean3_w_m2", "rain_mm_24h", "rain_mm_72h"]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError("Explicit timezone-aware UTC timestamp required")
    return stamp.tz_convert("UTC")


def _public_directory(path: Path) -> int:
    """Create only new public-cache directories, inheriting the existing group."""
    missing, ancestor = [], path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    group = ancestor.stat().st_gid
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        os.chown(directory,-1,group)
        directory.chmod(0o2775)
    return group


def _atomic(path: Path, payload: bytes) -> None:
    group = _public_directory(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchown(stream.fileno(),-1,group)
            os.fchmod(stream.fileno(),0o664)
            stream.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class BoundedHTTP:
    """Single-job public source budget; no credentials, redirects, retries."""

    def __init__(self, max_requests=64, max_bytes=16 * 1024 * 1024):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["User-Agent"] = "LST-global-weather-research/1.0"
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.requests, self.received_bytes = 0, 0
        self.receipts = []

    def get(self, url, *, params=None, limit=1_000_000, allow_empty=False):
        if self.requests >= self.max_requests:
            raise ValueError("Weather HTTP request budget exhausted")
        self.requests += 1
        with self.session.get(url, params=params, timeout=(10, 45), stream=True,
                              allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code == 204 and allow_empty:
                self.receipts.append({"url": response.url, "bytes": 0, "status":204,
                                      "sha256":_sha(b""), "retrieved_utc":pd.Timestamp.now(tz="UTC").isoformat()})
                return b""
            if response.status_code != 200:
                raise ValueError("Unexpected weather HTTP status")
            remaining = min(limit, self.max_bytes - self.received_bytes)
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > remaining:
                raise ValueError("Weather object exceeds remaining byte budget")
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=8192):
                self.received_bytes += len(chunk)
                size += len(chunk)
                if size > limit or self.received_bytes > self.max_bytes:
                    raise ValueError("Weather byte budget exceeded during response")
                chunks.append(chunk)
            payload = b"".join(chunks)
            self.receipts.append({"url": response.url, "bytes": size, "sha256": _sha(payload),
                                  "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()})
            return payload


def inspect_availability(cache, *, now=None, http=None, refresh=False) -> dict:
    """Probe only the small official ARCO index; keep immutable source snapshots."""
    now = _utc(now or pd.Timestamp.now(tz="UTC"))
    cache = Path(cache) / "weather_access" / "availability"
    current = cache / "latest.json"
    if current.exists() and not refresh:
        record = json.loads(current.read_text())
        age = (now - _utc(record["checked_utc"])).total_seconds()
        source = Path(record["metadata_path"])
        if 0 <= age <= 10800 and source.exists() and _sha(source.read_bytes()) == record["metadata_sha256"]:
            return record
    http = http or BoundedHTTP(max_requests=1, max_bytes=1_000_000)
    payload = http.get(METADATA_URL)
    metadata = json.loads(payload)["metadata"]
    legacy_radiation._validate_metadata(metadata)
    attrs = metadata[".zattrs"]
    start = pd.Timestamp(attrs["valid_time_start"], tz="UTC")
    final = pd.Timestamp(attrs["valid_time_stop"], tz="UTC") + pd.Timedelta(hours=23)
    provisional = pd.Timestamp(attrs.get("valid_time_stop_era5t", attrs["valid_time_stop"]), tz="UTC") + pd.Timedelta(hours=23)
    if not start <= final <= provisional <= now:
        raise ValueError("Nonmonotonic or future ARCO publication bounds")
    digest = _sha(payload)
    path = cache / (digest + ".zmetadata")
    _atomic(path, payload)
    record = {"version": VERSION, "checked_utc": now.isoformat(), "source_url": METADATA_URL,
              "metadata_path": str(path.resolve()), "metadata_sha256": digest,
              "metadata_bytes": len(payload), "publisher_last_updated": attrs.get("last_updated"),
              "first_valid_hour_utc": start.isoformat(), "final_last_valid_hour_utc": final.isoformat(),
              "provisional_last_valid_hour_utc": provisional.isoformat(),
              "operational_route": "experimental_gfs", "operational_compatibility_validated": False,
              "station_availability_verified": False}
    _atomic(current, (json.dumps(record, indent=2) + "\n").encode())
    return record


def choose_weather_source(timestamp, availability, *, now=None, allow_operational=False) -> dict:
    """Select by actual publication bounds, never by a fixed assumed lag."""
    target, now = _utc(timestamp), _utc(now or pd.Timestamp.now(tz="UTC"))
    if target != target.floor("h"):
        raise ValueError("Global jobs require exact hourly UTC timestamps")
    if target > now.floor("h"):
        raise ValueError("Future valid times are outside the historical-to-present product")
    if target < pd.Timestamp("2021-01-01", tz="UTC"):
        raise ValueError("Current surface/model contract begins in 2021")
    checked = _utc(availability["checked_utc"])
    if not pd.Timedelta(0) <= now - checked <= pd.Timedelta(hours=24):
        raise ValueError("Availability proof expired; refresh before selecting a date")
    final = _utc(availability["final_last_valid_hour_utc"])
    provisional = _utc(availability["provisional_last_valid_hour_utc"])
    source = "era5_final" if target <= final else "era5_provisional" if target <= provisional else "experimental_gfs"
    enabled = source != "experimental_gfs" or allow_operational
    if source == "experimental_gfs" and now - target > pd.Timedelta(days=7):
        enabled = False
    return {"version": VERSION, "requested_time_utc": target.isoformat(), "valid_time_utc": target.isoformat(),
            "source": source, "enabled": enabled, "availability": availability,
            "temporal_substitution": False, "is_station_observation": False,
            "is_reanalysis": source != "experimental_gfs",
            "provisional": source == "era5_provisional", "compatibility":
            "same_ERA5_feature_definitions_revision_possible" if source == "era5_provisional" else
            "same_ERA5_feature_definitions" if source == "era5_final" else "unvalidated_operational_distribution",
            "source_limitations": [] if source != "experimental_gfs" else [
                "GFS soil moisture represents 0–10 cm instead of trained ERA5 0–7 cm",
                "Open-Meteo GFS delivery does not expose the exact contributing model cycle",
                "Native GFS longwave/SWE companion has a separately recorded cycle and 0.25 degree grid",
                "Operational source change has not been validated against LST observations"]}


def _check_frame(frame, plan):
    if not plan["enabled"]:
        raise ValueError("Requested weather route is disabled/unavailable; no older time is substituted")
    if len(frame) > 262144:
        raise ValueError("Weather preparation is capped at one canonical tile")
    times = pd.to_datetime(frame.datetime_utc, utc=True, errors="raise")
    if not times.eq(_utc(plan["valid_time_utc"])).all():
        raise ValueError("Every row must match the planned exact valid hour")
    for column, low, high in [("latitude", -90, 90), ("longitude", -180, 180)]:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(low, high).all():
            raise ValueError("Invalid requested weather coordinate")


def _gfs_table(data):
    hourly, units = data.get("hourly", {}), data.get("hourly_units", {})
    if data.get("error") or data.get("utc_offset_seconds") != 0:
        raise ValueError("GFS delivery must be successful and UTC")
    table = pd.DataFrame(hourly).rename(columns={"time": "weather_datetime_utc", **GFS_VARIABLES})
    for variable, unit in GFS_UNITS.items():
        if variable not in hourly or legacy_weather._normalise_unit(units.get(variable)) != legacy_weather._normalise_unit(unit):
            raise ValueError(f"GFS variable or unit mismatch: {variable}")
    times = pd.to_datetime(table.weather_datetime_utc, utc=True, errors="raise")
    if times.duplicated().any() or not times.is_monotonic_increasing or not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("GFS delivery has non-contiguous/duplicate hourly coordinates")
    table["weather_datetime_utc"] = times
    for column in GFS_VARIABLES.values():
        table[column] = pd.to_numeric(table[column], errors="raise")
        if np.isinf(table[column]).any():
            raise ValueError("GFS contains infinite values")
    ranges = {"background_air_temperature_c":(-100,65), "relative_humidity_pct":(0,100),
              "dewpoint_c":(-120,65), "wind_speed_m_s":(0,150), "wind_direction_deg":(0,360),
              "surface_pressure_hpa":(100,1100), "cloud_cover_pct":(0,100),
              "soil_moisture_m3_m3":(0,1), "precipitation_mm_h":(0,1000), "rain_mm_h":(0,1000),
              "shortwave_down_w_m2":(0,1600), "direct_shortwave_w_m2":(0,1600), "diffuse_shortwave_w_m2":(0,1600)}
    for column,(low,high) in ranges.items():
        if not (table[column].isna()|table[column].between(low,high)).all():
            raise ValueError(f"GFS physical-domain failure: {column}")
    table["cloud_cover_fraction"] = table.pop("cloud_cover_pct") / 100
    angle = np.deg2rad(table.pop("wind_direction_deg"))
    table["wind_direction_sin"], table["wind_direction_cos"] = np.sin(angle), np.cos(angle)
    for lag in [1, 3, 24]:
        table[f"air_temperature_lag{lag}_c"] = table.background_air_temperature_c.shift(lag)
    table["shortwave_down_lag1_w_m2"] = table.shortwave_down_w_m2.shift(1)
    table["shortwave_down_mean3_w_m2"] = table.shortwave_down_w_m2.rolling(3, min_periods=3).mean()
    table["rain_mm_24h"] = table.rain_mm_h.rolling(24, min_periods=24).sum()
    table["rain_mm_72h"] = table.rain_mm_h.rolling(72, min_periods=72).sum()
    for coordinate in ("latitude", "longitude"):
        if not np.isfinite(float(data[coordinate])):
            raise ValueError("Missing native GFS delivery coordinate")
        table["weather_grid_" + coordinate] = float(data[coordinate])
    return table


def prepare_background(frame, cache, plan, *, max_requests=16, http=None):
    """Append source-explicit F40 meteorology; station correction remains separate."""
    _check_frame(frame, plan)
    if plan["source"] != "experimental_gfs":
        result, audit = legacy_weather.enrich_weather(frame, cache, max_requests=max_requests)
        if not pd.to_datetime(result.weather_datetime_utc, utc=True).eq(_utc(plan["valid_time_utc"])).all():
            raise ValueError("ERA5 delivery did not provide the exact requested hour")
        result["global_weather_product"] = plan["source"]
        return result, {"plan": plan, "weather_requests": audit, "station_correction_applied": False}
    http = http or BoundedHTTP(max_requests=max_requests, max_bytes=max_requests * 256_000)
    target = _utc(plan["valid_time_utc"])
    data = frame.copy()
    data["_query_lat"], data["_query_lon"] = np.round(data.latitude * 4) / 4, np.round(data.longitude * 4) / 4
    groups = data.groupby(["_query_lat", "_query_lon"], sort=True)
    if groups.ngroups > max_requests:
        raise ValueError("GFS request group cap exceeded")
    outputs, evidence = [], []
    for (lat, lon), group in groups:
        params = dict(latitude=float(lat), longitude=float(lon), start_date=str((target-pd.Timedelta(days=4)).date()),
                      end_date=str(target.date()), hourly=",".join(GFS_VARIABLES), models="gfs_global",
                      timezone="UTC", wind_speed_unit="ms", temperature_unit="celsius", precipitation_unit="mm",
                      elevation="nan", cell_selection="nearest")
        # New source snapshots each checked hour; never overwrite a prior delivery.
        key = _sha(json.dumps({"params": params, "checked_utc": plan["availability"]["checked_utc"]}, sort_keys=True).encode())
        path = Path(cache) / "weather_access" / "gfs" / (key + ".json")
        if not path.exists():
            payload = http.get(FORECAST_URL, params=params, limit=256_000)
            _gfs_table(json.loads(payload))
            record = {"request": params, "response": json.loads(payload), "source": FORECAST_URL,
                      "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(), "model": "gfs_global",
                      "exact_model_cycle": None, "response_sha256": _sha(payload)}
            _atomic(path, (json.dumps(record, sort_keys=True) + "\n").encode())
        record = json.loads(path.read_text())
        if record["request"] != params or record["model"] != "gfs_global":
            raise ValueError("GFS cache source identity mismatch")
        table = _gfs_table(record["response"])
        selected = table.loc[table.weather_datetime_utc.eq(target)]
        if len(selected) != 1:
            raise ValueError("GFS did not deliver the exact requested valid hour")
        joined = group.drop(columns=["_query_lat", "_query_lon"]).copy()
        for column, value in selected.iloc[0].items():
            joined[column] = value
        joined["weather_source"] = "GFS via Open-Meteo; explicit gfs_global; cycle not exposed"
        joined["weather_cache_key"] = key
        joined["global_weather_product"] = "experimental_gfs"
        joined["weather_soil_depth_cm"] = "0–10"
        joined["weather_age_minutes"] = 0.0
        joined["weather_grid_distance_km"] = legacy_weather.grid_distance_km(joined.latitude, joined.longitude, joined.weather_grid_latitude, joined.weather_grid_longitude)
        outputs.append(joined)
        evidence.append({"path": str(path.resolve()), "sha256": _sha(path.read_bytes()), "requested_grid": [lat, lon],
                         "returned_grid": [record["response"]["latitude"], record["response"]["longitude"]]})
    return pd.concat(outputs, ignore_index=True), {"plan": plan, "files": evidence, "http": http.receipts,
                                                 "station_correction_applied": False}


def _index_records(payload, cycle, lead):
    lines = payload.decode().splitlines()
    result = {}
    for variable in ("WEASD", "DLWRF"):
        found = [line for line in lines if f":{variable}:surface:" in line]
        if len(found) != 1 or f":d={cycle.strftime('%Y%m%d%H')}:" not in found[0]:
            raise ValueError(f"GFS index lacks unique matching {variable}")
        description = found[0].split(":")[5]
        if variable == "WEASD":
            if description != f"{lead} hour fcst":
                raise ValueError("Unexpected snow time support")
            result[variable] = {"line": found[0], "start": lead, "end": lead}
        else:
            match = re.fullmatch(r"(\d+)-(\d+) hour ave fcst", description)
            if not match or int(match[2]) != lead or not 0 <= int(match[1]) < lead:
                raise ValueError("Unexpected longwave averaging support")
            result[variable] = {"line": found[0], "start": int(match[1]), "end": lead}
    return result


def hourly_longwave(current, previous, current_start, lead, previous_start=None):
    """Deaverage matching same-run cumulative means; never interpolate time."""
    width = lead - current_start
    if width == 1:
        value = np.asarray(current, float)
    elif width > 1 and previous is not None and previous_start == current_start:
        value = width * np.asarray(current, float) - (width - 1) * np.asarray(previous, float)
    else:
        raise ValueError("Longwave means do not cover the same accumulation origin")
    return np.where(np.isfinite(value) & (value > 0) & (value < 1000), value, np.nan)


def _native_rowcol(transform, lons, lats, width, height):
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, lons, lats)
    rows, cols = np.asarray(rows), np.asarray(cols)
    if ((rows<0)|(rows>=height)|(cols<0)|(cols>=width)).any():
        raise ValueError("GFS subset does not cover requested cells")
    return rows, cols


def add_radiation(frame, cache, plan, *, http=None):
    """Exact-hour ERA5 fields, or true native GFS SWE + deaveraged hourly LW."""
    _check_frame(frame, plan)
    if plan["source"] != "experimental_gfs":
        # Provisional chunks may be revised: isolate by immutable metadata snapshot.
        root = Path(cache) / "radiation"
        if plan["provisional"]:
            root = Path(cache) / "weather_access" / "era5t" / plan["availability"]["metadata_sha256"]
            _public_directory(root/"arco-era5-v3")
        result = legacy_radiation.add_radiation(frame, root, max_hours=1, allow_provisional=plan["provisional"])
        return result, result.attrs["radiation_context"]
    import rasterio
    from rasterio.io import MemoryFile
    target = _utc(plan["valid_time_utc"])
    now = _utc(plan["availability"]["checked_utc"])
    # Run at least six hours old, hence normally published; its existence is still verified.
    cycle = min(target - pd.Timedelta(hours=1), now - pd.Timedelta(hours=6)).floor("6h")
    lead = int((target-cycle)/pd.Timedelta(hours=1))
    if not 1 <= lead <= 24:
        raise ValueError("Current GFS companion lead outside bounded1–24h contract")
    http = http or BoundedHTTP(max_requests=4, max_bytes=4_000_000)
    lats, lons = np.asarray(frame.latitude, float), np.asarray(frame.longitude, float)
    # Half-cell halo guarantees nearest centres. Dateline splitting remains explicit future work.
    left, right = math.floor(lons.min()*4)/4, math.ceil(lons.max()*4)/4
    bottom, top = math.floor(lats.min()*4)/4, math.ceil(lats.max()*4)/4
    if right-left > 2 or top-bottom > 2 or left < -180 or right > 180:
        raise ValueError("Native GFS companion requires a <=2degree non-crossing region")
    left, right, bottom, top = left-.25, right+.25, max(-90,bottom-.25), min(90,top+.25)
    evidence = []
    def retrieve(hour):
        stem = f"gfs.t{cycle.hour:02d}z.pgrb2.0p25.f{hour:03d}"
        directory = f"gfs.{cycle.strftime('%Y%m%d')}/{cycle.hour:02d}/atmos"
        idx_url = f"{GFS_ROOT}/{directory}/{stem}.idx"
        index = http.get(idx_url, limit=100_000)
        records = _index_records(index, cycle, hour)
        params = {"file": stem, "dir": "/"+directory, "lev_surface": "on", "var_DLWRF": "on",
                  "var_WEASD": "on", "subregion": "", "leftlon": left % 360, "rightlon": right % 360,
                  "toplat": top, "bottomlat": bottom}
        if params["leftlon"] > params["rightlon"]:
            # NOMADS also accepts negative longitude ranges, including Greenwich.
            params["leftlon"], params["rightlon"] = left, right
        key = _sha(json.dumps(params, sort_keys=True).encode())
        path = Path(cache) / "weather_access" / "gfs_native" / (key+".grib2")
        if not path.exists():
            payload = http.get(GFS_FILTER, params=params, limit=1_000_000)
            if payload[:4] != b"GRIB":
                raise ValueError("Native GFS response is not GRIB")
            _atomic(path, payload)
        payload = path.read_bytes()
        values, tags_out = {}, {}
        with MemoryFile(payload) as memory, memory.open() as dataset:
            if not np.allclose([dataset.transform.a,dataset.transform.e],[.25,-.25]):
                raise ValueError("Unexpected native GFS grid")
            if dataset.width > 16 or dataset.height > 16:
                raise ValueError("GFS server did not honor the bounded native subset")
            for band in range(1,dataset.count+1):
                tags = dataset.tags(band)
                variable = tags.get("GRIB_ELEMENT")
                if variable not in records or variable in values or tags.get("GRIB_SHORT_NAME") != "0-SFC":
                    raise ValueError("Unexpected/duplicate native GFS field")
                valid = int((cycle+pd.Timedelta(hours=hour)).timestamp())
                if int(tags["GRIB_VALID_TIME"]) != valid or int(tags["GRIB_REF_TIME"]) != int(cycle.timestamp()):
                    raise ValueError("Native GFS run/valid time differs from plan")
                expected_unit = "[kg/(m^2)]" if variable == "WEASD" else "[W/(m^2)]"
                if tags["GRIB_UNIT"] != expected_unit:
                    raise ValueError("Native GFS unit mismatch")
                if variable == "DLWRF" and (tags.get("GRIB_PDS_PDTN") != "8" or int(tags["GRIB_FORECAST_SECONDS"]) != records[variable]["start"]*3600):
                    raise ValueError("Native GFS average origin does not match index")
                rows, cols = _native_rowcol(dataset.transform,lons,lats,dataset.width,dataset.height)
                raw = dataset.read(band, masked=True).filled(np.nan)
                values[variable] = raw[rows, cols]
                tags_out[variable] = tags
            if set(values) != {"WEASD", "DLWRF"}:
                raise ValueError("Missing native GFS companion field")
            xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
        evidence.append({"path": str(path.resolve()), "sha256": _sha(payload), "bytes":len(payload),
                         "index_sha256":_sha(index), "index_records": records, "grib_tags":tags_out,
                         "request":params})
        return values, records, np.asarray(xs), np.asarray(ys)
    values, records, xs, ys = retrieve(lead)
    previous, previous_start = None, None
    if records["DLWRF"]["start"] < lead-1:
        prior, prior_records, px, py = retrieve(lead-1)
        if not np.array_equal(xs,px) or not np.array_equal(ys,py):
            raise ValueError("GFS hourly deaveraging grids differ")
        previous, previous_start = prior["DLWRF"], prior_records["DLWRF"]["start"]
    result = frame.copy()
    result["era5_longwave_down_w_m2"] = hourly_longwave(values["DLWRF"],previous,records["DLWRF"]["start"],lead,previous_start)
    snow = values["WEASD"].astype(float)/1000
    result["era5_snow_water_equivalent_m"] = np.where(np.isfinite(snow)&(snow>=0),snow,np.nan)
    for column in legacy_radiation.VARIABLES.values():
        result[column+"_status"] = np.where(result[column].notna(),"experimental_native_GFS", "missing_or_invalid_value")
    result["radiation_era5_time_utc"] = target
    result["radiation_era5_latitude"], result["radiation_era5_longitude"] = ys, xs
    result["radiation_actual_source"] = "NOAA GFS0.25; legacy era5 column names retained only for F40 schema"
    result["radiation_gfs_cycle_utc"] = cycle
    receipt = {"source":"NOAA GFS0.25", "cycle_utc":cycle.isoformat(), "valid_time_utc":target.isoformat(),
               "is_reanalysis":False, "compatibility_validated":False, "files":evidence, "http":http.receipts,
               "longwave_support":"deaveraged one-hour mean ending at requested valid hour",
               "snow_support":"instantaneous WEASD kg/m2 divided by1000 to m water equivalent",
               "legacy_column_names_do_not_identify_source":True}
    return result, receipt
