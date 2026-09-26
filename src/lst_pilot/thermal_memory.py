"""Research-only causal ERA5 heating/cooling history; no model or daylight gate.

Fluxes labelled H describe (H-1h, H]. A request at 12:30 uses complete
intervals ending no later than 12:00; no part of (12:00, 12:30] is invented.
Air values are instantaneous ERA5 background samples at hourly endpoints,
not an observed weather-station history or continuous-time extrema.

This module does not change serving, fetch thermal labels, interpolate gaps,
borrow weather from another date, or apply a current station/manual override
to the past. Its features need an independent training/evaluation experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import requests

from . import radiation, weather

VERSION = "thermal-memory-v1"
HOUR = pd.Timedelta(hours=1)
HISTORY_HOURS = 24
MAX_ADDITIONAL_CACHE_BYTES = 512 * 1024**2
WEATHER_RESPONSE_RESERVE_BYTES = 1024**2
INPUT_UNITS = {
    "shortwave_down_w_m2": "W/m²",
    "era5_longwave_down_w_m2": "W/m²",
    "background_air_temperature_c": "°C",
}
FEATURE_UNITS = {
    **{f"memory_shortwave_energy_{h}h_j_m2": "J/m²" for h in (6, 12, 24)},
    **{f"memory_longwave_mean_{h}h_w_m2": "W/m²" for h in (6, 24)},
    **{f"memory_air_{stat}_{h}h_c": "°C" for h in (6, 24) for stat in ("mean", "range")},
    **{f"memory_air_change_{h}h_c": "°C" for h in (3, 6)},
}
DOCUMENTATION = {
    "shortwave_and_air": "https://open-meteo.com/en/docs/historical-weather-api",
    "era5_hourly_mean_rates": "https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810",
    "arco": radiation.DOC_URL,
    "longwave_parameter": "https://codes.ecmwf.int/grib/param-db/235036",
}


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Thermal memory requires an explicit timezone-aware timestamp.")
    return stamp.tz_convert("UTC")


def history_end(target) -> pd.Timestamp:
    return _utc(target).floor("h")


def aggregate_history(history: pd.DataFrame, target, *, units: dict) -> dict:
    """Aggregate one location's hourly history with explicit unit validation.

    Required time column is hour_ending_utc, unique and exactly on UTC hours.
    Missing timestamps/NaNs remain missing. A window requires all its values;
    change features require the two endpoints, not intermediate observations.
    Extra past/future rows may exist in a day-sized response but are never used.
    """
    stamp, cutoff = _utc(target), history_end(target)
    for name, expected in INPUT_UNITS.items():
        if weather._normalise_unit(units.get(name)) != weather._normalise_unit(expected):
            raise weather.WeatherResponseError(f"Thermal-memory unit mismatch for {name}: expected {expected!r}.")
    if not {"hour_ending_utc", *INPUT_UNITS}.issubset(history.columns):
        raise ValueError("History must contain hour_ending_utc and all three source variables.")
    times = pd.DatetimeIndex([_utc(value) for value in history.hour_ending_utc], tz="UTC")
    if times.duplicated().any() or not times.equals(times.floor("h")):
        raise ValueError("History timestamps must be unique whole UTC hours.")
    table = history.loc[:, list(INPUT_UNITS)].copy()
    table.index = times
    for name in INPUT_UNITS:
        table[name] = pd.to_numeric(table[name], errors="raise").astype(float)
        if np.isinf(table[name]).any():
            raise weather.WeatherResponseError(f"Infinite history value in {name}.")
    # Invalid physical flux values count as missing, not zero-night energy.
    table.loc[table.shortwave_down_w_m2.lt(0), "shortwave_down_w_m2"] = np.nan
    table.loc[table.era5_longwave_down_w_m2.le(0), "era5_longwave_down_w_m2"] = np.nan
    output = {
        "memory_history_end_utc": cutoff,
        "memory_interval_start_utc": cutoff - HISTORY_HOURS * HOUR,
        "memory_excluded_partial_hour_minutes": (stamp - cutoff).total_seconds() / 60,
        "memory_air_history_source": "ERA5 background air; independent of current station/manual temperature",
        "memory_future_rows_excluded": int((times > cutoff).sum()),
    }

    def values(column, hours, prefix):
        expected = pd.date_range(cutoff - (hours - 1) * HOUR, cutoff, freq="h")
        result = table[column].reindex(expected)
        count = int(result.notna().sum())
        output[prefix + "_valid_hours"] = count
        output[prefix + "_coverage_fraction"] = count / hours
        output[prefix + "_status"] = "complete" if count == hours else "missing_history"
        return result if count == hours else None

    for hours in (6, 12, 24):
        found = values("shortwave_down_w_m2", hours, f"memory_shortwave_{hours}h")
        output[f"memory_shortwave_energy_{hours}h_j_m2"] = float(found.sum() * 3600) if found is not None else np.nan
    for hours in (6, 24):
        found = values("era5_longwave_down_w_m2", hours, f"memory_longwave_{hours}h")
        output[f"memory_longwave_mean_{hours}h_w_m2"] = float(found.mean()) if found is not None else np.nan
        found = values("background_air_temperature_c", hours, f"memory_air_{hours}h")
        output[f"memory_air_mean_{hours}h_c"] = float(found.mean()) if found is not None else np.nan
        output[f"memory_air_range_{hours}h_c"] = float(found.max() - found.min()) if found is not None else np.nan
    for hours in (3, 6):
        found = table.background_air_temperature_c.reindex([cutoff - hours * HOUR, cutoff])
        count = int(found.notna().sum())
        prefix = f"memory_air_change_{hours}h"
        output[prefix + "_valid_endpoints"] = count
        output[prefix + "_coverage_fraction"] = count / 2
        output[prefix + "_status"] = "complete" if count == 2 else "missing_history"
        output[prefix + "_c"] = float(found.iloc[1] - found.iloc[0]) if count == 2 else np.nan
    output["memory_status"] = "complete" if all(np.isfinite(output[name]) for name in FEATURE_UNITS) else "missing_history"
    return output


def _plan(samples, max_hours, max_weather_requests, max_samples, cache_budget):
    if len(samples) < 1 or len(samples) > max_samples:
        raise ValueError(f"Thermal memory accepts 1–{max_samples} rows per bounded call.")
    if not {"datetime_utc", "latitude", "longitude"}.issubset(samples.columns):
        raise ValueError("Expected datetime_utc, latitude and longitude.")
    if any(str(name).startswith("memory_") for name in samples.columns):
        raise ValueError("Input already contains memory features; refusing to overwrite them.")
    if min(max_hours, max_weather_requests, cache_budget) < 1:
        raise ValueError("Retrieval caps must be positive.")
    requests_by_row, required_hours, weather_groups, radiation_rows = [], set(), {}, set()
    for position, (_, row) in enumerate(samples.iterrows()):
        target = _utc(row.datetime_utc)
        lat, lon = float(row.latitude), float(row.longitude)
        if not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid WGS84 history coordinates.")
        cutoff = history_end(target)
        hours = pd.date_range(cutoff - (HISTORY_HOURS - 1) * HOUR, cutoff, freq="h")
        required_hours.update(hours)
        wlat, wlon = float(np.round(lat / .25) * .25), float(np.round(lon / .25) * .25)
        key = (wlat, wlon, cutoff.strftime("%Y-%m-%d"))
        weather_groups.setdefault(key, []).append(position)
        # Separate native grid sampling is preserved for each radiation point.
        r, c = radiation._grid_indices([lat], [lon])
        rlat, rlon = float(90 - r[0] / 4), float(((c[0] / 4 + 180) % 360) - 180)
        radiation_rows.update((hour, rlat, rlon) for hour in hours)
        requests_by_row.append({"position": position, "target": target, "cutoff": cutoff,
                                "weather_key": key, "radiation_coordinate": (rlat, rlon)})
    if len(required_hours) > max_hours:
        raise ValueError(f"History needs {len(required_hours)} global radiation hours, exceeding cap {max_hours}; no source reads made.")
    if len(weather_groups) > max_weather_requests:
        raise ValueError(f"History needs {len(weather_groups)} weather requests, exceeding cap {max_weather_requests}; no source reads made.")
    # Existing radiation adapter fetches snow as well as longwave. Account for
    # both even though snow is not one of this module's eleven predictors.
    upper_bound = (len(required_hours) * len(radiation.VARIABLES) * radiation.MAX_OBJECT_BYTES
                   + 4 * radiation.MAX_METADATA_BYTES + len(weather_groups) * WEATHER_RESPONSE_RESERVE_BYTES)
    if upper_bound > cache_budget:
        raise ValueError(f"Conservative retrieval reserve {upper_bound} bytes exceeds cache budget {cache_budget}; no source reads made.")
    return requests_by_row, weather_groups, sorted(radiation_rows), {
        "rows": len(samples), "radiation_hours": len(required_hours), "weather_requests": len(weather_groups),
        "maximum_radiation_hours": max_hours, "maximum_weather_requests": max_weather_requests,
        "conservative_retrieval_reserve_bytes": upper_bound, "additional_cache_target_bytes": cache_budget,
        "weather_response_reserve_bytes_each": WEATHER_RESPONSE_RESERVE_BYTES,
        "weather_reserve_is_estimate_not_streaming_limit": True,
    }


def add_thermal_memory(samples: pd.DataFrame, cache: str | Path, *, max_hours: int = 24,
                       max_weather_requests: int = 4, max_samples: int = 256,
                       cache_budget_bytes: int = MAX_ADDITIONAL_CACHE_BYTES,
                       allow_provisional: bool = False) -> tuple[pd.DataFrame, dict]:
    """Append research features while preserving row/index identity and input air.

    Cache follows production adapters: cache/weather and cache/radiation/arco-era5-v3.
    All limits are checked before retrieval. Missing source data is explicit;
    unit/schema contract errors are not silently repaired. No date substitution.
    """
    rows, groups, radiation_rows, plan = _plan(samples, max_hours, max_weather_requests, max_samples, cache_budget_bytes)
    cache = Path(cache)
    report = {
        "version": VERSION, "research_only": True, "not_a_temperature_prediction": True,
        "feature_units": FEATURE_UNITS, "documentation": DOCUMENTATION, "plan": plan,
        "temporal_rule": "Complete hourly intervals ending no later than floor(requested UTC hour).",
        "missingness_rule": "Full windows required; no interpolation, partial rescaling or imputation.",
        "air_statistic_rule": "Means/ranges of hourly instantaneous ERA5 background-air samples; changes use two endpoints.",
        "current_station_or_manual_override_applied_to_history": False,
        "availability_rule": "Reanalysis valid-time causality; archive publication occurs later, so this is not an as-issued realtime backtest.",
        "snow_retrieved_by_legacy_adapter_but_not_used": True,
        "weather": [], "errors": [],
    }
    weather_tables = {}
    for (lat, lon, date), positions in groups.items():
        cutoffs = [rows[position]["cutoff"] for position in positions]
        start, end = (min(cutoffs) - HISTORY_HOURS * HOUR).floor("D"), max(cutoffs).floor("D")
        entry = {"requested_latitude": lat, "requested_longitude": lon,
                 "requested_start_date": start.date().isoformat(), "requested_end_date": end.date().isoformat()}
        try:
            table, record = weather.fetch_archive(lat, lon, start, end, cache)
            # Preserve the exact validated response identity without duplicating
            # all fetched future hours into the feature manifest.
            payload = json.dumps(record["response"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            entry.update({"status": "available", "source": record.get("source", weather.URL),
                          "retrieved_utc": record.get("retrieved_utc"), "request": record.get("request"),
                          "response_sha256": hashlib.sha256(payload).hexdigest(),
                          "hourly_units": record["response"].get("hourly_units"), "validation": record.get("validation"),
                          "grid_latitude": record["response"].get("latitude"), "grid_longitude": record["response"].get("longitude"),
                          "weather_cache_keys": sorted(table.weather_cache_key.dropna().unique().tolist()) if "weather_cache_key" in table else []})
            weather_tables[(lat, lon, date)] = table.rename(columns={"weather_datetime_utc": "hour_ending_utc"})
        except weather.WeatherResponseError:
            raise
        except (requests.RequestException, OSError) as exc:
            entry.update(status="source_unavailable", error=f"{type(exc).__name__}: {exc}")
            report["errors"].append(entry["error"])
            weather_tables[(lat, lon, date)] = pd.DataFrame(columns=["hour_ending_utc", "shortwave_down_w_m2", "background_air_temperature_c"])
        report["weather"].append(entry)
    rad_inputs = pd.DataFrame(radiation_rows, columns=["datetime_utc", "latitude", "longitude"])
    rad = radiation.add_radiation(rad_inputs, cache / "radiation", max_hours=max_hours, allow_provisional=allow_provisional)
    report["radiation"] = rad.attrs.get("radiation_context", {})
    report["errors"].extend(report["radiation"].get("errors", []))
    result_rows = []
    for request in rows:
        table = weather_tables[request["weather_key"]]
        wanted = pd.date_range(request["cutoff"] - (HISTORY_HOURS - 1) * HOUR, request["cutoff"], freq="h")
        # Reindex exact UTC timestamps. Never use a nearest future sample, a
        # shifted rolling window, or values from a different requested location.
        table = table.set_index("hour_ending_utc").reindex(wanted)
        rlat, rlon = request["radiation_coordinate"]
        rt = rad.loc[rad.latitude.eq(rlat) & rad.longitude.eq(rlon)].set_index("datetime_utc").reindex(wanted)
        history = pd.DataFrame({"hour_ending_utc": wanted,
                                "shortwave_down_w_m2": table.shortwave_down_w_m2.to_numpy(),
                                "background_air_temperature_c": table.background_air_temperature_c.to_numpy(),
                                "era5_longwave_down_w_m2": rt.era5_longwave_down_w_m2.to_numpy()})
        result = aggregate_history(history, request["target"], units=INPUT_UNITS)
        result["memory_weather_query_latitude"], result["memory_weather_query_longitude"] = request["weather_key"][:2]
        result["memory_radiation_grid_latitude"], result["memory_radiation_grid_longitude"] = rlat, rlon
        result["memory_longwave_source_status"] = ";".join(sorted(set(rt.era5_longwave_down_w_m2_status.fillna("missing_hour").astype(str))))
        result_rows.append(result)
    output = samples.copy()
    additions = pd.DataFrame(result_rows)
    for name in additions:
        output[name] = additions[name].to_numpy()
    report["complete_rows"] = int(output.memory_status.eq("complete").sum())
    report["feature_nonmissing_rows"] = {name: int(output[name].notna().sum()) for name in FEATURE_UNITS}
    report["radiation_downloaded_bytes"] = report["radiation"].get("downloaded_bytes", 0)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datetime", required=True, help="Explicit UTC or offset-aware acquisition time")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--output", required=True, help="New research directory; never overwrites a run")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    sample = pd.DataFrame([{"datetime_utc": _utc(args.datetime), "latitude": args.latitude, "longitude": args.longitude}])
    if args.plan_only:
        print(json.dumps(_plan(sample, 24, 1, 1, MAX_ADDITIONAL_CACHE_BYTES)[-1], indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result, report = add_thermal_memory(sample, args.cache, max_weather_requests=1)
    report["runtime"] = {"wall_seconds": time.monotonic() - started,
                         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    result.to_parquet(output / "features.parquet", index=False)
    report["feature_table_sha256"] = hashlib.sha256((output / "features.parquet").read_bytes()).hexdigest()
    (output / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    result.to_json(output / "features.json", orient="records", date_format="iso", indent=2)
    print(json.dumps({"complete_rows": report["complete_rows"], "features": len(FEATURE_UNITS),
                      "radiation_downloaded_bytes": report["radiation_downloaded_bytes"], "errors": report["errors"]}, indent=2))


if __name__ == "__main__":
    main()
