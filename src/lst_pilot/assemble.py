"""Join independent surface descriptors, ERA5 context, and observed station air.

Each stage writes a resumable parquet checkpoint and an audit JSON. All work is
intended for the remote server. Station residuals correct the coarse background;
missing/distant/stale stations are explicitly flagged, never presented as measured.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version as package_version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from .context import add_context
from .satellite import region_bbox
from .stations import station_inventory, nearby_stations, station_year_available, download_station_year, parse_station_file
from .weather import WeatherResponseError, enrich_weather


PROCESSING_VERSION = "assembly-v2-explicit-provenance"
EXCLUDED_AIR_STATION_IDS = frozenset({"USW00094044", "USW00054918"})
EXCLUSION_REASON = "SURFRAD reference facility: 10 m air measurement, excluded from nominal 2 m station inputs and reference independence."


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def assembly_signature(input_path, areas_path, options):
    """Bind every resumable stage to its inputs, options and processing code."""
    source = Path(__file__).parent
    specification = {
        "processing_version": PROCESSING_VERSION,
        "input_sha256": file_sha256(input_path),
        "areas_sha256": file_sha256(areas_path),
        "options": options,
        "excluded_air_station_ids": sorted(EXCLUDED_AIR_STATION_IDS),
        "processing_source_sha256": {name: file_sha256(source / name) for name in ("assemble.py", "context.py", "weather.py", "stations.py", "satellite.py", "terrain.py")},
        "dependency_versions": {name: package_version(name) for name in ("pandas", "numpy", "rasterio", "pvlib", "pyproj")},
    }
    digest = sha256(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "specification": specification}


def guard_checkpoints(output, signature):
    output = Path(output)
    state_path = output / "assembly_fingerprint.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("sha256") != signature["sha256"]:
            raise ValueError("Assembly input, processing options or source version changed; select a new output directory to preserve checkpoints.")
    elif any((output / name).exists() for name in ("input_fingerprint.txt", "context.parquet", "weather.parquet", "training_data.parquet")):
        raise ValueError("Legacy/unversioned assembly checkpoints cannot be reused safely; select a new output directory.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".partial")
        temporary.write_text(json.dumps(signature, indent=2) + "\n")
        temporary.replace(state_path)


def normalise_terrain_columns(data):
    """Keep explicit source units while exposing the model's canonical names."""
    result = data.copy()
    for source, feature in (("elevation_m", "elevation"), ("slope_deg", "slope")):
        if source not in result:
            continue
        values = pd.to_numeric(result[source], errors="raise")
        if feature in result and not np.allclose(values, pd.to_numeric(result[feature], errors="raise"), equal_nan=True):
            raise ValueError(f"Conflicting terrain columns {source!r} and {feature!r}.")
        result[feature] = values
    return result


def select_land_samples(source):
    """Conservative QA selection, not a comprehensive land/coastline mask."""
    if "water_fraction" not in source:
        raise ValueError("Satellite sample water_fraction is required for the declared land-only selection.")
    fraction = pd.to_numeric(source["water_fraction"], errors="raise")
    if (fraction.notna() & ~fraction.between(0, 1)).any():
        raise ValueError("water_fraction must be between 0 and 1 or missing.")
    keep = fraction.eq(0)
    audit = {"criterion": "QA water_fraction == 0; conservative sampled land-cell selection",
             "input_rows": len(source), "kept_rows": int(keep.sum()), "removed_rows": int((~keep).sum()),
             "missing_water_fraction_rows": int(fraction.isna().sum()),
             "limitation": "Mixed shore/water cells and unknown QA are removed; this is not a comprehensive global coastline mask. Inland-water prediction is deferred."}
    return source.loc[keep].copy(), audit


def station_series(station_id, years, cache):
    frames = []
    for year in years:
        raw = download_station_year(station_id, int(year), cache)
        provenance = {"schema_version": 1, "raw_sha256": file_sha256(raw),
                      "parser_source_sha256": file_sha256(Path(__file__).with_name("stations.py")),
                      "keep_flags": True}
        parser_key = sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
        parsed = Path(cache) / str(year) / f"{station_id}_parsed_{parser_key}.parquet"
        if parsed.exists():
            frame = pd.read_parquet(parsed)
        else:
            frame = parse_station_file(raw)
            parsed.parent.mkdir(parents=True, exist_ok=True)
            temporary = parsed.with_suffix(".partial.parquet")
            frame.to_parquet(temporary, index=False)
            temporary.replace(parsed)
            parsed.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.dropna(subset=["air_temperature_c"]).sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def attach_stations(data, areas, cache_dir, stations_per_region=2, max_distance_km=100, max_age_minutes=90):
    if stations_per_region < 0 or max_distance_km <= 0 or max_age_minutes <= 0:
        raise ValueError("Station count must be nonnegative; distance and age limits must be positive.")
    cache = Path(cache_dir) / "stations"
    inventory = station_inventory(cache)
    geod = Geod(ellps="WGS84")
    outputs, audit = [], []
    area_map = {a["id"]: a for a in areas}
    for region_id, group in data.groupby("region_id", sort=True):
        group = group.copy().reset_index(drop=True)
        area = area_map[region_id]
        # A backward match just after New Year may need December's reports.
        years = sorted(set(group.datetime_utc.dt.year) | set((group.datetime_utc-pd.Timedelta(minutes=max_age_minutes)).dt.year))
        candidates = nearby_stations(inventory, tuple(region_bbox(area)), max_distance_km=max_distance_km, limit=16)
        for _, excluded in candidates.loc[candidates.station_id.isin(EXCLUDED_AIR_STATION_IDS)].iterrows():
            audit.append({"region_id": region_id, "station_id": excluded.station_id, "status": "excluded_reference_station", "reason": EXCLUSION_REASON})
        candidates = candidates.loc[~candidates.station_id.isin(EXCLUDED_AIR_STATION_IDS)].copy()
        # ICAO reporting stations are useful archive candidates, but still need
        # explicit year/QC checks. Geographic inventory presence is insufficient.
        candidates["_has_icao"] = candidates.icao.astype(str).str.len().eq(4)
        candidates = candidates.sort_values(["_has_icao", "distance_to_center_km"], ascending=[False, True])
        options = []
        for _, station in candidates.iterrows():
            if len(options) >= stations_per_region:
                break
            entry = {"region_id":region_id, "station_id":station.station_id, "name":station["name"], "years":list(map(int,years))}
            try:
                available_years = [int(y) for y in years if station_year_available(station.station_id,int(y))["available"]]
                if not available_years:
                    entry["status"] = "no_requested_years"
                    audit.append(entry)
                    continue
                series = station_series(station.station_id, available_years, cache)
                observations = series[["timestamp_utc","air_temperature_c"]].rename(columns={"air_temperature_c":"observed_station_air_temperature_c"})
                matched = pd.merge_asof(group[["datetime_utc"]].drop_duplicates().sort_values("datetime_utc"), observations,
                                        left_on="datetime_utc",right_on="timestamp_utc",direction="backward",tolerance=pd.Timedelta(minutes=max_age_minutes))
                matched = matched.dropna(subset=["observed_station_air_temperature_c"]).drop_duplicates("datetime_utc")
                if matched.empty:
                    entry["status"] = "no_timely_usable_observations"
                    audit.append(entry)
                    continue
                # Request background weather at the station observation time,
                # so the correction compares measured and background air fairly.
                query = matched[["timestamp_utc"]].drop_duplicates().rename(columns={"timestamp_utc":"datetime_utc"})
                query["latitude"], query["longitude"] = station.latitude, station.longitude
                background, _ = enrich_weather(query, cache_dir, max_requests=100)
                background = background[["datetime_utc","background_air_temperature_c"]].rename(columns={"datetime_utc":"timestamp_utc","background_air_temperature_c":"station_background_air_temperature_c"})
                matched = matched.merge(background,on="timestamp_utc",how="left",validate="many_to_one")
                joined = group[["datetime_utc","latitude","longitude","background_air_temperature_c"]].merge(matched,on="datetime_utc",how="left",validate="many_to_one")
                _, _, distance = geod.inv(joined.longitude.to_numpy(),joined.latitude.to_numpy(),
                                         np.full(len(joined),float(station.longitude)),np.full(len(joined),float(station.latitude)))
                joined["station_distance_km"] = np.asarray(distance) / 1000
                joined["station_age_minutes"] = (joined.datetime_utc-joined.timestamp_utc).dt.total_seconds()/60
                joined["station_id"] = station.station_id
                joined["station_elevation_m"] = station.elevation_m
                joined["station_air_correction_c"] = joined.observed_station_air_temperature_c-joined.station_background_air_temperature_c
                # Outlier screen protects against unmatched reports/units; this
                # is not a statistical guarantee and the rejected count is saved.
                valid = joined.station_air_correction_c.abs().le(20) & joined.station_distance_km.le(max_distance_km) & joined.station_age_minutes.between(0, max_age_minutes)
                joined.loc[~valid,"station_air_correction_c"] = np.nan
                joined["air_temperature_c"] = joined.background_air_temperature_c+joined.station_air_correction_c
                entry.update(status="usable_candidate" if valid.any() else "no_usable_corrections",matched_timestamps=len(matched),available_years=available_years,
                             candidate_rows=int(valid.sum()),rejected_rows=int((~valid).sum()),qc_usable_reports=len(series))
                if valid.any():
                    options.append(joined)
            except WeatherResponseError:
                # A unit/temperature contract failure must not silently become
                # an apparently valid station-free training dataset.
                raise
            except Exception as error:
                entry.update(status="failed",error=f"{type(error).__name__}: {str(error)[:300]}")
            audit.append(entry)
            print("station",entry,flush=True)
        group["air_temperature_c"] = group.background_air_temperature_c
        group["air_temperature_source"] = "ERA5_only_no_timely_station"
        group["station_id"] = ""
        for col in ("station_distance_km","station_age_minutes","station_elevation_m","station_air_correction_c","observed_station_air_temperature_c"):
            group[col] = np.nan
        best_distance = np.full(len(group),np.inf)
        for option in options:
            # option preserves original group row order through the left merge.
            use = option.air_temperature_c.notna().to_numpy() & (option.station_distance_km.to_numpy()<best_distance)
            for col in ("air_temperature_c","station_id","station_distance_km","station_age_minutes","station_elevation_m","station_air_correction_c","observed_station_air_temperature_c"):
                group.loc[use,col] = option.loc[use,col].to_numpy()
            group.loc[use,"air_temperature_source"] = "observed_station_residual_plus_ERA5_spatial_background"
            best_distance[use] = option.loc[use,"station_distance_km"].to_numpy()
        outputs.append(group)
    return pd.concat(outputs,ignore_index=True),audit


def assemble(input_path, output_dir, areas_path, cache_dir, max_weather_requests=200, stations_per_region=2, terrain=False, max_station_distance_km=100, max_station_age_minutes=90):
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    areas = json.loads(Path(areas_path).read_text())["areas"]
    source = pd.read_parquet(input_path)
    if source.empty:
        raise ValueError("No satellite observations to assemble")
    source, land_audit = select_land_samples(source)
    if source.empty:
        raise ValueError("No land-only satellite observations remain after QA water_fraction == 0 selection.")
    source["datetime_utc"] = pd.to_datetime(source.datetime_utc,utc=True)
    source["label_source"] = "USGS Landsat C2 L2 via Planetary Computer"
    source["source_scene_id"] = source.scene_id
    metadata = {"input_path":str(Path(input_path).resolve()),"source_rows":len(source),
                "scope":"clear-sky daytime training labels; all-weather performance is not established",
                "land_selection":land_audit,
                "optical_feature_timing":"Same-scene optical descriptors support a retrospective experiment, not proof of real-time feature availability. QA snow_fraction is retained for audit only and excluded from model predictors.",
                "station_method":f"ERA5 pixel background plus nearest timely station minus ERA5 station background; max{max_station_age_minutes}min/max{max_station_distance_km}km; missing observations explicitly fall back to ERA5",
                "weather_lags":"Retrospective ERA5 background history, not observed station history or a real-time availability claim",
                "excluded_air_station_ids":sorted(EXCLUDED_AIR_STATION_IDS),"station_exclusion_reason":EXCLUSION_REASON,
                "terrain_model_units":{"elevation":"metres, DSM above EGM2008", "slope":"degrees"}}
    options = {"terrain": bool(terrain), "max_weather_requests": int(max_weather_requests),
               "stations_per_region": int(stations_per_region), "max_station_distance_km": float(max_station_distance_km),
               "max_station_age_minutes": float(max_station_age_minutes),
               "land_selection": "qa_water_fraction_exactly_zero_v1"}
    signature = assembly_signature(input_path, areas_path, options)
    guard_checkpoints(output, signature)
    context_path,weather_path,final_path = [output/p for p in ("context.parquet","weather.parquet","training_data.parquet")]
    if context_path.exists():
        data = pd.read_parquet(context_path)
    else:
        data = add_context(source)
        if terrain:
            from .terrain import add_terrain
            data = add_terrain(data,cache_dir)
        data = normalise_terrain_columns(data)
        data.to_parquet(context_path,index=False)
    if weather_path.exists():
        data = pd.read_parquet(weather_path)
    else:
        data,weather_audit = enrich_weather(data,cache_dir,max_requests=max_weather_requests)
        data.to_parquet(weather_path,index=False)
        (output/"weather_audit.json").write_text(json.dumps(weather_audit,indent=2))
    if final_path.exists():
        data = pd.read_parquet(final_path)
    else:
        data,station_audit = attach_stations(data,areas,cache_dir,stations_per_region=stations_per_region,
                                            max_distance_km=max_station_distance_km,max_age_minutes=max_station_age_minutes)
        data.to_parquet(final_path,index=False)
        (output/"station_audit.json").write_text(json.dumps(station_audit,indent=2,default=str))
    metadata.update(rows=len(data),regions=data.region_id.value_counts().to_dict(),
                    climate_classes=data.climate_class.value_counts().to_dict(),
                    air_temperature_sources=data.air_temperature_source.value_counts().to_dict(),
                    missing_air_temperature=int(data.air_temperature_c.isna().sum()),
                    unique_region_dates=int(data.assign(day=data.datetime_utc.dt.date).groupby(["region_id","day"]).ngroups),
                    features_available=list(data.columns),input_sha256=signature["specification"]["input_sha256"],
                    assembly_fingerprint=signature["sha256"],processing_options=options,
                    processing_version=PROCESSING_VERSION)
    (output/"assembly_summary.json").write_text(json.dumps(metadata,indent=2,default=str))
    print(json.dumps(metadata,indent=2,default=str),flush=True)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--areas",default="pilot/areas_resolved.json")
    parser.add_argument("--cache",default="cache")
    parser.add_argument("--max-weather-requests",type=int,default=200)
    parser.add_argument("--stations-per-region",type=int,default=2)
    parser.add_argument("--terrain",action="store_true")
    parser.add_argument("--max-station-distance-km",type=float,default=100)
    parser.add_argument("--max-station-age-minutes",type=float,default=90)
    args=parser.parse_args()
    assemble(args.input,args.output,args.areas,args.cache,args.max_weather_requests,args.stations_per_region,args.terrain,args.max_station_distance_km,args.max_station_age_minutes)


if __name__=="__main__":
    main()
