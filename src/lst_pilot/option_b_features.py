"""Research feature pairing for fixed-grid LST samples; never fits or serves a model.

Every output row retains its sample_id. Labels and their quality fields are audit
columns only: optical search, masks, composites, land classes and weather never
consult them. Source valid times are causal; later archive publication means this
is a retrospective experiment, not a forecast available as issued.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import transform_bounds

from . import assemble, radiation, raster, satellite, thermal_memory, weather

VERSION = "option-b-features-v1"
BASE_FEATURES = (
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
COVER_CLASSES = {"tree": 10, "grass": 30, "crop": 40, "built": 50, "bare": 60}
COVER_FEATURES = tuple(f"worldcover_{name}_class_fraction" for name in COVER_CLASSES)
MEMORY_FEATURES = tuple(thermal_memory.FEATURE_UNITS)
TERRAIN_FEATURES = ("elevation", "slope", "aspect_sin", "aspect_cos", "terrain_relief_300m")
SR_FIELDS = tuple(f"sr_{band}" for band in satellite.SR_BANDS)
REQUIRED = ("sample_id", "region_id", "datetime_utc", "latitude", "longitude", "grid_row", "grid_col",
            "epsg", "lst_c", "label_product", "acquisition_id")
TILE_SIZE = 128
WORLD_COVER_END = pd.Timestamp("2020-12-31T23:59:59Z")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=raster._json, allow_nan=False) + "\n")
    temp.replace(path)


def validate_samples(frame, areas):
    missing = set(REQUIRED) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Nonempty sample table required; missing columns: {sorted(missing)}")
    data = frame.copy()
    for field in ("sample_id", "region_id", "acquisition_id", "label_product"):
        if data[field].isna().any() or data[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identity field: {field}")
    if data.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique across the entire input.")
    times = [thermal_memory._utc(value) for value in data.datetime_utc]
    data["datetime_utc"] = pd.DatetimeIndex(times)
    for name in ("grid_row", "grid_col", "epsg"):
        values = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain finite integers.")
    for region_id, group in data.groupby("region_id", sort=False):
        if region_id not in areas:
            raise ValueError(f"Unknown pilot: {region_id}")
        area = areas[region_id]
        height, width = area["grid_shape"]
        if not (group.epsg.eq(area["epsg"]) & group.grid_row.between(0, height - 1)
                & group.grid_col.between(0, width - 1)).all():
            raise ValueError("Sample grid coordinates/EPSG differ from the fixed pilot grid.")
        left, _, _, top = area["extent_m"]
        x, y = Transformer.from_crs(4326, area["epsg"], always_xy=True).transform(group.longitude, group.latitude)
        distance = np.hypot(np.asarray(x) - (left + (group.grid_col.to_numpy() + .5) * 100),
                            np.asarray(y) - (top - (group.grid_row.to_numpy() + .5) * 100))
        if not np.isfinite(distance).all() or (distance > 2.).any():
            raise ValueError("Latitude/longitude must identify the fixed 100 m cell centre within 2 m.")
    return data.reset_index(drop=True)


def preserve_identity(before, after):
    """Validate one-to-one cardinality and restore input order after adapters."""
    if after.sample_id.duplicated().any() or len(before) != len(after) or set(before.sample_id) != set(after.sample_id):
        raise ValueError("An adapter lost, added, or duplicated sample_id values.")
    ordered = after.set_index("sample_id").loc[before.sample_id].reset_index()
    for field in REQUIRED:
        if field != "sample_id" and not before[field].reset_index(drop=True).equals(ordered[field]):
            raise ValueError(f"An adapter changed immutable sample field {field}.")
    return ordered


def tile_grid(area, row, col):
    """Tiles anchored to the pilot origin, independent of samples/ROI order."""
    row, col = int(row) // TILE_SIZE * TILE_SIZE, int(col) // TILE_SIZE * TILE_SIZE
    height = min(TILE_SIZE, area["grid_shape"][0] - row)
    width = min(TILE_SIZE, area["grid_shape"][1] - col)
    transform = from_origin(area["extent_m"][0] + col * 100, area["extent_m"][3] - row * 100, 100, 100)
    return raster.RasterGrid(area["epsg"], transform, height, width,
                             array_bounds(height, width, transform), None, None, None, row, col)


def choose_optical(items, target, lookback_days=32, max_scenes=16):
    """Only metadata independent of the temperature label determines inclusion."""
    target = thermal_memory._utc(target)
    lower = target - pd.Timedelta(days=lookback_days)
    unique = {}
    for item in items:
        props = item.get("properties", {})
        if item.get("collection") != "landsat-c2-l2" or props.get("platform") not in ("landsat-8", "landsat-9"):
            continue
        if props.get("landsat:collection_category") != "T1":
            continue
        if not set((*satellite.SR_BANDS, "qa_pixel", "qa_radsat")).issubset(item.get("assets", {})):
            continue
        timestamp = thermal_memory._utc(props.get("datetime"))
        if lower <= timestamp <= target:
            unique[item["id"]] = item
    # Deterministic latest observations, not cloud/temperature/error ranking.
    found = sorted(unique.values(), key=lambda item: (item["properties"]["datetime"], item["id"]), reverse=True)
    return found[:max_scenes], {"eligible_scenes": len(found), "selected_scenes": min(len(found), max_scenes),
                                "scene_cap_truncated": len(found) > max_scenes}


def search_optical(area, target, cache, lookback_days=32, max_scenes=16):
    target = thermal_memory._utc(target)
    request = {"collections": ["landsat-c2-l2"], "bbox": satellite.region_bbox(area), "limit": 100,
               "datetime": f"{(target-pd.Timedelta(days=lookback_days)).isoformat()}/{target.isoformat()}",
               "query": {"platform": {"in": ["landsat-8", "landsat-9"]},
                         "landsat:collection_category": {"eq": "T1"}}}
    path = Path(cache) / "option-b-stac" / (_hash(request) + ".json")
    if not path.exists():
        with satellite._session() as session:
            response = session.post(satellite.STAC_URL + "/search", json=request, timeout=(15, 90))
            response.raise_for_status()
            payload = response.json()
        if any(link.get("rel") == "next" for link in payload.get("links", [])):
            raise ValueError("Optical search exceeds the 100-item bound; no partial search accepted.")
        _write_json(path, payload)
    payload = json.loads(path.read_text())
    if any(link.get("rel") == "next" for link in payload.get("links", [])):
        raise ValueError("Incomplete optical-search cache.")
    items, audit = choose_optical(payload.get("features", []), target, lookback_days, max_scenes)
    return items, {**audit, "request": request, "source_snapshot_sha256": _sha(path), "cache_path": str(path),
                   "thermal_assets_requested": False, "selection": "latest valid-time SR observations, T1 L8/9; no LST quality selection"}


def composite_optical(observations, timestamps, target, shape):
    """Median reflectance per band, then descriptors; all six bands share QA."""
    target = thermal_memory._utc(target)
    if any(thermal_memory._utc(value) > target for value in timestamps):
        raise ValueError("Future optical observations are forbidden.")
    if not observations:
        out = {name: np.full(shape, np.nan, np.float32) for name in (*satellite.SURFACE_FEATURES, "water_fraction")}
        out.update(optical_observation_count=np.zeros(shape, np.int16),
                   optical_earliest_epoch_s=np.full(shape, np.nan), optical_latest_epoch_s=np.full(shape, np.nan))
        return out
    accepted = np.stack([np.logical_and.reduce([np.isfinite(obs[name]) for name in SR_FIELDS])
                         & (obs["optical_valid_fraction"] >= .8) for obs in observations])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reflectance = {band: np.nanmedian(np.where(accepted, np.stack([obs[f"sr_{band}"] for obs in observations]), np.nan), axis=0)
                       for band in satellite.SR_BANDS}
        water = np.nanmedian(np.where(accepted, np.stack([obs["water_fraction"] for obs in observations]), np.nan), axis=0)
    result = satellite.surface_descriptors(reflectance)
    result["water_fraction"] = water
    times = np.asarray([thermal_memory._utc(t).timestamp() for t in timestamps])[:, None, None]
    count = accepted.sum(axis=0)
    result["optical_observation_count"] = count.astype(np.int16)
    result["optical_earliest_epoch_s"] = np.where(count, np.where(accepted, times, np.inf).min(axis=0), np.nan)
    result["optical_latest_epoch_s"] = np.where(count, np.where(accepted, times, -np.inf).max(axis=0), np.nan)
    return result


def cover_arrays(classes, transform, crs, grid):
    classified = np.isin(classes, (*raster.LAND_CLASSES, 80))
    support = raster._warp(classified, transform, crs, grid)
    result = {f"worldcover_{name}_class_fraction": np.where(support >= .95, raster._warp(classes == code, transform, crs, grid), np.nan)
              for name, code in COVER_CLASSES.items()}
    result["worldcover_classified_fraction"] = support
    result["worldcover_land_fraction"] = raster._warp(np.isin(classes, raster.LAND_CLASSES), transform, crs, grid)
    result["worldcover_water_fraction"] = raster._warp(classes == 80, transform, crs, grid)
    return result


def read_cover(grid, cache):
    bbox = transform_bounds(f"EPSG:{grid.epsg}", 4326, *grid.bounds, densify_pts=21)
    items = raster._stac_tiles("esa-worldcover", bbox, cache, "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z")
    if any(item["properties"].get("esa_worldcover:product_version") != "1.0.0" for item in items):
        raise ValueError("Expected fixed WorldCover 2020 v100; other reference years are forbidden.")
    classes, transform, crs = raster._mosaic(items, "map", grid, 0, "uint8")
    result = cover_arrays(classes, transform, crs, grid)
    audit = {"dataset": "ESA WorldCover 2020 v100", "source_valid_time_end_utc": WORLD_COVER_END.isoformat(),
             "source_stac_sha256": _hash(items), "items": [item["id"] for item in items],
             "urls": [item["assets"]["map"]["href"].split("?")[0] for item in items],
             "documentation": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/esa-worldcover",
             "method": "Area-average 10 m class indicators; >=95% classified support; other classes remain in denominator",
             "limitation": "Mapped class fractions, not physical canopy/building/impervious area; static 2020 snapshot published later",
             "attribution": "© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium"}
    return result, audit


def surface_group(frame):
    """Independent class dominance; other classes and weak dominance stay explicit."""
    values = frame.reindex(columns=COVER_FEATURES).to_numpy(dtype=float)
    complete = np.isfinite(values).all(axis=1)
    safe = np.where(np.isfinite(values), values, -1.)
    maximum = safe.max(axis=1)
    labels = np.asarray(list(COVER_CLASSES), object)[safe.argmax(axis=1)]
    labels[maximum < .5] = "other_or_mixed"
    labels[~complete] = "unknown"
    return labels


def _cached_tile(kind, grid, cache, source_identity, reader):
    specification = {"version": VERSION, "kind": kind, "epsg": grid.epsg, "bounds": grid.bounds,
                     "shape": grid.shape, "source": source_identity,
                     "raster_code_sha256": _sha(Path(raster.__file__)), "builder_code_sha256": _sha(Path(__file__))}
    path = Path(cache) / "option-b-tiles" / kind / (_hash(specification) + ".npz")
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        audit = json.loads(metadata_path.read_text())
        if audit["array_sha256"] != _sha(path):
            raise ValueError("Cached feature tile hash differs from its checkpoint.")
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}, audit
    values, audit = reader()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)
    audit = {**audit, "array_path": str(path), "array_sha256": _sha(path), "specification": specification}
    _write_json(metadata_path, audit)
    return values, audit


def _reuse_optical(data, lookback_days):
    required = (*SR_FIELDS, "water_fraction", "optical_source_datetime_utc", "optical_source_id")
    if not set(required).issubset(data):
        return False
    times = pd.Series([thermal_memory._utc(value) for value in data.optical_source_datetime_utc], index=data.index)
    age = (data.datetime_utc - times).dt.total_seconds() / 86400
    if not age.between(0, lookback_days).all():
        raise ValueError("Supplied optical predictors are future or outside the trailing 32-day interval.")
    return np.isfinite(data[list(SR_FIELDS)].to_numpy(float)).all() and data.optical_source_id.notna().all()


def append_surfaces(data, area, cache, *, preserve_existing_base=False, lookback_days=32, max_scenes=16):
    data = data.copy()
    target = data.datetime_utc.iloc[0]
    if not data.datetime_utc.eq(target).all():
        raise ValueError("Surface batch must contain one exact acquisition time.")
    if target <= WORLD_COVER_END:
        raise ValueError("WorldCover 2020 is not a past-only reference for this target.")
    audit = {"tiles": [], "errors": [], "optical_method": "preserved_legacy" if preserve_existing_base else "past_only_median_SR"}
    supplied_optical = False if preserve_existing_base else _reuse_optical(data, lookback_days)
    supplied_terrain = set(TERRAIN_FEATURES).issubset(data) and np.isfinite(data[list(TERRAIN_FEATURES)].to_numpy(float)).all()
    needs_optical = not preserve_existing_base and not supplied_optical
    scenes = []
    if needs_optical:
        scenes, audit["optical_search"] = search_optical(area, target, cache, lookback_days, max_scenes)
    if supplied_optical:
        for name, value in satellite.surface_descriptors({band: data[f"sr_{band}"].to_numpy() for band in satellite.SR_BANDS}).items():
            data[name] = value
        data["optical_observation_count"] = 1
        data["optical_earliest_source_utc"] = data.optical_source_datetime_utc
        data["optical_latest_source_utc"] = data.optical_source_datetime_utc
        data["optical_pairing_status"] = "supplied_independent_SR"
    for name in (*COVER_FEATURES, "worldcover_classified_fraction", "worldcover_land_fraction", "worldcover_water_fraction"):
        data[name] = np.nan
    data["worldcover_status"] = "unavailable"
    groups = data.groupby([data.grid_row // TILE_SIZE, data.grid_col // TILE_SIZE], sort=True)
    for _, part in groups:
        grid = tile_grid(area, part.grid_row.iloc[0], part.grid_col.iloc[0])
        rr = part.grid_row.to_numpy(int) - grid.row_offset
        cc = part.grid_col.to_numpy(int) - grid.col_offset
        entry = {"row_offset": grid.row_offset, "col_offset": grid.col_offset, "sample_count": len(part)}
        try:
            arrays, entry["worldcover"] = _cached_tile("worldcover2020", grid, cache, "2020-v100", lambda: read_cover(grid, cache))
            for name, values in arrays.items():
                data.loc[part.index, name] = values[rr, cc]
            data.loc[part.index, "worldcover_status"] = np.where(arrays["worldcover_classified_fraction"][rr, cc] >= .95,
                                                                  "complete", "insufficient_classified_support")
        except Exception as error:
            entry["worldcover_error"] = satellite._safe_error(error)
            audit["errors"].append({"stage": "worldcover", "error": entry["worldcover_error"]})
        if not preserve_existing_base and not supplied_terrain:
            arrays, entry["terrain"] = _cached_tile("terrain", grid, cache, "copdem-glo30", lambda: raster.read_terrain(grid, cache))
            for name, values in arrays.items():
                data.loc[part.index, name] = values[rr, cc]
        if needs_optical:
            observations, times, entry["optical_sources"] = [], [], []
            for scene in scenes:
                # Tiles outside a scene's footprint are allowed to have no support.
                try:
                    def read(scene=scene):
                        features, _, record = raster.read_optical(scene, grid, include_observed=False)
                        return features, record
                    arrays, record = _cached_tile("optical", grid, cache, _hash(scene), read)
                    observations.append(arrays)
                    times.append(scene["properties"]["datetime"])
                    entry["optical_sources"].append(record)
                except Exception as error:
                    entry["optical_sources"].append({"id": scene["id"], "status": "unavailable", "error": satellite._safe_error(error)})
            arrays = composite_optical(observations, times, target, grid.shape)
            for name, values in arrays.items():
                data.loc[part.index, name] = values[rr, cc]
        audit["tiles"].append(entry)
    if needs_optical:
        data["optical_earliest_source_utc"] = pd.to_datetime(data.pop("optical_earliest_epoch_s"), unit="s", utc=True)
        data["optical_latest_source_utc"] = pd.to_datetime(data.pop("optical_latest_epoch_s"), unit="s", utc=True)
        data["optical_pairing_status"] = np.where(data.optical_observation_count.gt(0), "complete_trailing_SR", "no_valid_prior_SR")
    if preserve_existing_base:
        data["optical_pairing_status"] = "legacy_joint_optical_label_mask"
        data["legacy_pairing_limitation"] = "Frozen original base values; original sampling jointly selected optical and thermal validity; partial legacy station provenance"
    else:
        data["optical_max_age_days"] = (data.datetime_utc - pd.to_datetime(data.optical_earliest_source_utc, utc=True)).dt.total_seconds() / 86400
    data["worldcover_source_valid_time_end_utc"] = WORLD_COVER_END
    data["weight_surface_group"] = surface_group(data)
    return data, audit


def append_memory(data, cache):
    # Both adapters use 0.25 degrees, with separate rounding rules at exact ties.
    # Deduplicate their joint native-grid identity, never merely one of the grids.
    r, c = radiation._grid_indices(data.latitude.to_numpy(), data.longitude.to_numpy())
    keys = pd.DataFrame({"wlat": np.round(data.latitude / .25), "wlon": np.round(data.longitude / .25),
                         "r": r, "c": c, "datetime_utc": data.datetime_utc}).reset_index(drop=True)
    keys["key"] = keys.groupby(list(keys.columns), sort=False).ngroup()
    representatives = data.reset_index(drop=True).loc[~keys.key.duplicated(), ["datetime_utc", "latitude", "longitude"]].copy()
    representatives["key"] = keys.loc[~keys.key.duplicated(), "key"].to_numpy()
    additions, report = thermal_memory.add_thermal_memory(representatives, cache, max_samples=256,
                                                          max_weather_requests=64, max_hours=24)
    fields = [name for name in additions if name.startswith("memory_")]
    paired = keys[["key"]].merge(additions[["key", *fields]], on="key", how="left", validate="many_to_one", sort=False)
    result = data.copy()
    for field in fields:
        result[field] = paired[field].to_numpy()
    report["sample_rows"] = len(data)
    report["unique_joint_weather_radiation_cells"] = len(representatives)
    return result, report


def _complete(data, names):
    if not set(names).issubset(data):
        return np.zeros(len(data), bool)
    numeric = [name for name in names if name != "climate_class"]
    complete = np.isfinite(data[numeric].to_numpy(float)).all(axis=1)
    if "climate_class" in names:
        complete &= data.climate_class.notna().to_numpy() & ~data.climate_class.astype(str).isin(["unknown", "__unknown__", ""]).to_numpy()
    return complete


def assemble_acquisition(data, areas, cache, *, preserve_existing_base=False, max_optical_scenes=16):
    original = data.copy()
    audit = {"version": VERSION, "region_id": str(data.region_id.iloc[0]), "acquisition_id": str(data.acquisition_id.iloc[0]),
             "datetime_utc": data.datetime_utc.iloc[0].isoformat(), "rows": len(data), "preserve_existing_base": preserve_existing_base}
    data, audit["surfaces"] = append_surfaces(data, areas[data.region_id.iloc[0]], cache,
                                             preserve_existing_base=preserve_existing_base, max_scenes=max_optical_scenes)
    data = preserve_identity(original, data)
    if not preserve_existing_base:
        data = preserve_identity(data, raster.add_raster_context(data))
        before = data
        data, audit["weather"] = weather.enrich_weather(data, cache, max_requests=64)
        data = preserve_identity(before, data)
        before = data
        data, audit["stations"] = assemble.attach_stations(data, list(areas.values()), cache, stations_per_region=2,
                                                          max_distance_km=100, max_age_minutes=90)
        data = preserve_identity(before, data)
        # Old adapter lacks raw report fields; preserve any supplied by newer adapters.
        if "station_observation_datetime_utc" not in data:
            data["station_observation_datetime_utc"] = data.datetime_utc - pd.to_timedelta(data.station_age_minutes, unit="m")
        data["station_pairing_provenance"] = "Backward QC-usable NOAA report; source timestamp derived from age where raw report metadata absent"
        data = radiation.add_radiation(data, Path(cache) / "radiation", max_hours=1)
        audit["radiation"] = data.attrs.get("radiation_context", {})
    data, audit["memory"] = append_memory(data, cache)
    data = preserve_identity(original, data)
    if preserve_existing_base:
        pd.testing.assert_frame_equal(original[list(BASE_FEATURES)], data[list(BASE_FEATURES)], check_exact=True)
    data["features_A_complete"] = _complete(data, BASE_FEATURES)
    data["features_B_complete"] = data.features_A_complete & _complete(data, MEMORY_FEATURES)
    data["features_C_complete"] = data.features_A_complete & _complete(data, COVER_FEATURES)
    data["features_D_complete"] = data.features_B_complete & data.features_C_complete
    data["station_pair_available"] = data.get("station_id", pd.Series("", index=data.index)).fillna("").ne("")
    data["feature_assembly_version"] = VERSION
    audit["completeness"] = {name: int(data[name].sum()) for name in ("features_A_complete", "features_B_complete", "features_C_complete", "features_D_complete", "station_pair_available")}
    audit["eligibility_note"] = "Completeness diagnostics only. No rows dropped or training eligibility assigned; root cohort protocol decides intersection."
    return data, audit


def build(input_path, output_dir, areas_path, cache, *, preserve_existing_base=False, max_optical_scenes=16, max_acquisitions=None):
    if not 1 <= max_optical_scenes <= 32:
        raise ValueError("Require 1–32 optical scenes per bounded acquisition.")
    output = Path(output_dir)
    areas = {area["id"]: area for area in json.loads(Path(areas_path).read_text())["areas"]}
    data = validate_samples(pd.read_parquet(input_path), areas)
    if preserve_existing_base and not set(BASE_FEATURES).issubset(data):
        raise ValueError("Legacy preservation requires all frozen base-40 fields.")
    sources = {Path(module.__file__).name: _sha(module.__file__) for module in
               (assemble, radiation, raster, satellite, thermal_memory, weather)}
    signature = {"version": VERSION, "input_sha256": _sha(input_path), "areas_sha256": _sha(areas_path),
                 "builder_sha256": _sha(__file__), "adapter_source_sha256": sources,
                 "preserve_existing_base": preserve_existing_base, "max_optical_scenes": max_optical_scenes,
                 "lookback_days": 32, "base_features": BASE_FEATURES, "cover_features": COVER_FEATURES,
                 "memory_features": MEMORY_FEATURES}
    fingerprint = _hash(signature)
    output.mkdir(parents=True, exist_ok=True)
    guard = output / "signature.json"
    if guard.exists() and json.loads(guard.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output checkpoints belong to different inputs, code, or options; choose a fresh output directory.")
    _write_json(guard, {"fingerprint": fingerprint, "specification": signature})
    parts, records = [], []
    groups = data.groupby(["region_id", "acquisition_id", "datetime_utc"], sort=True)
    started = time.monotonic()
    for number, (key, part) in enumerate(groups, 1):
        if max_acquisitions is not None and number > max_acquisitions:
            break
        checkpoint = output / "acquisitions" / (_hash(tuple(map(str, key))) + ".parquet")
        audit_path = checkpoint.with_suffix(".json")
        print(json.dumps({"acquisition": number, "total": groups.ngroups, "key": list(map(str, key)), "rows": len(part)}), flush=True)
        part = part.reset_index(drop=True)
        if checkpoint.exists() and audit_path.exists():
            report = json.loads(audit_path.read_text())
            if report["output_sha256"] != _sha(checkpoint):
                raise ValueError("Acquisition checkpoint hash mismatch.")
            result = preserve_identity(part, pd.read_parquet(checkpoint))
        else:
            result, report = assemble_acquisition(part, areas, cache, preserve_existing_base=preserve_existing_base,
                                                  max_optical_scenes=max_optical_scenes)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".tmp.parquet")
            result.to_parquet(temporary, index=False)
            temporary.replace(checkpoint)
            report["output_sha256"] = _sha(checkpoint)
            _write_json(audit_path, report)
        parts.append(result)
        records.append({"key": list(map(str, key)), "checkpoint": str(checkpoint), "audit_path": str(audit_path),
                        "output_sha256": report["output_sha256"], "audit_sha256": _sha(audit_path), "completeness": report["completeness"]})
    final = pd.concat(parts, ignore_index=True)
    subset = data[data.sample_id.isin(final.sample_id)]
    final = preserve_identity(subset.reset_index(drop=True), final)
    final_path = output / "features.parquet"
    final.to_parquet(final_path.with_suffix(".tmp.parquet"), index=False)
    final_path.with_suffix(".tmp.parquet").replace(final_path)
    manifest = {"version": VERSION, "fingerprint": fingerprint, "input_path": str(input_path), "output_path": str(final_path),
                "output_sha256": _sha(final_path), "input_rows": len(data), "output_rows": len(final), "acquisitions": records,
                "complete_input_processed": len(final) == len(data), "elapsed_seconds": time.monotonic() - started,
                "research_only": True, "model_fitted": False, "row_eligibility_assigned": False,
                "completeness": {name: int(final[name].sum()) for name in ("features_A_complete", "features_B_complete", "features_C_complete", "features_D_complete", "station_pair_available")},
                "valid_time_causal": True, "as_issued_realtime_backtest": False}
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "acquisitions"}, indent=2), flush=True)
    return final, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--areas", default="pilot/areas_resolved.json")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--preserve-existing-base", action="store_true")
    parser.add_argument("--max-optical-scenes", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int)
    args = parser.parse_args()
    satellite.configure_safe_logging()
    build(args.input, args.output_dir, args.areas, args.cache, preserve_existing_base=args.preserve_existing_base,
          max_optical_scenes=args.max_optical_scenes, max_acquisitions=args.max_acquisitions)


if __name__ == "__main__":
    main()
