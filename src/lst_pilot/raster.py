"""Bounded, real 100 m raster inference for the archived pilot website.

All source reads are COG windows or bounded weather queries. Optical predictors
never read thermal values; observed LST is an optional, separate comparison.
This retrospective daytime experiment is not an hourly all-weather product.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import reproject, transform_bounds, calculate_default_transform, Resampling
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as transform_geometry

from .satellite import (SR_BANDS, STAC_URL, qa_valid, scale_reflectance,
                        scale_temperature, surface_descriptors, _session,
                        configure_safe_logging)

VERSION = "pilot-raster-v3-daytime-only"
RESOLUTION = 100
MAX_PIXELS = 640000
MAX_AREA_KM2 = 6400
STATIC_TILE_SIZE = 128
FRAME_CHUNK_SIZE = 65536
MAX_WEATHER_REQUESTS = 64
BOUNDARY_TOLERANCE_M = 1.0
MAX_NATIVE_CELLS = 25_000_000
NODATA = -9999.0
LAND_CLASSES = (10, 20, 30, 40, 50, 60, 70, 90, 95, 100)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
                GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="90",
                GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_CACHEMAX=64 * 1024 * 1024,
                VSI_CACHE=False, GDAL_NUM_THREADS="1")


@dataclass
class RasterGrid:
    epsg: int
    transform: object
    height: int
    width: int
    bounds: tuple
    polygon: object
    polygon_wgs84: object
    inside: np.ndarray
    row_offset: int
    col_offset: int

    @property
    def shape(self):
        return self.height, self.width


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported provenance type: {type(value).__name__}")


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_grid(region, polygon):
    """Validate before any source reads; align the bounding box to the pilot grid.

    Polygon membership uses pixel centres. The complete bounding box, including
    outside-polygon cells, counts against the 640,000-cell computation limit.
    """
    if polygon.get("type") == "Feature":
        polygon = polygon["geometry"]
    geom = shape(polygon)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid:
        raise ValueError("Selection must be a valid, nonempty WGS84 Polygon or MultiPolygon.")
    bounds = np.asarray(geom.bounds)
    if not np.isfinite(bounds).all() or not (-180 <= bounds[0] <= bounds[2] <= 180 and -90 <= bounds[1] <= bounds[3] <= 90):
        raise ValueError("Selection has invalid WGS84 coordinates.")
    # None of the pilot areas crosses the antimeridian; bound densification work.
    if bounds[2] - bounds[0] > 2 or bounds[3] - bounds[1] > 1:
        raise ValueError("Selection is too large for a bounded pilot request.")
    from shapely import get_num_coordinates
    if get_num_coordinates(geom) > 10000:
        raise ValueError("Selection exceeds 10,000 input vertices.")
    projected = transform_geometry(Transformer.from_crs(4326, region["epsg"], always_xy=True).transform,
                                   geom.segmentize(.005))
    extent = tuple(map(float, region["extent_m"]))
    pilot_boundary = box(*extent)
    if not pilot_boundary.buffer(BOUNDARY_TOLERANCE_M).covers(projected):
        raise ValueError("Selection must lie fully inside the selected pilot area.")
    # Canonical pilot polygons have a densified WGS84 boundary. Straight WGS84
    # segments can bow sub-metre distances after projection; clip that tolerance
    # back to the exact pilot extent instead of allocating an extra pixel row.
    projected = projected.intersection(pilot_boundary)
    left, bottom, right, top = projected.bounds
    # Millimetre tolerance prevents round-trip CRS noise adding a complete cell.
    col0 = math.floor((left - extent[0] + .001) / RESOLUTION)
    col1 = math.ceil((right - extent[0] - .001) / RESOLUTION)
    row0 = math.floor((extent[3] - top + .001) / RESOLUTION)
    row1 = math.ceil((extent[3] - bottom - .001) / RESOLUTION)
    height, width = row1-row0, col1-col0
    if height < 1 or width < 1 or height * width > MAX_PIXELS:
        raise ValueError("Aligned bounding box must contain 1–640,000 pixels (at most 6,400 km² at 100 m).")
    transform = from_origin(extent[0]+col0*RESOLUTION, extent[3]-row0*RESOLUTION, RESOLUTION, RESOLUTION)
    inside = geometry_mask([mapping(projected)], (height, width), transform, invert=True, all_touched=False)
    if not inside.any():
        raise ValueError("Selection contains no 100 m pixel centres.")
    return RasterGrid(int(region["epsg"]), transform, height, width,
                      array_bounds(height, width, transform), projected, geom, inside, row0, col0)


def grid_tiles(grid, tile_size=STATIC_TILE_SIZE):
    """Partition an aligned grid exactly, without changing any pixel centres."""
    if tile_size < 1:
        raise ValueError("Tile size must be positive.")
    for row in range(0, grid.height, tile_size):
        for col in range(0, grid.width, tile_size):
            height, width = min(tile_size, grid.height-row), min(tile_size, grid.width-col)
            slices = (slice(row,row+height), slice(col,col+width))
            transform = grid.transform * rasterio.Affine.translation(col,row)
            tile = RasterGrid(grid.epsg, transform, height, width,
                              array_bounds(height,width,transform), grid.polygon, grid.polygon_wgs84,
                              grid.inside[slices] if grid.inside is not None else None,
                              grid.row_offset+row, grid.col_offset+col)
            yield slices, tile


def validate_time(scene, requested_datetime, mode):
    if mode not in ("observed", "experimental", "scenario"):
        raise ValueError("Mode must be observed, experimental or scenario.")
    target = pd.Timestamp(requested_datetime)
    source = pd.Timestamp(scene["properties"]["datetime"])
    if target.tzinfo is None or source.tzinfo is None or pd.isna(target) or pd.isna(source):
        raise ValueError("Scene and requested datetimes require explicit timezone offsets.")
    target, source = target.tz_convert("UTC"), source.tz_convert("UTC")
    if mode == "scenario":
        if not 1900 <= target.year <= 2100:
            raise ValueError("Reported-air scenarios accept dates from 1900 through 2100.")
        return target, source, float((target-source).total_seconds()/86400)
    if target.year not in (2021, 2022, 2023, 2024) or source.year not in (2021, 2022, 2023, 2024):
        raise ValueError("This pilot supports historical 2021–2024 dates only.")
    age = (target-source).total_seconds() / 86400
    if age < 0:
        raise ValueError("A future optical scene cannot be used as an input.")
    if mode == "observed" and target != source:
        raise ValueError("Observed mode requires the exact source-scene timestamp; choose experimental for another time.")
    if age > 90:
        raise ValueError("Optical source age exceeds the 90-day experimental limit.")
    return target, source, float(age)


def _warp(values, src_transform, src_crs, grid, nodata=None, resampling=Resampling.average):
    result = np.full(grid.shape, np.nan, np.float32)
    reproject(np.asarray(values, dtype=np.float32), result,
              src_transform=src_transform, src_crs=src_crs, src_nodata=nodata,
              dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", dst_nodata=np.nan,
              resampling=resampling, num_threads=1, tolerance=0.)
    return result


def aggregate_optical(raw, src_transform, src_crs, grid, minimum_fraction=.8):
    """Optical-only QA and aggregation: no thermal band or uncertainty inputs."""
    valid = qa_valid(raw["qa_pixel"], raw["qa_radsat"])
    for band in SR_BANDS:
        valid &= (raw[band] >= 7273) & (raw[band] <= 43636)
    fraction = _warp(valid, src_transform, src_crs, grid)
    accepted = np.isfinite(fraction) & (fraction >= minimum_fraction)
    sr = {}
    for band in SR_BANDS:
        avg = _warp(np.where(valid, scale_reflectance(raw[band]), np.nan), src_transform, src_crs, grid, np.nan)
        sr[band] = np.where(accepted, avg, np.nan)
    result = surface_descriptors(sr)
    result["optical_valid_fraction"] = fraction
    # Keep the trained feature's QA-water semantics, independently of WorldCover.
    result["water_fraction"] = _warp(np.where(valid, ((raw["qa_pixel"] >> 7) & 1), np.nan), src_transform, src_crs, grid, np.nan)
    return result, valid


def _native_window(source, grid, padding=2):
    from rasterio.windows import from_bounds, Window
    bounds = transform_bounds(f"EPSG:{grid.epsg}", source.crs, *grid.bounds, densify_pts=21)
    win = from_bounds(*bounds, source.transform)
    left, top = math.floor(win.col_off)-padding, math.floor(win.row_off)-padding
    right, bottom = math.ceil(win.col_off+win.width)+padding, math.ceil(win.row_off+win.height)+padding
    if (right-left)*(bottom-top) > MAX_NATIVE_CELLS:
        raise ValueError("Source window exceeds the native-cell safety bound.")
    return Window(left, top, right-left, bottom-top)


def read_optical(scene, grid, include_observed=False):
    import planetary_computer
    required = [*SR_BANDS, "qa_pixel", "qa_radsat"]
    if scene.get("collection") != "landsat-c2-l2" or scene["properties"].get("platform") not in ("landsat-8", "landsat-9"):
        raise ValueError("Only catalogued Landsat 8/9 Collection 2 Level 2 scenes are supported.")
    missing = set(required)-set(scene.get("assets", {}))
    if missing:
        raise ValueError(f"Source scene lacks required optical assets: {sorted(missing)}")
    bands = required + ([b for b in ("lwir11", "qa") if b in scene["assets"]] if include_observed else [])
    with rasterio.Env(**GDAL_ENV), ExitStack() as stack:
        sources = {b: stack.enter_context(rasterio.open(planetary_computer.sign(scene["assets"][b]["href"]))) for b in bands}
        reference = sources["red"]
        win = _native_window(reference, grid)
        raw = {}
        for band, source in sources.items():
            if (source.crs, source.transform, source.shape) != (reference.crs, reference.transform, reference.shape):
                raise ValueError("Scene asset grids differ; refusing implicit band alignment.")
            raw[band] = source.read(1, window=win, boundless=True, fill_value=1 if band == "qa_pixel" else 0)
        src_transform, src_crs = reference.window_transform(win), reference.crs
    features, valid = aggregate_optical(raw, src_transform, src_crs, grid)
    observed = None
    if include_observed and "lwir11" in raw and "qa" in raw:
        thermal_valid = valid & (raw["lwir11"] > 0) & (raw["qa"]*.01 <= 3.0)
        fraction = _warp(thermal_valid, src_transform, src_crs, grid)
        observed = _warp(np.where(thermal_valid, scale_temperature(raw["lwir11"]), np.nan), src_transform, src_crs, grid, np.nan)
        observed = np.where(fraction >= .8, observed, np.nan)
    audit = {"id": scene["id"], "collection": scene["collection"],
             "url": f"{STAC_URL}/collections/landsat-c2-l2/items/{scene['id']}",
             "stac_sha256": hashlib.sha256(json.dumps(scene, sort_keys=True, default=_json).encode()).hexdigest(),
             "assets": {b: scene["assets"][b]["href"].split("?")[0] for b in bands},
             "native_window": [int(win.col_off), int(win.row_off), int(win.width), int(win.height)],
             "optical_qa": "Reject QA fill, dilated cloud, cirrus, cloud/shadow, medium cloud confidence, optical saturation/terrain occlusion; all six SR DN 7273–43636; >=80% valid area per 100 m cell.",
             "thermal_independence": "Predictor masks and values never depend on lwir11 or ST uncertainty.",
             "aggregation": "Mask native 30 m optical grid; area-average six reflectances to pilot 100 m grid, then compute indices.",
             "observed_qa": "Separate thermal DN>0 and ST uncertainty<=3 K, >=80% area; thermal gaps never remove predictions." if include_observed else "Not requested"}
    return features, observed, audit


def _stac_tiles(collection, bbox, cache, date=None):
    request = {"collections": [collection], "bbox": list(map(float, bbox)), "limit": 10}
    if date:
        request["datetime"] = date
    path = Path(cache)/"raster-stac"/(hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()+".json")
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        with _session() as session:
            response = session.post(STAC_URL+"/search", json=request, timeout=(15, 90))
            response.raise_for_status()
            payload = response.json()
        if any(link.get("rel") == "next" for link in payload.get("links", [])):
            raise ValueError("Static-data request exceeds the ten-tile bound.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    items = payload.get("features", [])
    if not items or len(items) > 10:
        raise ValueError(f"No bounded {collection} coverage exists for this selection.")
    return items


def _mosaic(items, asset, grid, nodata, dtype, sources=None):
    """Read a single bounded native mosaic, then warp once for the full request."""
    import planetary_computer
    from rasterio.merge import merge
    with rasterio.Env(**GDAL_ENV), ExitStack() as stack:
        if sources is None:
            sources = [stack.enter_context(rasterio.open(planetary_computer.sign(item["assets"][asset]["href"]))) for item in items]
        crs = sources[0].crs
        if any(source.crs != crs for source in sources):
            raise ValueError("Static source tiles have incompatible CRSs.")
        bounds = transform_bounds(f"EPSG:{grid.epsg}", crs, *grid.bounds, densify_pts=21)
        rx = min(abs(source.res[0]) for source in sources)
        ry = min(abs(source.res[1]) for source in sources)
        bounds = (bounds[0]-2*rx, bounds[1]-2*ry, bounds[2]+2*rx, bounds[3]+2*ry)
        if math.ceil((bounds[2]-bounds[0])/rx)*math.ceil((bounds[3]-bounds[1])/ry) > MAX_NATIVE_CELLS:
            raise ValueError("Static raster read exceeds the native-cell safety bound.")
        values, transform = merge(sources, bounds=bounds, res=(rx, ry), nodata=nodata,
                                  dtype=dtype, indexes=[1], target_aligned_pixels=True)
    return values[0], transform, crs


def worldcover_fractions(classes, src_transform, src_crs, grid):
    # Zero/unknown pixels remain in the denominator, never treated as valid land.
    land = np.nan_to_num(_warp(np.isin(classes, LAND_CLASSES), src_transform, src_crs, grid), nan=0.)
    water = np.nan_to_num(_warp(classes == 80, src_transform, src_crs, grid), nan=0.)
    return land, water


def read_land_mask(grid, cache):
    bbox = transform_bounds(f"EPSG:{grid.epsg}", 4326, *grid.bounds, densify_pts=21)
    items = _stac_tiles("esa-worldcover", bbox, cache, "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z")
    if any(item["properties"].get("esa_worldcover:product_version") != "2.0.0" for item in items):
        raise ValueError("Expected ESA WorldCover 2021 v200.")
    import planetary_computer
    land = np.empty(grid.shape, np.float32)
    water = np.empty(grid.shape, np.float32)
    tiles = 0
    # Keep sources open across tiles, reusing GDAL's bounded COG block cache.
    # At polar pilot latitudes even a full 50 km WorldCover window can exceed
    # 90 million cells. Only 128x128 target cells are materialized at a time.
    with rasterio.Env(**GDAL_ENV), ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(planetary_computer.sign(item["assets"]["map"]["href"]))) for item in items]
        for slices, tile in grid_tiles(grid):
            classes, transform, crs = _mosaic(items, "map", tile, 0, "uint8", sources=sources)
            land[slices], water[slices] = worldcover_fractions(classes, transform, crs, tile)
            del classes
            tiles += 1
    return land, water, {"dataset": "ESA WorldCover 2021 v200", "items": [item["id"] for item in items],
                         "processing_tiles": tiles, "maximum_tile_shape": [STATIC_TILE_SIZE,STATIC_TILE_SIZE],
                         "maximum_native_cells_per_read": MAX_NATIVE_CELLS,
                         "urls": [item["assets"]["map"]["href"].split("?")[0] for item in items],
                         "documentation": "https://esa-worldcover.org/en/data-access",
                         "attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
                         "criterion": "Area-average class membership; land fraction >=0.8. Exclude permanent water 80 and nodata/unknown; retain snow/ice70, wetlands90 and mangroves95.",
                         "limitation": "Static 2021 classification; seasonal water/coastline errors remain possible. Not a contemporary comprehensive coastline mask."}


def terrain_arrays(z, cell_size=100.):
    """Vectorized equivalent of terrain.terrain_descriptors, including its halo."""
    z = np.asarray(z, float)
    if z.ndim != 2 or min(z.shape) < 3:
        raise ValueError("Terrain needs a 2D raster with a one-cell halo.")
    windows = np.lib.stride_tricks.sliding_window_view(z, (3, 3))
    valid = np.isfinite(windows).all(axis=(-1, -2))
    dx = ((z[:-2, 2:]+2*z[1:-1, 2:]+z[2:, 2:])-(z[:-2, :-2]+2*z[1:-1, :-2]+z[2:, :-2]))/(8*cell_size)
    dy = ((z[:-2, :-2]+2*z[:-2, 1:-1]+z[:-2, 2:])-(z[2:, :-2]+2*z[2:, 1:-1]+z[2:, 2:]))/(8*cell_size)
    gradient = np.hypot(dx, dy)
    sine = np.divide(-dx, gradient, out=np.zeros_like(dx), where=gradient > 1e-8)
    cosine = np.divide(-dy, gradient, out=np.zeros_like(dy), where=gradient > 1e-8)
    return {"elevation": z[1:-1, 1:-1], "slope": np.where(valid, np.degrees(np.arctan(gradient)), np.nan),
            "aspect_sin": np.where(valid, sine, np.nan), "aspect_cos": np.where(valid, cosine, np.nan),
            "terrain_relief_300m": np.where(valid, windows.max(axis=(-1, -2))-windows.min(axis=(-1, -2)), np.nan)}


def read_terrain(grid, cache):
    transform = grid.transform * rasterio.Affine.translation(-1, -1)
    halo = RasterGrid(grid.epsg, transform, grid.height+2, grid.width+2,
                      array_bounds(grid.height+2, grid.width+2, transform), None, None, None, 0, 0)
    bbox = transform_bounds(f"EPSG:{halo.epsg}", 4326, *halo.bounds, densify_pts=21)
    items = _stac_tiles("cop-dem-glo-30", bbox, cache)
    z, native_transform, crs = _mosaic(items, "data", halo, np.nan, "float32")
    averaged = _warp(z, native_transform, crs, halo, np.nan)
    return terrain_arrays(averaged), {"dataset": "Copernicus GLO-30 DSM", "items": [item["id"] for item in items],
                                      "urls": [item["assets"]["data"]["href"].split("?")[0] for item in items],
                                      "units": "Elevation metres above EGM2008, slope degrees, relief metres",
                                      "method": "Area-average full DSM window to 100 m plus one-cell halo; Horn slope/aspect and 300 m relief.",
                                      "limitation": "Static DSM includes vegetation/buildings; no explicit cast-shadow model.",
                                      "attribution": "Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018; provided under COPERNICUS by the EU and ESA."}


def feature_frame(region, grid, arrays, target):
    rows, cols = np.indices(grid.shape)
    x = grid.transform.c+(cols+.5)*RESOLUTION
    y = grid.transform.f-(rows+.5)*RESOLUTION
    lon, lat = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform(x.ravel(), y.ravel())
    data = pd.DataFrame({"_raster_position": np.arange(grid.width*grid.height), "region_id": region["id"],
                         "datetime_utc": target, "longitude": lon, "latitude": lat,
                         "pixel_x": x.ravel(), "pixel_y": y.ravel(), "pixel_epsg": grid.epsg,
                         "pixel_id": [f"{region['id']}:{r+grid.row_offset}:{c+grid.col_offset}" for r, c in zip(rows.ravel(), cols.ravel())]})
    for name, values in arrays.items():
        data[name] = values.ravel()
    return data


def add_raster_context(frame, climate_raster=None, chunk_size=FRAME_CHUNK_SIZE):
    """Exact point-to-climate lookup and chunked SPA, without per-pixel I/O.

    The climate lookup uses the same floor-to-source-cell rule as
    rasterio.sample in context.add_context. Solar/time features retain that
    module's formulas and per-pixel positions, with bounded temporary arrays.
    """
    from .context import CLIMATE_URL, climate_label
    import pvlib
    from rasterio.windows import Window
    if chunk_size < 1:
        raise ValueError("Context chunk size must be positive.")
    data = frame.copy()
    data["datetime_utc"] = pd.to_datetime(data.datetime_utc,utc=True)
    labels = np.full(len(data),"unknown",dtype=object)
    with rasterio.Env(**GDAL_ENV), rasterio.open(str(climate_raster or CLIMATE_URL)) as source:
        x,y = Transformer.from_crs(4326,source.crs,always_xy=True).transform(data.longitude.to_numpy(),data.latitude.to_numpy())
        rows,cols = rasterio.transform.rowcol(source.transform,x,y)
        rows,cols = np.asarray(rows),np.asarray(cols)
        valid = (rows >= 0) & (rows < source.height) & (cols >= 0) & (cols < source.width)
        if valid.any():
            row0,col0 = int(rows[valid].min()),int(cols[valid].min())
            height,width = int(rows[valid].max())-row0+1,int(cols[valid].max())-col0+1
            if height*width > MAX_NATIVE_CELLS:
                raise ValueError("Climate window exceeds the native-cell safety bound.")
            values = source.read(1,window=Window(col0,row0,width,height),masked=True)
            sampled = np.ma.asarray(values,dtype=float).filled(np.nan)[rows[valid]-row0,cols[valid]-col0]
            labels[valid] = [climate_label(value) for value in sampled]
    data["climate_class"] = labels
    data["climate_sampling_status"] = np.where(data.climate_class.eq("unknown"),"unknown_or_nodata","classified")
    data["climate_source"] = "Beck et al. 2023, 1991-2020; NatCap COG conversion"
    elevation = np.empty(len(data),float)
    sine = np.empty(len(data),float)
    cosine = np.empty(len(data),float)
    for start in range(0,len(data),chunk_size):
        stop = min(start+chunk_size,len(data))
        part = data.iloc[start:stop]
        position = pvlib.solarposition.spa_python(pd.DatetimeIndex(part.datetime_utc),
                                                part.latitude.to_numpy(),part.longitude.to_numpy(),how="numpy")
        elevation[start:stop] = position.elevation.to_numpy()
        azimuth = np.deg2rad(position.azimuth.to_numpy())
        sine[start:stop],cosine[start:stop] = np.sin(azimuth),np.cos(azimuth)
    data["solar_elevation_deg"] = elevation
    data["solar_azimuth_sin"],data["solar_azimuth_cos"] = sine,cosine
    hour = data.datetime_utc.dt.hour+data.datetime_utc.dt.minute/60+data.longitude/15
    data["hour_sin"],data["hour_cos"] = np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)
    day = data.datetime_utc.dt.dayofyear
    data["day_of_year_sin"],data["day_of_year_cos"] = np.sin(2*np.pi*day/365.2425),np.cos(2*np.pi*day/365.2425)
    return data


def predict_chunks(frame, bundle, chunk_size=FRAME_CHUNK_SIZE):
    """Keep sklearn's feature transforms bounded while preserving row identity."""
    from .model import predict_frame
    if chunk_size < 1:
        raise ValueError("Prediction chunk size must be positive.")
    return pd.concat([predict_frame(frame.iloc[start:start+chunk_size],bundle)
                      for start in range(0,len(frame),chunk_size)],axis=0)


def manual_air_reference(frame, scalar, grid, target, cache):
    from .weather import enrich_weather
    if isinstance(scalar, bool) or not np.isfinite(float(scalar)) or not -90 <= float(scalar) <= 65:
        raise ValueError("Manual air temperature must be a finite Celsius value from -90 to 65.")
    lon, lat = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform(grid.polygon.centroid.x, grid.polygon.centroid.y)
    anchor = pd.DataFrame({"datetime_utc": [target], "longitude": [lon], "latitude": [lat]})
    background, audit = enrich_weather(anchor, cache, max_requests=1)
    correction = float(scalar)-float(background.background_air_temperature_c.iloc[0])
    result = frame.copy()
    result["air_temperature_c"] = result.background_air_temperature_c+correction
    result["air_temperature_source"] = "user_air_at_polygon_centroid_plus_ERA5_spatial_background"
    result["station_id"] = ""
    result["station_air_correction_c"] = correction
    return result, {"source": "user-provided temperature; no instrument/QC assertion", "air_temperature_c": float(scalar),
                    "anchor_longitude": lon, "anchor_latitude": lat, "anchor": "projected polygon centroid",
                    "method": "User air temperature minus ERA5 at centroid plus ERA5 at each pixel; weather history remains ERA5.",
                    "weather": audit}


def _write_tif(path, arrays, grid):
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=len(arrays),
                       dtype="float32", crs=f"EPSG:{grid.epsg}", transform=grid.transform, nodata=NODATA,
                       compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256) as dest:
        for band, (name, values) in enumerate(arrays.items(), 1):
            dest.write(np.where(np.isfinite(values), values, NODATA).astype(np.float32), band)
            dest.set_band_description(band, name)
        dest.update_tags(software=VERSION, grid_resolution_m=100)


def _overlay(path, values, grid, minimum, maximum, cmap="inferno"):
    from matplotlib import colormaps
    from PIL import Image
    transform, width, height = calculate_default_transform(f"EPSG:{grid.epsg}", "EPSG:3857", grid.width, grid.height, *grid.bounds)
    target = np.full((height, width), np.nan, np.float32)
    reproject(values.astype(np.float32), target, src_transform=grid.transform, src_crs=f"EPSG:{grid.epsg}", src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:3857", dst_nodata=np.nan, resampling=Resampling.nearest, num_threads=1)
    normalized = np.clip((target-minimum)/max(maximum-minimum, .01), 0, 1)
    rgba = colormaps[cmap](np.nan_to_num(normalized), bytes=True)
    rgba[..., 3] = np.where(np.isfinite(target), 230, 0).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return list(map(float, transform_bounds(3857, 4326, *array_bounds(height, width, transform), densify_pts=21)))


def comparison_legends(prediction, observed=None):
    """Use the union of valid predictions/references for both temperature maps."""
    prediction = np.asarray(prediction, dtype=float)
    valid_prediction = prediction[np.isfinite(prediction)]
    if valid_prediction.size == 0:
        raise ValueError("Cannot define a legend without finite predictions.")
    minimum, maximum = float(valid_prediction.min()), float(valid_prediction.max())
    reference = np.asarray(observed, dtype=float) if observed is not None else None
    comparable = (np.isfinite(prediction) & np.isfinite(reference)) if reference is not None else None
    has_comparison = comparable is not None and comparable.any()
    if has_comparison:
        minimum = min(minimum, float(reference[comparable].min()))
        maximum = max(maximum, float(reference[comparable].max()))
    # A sub-degree range must not consume the whole thermal colour ramp.
    # This changes display limits only; predictions and temperature stats stay exact.
    if maximum-minimum < 5.0:
        middle = (minimum+maximum)/2
        minimum, maximum = middle-2.5, middle+2.5
    legends = {"prediction": {"min_c": minimum, "max_c": maximum, "cmap": "inferno"}}
    if has_comparison:
        legends["observed"] = dict(legends["prediction"])
        limit = max(1., float(np.abs(prediction[comparable]-reference[comparable]).max()))
        legends["residual"] = {"min_c": -limit, "max_c": limit, "cmap": "RdBu_r"}
    return legends


def _counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def summarize_features(frame, features):
    """Summarize actual model-input rows, counting nonfinite values as missing."""
    summary = {}
    for name in features:
        if name == "climate_class":
            counts = frame[name].astype("string").fillna("__missing__").value_counts()
            summary[name] = {"counts": {str(key): int(counts[key]) for key in sorted(counts.index)}}
            continue
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        actual = values[finite]
        summary[name] = {"min": float(actual.min()) if len(actual) else None,
                         "max": float(actual.max()) if len(actual) else None,
                         "mean": float(actual.mean()) if len(actual) else None,
                         "missing": int((~finite).sum())}
    return summary


def render_raster(region, polygon, scene, requested_datetime, output_dir, cache_dir,
                  model_path, progress=None, air_override=None, mode="observed"):
    """Create a single bounded raster using trusted server region/scene/model paths.

    progress(dict) receives stage/message/fraction. Returns JSON-serializable
    paths and audit metadata; the API owns URL publication and access control.
    """
    started = time.monotonic()
    def notify(stage, fraction, message):
        if progress:
            progress({"stage": stage, "fraction": fraction, "message": message})
    notify("validate", .01, "Validating selected area, source scene and time")
    grid = build_grid(region, polygon)
    target, source_time, age_days = validate_time(scene, requested_datetime, mode)
    if mode == "scenario":
        from .scenario import require_daylight_area
        require_daylight_area(target, grid.polygon_wgs84)
    output, cache = Path(output_dir), Path(cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ("prediction.tif", "overlay.png", "provenance.json")):
        raise ValueError("Output directory already contains raster results; choose a new job directory.")
    bundle = joblib.load(model_path)  # Trusted deployment configuration only.
    if bundle.get("smoke_only"):
        raise ValueError("A smoke-only unknown-climate model cannot serve predictions.")
    configure_safe_logging()
    notify("optical", .1, "Reading actual Landsat optical windows and independent quality masks")
    optical, observed, scene_audit = read_optical(scene, grid, include_observed=(target == source_time))
    notify("land", .25, "Aggregating ESA WorldCover land cover to the 100 m grid")
    land, water, land_audit = read_land_mask(grid, cache)
    valid = grid.inside & (land >= .8)
    if mode != "scenario":
        valid &= optical["optical_valid_fraction"] >= .8
    counts = {"total_pixels": grid.width*grid.height, "outside_polygon": int((~grid.inside).sum()),
              "insufficient_land": int((grid.inside & (land < .8)).sum()),
              "insufficient_optical": int((grid.inside & (land >= .8) & ~(optical["optical_valid_fraction"] >= .8)).sum())}
    if not valid.any():
        raise ValueError("No pixels pass the polygon, >=80% independent land and >=80% optical-quality masks.")
    notify("terrain", .38, "Reading full DSM window and computing 100 m terrain descriptors")
    terrain, terrain_audit = read_terrain(grid, cache)
    frame = feature_frame(region, grid, {**optical, **terrain}, target)
    frame = frame.loc[valid.ravel()].reset_index(drop=True)
    from .context import CLIMATE_URL
    frame = add_raster_context(frame)
    unknown = frame.climate_class.isin(["unknown", "__unknown__", ""]) | frame.climate_class.isna()
    counts["unknown_climate"] = int(unknown.sum())
    frame = frame.loc[~unknown].copy()
    if frame.empty:
        raise ValueError("No selected pixels have a known climate classification.")
    if not np.isfinite(frame.solar_elevation_deg).all():
        raise ValueError("Solar geometry is unavailable for the requested time.")
    if mode != "scenario" and frame.solar_elevation_deg.le(0).any():
        raise ValueError("Use reported-air scenario mode for a clearly labelled coarse night baseline.")
    if mode == "scenario":
        # The night baseline uses no optical predictors; do not turn an old
        # optical cloud/thermal gap into an artificial nighttime data gap.
        frame = frame.loc[frame.solar_elevation_deg.le(0) | frame.optical_valid_fraction.ge(.8)].copy()
        if frame.empty:
            raise ValueError("No daytime pixels have valid optical surface features; choose another snapshot.")
    notify("weather", .5, "Joining retrospective weather and timely station air observations")
    station_audit, legacy_audit, manual_audit = [], None, None
    weather_context, skin_audit = None, None
    if mode == "scenario":
        from .scenario import prepare_scenario
        frame, weather_context, weather_audit, radiation_audit, skin_audit = prepare_scenario(
            frame, grid, target, cache, air_override)
        manual_audit = weather_context
    else:
        from .weather import enrich_weather
        from .assemble import attach_stations
        frame, weather_audit = enrich_weather(frame, cache, max_requests=MAX_WEATHER_REQUESTS)
        if air_override is None:
            frame, station_audit = attach_stations(frame, [region], cache, stations_per_region=2)
            if region["id"] == "darwin_howard_springs":
                from .legacy_isd import apply_darwin_fallback
                raw_dir = Path(model_path).resolve().parents[2]/"runs/darwin_qc_audit/legacy_isd"
                frame, legacy_audit = apply_darwin_fallback(frame, cache, raw_dir)
        else:
            frame, manual_audit = manual_air_reference(frame, air_override, grid, target, cache)
        notify("radiation", .66, "Fetching required ERA5 longwave and snow context")
        from .radiation import add_radiation, VARIABLES
        frame = add_radiation(frame, cache/"radiation", max_hours=1)
        radiation_audit = frame.attrs["radiation_context"]
        if any(not np.isfinite(frame[field]).all() for field in VARIABLES.values()):
            raise ValueError("Required ARCO longwave/snow context is unavailable: "+json.dumps(radiation_audit.get("errors", []))[:600])
    model_features = list(bundle["features"])
    night_features = ["air_temperature_c", "background_air_temperature_c", "era5_skin_temperature_c"]
    night_rows = frame.solar_elevation_deg.le(0)
    features = list(model_features) if (~night_rows).any() else list(night_features)
    if night_rows.any() and (~night_rows).any():
        features += [field for field in night_features if field not in features]
    missing = set(features)-set(frame.columns)
    if missing:
        raise ValueError(f"Prediction requires unavailable feature columns: {sorted(missing)}")
    numeric = [field for field in features if field != "climate_class"]
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="raise")
    missing_features = {field: int((~np.isfinite(numeric_frame[field])).sum()) for field in numeric if (~np.isfinite(numeric_frame[field])).any()}
    complete = pd.Series(True, index=frame.index)
    if (~night_rows).any():
        complete.loc[~night_rows] = np.isfinite(frame.loc[~night_rows, [f for f in model_features if f != "climate_class"]]).all(axis=1)
    if night_rows.any():
        complete.loc[night_rows] = np.isfinite(frame.loc[night_rows, night_features]).all(axis=1)
    counts["incomplete_features"] = int((~complete).sum())
    frame = frame.loc[complete].copy()
    if frame.empty:
        raise ValueError("No pixels have a complete supported input vector.")
    night_rows = frame.solar_elevation_deg.le(0)
    has_night, has_day = bool(night_rows.any()), bool((~night_rows).any())
    # Publish the inputs used by surviving rows, after completeness filtering.
    features = list(model_features) if has_day else list(night_features)
    if has_night and has_day:
        features += [field for field in night_features if field not in features]
    missing_features = {field: value for field, value in missing_features.items() if field in features}
    method = "mixed_day_night" if has_night and has_day else "nighttime_coarse_baseline" if has_night else "daytime_ml"
    frame["prediction_method"] = np.where(night_rows, "nighttime_coarse_baseline", "daytime_ml")
    notify("predict", .8, "Applying the daytime model or explicitly labelled coarse night baseline")
    del numeric_frame
    if has_night:
        from .scenario import predict_with_night_baseline
        predicted = predict_with_night_baseline(frame, bundle, predict_chunks)
    else:
        predicted = predict_chunks(frame, bundle)
    # A daytime calibration width is not an uncertainty estimate for a night
    # baseline. Mixed maps also omit interval bands rather than imply coverage.
    interval_valid = not has_night
    positions = frame._raster_position.to_numpy(dtype=int)
    arrays = {}
    export_fields = ("predicted_lst_c", "lower_lst_c", "upper_lst_c") if interval_valid else ("predicted_lst_c",)
    for field in export_fields:
        values = np.full(grid.width*grid.height, np.nan, np.float32)
        values[positions] = predicted[field].to_numpy(dtype=np.float32)
        arrays[field] = values.reshape(grid.shape)
    if not all(np.isfinite(values).sum() == len(frame) for values in arrays.values()):
        raise ValueError("The saved model produced nonfinite predictions or intervals.")
    effective = np.isfinite(arrays["predicted_lst_c"])
    if observed is not None:
        observed = np.where(effective, observed, np.nan)
        arrays["observed_lst_c"] = observed
        arrays["residual_prediction_minus_observed_c"] = arrays["predicted_lst_c"]-observed
    minimum, maximum = float(predicted.predicted_lst_c.min()), float(predicted.predicted_lst_c.max())
    counts.update(valid_pixels=len(frame), masked_pixels=grid.width*grid.height-len(frame),
                  observed_pixels=int(np.isfinite(observed).sum()) if observed is not None else 0,
                  climate_unseen_in_training=int(predicted.climate_unseen_in_training.sum()))
    summary = {"min_c": minimum, "max_c": maximum, "mean_c": float(predicted.predicted_lst_c.mean()),
               "valid_pixels": len(frame), "masked_pixels": counts["masked_pixels"], "total_pixels": counts["total_pixels"],
               "interval_radius_c": float(bundle["interval_radius_c"]) if interval_valid else None,
               "daytime_ml_pixels": int((~night_rows).sum()), "night_baseline_pixels": int(night_rows.sum()),
               "compared_pixels": counts["observed_pixels"],
               "compared_coverage_fraction": counts["observed_pixels"]/len(frame),
               "compared_coverage_denominator": "valid prediction pixels"}
    legends = comparison_legends(arrays["predicted_lst_c"], observed)
    if counts["observed_pixels"]:
        summary.update(observed_min_c=float(np.nanmin(observed)), observed_max_c=float(np.nanmax(observed)),
                       observed_mean_c=float(np.nanmean(observed)))
    warnings = ["Experimental retrospective clear-sky daytime model; hourly, nighttime and all-weather accuracy are not established.",
                "100 m is the output grid spacing, not a claim of independently measured 100 m thermal detail.",
                "Calibration intervals have no guaranteed coverage for a new area, weather regime or manual air input.",
                "The albedo field is a directional-reflectance spectral proxy; building/tree cast shadows are not explicitly modelled.",
                "Weather and longwave/snow context are coarse retrospective reanalysis; historical inputs do not prove real-time availability.",
                "WorldCover is a static 2021 land classification, including for dates earlier in 2021; land-use and shoreline changes are unmodelled."]
    if has_night:
        warnings.append("Night pixels use reported air plus the coarse ERA5 skin–air temperature difference, not the daylight ML model. There is no validated 100 m nighttime detail or calibrated error interval.")
    if weather_context:
        warnings.extend(weather_context["warnings"])
        if weather_context["weather_basis"] == "seasonal_reference":
            warnings.append("Weather other than the reported air temperature comes from one same-season/hour reference day in 2023. It is not weather observed at the requested time and not a climatological average.")
        warnings.append("The optical and land-cover dates describe a fixed surface snapshot; this scenario does not reconstruct historical or future surface changes.")
    if age_days < 0:
        warnings.append(f"The surface snapshot is {abs(age_days):.2f} days after the requested time; it is a fixed-property scenario, not a historical reconstruction.")
    if age_days > 0:
        warnings.append(f"Optical surface descriptors are {age_days:.2f} days old; this changed-time prediction is unvalidated and has no same-time satellite reference.")
    if counts["climate_unseen_in_training"]:
        warnings.append("Some known climate classes were absent from model training; these predictions are extrapolations.")
    if frame.water_fraction.gt(0).any():
        warnings.append("Some WorldCover land cells contain Landsat QA-water fractions; training used QA water_fraction==0, so these mixed cells differ from training selection.")
    if manual_audit:
        warnings.append("Manual air temperature is assumed to represent the polygon centroid at the requested time; instrument accuracy and representativeness are unverified.")
    if counts["observed_pixels"]:
        warnings.append("The scene-time satellite comparison is not a new held-out validation; selected archived scenes may have contributed training samples.")
    notify("export", .9, "Writing GeoTIFF, georeferenced PNG overlays and source audit")
    files = {"prediction_tif": str(output/"prediction.tif"), "overlay_png": str(output/"overlay.png"),
             "provenance_json": str(output/"provenance.json"), "quality_tif": str(output/"quality.tif"),
             "features_parquet": str(output/"features.parquet")}
    _write_tif(files["prediction_tif"], arrays, grid)
    _write_tif(files["quality_tif"], {"worldcover_land_fraction": land, "worldcover_water_fraction": water,
                                     "optical_valid_fraction": optical["optical_valid_fraction"],
                                     "prediction_valid": effective.astype(float), "polygon_pixel_center": grid.inside.astype(float)}, grid)
    temperature_legend = legends["prediction"]
    bounds = _overlay(files["overlay_png"], arrays["predicted_lst_c"], grid,
                      temperature_legend["min_c"], temperature_legend["max_c"])
    if counts["observed_pixels"]:
        files["observed_overlay_png"] = str(output/"observed_overlay.png")
        files["residual_overlay_png"] = str(output/"residual_overlay.png")
        # Identical predicted/observed colour scales make the comparison honest.
        _overlay(files["observed_overlay_png"], observed, grid, temperature_legend["min_c"], temperature_legend["max_c"])
        residual = arrays["residual_prediction_minus_observed_c"]
        limit = legends["residual"]["max_c"]
        _overlay(files["residual_overlay_png"], residual, grid, -limit, limit, "RdBu_r")
        summary["residual_display_limit_c"] = limit
        summary["scene_comparison_mae_c"] = float(np.nanmean(np.abs(residual)))
    frame.to_parquet(files["features_parquet"], index=False)
    sources = ([scene_audit, terrain_audit] if has_day else []) + [land_audit,
               {"dataset": "Köppen–Geiger 1991–2020 climate", "url": CLIMATE_URL, "sampling": "Nearest classified 1 km cell"},
               {"dataset": "ERA5 via Open-Meteo", "status": "used", "audit": weather_audit,
                "usage": "Weather and history for daytime ML" if has_day else "Background air temperature only; other returned weather fields are context, not night predictors"}]
    if has_day:
        sources.append({"dataset": "ARCO ERA5", "status": "used", "audit": radiation_audit})
    if skin_audit:
        sources.append(skin_audit)
    ghcnh_pixels = int(frame.air_temperature_source.eq("observed_station_residual_plus_ERA5_spatial_background").sum())
    if ghcnh_pixels:
        sources.append({"dataset": "NOAA GHCNh air observations", "status": "used", "used_pixels": ghcnh_pixels, "audit": station_audit})
    legacy_pixels = int(frame.air_temperature_source.eq("observed_legacy_ISD_station_residual_plus_ERA5_spatial_background").sum())
    if legacy_audit and legacy_pixels:
        sources.append({"dataset": "Darwin legacy ISD QC1", "status": "used", "used_pixels": legacy_pixels, "audit": legacy_audit})
    if manual_audit:
        sources.append({"dataset": "Manual air reference", "status": "used", "used_pixels": len(frame), "audit": manual_audit})
    result = {"version": VERSION, "status": "complete", "region_id": region["id"], "source_scene_id": scene["id"],
              "datetime_utc": target.isoformat(), "scene_datetime_utc": source_time.isoformat(), "mode": mode,
              "prediction_method": method, "weather_context": weather_context,
              "source_age_days": age_days, "bounds": bounds, "files": files, "summary": summary, "counts": counts, "legends": legends,
              "grid": {"epsg": grid.epsg, "resolution_m": 100, "shape": list(grid.shape), "bounds_m": list(grid.bounds),
                       "transform": list(grid.transform)[:6], "overlay_crs": "EPSG:3857", "polygon_rule": "Pixel centre inside polygon; grid bounding box clipped by nodata", "boundary_tolerance_m": BOUNDARY_TOLERANCE_M},
              "polygon": mapping(grid.polygon_wgs84), "sources": sources, "features": features,
              "feature_summary": summarize_features(frame, features),
              "feature_missing_counts": missing_features, "air_sources": _counts(frame.air_temperature_source),
              "station_ids": _counts(frame.station_id), "climate_classes": _counts(frame.climate_class),
              "station_candidate_audit": station_audit, "legacy_station_candidate_audit": legacy_audit,
              "model_sha256": _digest(model_path), "module_sha256": _digest(__file__),
              "resource_limits": {"maximum_pixels": MAX_PIXELS, "maximum_area_km2": MAX_AREA_KM2,
                                  "worldcover_tile_size": STATIC_TILE_SIZE, "frame_chunk_size": FRAME_CHUNK_SIZE,
                                  "maximum_native_cells_per_read": MAX_NATIVE_CELLS, "maximum_weather_requests": MAX_WEATHER_REQUESTS},
              "feature_units": {"temperature": "degrees Celsius", "radiation": "W/m²", "snow_water_equivalent": "metres of water equivalent", "elevation": "metres above EGM2008"},
              "warnings": warnings, "elapsed_seconds": round(time.monotonic()-started, 2)}
    for name, path in files.items():
        if name != "provenance_json":
            result.setdefault("artifact_sha256", {})[name] = _digest(path)
    Path(files["provenance_json"]).write_text(json.dumps(result, indent=2, default=_json, allow_nan=False)+"\n")
    notify("complete", 1., "Raster and source audit are ready")
    return result
