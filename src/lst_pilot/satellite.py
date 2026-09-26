"""Bounded Landsat 8/9 C2 L2 acquisition for the 100 m pilot.

Run raster acquisition on the remote pilot server. Metadata and compact samples
are cached; full scenes are never downloaded. These labels are clear-sky daytime
surface temperature, not observations of nighttime or cloudy surfaces.

Sources: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
VERSION = "landsat-100m-v1"
SR_BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
REQUIRED_ASSETS = (*SR_BANDS, "lwir11", "qa_pixel", "qa_radsat")
# This is the feature allowlist. Thermal auxiliary assets are never features.
SURFACE_FEATURES = [*(f"sr_{b}" for b in SR_BANDS), "ndvi", "ndbi", "ndwi", "albedo_proxy"]
LOG = logging.getLogger(__name__)


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "lst-pilot/1.0 (bounded scientific sampling)"
    return session


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _redact_url_queries(message: str) -> str:
    return re.sub(r"https?://[^\s'\"]+", lambda m: m.group(0).split("?")[0] +
                  ("?[redacted]" if "?" in m.group(0) else ""), message)


def _safe_error(exc: Exception) -> str:
    """Keep useful diagnostics without persisting signed URL query strings."""
    return _redact_url_queries(str(exc))[:600]


class _SafeUrlFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_url_queries(record.getMessage())
        record.args = ()
        if record.exc_text:
            record.exc_text = _redact_url_queries(record.exc_text)
        return True


class _SafeUrlFormatter(logging.Formatter):
    def __init__(self, original: logging.Formatter):
        super().__init__()
        self.original = original

    def format(self, record: logging.LogRecord) -> str:
        # Formatting happens after filters and may materialize traceback text.
        return _redact_url_queries(self.original.format(record))


def configure_safe_logging() -> None:
    """Redact URL queries in library logs as well as our own error messages."""
    logging.getLogger("rasterio.merge").setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(value, _SafeUrlFilter) for value in handler.filters):
            handler.addFilter(_SafeUrlFilter())
        if not isinstance(handler.formatter, _SafeUrlFormatter):
            handler.setFormatter(_SafeUrlFormatter(handler.formatter or logging.Formatter()))


def region_bbox(region: dict) -> list[float]:
    """Densified projected rectangle transformed to a WGS84 search envelope."""
    from rasterio.warp import transform_bounds
    return list(transform_bounds(f"EPSG:{region['epsg']}", "EPSG:4326",
                                 *region["extent_m"], densify_pts=21))


def _datetime(value: str) -> datetime:
    value = value.strip()
    if len(value) == 10:
        value += "T00:00:00+00:00"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def scene_coverage_fraction(item: dict, region: dict) -> float:
    """Approximate scene footprint coverage on an 80x80 regional metadata grid."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_geom
    geometry = transform_geom("EPSG:4326", f"EPSG:{region['epsg']}", item["geometry"])
    mask = rasterize([(geometry, 1)], out_shape=(80, 80),
                     transform=from_bounds(*region["extent_m"], 80, 80), fill=0)
    return float(mask.mean())


def _rank_candidates(items: list[dict], region: dict) -> list[dict]:
    for item in items:
        item["_pilot_roi_coverage_fraction"] = scene_coverage_fraction(item, region)
    strata = sorted({item.get("_pilot_temporal_stratum", 0) for item in items})
    queues = [sorted((x for x in items if x.get("_pilot_temporal_stratum", 0) == i),
                     key=lambda x: (-x["_pilot_roi_coverage_fraction"],
                                    x["properties"].get("eo:cloud_cover", 100))) for i in strata]
    ordered = []
    for j in range(max((len(q) for q in queues), default=0)):
        ordered.extend(q[j] for q in queues if j < len(q))
    return ordered


def search_scenes(region: dict, date_range: str, cache_dir: Path,
                  max_scenes: int = 3, max_candidates: int = 36,
                  cloud_cover_max: float = 80) -> list[dict]:
    """Search bounded temporal strata; cache unsigned STAC, never SAS tokens.

    For each equal-length time stratum, inspect low-cloud Tier 1 L2SP items.
    Cloud filtering is label quality selection, not all-weather coverage.
    There is no unbounded item pagination or whole-archive enumeration.
    """
    if not 1 <= max_scenes <= 64 or not max_scenes <= max_candidates <= 512:
        raise ValueError("Require 1 <= max_scenes <= 64 and max_scenes <= max_candidates <= 512")
    start_s, end_s = date_range.split("/")
    start, end = _datetime(start_s), _datetime(end_s)
    if end <= start:
        raise ValueError("date_range end must follow start")
    config = dict(version=VERSION, bbox=region_bbox(region), date_range=date_range,
                  max_scenes=max_scenes, max_candidates=max_candidates, cloud_cover_max=cloud_cover_max)
    path = Path(cache_dir) / f"{region['id']}_{_hash(config)}.json"
    if path.exists():
        return _rank_candidates(json.loads(path.read_text())["items"], region)
    strata = min(12, max_scenes)
    per_stratum = max_candidates // strata
    items = {}
    with _session() as session:
        for i in range(strata):
            lower = start + (end - start) * (i / strata)
            upper = start + (end - start) * ((i + 1) / strata)
            payload = {
                "collections": [COLLECTION], "bbox": config["bbox"],
                "datetime": f"{_iso(lower)}/{_iso(upper)}", "limit": per_stratum,
                "query": {"platform": {"in": ["landsat-8", "landsat-9"]},
                          "landsat:collection_category": {"eq": "T1"},
                          "landsat:correction": {"eq": "L2SP"},
                          "eo:cloud_cover": {"lte": cloud_cover_max}},
                "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
            }
            response = session.post(STAC_URL + "/search", json=payload, timeout=(15, 90))
            response.raise_for_status()
            for item in response.json().get("features", [])[:per_stratum]:
                if (all(b in item.get("assets", {}) for b in REQUIRED_ASSETS)
                        and item["properties"].get("platform") in ("landsat-8", "landsat-9")):
                    item["_pilot_temporal_stratum"] = i
                    items[item["id"]] = item
    # Round robin strata, so retrying after an unusable scene retains time spread.
    ordered = _rank_candidates(list(items.values()), region)
    _save_json(path, {"request": config, "created_utc": _iso(datetime.now(timezone.utc)), "items": ordered})
    return ordered


def qa_valid(qa_pixel: np.ndarray, qa_radsat: np.ndarray) -> np.ndarray:
    """Reject fill/cloud/cirrus/cloud-shadow/saturation/occlusion, retain snow.

    Landsat 8/9 QA_PIXEL bits 0..4; medium-or-higher cloud confidence too.
    QA_RADSAT rejects the six optical bands used here and terrain occlusion.
    Water is retained and flagged; a separate land mask must exclude oceans.
    """
    bad_pixel = (qa_pixel.astype(np.uint16) & 0b11111) != 0
    medium_cloud = ((qa_pixel.astype(np.uint16) >> 8) & 3) >= 2
    bad_radsat = (qa_radsat.astype(np.uint16) & (0b1111110 | (1 << 11))) != 0
    return ~(bad_pixel | medium_cloud | bad_radsat)


def scale_reflectance(dn: np.ndarray) -> np.ndarray:
    return np.asarray(dn, dtype=np.float32) * 0.0000275 - 0.2


def scale_temperature(dn: np.ndarray) -> np.ndarray:
    return np.asarray(dn, dtype=np.float32) * 0.00341802 + 149.0 - 273.15


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    return np.divide(a - b, denominator, out=np.full_like(a, np.nan, dtype=np.float32),
                     where=np.abs(denominator) > 1e-6)


def surface_descriptors(sr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {f"sr_{key}": value for key, value in sr.items()}
    result.update(ndvi=normalized_difference(sr["nir08"], sr["red"]),
                  ndbi=normalized_difference(sr["swir16"], sr["nir08"]),
                  ndwi=normalized_difference(sr["green"], sr["nir08"]))
    # Liang narrowband weights applied to directional OLI surface reflectance.
    # A spectral proxy, NOT a BRDF-corrected hemispherical albedo measurement.
    # Landsat 8 application: https://doi.org/10.3390/rs13040799
    result["albedo_proxy"] = (0.356 * sr["blue"] + 0.130 * sr["red"] +
                              0.373 * sr["nir08"] + 0.085 * sr["swir16"] +
                              0.072 * sr["swir22"] - 0.0018)
    return result


def aggregate_patch(raw: dict[str, np.ndarray], src_transform: Any, src_crs: Any,
                    dst_transform: Any, dst_crs: Any, shape: tuple[int, int],
                    min_valid_fraction: float = 0.8,
                    max_lst_uncertainty_k: float = 3.0) -> dict[str, np.ndarray]:
    """Mask on the original grid BEFORE area-weighted 100 m aggregation."""
    from rasterio.warp import reproject, Resampling
    valid = qa_valid(raw["qa_pixel"], raw["qa_radsat"]) & (raw["lwir11"] > 0)
    for band in SR_BANDS:
        valid &= (raw[band] >= 7273) & (raw[band] <= 43636)
    if "qa" in raw:
        valid &= (raw["qa"] >= 0) & (raw["qa"] * 0.01 <= max_lst_uncertainty_k)

    def warp(values: np.ndarray, nodata: float | None = None) -> np.ndarray:
        dest = np.full(shape, np.nan, dtype=np.float32)
        reproject(np.asarray(values, dtype=np.float32), dest, src_transform=src_transform,
                  src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs,
                  src_nodata=nodata, dst_nodata=np.nan, resampling=Resampling.average,
                  num_threads=1)
        return dest

    fraction = warp(valid.astype(np.float32))
    accepted = np.isfinite(fraction) & (fraction >= min_valid_fraction)
    def masked_warp(values: np.ndarray) -> np.ndarray:
        output = warp(np.where(valid, values, np.nan), np.nan)
        return np.where(accepted, output, np.nan)

    sr = {band: masked_warp(scale_reflectance(raw[band])) for band in SR_BANDS}
    result = surface_descriptors(sr)
    result.update(lst_c=masked_warp(scale_temperature(raw["lwir11"])),
                  valid_fraction=fraction,
                  snow_fraction=masked_warp(((raw["qa_pixel"] >> 5) & 1).astype(np.float32)),
                  water_fraction=masked_warp(((raw["qa_pixel"] >> 7) & 1).astype(np.float32)))
    if "qa" in raw:
        result["label_lst_uncertainty_k"] = masked_warp(raw["qa"] * 0.01)
    if "qa_aerosol" in raw:
        result["qa_aerosol_high_fraction"] = masked_warp(((raw["qa_aerosol"] >> 6) & 3) == 3)
    return result


def patch_origins(region: dict, count: int, patch_size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Spatially stratified patch origins, expressed in the region's 100 m grid."""
    nrows, ncols = region["grid_shape"]
    sides = math.ceil(math.sqrt(count))
    candidates = []
    for i in range(sides):
        for j in range(sides):
            lo_r, hi_r = int(i * nrows / sides), int((i + 1) * nrows / sides) - patch_size
            lo_c, hi_c = int(j * ncols / sides), int((j + 1) * ncols / sides) - patch_size
            if hi_r >= lo_r and hi_c >= lo_c:
                candidates.append((int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))))
    rng.shuffle(candidates)
    return candidates[:count]


def sample_scene(item: dict, region: dict, samples_per_scene: int = 200, seed: int = 42,
                 max_patches: int = 24, patch_size: int = 8,
                 min_valid_fraction: float = 0.8,
                 max_lst_uncertainty_k: float = 3.0) -> pd.DataFrame:
    """Read small signed COG windows, aggregate, and sample aligned 100 m cells.

    max_patches bounds network reads. Samples may be fewer than requested if
    cloud/scene boundaries leave insufficient valid cells. Never silently invent.
    """
    import planetary_computer
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    configure_safe_logging()
    if not 1 <= samples_per_scene <= 2000 or not 1 <= max_patches <= 64:
        raise ValueError("Require samples_per_scene 1..2000 and max_patches 1..64")
    entropy = int(hashlib.sha256(f"{seed}:{region['id']}:{item['id']}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(entropy)
    x0, _, _, y1 = region["extent_m"]
    dst_crs = f"EPSG:{region['epsg']}"
    to_lonlat = Transformer.from_crs(dst_crs, 4326, always_xy=True)
    asset_keys = [*REQUIRED_ASSETS, *(b for b in ("qa", "qa_aerosol") if b in item["assets"])]
    rows, reserve = [], []
    quota = max(1, math.ceil(samples_per_scene / min(max_patches, 8)))
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
               GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="90",
               GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_CACHEMAX=64 * 1024 * 1024,
               VSI_CACHE=False, GDAL_NUM_THREADS="1")
    with rasterio.Env(**env), ExitStack() as stack:
        datasets = {b: stack.enter_context(rasterio.open(planetary_computer.sign(item["assets"][b]["href"])))
                    for b in asset_keys}
        base = datasets["lwir11"]
        if any(ds.crs != base.crs or ds.transform != base.transform or ds.shape != base.shape
               for ds in datasets.values()):
            raise ValueError("Assets do not share an aligned native grid")
        for row0, col0 in patch_origins(region, max_patches, patch_size, rng):
            left, top = x0 + col0 * 100, y1 - row0 * 100
            dst_transform = from_origin(left, top, 100, 100)
            bounds = transform_bounds(dst_crs, base.crs, left, top - patch_size * 100,
                                      left + patch_size * 100, top, densify_pts=21)
            window_float = from_bounds(*bounds, transform=base.transform)
            c0, r0 = math.floor(window_float.col_off) - 2, math.floor(window_float.row_off) - 2
            c1 = math.ceil(window_float.col_off + window_float.width) + 2
            r1 = math.ceil(window_float.row_off + window_float.height) + 2
            if c1 <= 0 or r1 <= 0 or c0 >= base.width or r0 >= base.height:
                continue
            window = Window(c0, r0, c1 - c0, r1 - r0)
            if window.width * window.height > 100000:
                raise ValueError("Native patch exceeds the safety bound; check region CRS")
            raw = {b: ds.read(1, window=window, boundless=True, fill_value=(1 if b == "qa_pixel" else 0))
                   for b, ds in datasets.items()}
            values = aggregate_patch(raw, base.window_transform(window), base.crs,
                                     dst_transform, dst_crs, (patch_size, patch_size),
                                     min_valid_fraction, max_lst_uncertainty_k)
            valid = np.isfinite(values["lst_c"])
            for name in SURFACE_FEATURES:
                valid &= np.isfinite(values[name])
            choices = np.argwhere(valid)
            rng.shuffle(choices)
            primary_count = min(quota, samples_per_scene - len(rows))
            for choice_index, (rr, cc) in enumerate(choices):
                grid_row, grid_col = row0 + int(rr), col0 + int(cc)
                xx, yy = x0 + (grid_col + .5) * 100, y1 - (grid_row + .5) * 100
                lon, lat = to_lonlat.transform(xx, yy)
                props = item["properties"]
                record = {"region_id": region["id"], "scene_id": item["id"],
                          "datetime_utc": props["datetime"], "longitude": lon, "latitude": lat,
                          "grid_row": grid_row, "grid_col": grid_col,
                          "pixel_id": f"{region['id']}:{grid_row}:{grid_col}",
                          "pixel_x": xx, "pixel_y": yy, "pixel_epsg": region["epsg"],
                          "platform": props["platform"], "label_resolution_m": 100,
                          "label_condition": "clear_sky_daytime",
                          "scene_cloud_cover": props.get("eo:cloud_cover", np.nan),
                          "scene_sun_elevation": props.get("view:sun_elevation", np.nan)}
                record.update({k: float(v[rr, cc]) for k, v in values.items()})
                if choice_index < primary_count:
                    rows.append(record)
                else:
                    reserve.append(record)
            if len(rows) >= samples_per_scene:
                break
    if len(rows) < samples_per_scene:
        # Fill from valid cells already read when only a few patches intersect
        # usable data, instead of discarding those cells due to the patch quota.
        rng.shuffle(reserve)
        rows.extend(reserve[:samples_per_scene - len(rows)])
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates("pixel_id")
        result["datetime_utc"] = pd.to_datetime(result["datetime_utc"], utc=True)
    return result


def acquire_samples(areas_path: str | Path, output_dir: str | Path,
                    region_ids: list[str] | None = None,
                    date_range: str = "2023-01-01/2025-01-01",
                    max_scenes_per_region: int = 3, samples_per_scene: int = 200,
                    max_candidates_per_region: int = 36, seed: int = 42,
                    max_patches: int = 24, min_valid_fraction: float = .8,
                    max_lst_uncertainty_k: float = 3.0) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regions = json.loads(Path(areas_path).read_text())["areas"]
    selected = [r for r in regions if region_ids is None or r["id"] in region_ids]
    if region_ids and set(region_ids) - {r["id"] for r in selected}:
        raise ValueError("Unknown region ID")
    config = dict(version=VERSION, date_range=date_range, max_scenes_per_region=max_scenes_per_region,
                  samples_per_scene=samples_per_scene, max_candidates_per_region=max_candidates_per_region,
                  seed=seed, max_patches=max_patches, min_valid_fraction=min_valid_fraction,
                  max_lst_uncertainty_k=max_lst_uncertainty_k)
    manifest = {"configuration": config, "regions": selected, "scenes": [], "failures": [],
                "surface_feature_allowlist": SURFACE_FEATURES,
                "label_limitations": ["Clear-sky daytime satellite labels only",
                    "Area averages of 30 m resampled Landsat thermal product; native thermal information is ~100 m",
                    "Arithmetic area-average retrieved temperature approximates 100 m LST; it is not emissivity-weighted radiometric T^4 aggregation",
                    "Snow retained; water flagged but ocean masking remains a separate step",
                    "Low-cloud scene selection is not representative of all weather",
                    "albedo_proxy is directional reflectance weighted by Liang coefficients, without BRDF correction",
                    "label uncertainty and QA columns must not enter the model predictor matrix"]}
    frames = []
    for region in selected:
        items = search_scenes(region, date_range, output / "stac", max_scenes_per_region,
                              max_candidates_per_region)
        successful = 0
        # Acquisition attempts are bounded separately from cheap metadata search.
        for item in items[:max_scenes_per_region * 3]:
            if successful >= max_scenes_per_region:
                break
            # Expanding dates/scene counts reuses already sampled scenes.
            key = _hash({"version": VERSION, "region": region, "scene_id": item["id"],
                         "sampling_revision": "coverage-and-reserve-v2",
                         "samples_per_scene": samples_per_scene, "seed": seed,
                         "max_patches": max_patches, "min_valid_fraction": min_valid_fraction,
                         "max_lst_uncertainty_k": max_lst_uncertainty_k})
            path = output / "scenes" / f"{region['id']}_{item['id']}_{key}.parquet"
            try:
                if path.exists():
                    frame = pd.read_parquet(path)
                else:
                    frame = sample_scene(item, region, samples_per_scene, seed, max_patches,
                                         min_valid_fraction=min_valid_fraction,
                                         max_lst_uncertainty_k=max_lst_uncertainty_k)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp.parquet")
                    frame.to_parquet(tmp, index=False)
                    tmp.replace(path)
                manifest["scenes"].append({"region_id": region["id"], "scene_id": item["id"],
                                            "samples": len(frame), "file": str(path),
                                            "requested_samples": samples_per_scene,
                                            "underfilled": len(frame) < samples_per_scene,
                                            "roi_footprint_fraction": item.get("_pilot_roi_coverage_fraction"),
                                            "stac_item": f"{STAC_URL}/collections/{COLLECTION}/items/{item['id']}"})
                LOG.info("%s %s: %d samples", region["id"], item["id"], len(frame))
                if not frame.empty:
                    successful += 1
                    frames.append(frame)
            except Exception as exc:
                # Strip signed URLs from error text to avoid storing SAS tokens.
                manifest["failures"].append({"region_id": region["id"], "scene_id": item["id"],
                                               "error_type": type(exc).__name__, "detail": _safe_error(exc)})
                LOG.warning("%s %s failed: %s: %s", region["id"], item["id"], type(exc).__name__, _safe_error(exc))
            _save_json(output / "satellite_manifest.json", manifest)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(["region_id", "scene_id", "pixel_id"])
    combined.to_parquet(output / "satellite_samples.parquet", index=False)
    manifest["total_samples"] = len(combined)
    manifest["completed_utc"] = _iso(datetime.now(timezone.utc))
    _save_json(output / "satellite_manifest.json", manifest)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areas", type=Path, default=Path("pilot/areas_resolved.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", action="append", dest="regions")
    parser.add_argument("--date-range", default="2023-01-01/2025-01-01")
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--samples-per-scene", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=36)
    parser.add_argument("--max-patches", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_safe_logging()
    frame = acquire_samples(args.areas, args.output, args.regions, args.date_range,
                            args.max_scenes, args.samples_per_scene, args.max_candidates,
                            args.seed, args.max_patches)
    print(json.dumps({"samples": len(frame), "output": str(args.output / "satellite_samples.parquet")}))
    if frame.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
