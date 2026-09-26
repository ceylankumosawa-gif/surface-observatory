"""Small-window Copernicus GLO-30 surface elevation descriptors for pilot samples.

The source is a DSM, including vegetation/buildings, not a bare-ground DTM.
https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-30
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .satellite import STAC_URL, _hash, _save_json, _session, _safe_error, configure_safe_logging

COLLECTION = "cop-dem-glo-30"
VERSION = "copdem-100m-v1"
FEATURES = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "terrain_relief_300m"]
LOG = logging.getLogger(__name__)


def terrain_descriptors(elevations: np.ndarray, cell_size: float = 100.) -> dict[str, float]:
    """Horn gradient on a 3x3 north-up 100 m DSM; aspect is downslope clockwise from north.

    A flat surface receives zero sine/cosine, avoiding a fictional orientation.
    """
    z = np.asarray(elevations, dtype=float)
    empty = {name: float("nan") for name in FEATURES}
    if z.shape != (3, 3):
        raise ValueError("Expected a 3x3 elevation stencil")
    if not np.isfinite(z[1, 1]):
        return empty
    empty["elevation_m"] = float(z[1, 1])
    if not np.all(np.isfinite(z)):
        return empty
    dx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8*cell_size)
    # Northward rise: row index decreases northward.
    dy = ((z[0, 0] + 2*z[0, 1] + z[0, 2]) - (z[2, 0] + 2*z[2, 1] + z[2, 2])) / (8*cell_size)
    gradient = float(np.hypot(dx, dy))
    return {"elevation_m": float(z[1, 1]), "slope_deg": float(np.degrees(np.arctan(gradient))),
            "aspect_sin": float(-dx / gradient) if gradient > 1e-8 else 0.,
            "aspect_cos": float(-dy / gradient) if gradient > 1e-8 else 0.,
            "terrain_relief_300m": float(z.max() - z.min())}


def _items(points: pd.DataFrame, cache: Path, max_tiles: int = 24) -> list[dict]:
    # A 0.01 degree envelope includes the 300 m stencil, even at polar pilot sites.
    bbox = [float(points.longitude.min() - .02), float(points.latitude.min() - .01),
            float(points.longitude.max() + .02), float(points.latitude.max() + .01)]
    path = cache / f"stac_{_hash(bbox)}.json"
    if path.exists():
        return json.loads(path.read_text())["items"]
    with _session() as session:
        response = session.post(STAC_URL + "/search", json={"collections": [COLLECTION],
                                "bbox": bbox, "limit": max_tiles + 1}, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
    items = data.get("features", [])
    if len(items) > max_tiles or any(link.get("rel") == "next" for link in data.get("links", [])):
        raise ValueError("Terrain request exceeds bounded tile budget; split the region")
    items = [item for item in items if "data" in item.get("assets", {})]
    _save_json(path, {"bbox": bbox, "items": items})
    return items


def _sample_pixels(points: pd.DataFrame, items: list[dict]) -> pd.DataFrame:
    import planetary_computer
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform_bounds, Resampling
    configure_safe_logging()
    if not items:
        return pd.DataFrame([{**{"pixel_id": p}, **{k: np.nan for k in FEATURES}}
                             for p in points.pixel_id])
    output = []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
               GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="90",
               GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_CACHEMAX=64 * 1024 * 1024,
               VSI_CACHE=False, GDAL_NUM_THREADS="1")
    with rasterio.Env(**env), ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(planetary_computer.sign(item["assets"]["data"]["href"])))
                   for item in items]
        source_crs = sources[0].crs
        if any(src.crs != source_crs for src in sources):
            raise ValueError("DEM tiles have differing CRS")
        for row in points.itertuples(index=False):
            dst_crs = f"EPSG:{int(row.pixel_epsg)}"
            x, y = float(row.pixel_x), float(row.pixel_y)
            # Native windows include a border around the 3x3 target cells.
            bounds = transform_bounds(dst_crs, source_crs, x - 180, y - 180,
                                      x + 180, y + 180, densify_pts=21)
            rx = min(abs(src.res[0]) for src in sources)
            ry = min(abs(src.res[1]) for src in sources)
            if ((bounds[2]-bounds[0])/rx) * ((bounds[3]-bounds[1])/ry) > 100000:
                raise ValueError("DEM read exceeds native-window bound")
            mosaic, src_transform = merge(sources, bounds=bounds, res=(rx, ry),
                                          nodata=np.nan, dtype="float32", indexes=[1])
            target = np.full((3, 3), np.nan, dtype=np.float32)
            reproject(mosaic[0], target, src_transform=src_transform, src_crs=source_crs,
                      src_nodata=np.nan, dst_transform=from_origin(x - 150, y + 150, 100, 100),
                      dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.average,
                      num_threads=1)
            output.append({"pixel_id": row.pixel_id, **terrain_descriptors(target)})
    return pd.DataFrame(output)


def add_terrain(frame: pd.DataFrame, cache_dir: str | Path, max_unique_pixels: int = 50000) -> pd.DataFrame:
    """Return samples with static 100 m elevation/slope/aspect, cached per pixel.

    Raster reads run only where uncached unique pixels exist. Public COG requests
    use small window ranges, never a bulk DEM download. Missing coverage is NaN.
    """
    required = {"pixel_id", "region_id", "pixel_x", "pixel_y", "pixel_epsg", "longitude", "latitude"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing terrain inputs: {sorted(required - set(frame.columns))}")
    points = frame[list(sorted(required))].drop_duplicates("pixel_id")
    if len(points) > max_unique_pixels:
        raise ValueError("Too many unique pixels for bounded pilot terrain acquisition")
    cache = Path(cache_dir) / VERSION
    cache.mkdir(parents=True, exist_ok=True)
    records, provenance = [], {"version": VERSION, "sources": [], "failures": [],
        "dataset": COLLECTION, "vertical_reference": "EGM2008 metres",
        "acquisition_period": "Main TanDEM-X source acquisition 2011-2015, with older gap fills; static DSM",
        "interpretation": "DSM includes trees/buildings; not bare-ground height or a building shadow model",
        "processing": "Area-average to 100 m; central cell elevation; Horn slope/aspect over 3x3 of those cells",
        "features": FEATURES,
        "source_documentation": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
        "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"}
    previous_provenance = cache / "terrain_provenance.json"
    if previous_provenance.exists():
        previous = json.loads(previous_provenance.read_text())
        provenance["sources"] = previous.get("sources", [])
    for region_id, region_points in points.groupby("region_id", sort=False):
        path = cache / f"{region_id}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["pixel_id", *FEATURES])
        pending = region_points[~region_points.pixel_id.isin(old.pixel_id)]
        if not pending.empty:
            try:
                items = _items(pending, cache)
                new = _sample_pixels(pending, items)
                old = pd.concat([old, new], ignore_index=True).drop_duplicates("pixel_id", keep="last")
                tmp = path.with_suffix(".tmp.parquet")
                old.to_parquet(tmp, index=False)
                tmp.replace(path)
                provenance["sources"].extend({"region_id": region_id, "item_id": item["id"],
                    "stac_item": f"{STAC_URL}/collections/{COLLECTION}/items/{item['id']}"} for item in items)
                LOG.info("Terrain %s: %d new pixels", region_id, len(new))
            except Exception as exc:
                provenance["failures"].append({"region_id": region_id, "error_type": type(exc).__name__, "detail": _safe_error(exc)})
                LOG.warning("Terrain %s failed: %s: %s", region_id, type(exc).__name__, _safe_error(exc))
        records.append(old)
        _save_json(cache / "terrain_provenance.json", provenance)
    terrain = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["pixel_id", *FEATURES])
    result = frame.drop(columns=[x for x in FEATURES if x in frame], errors="ignore").merge(
        terrain, on="pixel_id", how="left", validate="many_to_one", sort=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_safe_logging()
    result = add_terrain(pd.read_parquet(args.input), args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(result), "with_elevation": int(result.elevation_m.notna().sum())}))


if __name__ == "__main__":
    main()
