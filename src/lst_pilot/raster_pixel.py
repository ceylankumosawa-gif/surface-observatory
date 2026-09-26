"""Read one native cell from a local, canonical temperature GeoTIFF.

This helper does no resampling, rendering, source retrieval or model inference.
Band descriptions, not display colours or presumed band positions, name values.
"""
from __future__ import annotations

import math
from pathlib import Path

from pyproj import Transformer

BANDS = {
    "prediction": "predicted_lst_c",
    "observed": "observed_lst_c",
    "residual": "residual_prediction_minus_observed_c",
}


def _pixel_coordinate(value: float) -> float:
    # A CRS round-trip can move an exact grid edge by a few floating-point bits.
    # Snap within one millionth of a metre on the pilot's 100 m grid; otherwise
    # retain ordinary floor / half-open cell ownership, including outer edges.
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= 1e-8 else value


def read_pixel(path: Path, lon: float, lat: float) -> dict:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    if not (math.isfinite(lon) and math.isfinite(lat) and
            -180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Coordinates must be finite longitude/latitude values.")
    response = {
        "query": {"lon": lon, "lat": lat}, "status": "outside", "pixel": None,
        "values": {key: None for key in BANDS}, "units": "degC",
    }
    # Accept only the expected local file format, never an XML/VRT disguised as
    # a TIFF that might resolve other local or remote data sources.
    with rasterio.open(path, driver="GTiff") as raster:
        if raster.crs is None or not all(math.isfinite(x) for x in raster.transform):
            raise ValueError("The temperature raster has no usable georeferencing.")
        project = Transformer.from_crs(4326, raster.crs, always_xy=True)
        x, y = project.transform(lon, lat)
        if not all(math.isfinite(v) for v in (x, y)):
            return response
        column, row = (~raster.transform) * (x, y)
        if not all(math.isfinite(v) for v in (column, row)):
            return response
        column, row = _pixel_coordinate(column), _pixel_coordinate(row)
        if not (0 <= column < raster.width and 0 <= row < raster.height):
            return response
        column, row = math.floor(column), math.floor(row)
        geographic = Transformer.from_crs(raster.crs, 4326, always_xy=True)

        def location(col, line):
            point = geographic.transform(*(raster.transform * (col, line)), errcheck=True)
            if not all(math.isfinite(v) for v in point):
                raise ValueError("The temperature cell cannot be located.")
            return [float(point[0]), float(point[1])]

        response["pixel"] = {
            "row": row, "column": column, "center": location(column + .5, row + .5),
            "corners": [location(col, line) for col, line in (
                (column, row), (column + 1, row), (column + 1, row + 1),
                (column, row + 1), (column, row))],
        }
        bands = []
        for role, description in BANDS.items():
            indexes = [index for index, name in enumerate(raster.descriptions, 1) if name == description]
            if len(indexes) > 1:
                raise ValueError("The temperature raster has ambiguous band descriptions.")
            if indexes:
                bands.append((role, indexes[0]))
        if bands:
            # GDAL may decompress its containing storage block, but the requested
            # numerical array is only one pixel per recognized temperature band.
            values = raster.read([index for _, index in bands],
                                 window=Window(column, row, 1, 1), masked=True)
            for index, (role, _) in enumerate(bands):
                value = values[index, 0, 0]
                if not np.ma.is_masked(value) and math.isfinite(float(value)):
                    response["values"][role] = float(value)
        response["status"] = "ok" if any(value is not None for value in response["values"].values()) else "nodata"
    return response
