"""Build small planning geometries only. No imagery, feature extraction or ML.

Requires pyproj and shapely. Run from any directory. Projected extents in
areas_resolved.json are authoritative; GeoJSON edges are densified for display.
"""
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import transform

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "areas.json").read_text())
resolution = manifest["resolution_m"]
features, resolved = [], []

for area in manifest["areas"]:
    forward = Transformer.from_crs(4326, area["epsg"], always_xy=True)
    inverse = Transformer.from_crs(area["epsg"], 4326, always_xy=True)
    width = area["width_km"] * 1000
    if "fixed_extent_m" in area:
        xmin, ymin, xmax, ymax = area["fixed_extent_m"]
    else:
        x, y = forward.transform(*area["center_lonlat"])
        xmin = round((x - width / 2) / resolution) * resolution
        ymin = round((y - width / 2) / resolution) * resolution
        xmax, ymax = xmin + width, ymin + width
    assert all(v % resolution == 0 for v in [xmin, ymin, xmax, ymax])
    assert xmax - xmin == ymax - ymin == width
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    ring = []
    for start, end in zip(corners, corners[1:]):
        for index in range(20):
            t = index / 20
            ring.append(inverse.transform(start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])))
    ring.append(ring[0])
    geom = Polygon(ring)
    assert geom.is_valid
    item = dict(area, extent_m=[xmin, ymin, xmax, ymax],
                grid_shape=[int(width / resolution)] * 2,
                pixels_before_mask=int((width / resolution) ** 2),
                resolved_center_lonlat=list(inverse.transform((xmin + xmax) / 2, (ymin + ymax) / 2)))
    features.append({"type": "Feature", "properties": item, "geometry": mapping(geom)})
    resolved.append(item)

source = json.loads((ROOT / "london_boroughs_27700.json").read_text())
assert len(source["features"]) == 33
# Esri ring order is not needed here: checking every ring's extent establishes
# containment of every borough, including multipart boundaries and islands.
all_points = [p for f in source["features"] for ring in f["geometry"]["rings"] for p in ring]
london = next(a for a in resolved if a["id"] == "greater_london")
tile = box(*london["extent_m"])
official_bounds = manifest["london_source_extent_epsg27700"]
assert tile.covers(box(*official_bounds).buffer(10000))
assert all(tile.covers(Point(p)) for p in all_points)
to_wgs84 = Transformer.from_crs(27700, 4326, always_xy=True).transform
boroughs = []
for feature in source["features"]:
    # Symmetric difference preserves Esri outer rings and interior holes without
    # relying on winding conventions; source was simplified to 50 m for display.
    rings = [Polygon(ring) for ring in feature["geometry"]["rings"]]
    geom = rings[0]
    for ring in rings[1:]:
        geom = geom.symmetric_difference(ring)
    boroughs.append({"type": "Feature", "properties": feature["attributes"], "geometry": mapping(transform(to_wgs84, geom))})

total = sum(a["pixels_before_mask"] for a in resolved)
summary = {
    "areas": len(resolved), "pixels_before_mask": total,
    "area_km2_before_mask": total * resolution ** 2 / 1e6,
    "float32_temperature_bytes_per_hour": total * 4,
    "float32_temperature_GB_per_365_day_year": total * 4 * 24 * 365 / 1e9,
    "london_boroughs_verified": 33,
    "london_official_extent_plus_10km_contained": True,
    "geometry_note": "Local projected grids at 100 m. WGS84 GeoJSON is for display; projected extents define cells. London display boundary simplified to 50 m; containment also verified against full source extent."
}
for filename, data in [
    ("pilot_tiles.geojson", {"type": "FeatureCollection", "features": features}),
    ("london_boroughs.geojson", {"type": "FeatureCollection", "features": boroughs}),
    ("areas_resolved.json", {**manifest, "areas": resolved, "summary": summary})
]:
    (ROOT / filename).write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(summary, indent=2))
