"""Display geometry must not draw a polar tile across most of the world."""
import math

import numpy as np
import pytest
from shapely.geometry import shape

from lst_global.grid import TILE_METRES, Tile, iter_tiles, zones


def zone(name):
    return next(z for z in zones() if z.id == name)


def polygons(geometry):
    return list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]


@pytest.mark.parametrize("name,row", [("ups-n", 40), ("ups-s", 18)])
def test_polar_dateline_footprints_split_into_valid_local_polygons(name, row):
    tile = Tile(zone(name), 39, row)
    feature = tile.display_feature()
    assert feature["properties"]["display_status"] == "ready"
    geometry = shape(feature["geometry"])
    assert geometry.is_valid and geometry.geom_type == "MultiPolygon"
    parts = polygons(geometry)
    assert any(p.bounds[0] <= -179 for p in parts)
    assert any(p.bounds[2] >= 179 for p in parts)
    assert all(p.bounds[2] - p.bounds[0] < 180 for p in parts)
    assert all(-180 <= p.bounds[0] <= p.bounds[2] <= 180 for p in parts)


@pytest.mark.parametrize("name,latitude", [("ups-n", 90), ("ups-s", -90)])
def test_pole_containing_footprint_is_explicitly_suppressed(name, latitude):
    z = zone(name)
    x, y = z.forward.transform(0, latitude)
    tile = Tile(z, math.floor(x / TILE_METRES), math.floor(y / TILE_METRES))
    before = tile.centres()
    feature = tile.display_feature()
    assert feature["geometry"] is None
    assert feature["properties"]["display_status"] == "pole_footprint_suppressed"
    assert feature["properties"]["projected_bounds"] == list(tile.bounds)
    after = tile.centres()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bounds", [
    (179.8, -17, -179.8, -16.8),
    (-.3, 51.4, -.1, 51.6),
    (-20, 88, 20, 89),
    (-20, -89, 20, -88),
])
def test_display_preserves_registry_and_returns_only_valid_local_parts(bounds):
    tiles = list(iter_tiles(bounds))
    ids = [tile.id for tile in tiles]
    displayed = 0
    for tile in tiles:
        feature = tile.display_feature()
        assert feature["properties"]["tile_id"] == tile.id
        if feature["geometry"] is not None:
            displayed += 1
            geometry = shape(feature["geometry"])
            assert geometry.is_valid
            assert all(p.bounds[2] - p.bounds[0] <= 180 for p in polygons(geometry))
    assert displayed > 0
    assert [tile.id for tile in iter_tiles(bounds)] == ids
