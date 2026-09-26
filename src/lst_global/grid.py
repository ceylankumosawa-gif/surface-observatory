"""A deterministic 100 m UTM/UPS tile registry, including both polar caps.

Standard six-degree strips are used consistently (no national UTM exceptions).
Tiles are 512 square cells, aligned to projected coordinate zero. Ownership is
decided by pixel centres in half-open geographic strips, not overlapping tile
rectangles. Ocean/land classification is a separate input preparation step.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from functools import cached_property

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, box, mapping
from shapely.ops import transform, unary_union

RESOLUTION_M = 100
TILE_CELLS = 512
TILE_METRES = RESOLUTION_M * TILE_CELLS
GRID_VERSION = "utm-ups-100m-512-v1"


@dataclass(frozen=True)
class Zone:
    id: str
    epsg: int
    west: float
    south: float
    east: float
    north: float

    @cached_property
    def forward(self):
        return Transformer.from_crs(4326, self.epsg, always_xy=True)

    @cached_property
    def inverse(self):
        return Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @cached_property
    def projected_domain(self):
        # 0.1 degree chords approximate registry boundaries only. Pixel-centre
        # ownership below is authoritative and uses the inverse CRS transform.
        geographic = box(self.west, self.south, self.east, self.north)
        return transform(self.forward.transform, geographic.segmentize(.1)).buffer(0)

    def owns(self, longitude, latitude):
        lon = (np.asarray(longitude) + 180) % 360 - 180
        lat = np.asarray(latitude)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if self.id == "ups-n":
            return valid & (lat >= 84) & (lat <= 90)
        if self.id == "ups-s":
            return valid & (lat >= -90) & (lat < -80)
        return (valid & (lon >= self.west) & (lon < self.east)
                & (lat >= self.south) & (lat < self.north))


def zones():
    for number in range(1, 61):
        west = -180 + (number - 1) * 6
        yield Zone(f"utm-{number:02d}n", 32600 + number, west, 0, west + 6, 84)
        yield Zone(f"utm-{number:02d}s", 32700 + number, west, -80, west + 6, 0)
    yield Zone("ups-n", 32661, -180, 84, 180, 90)
    yield Zone("ups-s", 32761, -180, -90, 180, -80)


def geographic_boxes(bounds):
    """WGS84 bbox; west > east explicitly requests an antimeridian crossing."""
    if len(bounds) != 4 or any(isinstance(x, bool) for x in bounds):
        raise ValueError("Use four finite longitude/latitude bounds.")
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(x) for x in (west, south, east, north)):
        raise ValueError("Bounds must be finite.")
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid geographic bounds.")
    if west == east or (west == 180 and east == -180):
        raise ValueError("The longitude extent must have positive width.")
    if west < east:
        return [box(west, south, east, north)]
    return [b for b in (box(west, south, 180, north), box(-180, south, east, north)) if b.area > 0]


@dataclass(frozen=True)
class Tile:
    zone: Zone
    column: int
    row: int

    @property
    def id(self):
        return f"g100-{self.zone.id}-{self.column}-{self.row}"

    @property
    def bounds(self):
        x, y = self.column * TILE_METRES, self.row * TILE_METRES
        return x, y, x + TILE_METRES, y + TILE_METRES

    def centres(self):
        """North-up raster centres and zone ownership; one bounded tile only."""
        left, bottom, right, top = self.bounds
        x = left + (np.arange(TILE_CELLS) + .5) * RESOLUTION_M
        y = top - (np.arange(TILE_CELLS) + .5) * RESOLUTION_M
        xx, yy = np.meshgrid(x, y)
        lon, lat = self.zone.inverse.transform(xx, yy)
        return lon, lat, self.zone.owns(lon, lat)

    def display_feature(self):
        """Small tile footprint for QA, clipped to this zone's ownership domain."""
        clipped = box(*self.bounds).intersection(self.zone.projected_domain)
        feature = {"type": "Feature", "properties": {
            "tile_id": self.id, "epsg": self.zone.epsg,
            "resolution_m": RESOLUTION_M, "support": "global_extrapolation",
            "projected_bounds": list(self.bounds), "display_status": "ready"},
            "geometry": None}
        if clipped.is_empty:
            feature["properties"]["display_status"] = "outside_projected_domain"
            return feature

        pieces = [clipped]
        if self.zone.id.startswith("ups-"):
            # A polygon enclosing a pole has no single finite longitude branch.
            # Keep the metric footprint authoritative until a dedicated cap
            # representation is available in the viewer.
            pole_x, pole_y = self.zone.forward.transform(
                0, 90 if self.zone.id == "ups-n" else -90)
            if clipped.intersects(Point(pole_x, pole_y)):
                feature["properties"]["display_status"] = "pole_footprint_suppressed"
                return feature
            # UPS meridians are straight lines through its projected pole.
            # Quadrants separate the antimeridian before inverse projection;
            # this prevents a short polar edge becoming a 359-degree chord.
            left, bottom, right, top = self.zone.projected_domain.bounds
            pieces = [clipped.intersection(box(*bounds)) for bounds in (
                (left, bottom, pole_x, pole_y), (pole_x, bottom, right, pole_y),
                (left, pole_y, pole_x, top), (pole_x, pole_y, right, top))]
            pieces = [piece for piece in pieces if piece.area > 0]

        geographic_pieces = []
        for piece in pieces:
            if self.zone.id.startswith("ups-"):
                point = piece.representative_point()
                anchor, _ = self.zone.inverse.transform(point.x, point.y)
            else:
                anchor = (self.zone.west + self.zone.east) / 2

            def inverse_on_branch(x, y, z=None):
                lon, lat = self.zone.inverse.transform(x, y)
                lon = anchor + (np.asarray(lon) - anchor + 180) % 360 - 180
                # Only endpoint roundoff can exceed the chosen quadrant/zone's
                # legal WGS84 range after longitude-branch normalization.
                return np.clip(lon, -180, 180), np.clip(lat, -90, 90)

            geographic_pieces.append(transform(inverse_on_branch, piece.segmentize(1000)))
        # Merge adjacent pieces on the same branch. Pieces on opposite sides
        # of the antimeridian remain separate valid MultiPolygon members.
        geographic = unary_union(geographic_pieces)
        polygons = list(geographic.geoms) if geographic.geom_type == "MultiPolygon" else [geographic]
        if (not geographic.is_valid or geographic.geom_type not in ("Polygon", "MultiPolygon")
                or any(p.bounds[2] - p.bounds[0] > 180 + 1e-8 for p in polygons)):
            feature["properties"]["display_status"] = "unrepresentable_geographic_footprint"
            return feature
        feature["geometry"] = mapping(geographic)
        return feature


def iter_tiles(bounds=(-180, -90, 180, 90)):
    """Stream candidate tiles without allocating a global raster or cell table.

    A 100 m guard makes chord approximations conservative at curved edges.
    Exact geographic ownership masks remove padding before raster production.
    Registry tiles may include ocean or have no selected pixel centres.
    """
    selections = geographic_boxes(bounds)
    for zone in zones():
        emitted = set()
        for selection in selections:
            clipped = selection.intersection(box(zone.west, zone.south, zone.east, zone.north))
            if clipped.is_empty or clipped.area == 0:
                continue
            projected = transform(zone.forward.transform, clipped.segmentize(.1)).buffer(0)
            if projected.is_empty or not all(math.isfinite(x) for x in projected.bounds):
                raise ValueError(f"Cannot project selection in {zone.id}.")
            # The 100 m guard is larger than the chord deviation for 0.1 degree
            # domain segments. It only admits extra candidate tiles.
            candidate = projected.buffer(RESOLUTION_M)
            left, bottom, right, top = candidate.bounds
            for row in range(math.floor(bottom / TILE_METRES), math.ceil(top / TILE_METRES)):
                for col in range(math.floor(left / TILE_METRES), math.ceil(right / TILE_METRES)):
                    tile = Tile(zone, col, row)
                    if tile.id not in emitted and box(*tile.bounds).intersects(candidate):
                        emitted.add(tile.id)
                        yield tile
