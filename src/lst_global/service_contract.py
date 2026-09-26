"""Bounded public request geometry; no downloads or model execution."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from pyproj import Transformer
from shapely.geometry import shape, box
from shapely.ops import transform

from lst_pilot.api import PolygonInput
from .grid import iter_tiles, zones

RESOLUTIONS = (100, 250, 500, 1000)
AREA_LIMITS = {100: 6400, 250: 14400, 500: 25600, 1000: 25600}
MAX_OUTPUT_PIXELS = 800_000
MAX_SOURCE_TILES = 16
MAX_PENDING = 3
JOB_TIMEOUT_SECONDS = 1800
MIN_DATE = "2021-02-01T00:00:00Z"


class GlobalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    polygon: PolygonInput
    datetime_utc: str = Field(max_length=48)
    resolution_m: StrictInt = 100


def date_range(now=None, operational=False):
    now = now or datetime.now(timezone.utc)
    # The exact source availability is also checked by the worker. Never
    # advertise forecast substitution as equivalent to historical ERA5.
    maximum = (now if operational else now - timedelta(days=7)).replace(minute=0, second=0, microsecond=0)
    return {"min": MIN_DATE, "max": maximum.isoformat().replace("+00:00", "Z"),
            "timezone": "UTC", "time_step_minutes": 60,
            "current_conditions_available": operational,
            "description": ('Historical ERA5/ERA5T; the most recent days use experimental GFS weather and verified station reports. Source changes are not accuracy-validated.'
                            if operational else 'Historical requests only; recent-date weather integration is under validation.')}


def validate(payload: GlobalInput, now=None, operational=False):
    resolution = payload.resolution_m
    if resolution not in RESOLUTIONS:
        raise ValueError("Choose 100, 250, 500 or 1000 metres.")
    try:
        stamp = datetime.fromisoformat(payload.datetime_utc.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError()
        stamp = stamp.astimezone(timezone.utc)
    except (TypeError, ValueError):
        raise ValueError("Choose a date and time with a UTC offset.") from None
    if stamp.minute or stamp.second or stamp.microsecond:
        raise ValueError("Choose an exact UTC hour; the interface shows its local equivalent.")
    dates = date_range(now,operational)
    if not datetime.fromisoformat(dates['min'].replace('Z', '+00:00')) <= stamp <= datetime.fromisoformat(dates['max'].replace('Z', '+00:00')):
        raise ValueError(f"Available request dates: {dates['min']} through {dates['max']}. Source availability is checked separately.")
    geo = payload.polygon.model_dump()
    rings = geo['coordinates']
    if len(rings) != 1 or not 4 <= len(rings[0]) <= 512:
        raise ValueError("Draw one polygon with 4–512 points and no holes.")
    ring = rings[0]
    if ring[0] != ring[-1] or any(len(p) != 2 or not all(math.isfinite(x) for x in p)
                               or not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90) for p in ring):
        raise ValueError("Use finite geographic coordinates and close the polygon.")
    poly = shape(geo)
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        raise ValueError("Draw a valid polygon without self intersections.")
    west, south, east, north = poly.bounds
    if west < -179 or east > 179 or south <= -80 or north >= 84 or east - west > 180:
        raise ValueError("Polar and dateline source windows are not ready yet; this request cannot currently be processed.")
    # Reject very large inputs before constructing candidate tiles.
    if east - west > 12 or north - south > 3:
        raise ValueError("The selection is too large for this server. Draw a smaller area.")
    center = poly.centroid
    zone = next(z for z in zones() if bool(z.owns(center.x, center.y)))
    # Bound densification even for a valid, highly winding 512-point outline.
    # An adversarial perimeter must not allocate millions of validation points.
    projected = transform(zone.forward.transform, poly.segmentize(max(.01,poly.length/4096)))
    left, bottom, right, top = projected.bounds
    bounds = [math.floor(left/resolution)*resolution, math.floor(bottom/resolution)*resolution,
              math.ceil(right/resolution)*resolution, math.ceil(top/resolution)*resolution]
    width, height = round((bounds[2]-bounds[0])/resolution), round((bounds[3]-bounds[1])/resolution)
    area = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])/1e6
    if width < 1 or height < 1 or width*height > MAX_OUTPUT_PIXELS or area > AREA_LIMITS[resolution]:
        raise ValueError(f"At {resolution} m, the bounding rectangle must be at most {AREA_LIMITS[resolution]:,} km² and {MAX_OUTPUT_PIXELS:,} output pixels.")
    # A coarse output cell can extend outside the drawn polygon even when its
    # centre lies inside. Prepare its complete native footprint before masking,
    # or redraws near a tile boundary would change the available-area average.
    native_bounds = [math.floor(bounds[0]/100)*100, math.floor(bounds[1]/100)*100,
                     math.ceil(bounds[2]/100)*100, math.ceil(bounds[3]/100)*100]
    source_envelope = transform(zone.inverse.transform, box(*native_bounds).segmentize(1000))
    tiles = []
    for tile in iter_tiles(source_envelope.bounds):
        local_poly = transform(tile.zone.forward.transform, source_envelope)
        footprint = box(*tile.bounds).intersection(tile.zone.projected_domain)
        if local_poly.intersection(footprint).area < .01:
            continue
        anchor = footprint.representative_point()
        lon, lat = tile.zone.inverse.transform(anchor.x, anchor.y)
        if abs(lon) > 179:
            raise ValueError("This source tile touches the currently unsupported dateline window.")
        tiles.append({'tile_id': tile.id, 'epsg': tile.zone.epsg,
                      'longitude': lon, 'latitude': lat})
        if len(tiles) > MAX_SOURCE_TILES:
            raise ValueError(f"The area needs more than {MAX_SOURCE_TILES} source tiles. Draw a smaller area; coarser exports still require the 100 m inputs.")
    if not tiles:
        raise ValueError("No source tiles intersect the area.")
    return {'polygon': geo, 'datetime_utc': stamp.isoformat().replace('+00:00', 'Z'),
            'resolution_m': resolution, 'epsg': zone.epsg, 'bounds_m': bounds,
            'shape': [height, width], 'bbox_area_km2': area, 'source_tiles': tiles,
            'grid_id': f'utm-{zone.epsg}-{resolution}m-origin0',
            'grid_note': 'Output UTM zone follows the area centre. Exact overlap equality applies to requests on the same grid; exports in another CRS represent different cell footprints.'}


def capabilities(now=None,operational=False):
    return {'status': 'experimental', 'resolutions_m': list(RESOLUTIONS),
            'date_range': date_range(now,operational),
            'limits': {'max_area_km2_by_resolution': AREA_LIMITS, 'max_output_pixels': MAX_OUTPUT_PIXELS,
                       'max_source_tiles': MAX_SOURCE_TILES, 'max_pending': MAX_PENDING,
                       'job_timeout_seconds': JOB_TIMEOUT_SECONDS, 'hourly_per_connection': 6,
                       'new_jobs_per_day': 48, 'source_tiles_per_day': 64, 'output_retention_days': 30},
            'accuracy': {'target_mae_c': 3, 'qualified': False,
                         'description': 'The 3°C regional day/night target is not yet met. These are experimental predictions.'},
            'spatial': {'description': 'Draw land areas beyond the pilots. Polar caps and dateline windows are not yet supported. Missing inputs remain transparent.',
                        'latitude_range': [-80, 84], 'longitude_range': [-179, 179]},
            'resolution_description': 'Coarser pixels average available 100 m predictions, requiring at least 80% valid area. They do not add training detail.'}
