"""Bounded label-free historical raster smoke runs outside the pilot catalog.

This engineering adapter reuses the F40 research feature definitions exactly;
it intentionally does not call the older website's scenario implementation.
The point-weather delivery adapter is suitable for these small runs only. Bulk
worldwide weather preparation remains a separate implementation milestone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from lst_pilot import assemble, option_b_features as surface, radiation, raster, weather, satellite, context, stations
from .grid import TILE_METRES, RESOLUTION_M, Tile, zones
from .planner import utc_hour, sha
from .model import FrozenF, FEATURES, NUMERIC_FEATURES, CLIMATE_CLASSES, MODEL_SHA256
from .receipts import STATION_OK

REASONS = {0: "predicted_global_extrapolation", 1: "outside_zone", 2: "missing_land_mask",
           3: "insufficient_land_fraction", 4: "water_or_mixed_water",
           5: "missing_predictor", 6: "climate_not_in_frozen_fit", 7: "twilight_not_in_fit",
           8: "no_verified_recent_station"}


def patch_area(longitude, latitude, cells):
    if (isinstance(cells, bool) or not isinstance(cells, int) or not 1 <= cells <= 512
            or not all(math.isfinite(x) for x in (longitude, latitude))
            or not -180 <= longitude <= 180 or not -90 <= latitude <= 90):
        raise ValueError("Use a finite location and 1–512 cells per side.")
    longitude = (longitude + 180) % 360 - 180
    zone = next(z for z in zones() if bool(z.owns(longitude, latitude)))
    # Source wrappers currently assume a local, non-dateline geographic bbox.
    # Registry coverage still includes these areas; do not make unsafe queries.
    if zone.id.startswith("ups") or abs(longitude) > 179:
        raise ValueError("This initial source adapter excludes polar/dateline patches; the global registry retains them as awaiting_inputs.")
    x, y = zone.forward.transform(longitude, latitude)
    tile = Tile(zone, math.floor(x / TILE_METRES), math.floor(y / TILE_METRES))
    tx, ty, tr, tt = tile.bounds
    left = max(tx, min(tr - cells * 100, (math.floor(x / 100) - cells // 2) * 100))
    bottom = max(ty, min(tt - cells * 100, (math.floor(y / 100) - cells // 2) * 100))
    right, top = left + cells * 100, bottom + cells * 100
    area = {"id": f"{tile.id}-patch-{int(left)}-{int(bottom)}-{cells}", "epsg": zone.epsg,
            "extent_m": [left, bottom, right, top], "grid_shape": [cells, cells],
            "tile_id": tile.id, "zone_id": zone.id}
    rows, cols = np.indices((cells, cells))
    lon, lat = zone.inverse.transform(left + (cols.ravel() + .5) * 100, top - (rows.ravel() + .5) * 100)
    frame = pd.DataFrame({"region_id": area["id"], "grid_row": rows.ravel(), "grid_col": cols.ravel(),
                          "longitude": lon, "latitude": lat, "epsg": zone.epsg,
                          "zone_owned": zone.owns(lon, lat)})
    frame["sample_id"] = [f"{area['id']}:{r}:{c}" for r, c in zip(frame.grid_row, frame.grid_col)]
    return area, frame


def restore_rows(before, after):
    if after.sample_id.duplicated().any() or len(after) != len(before) or set(after.sample_id) != set(before.sample_id):
        raise ValueError("Source adapter changed pixel identities.")
    result = after.set_index("sample_id").loc[before.sample_id].reset_index()
    for name in ("longitude", "latitude", "grid_row", "grid_col", "datetime_utc", "zone_owned", "region_id", "epsg"):
        pd.testing.assert_series_equal(before[name].reset_index(drop=True), result[name], check_names=False, check_exact=True)
    return result


def support_reasons(frame):
    reason = np.zeros(len(frame), np.uint8)
    def exclude(mask, code):
        reason[(reason == 0) & np.asarray(mask)] = code
    exclude(~frame.zone_owned, 1)
    exclude(~np.isfinite(frame.worldcover_classified_fraction) | (frame.worldcover_classified_fraction < .95), 2)
    exclude(~np.isfinite(frame.worldcover_land_fraction) | (frame.worldcover_land_fraction < .8), 3)
    exclude(np.isfinite(frame.water_fraction) & frame.water_fraction.ne(0), 4)
    exclude(~np.isfinite(frame.loc[:, list(NUMERIC_FEATURES)].to_numpy(float)).all(axis=1), 5)
    exclude(~frame.climate_class.isin(CLIMATE_CLASSES), 6)
    exclude(frame.solar_elevation_deg.gt(-6) & frame.solar_elevation_deg.lt(10), 7)
    verified_station = (frame.station_id.fillna('').ne('')
                        & frame.station_age_minutes.between(0, 90)
                        & frame.station_distance_km.between(0, 100)
                        & np.isfinite(frame.observed_station_air_temperature_c))
    if 'verified_station_report_status' in frame:
        from .weather_stations import STATION_OK as METAR_OK
        historical = (frame.air_temperature_source.eq('observed_station_residual_plus_ERA5_spatial_background')
                      & frame.verified_station_report_status.eq(STATION_OK))
        operational = (frame.air_temperature_source.eq('observed_METAR_residual_plus_GFS_spatial_background')
                       & frame.verified_station_report_status.eq(METAR_OK))
        verified_station &= historical | operational
    else:
        verified_station &= False
    exclude(~verified_station, 8)
    return reason


def _run_bounded(output, longitude, latitude, start, cells=32, cache=None, weather_plan=None):
    stamp = pd.Timestamp(utc_hour(start))
    if weather_plan is not None:
        from . import weather_access, weather_stations
        if (not weather_plan['enabled'] or pd.Timestamp(weather_plan['valid_time_utc']) != stamp
                or stamp > pd.Timestamp.now(tz='UTC') or stamp.year < 2021):
            raise ValueError('Weather plan must authorize the exact requested historical or present hour.')
    elif stamp.year < 2021 or stamp > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7):
        raise ValueError("Use historical dates from 2021 to at least seven days ago; surface availability is checked from source data.")
    area, frame = patch_area(longitude, latitude, cells)
    backend = os.environ.get('LST_F_INFERENCE_BACKEND', 'sklearn')
    # Both backends use the same pinned fitted model. Verify artifacts before
    # downloading any inputs; the native option is explicitly enabled by host.
    model = FrozenF.load() if backend == 'sklearn' else FrozenF.load(backend=backend)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(cache or "/opt/lst-pilot/cache")
    frame["datetime_utc"] = stamp
    frame["sample_id"] = frame.sample_id + ":" + stamp.isoformat()
    initial = frame.copy()
    started = time.monotonic()
    audit = {"status": "preparing", "area": area, "requested_datetime_utc": stamp.isoformat(),
             "model_sha256": MODEL_SHA256,
             "requested_inference_backend": backend,
             "input_semantics": "F40 trailing 32-day optical composite; WorldCover2020 land mask; requested-hour historical ERA5; backward observed-station residual when available",
             "source_provenance": {}, "model_training": False, "thermal_labels_read": False,
             "actual_weather_only": True, "reference_year_substitution": False,
             "station_required": True,
             "global_accuracy_validated": False, "uncertainty_interval_available": False,
             "source_code_sha256": {module.__name__: sha(module.__file__) for module in
                                    (assemble, surface, radiation, raster, weather, satellite, context, stations)},
             "patch_code_sha256": sha(__file__),
             "known_limitations": ["New places are unvalidated transfers.",
                                   "Night training is limited to London/Cfb and Sioux Falls/Dfa.",
                                   "Cold and snow errors remain unresolved.",
                                   "Static surfaces do not describe subsequent land-cover changes."],
             "weather_delivery_scope": "bounded engineering request; free point API is not the worldwide backend"}
    def checkpoint(stage):
        audit['stage'] = stage
        (output / 'provenance.json').write_text(json.dumps(audit, indent=2, default=raster._json, allow_nan=False) + '\n')
        print(json.dumps({'stage': stage, 'seconds': time.monotonic() - started}), flush=True)
    checkpoint('starting')
    if weather_plan is not None:
        audit['weather_plan'] = weather_plan
        audit['actual_weather_only'] = weather_plan['is_reanalysis']
        audit['weather_is_reanalysis'] = weather_plan['is_reanalysis']
        audit['input_semantics'] = 'F40 surface definitions; exact requested hour; explicit weather source in weather_plan; no date/year substitution'
        audit['known_limitations'] += weather_plan['source_limitations']
        audit['source_code_sha256'].update({module.__name__:sha(module.__file__)
                                           for module in (weather_access,weather_stations)})
    try:
        frame, audit['source_provenance']['surface'] = surface.append_surfaces(frame, area, cache, max_scenes=16)
        frame = restore_rows(initial, frame)
        checkpoint('surface_complete')
        frame = restore_rows(initial, raster.add_raster_context(frame))
        audit['source_provenance']['climate'] = {
            'source': frame.climate_source.iloc[0],
            'classes_sampled': frame.climate_class.value_counts().to_dict(),
            'sampled_class_sha256': hashlib.sha256('\n'.join(frame.climate_class).encode()).hexdigest()}
        if weather_plan is None:
            frame, audit['source_provenance']['weather'] = weather.enrich_weather(frame, cache, max_requests=16)
        else:
            frame, audit['source_provenance']['weather'] = weather_access.prepare_background(frame, cache, weather_plan, max_requests=64)
        frame = restore_rows(initial, frame)
        checkpoint('weather_complete')
        operational = weather_plan is not None and weather_plan['source']=='experimental_gfs'
        if operational:
            frame, audit['source_provenance']['station'] = weather_stations.attach_recent_stations(frame, [area], cache, weather_plan)
        else:
            frame, audit['source_provenance']['station'] = assemble.attach_stations(frame, [area], cache, stations_per_region=2,
                                                                                   max_distance_km=100, max_age_minutes=90)
        frame = restore_rows(initial, frame)
        if weather_plan is None:
            frame = radiation.add_radiation(frame, cache / 'radiation', max_hours=1)
            audit['source_provenance']['radiation'] = frame.attrs.get('radiation_context', {})
        else:
            frame, audit['source_provenance']['radiation'] = weather_access.add_radiation(frame, cache, weather_plan)
        frame = restore_rows(initial, frame)
        checkpoint('inputs_complete')
        from .receipts import bind_cached_sources
        if not operational:
            frame, audit['source_provenance']['cached_sources'] = bind_cached_sources(frame, cache)
        else:
            audit['source_provenance']['cached_sources'] = {
                'source': 'Explicit GFS and raw-verified METAR receipts above; ERA5 cache verifier is inapplicable',
                'not_validated_as_era5': True,
                'station_status_counts': frame.verified_station_report_status.value_counts().to_dict()}
        frame = restore_rows(initial, frame)
        # Preserve the complete label-free table: exact weather cells/times,
        # station age/distance/report, optical range/count and source statuses.
        frame.attrs = {}
        frame.to_parquet(output / 'inputs.parquet', index=False)
        reason = support_reasons(frame)
        eligible = reason == 0
        output_values = np.full(len(frame), -9999, np.float32)
        frame.loc[:, list(FEATURES)].to_parquet(output / 'features.parquet', index=False)
        if eligible.any():
            result = model.predict(frame.loc[eligible, list(FEATURES)], source_provenance={
                'request_datetime_utc': stamp.isoformat(), 'area': area,
                'feature_table_sha256': sha(output / 'features.parquet'),
                'detailed_sources': 'provenance.json'})
            output_values[eligible] = result.values.predicted_lst_c.to_numpy(np.float32)
            audit['inference'] = result.provenance
        profile = dict(driver='GTiff', width=cells, height=cells, count=1, crs=f"EPSG:{area['epsg']}",
                       transform=from_origin(area['extent_m'][0], area['extent_m'][3], 100, 100),
                       compress='deflate', tiled=True, blockxsize=128, blockysize=128)
        with rasterio.open(output / 'lst.tif', 'w', **profile, dtype='float32', nodata=-9999) as dest:
            dest.write(output_values.reshape(cells, cells), 1)
            dest.set_band_description(1, 'Experimental land surface temperature, degrees Celsius')
            dest.update_tags(support='global_extrapolation', model_sha256=MODEL_SHA256,
                             datetime_utc=stamp.isoformat(), uncertainty='not_calibrated', thermal_labels_read='false')
        with rasterio.open(output / 'support.tif', 'w', **profile, dtype='uint8') as dest:
            dest.write(reason.reshape(cells, cells), 1)
            dest.update_tags(reason_codes=json.dumps(REASONS))
        frame[['sample_id', 'longitude', 'latitude', 'grid_row', 'grid_col', 'air_temperature_c',
               'air_temperature_source', 'station_id', 'station_age_minutes', 'solar_elevation_deg']].assign(
                   support_code=reason, predicted_lst_c=np.where(eligible, output_values, np.nan),
                   climate_phase_seen_in_fit=(frame.solar_elevation_deg.ge(10) | (
                       frame.solar_elevation_deg.le(-6) & frame.climate_class.isin(['Cfb', 'Dfa']))),
                   station_observation_datetime_utc=frame.datetime_utc - pd.to_timedelta(frame.station_age_minutes, unit='m')
                   ).to_parquet(output / 'pixels.parquet', index=False)
        audit.update(status='complete', pixel_count=len(frame), predicted_pixels=int(eligible.sum()),
                     support_counts={label: int((reason == code).sum()) for code, label in REASONS.items()},
                     air_sources=frame.air_temperature_source.value_counts().to_dict(),
                     phase_counts={'day': int(frame.solar_elevation_deg.gt(0).sum()), 'night': int(frame.solar_elevation_deg.le(0).sum())},
                     elapsed_seconds=time.monotonic() - started,
                     process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     process_user_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                     process_system_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime)
        audit['artifacts'] = {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size}
                              for name in ('lst.tif', 'support.tif', 'inputs.parquet', 'features.parquet', 'pixels.parquet')}
        checkpoint('complete')
        return audit
    except Exception as exc:
        audit.update(status='failed', error=satellite._safe_error(exc), elapsed_seconds=time.monotonic() - started)
        checkpoint('failed')
        raise


def run(output, longitude, latitude, start, cells=512, cache=None, weather_plan=None):
    """Prepare one complete canonical tile; smaller outputs must crop it.

    Optical scene selection and station candidate acceptance both depend on
    preparation extent. A drawn ROI must not re-run them on a smaller extent.
    """
    if cells != 512 or isinstance(cells, bool):
        raise ValueError("Prepare the full 512-cell canonical tile, then use lst_global.crop for a smaller area.")
    return _run_bounded(output, longitude, latitude, start, cells, cache, weather_plan=weather_plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--longitude', type=float, required=True)
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--cells', type=int, default=512)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    run(args.output, args.longitude, args.latitude, args.start, args.cells, args.cache)


if __name__ == '__main__':
    main()
