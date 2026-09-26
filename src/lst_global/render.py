"""Assemble bounded user rasters from canonical full-tile predictions.

Every tile is prepared identically regardless of drawn polygon or output size.
Coarse exports aggregate the same native predictions, with explicit coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer

from lst_pilot.api import _clean_json
from lst_pilot.raster import _overlay, comparison_legends
from . import patch
from .model import MODEL_SHA256
from .planner import sha


def fingerprint(root):
    digest = hashlib.sha256()
    for package in ('lst_global', 'lst_pilot'):
        for source in sorted((Path(root)/'src'/package).glob('*.py')):
            digest.update(f'{package}/{source.name}'.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def verified_tile(directory, tile_id, stamp):
    try:
        provenance = json.loads((directory/'provenance.json').read_text())
        if (provenance['status'] != 'complete' or provenance['area']['tile_id'] != tile_id
                or provenance['model_sha256'] != MODEL_SHA256
                or provenance['area']['grid_shape'] != [512, 512]
                or provenance['requested_datetime_utc'].replace('+00:00', 'Z') != stamp):
            return None
        for name in ('lst.tif', 'support.tif'):
            if (directory/name).is_symlink() or sha(directory/name) != provenance['artifacts'][name]['sha256']:
                return None
        return provenance
    except (OSError, KeyError, ValueError):
        return None


def prepare_tile(spec, stamp, tile_cache, source_cache, pipeline_hash, tile_runner=patch.run, weather_plan=None):
    weather_identity = None if weather_plan is None else {
        'source':weather_plan['source'],
        'publication':weather_plan['availability']['metadata_sha256'],
        'operational_checked':weather_plan['availability']['checked_utc'] if weather_plan['source']=='experimental_gfs' else None}
    key = hashlib.sha256(json.dumps([spec['tile_id'], stamp, MODEL_SHA256, pipeline_hash,weather_identity],sort_keys=True).encode()).hexdigest()
    directory = tile_cache/key
    provenance = verified_tile(directory, spec['tile_id'], stamp)
    if provenance:
        return directory, provenance, True
    if directory.exists():
        # Only our dedicated, hash-named generation cache is eligible here.
        if directory.is_symlink():
            raise ValueError('Unsafe tile cache entry.')
        shutil.rmtree(directory)
    tile_runner(directory, spec['longitude'], spec['latitude'], stamp, cache=source_cache,weather_plan=weather_plan)
    provenance = verified_tile(directory, spec['tile_id'], stamp)
    if not provenance:
        raise ValueError('Canonical source tile failed its identity or artifact checks.')
    return directory, provenance, False


def assemble(request, tile_paths, output):
    resolution = request['resolution_m']
    left, bottom, right, top = request['bounds_m']
    height, width = request['shape']
    dst_crs = f"EPSG:{request['epsg']}"
    out_transform = from_origin(left, top, resolution, resolution)
    # Native buffer is independent of output polygon. Expand the extent to the
    # fixed 100 m grid before any coarse averaging, including 250 m exports.
    nl, nb, nr, nt = (math.floor(left/100)*100, math.floor(bottom/100)*100,
                      math.ceil(right/100)*100, math.ceil(top/100)*100)
    nw, nh = round((nr-nl)/100), round((nt-nb)/100)
    native_transform = from_origin(nl, nt, 100, 100)
    native = np.full((nh, nw), np.nan, np.float32)
    for directory in tile_paths:
        with rasterio.open(directory/'lst.tif') as source:
            candidate = np.full_like(native, np.nan)
            reproject(rasterio.band(source, 1), candidate, src_transform=source.transform,
                      src_crs=source.crs, src_nodata=-9999, dst_transform=native_transform,
                      dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.nearest, num_threads=1)
            # Deterministic tile order, and no blending of duplicated zone pixels.
            take = ~np.isfinite(native) & np.isfinite(candidate)
            native[take] = candidate[take]
    valid = np.isfinite(native)
    coverage = np.zeros((height, width), np.float32)
    values = np.full((height, width), np.nan, np.float32)
    common = dict(src_transform=native_transform, src_crs=dst_crs,
                  dst_transform=out_transform, dst_crs=dst_crs,
                  resampling=Resampling.average, num_threads=1)
    reproject(valid.astype(np.float32), coverage, **common)
    reproject(native, values, src_nodata=np.nan, dst_nodata=np.nan, **common)
    transformer = Transformer.from_crs(4326, dst_crs, always_xy=True)
    polygon = transform(transformer.transform, shape(request['polygon']).segmentize(.001))
    inside = geometry_mask([mapping(polygon)], out_shape=values.shape, transform=out_transform,
                           invert=True, all_touched=False)
    values[(coverage < .8-1e-6) | ~inside] = np.nan
    quality = np.where(~inside, 2, np.where(np.isfinite(values), 0, 1)).astype(np.uint8)
    profile = dict(driver='GTiff', height=height, width=width, count=1, crs=dst_crs,
                   transform=out_transform, compress='deflate', tiled=True,
                   blockxsize=128, blockysize=128)
    with rasterio.open(output/'prediction.tif', 'w', **profile, dtype='float32', nodata=-9999) as target:
        target.write(np.where(np.isfinite(values), values, -9999).astype(np.float32), 1)
        target.set_band_description(1, 'predicted_lst_c')
        target.update_tags(resolution_m=resolution, native_resolution_m=100, accuracy='unvalidated_global_transfer',
                           datetime_utc=request['datetime_utc'], model_sha256=MODEL_SHA256)
    with rasterio.open(output/'quality.tif', 'w', **profile, dtype='uint8') as target:
        target.write(quality, 1)
        target.set_band_description(1, 'support_code')
        target.update_tags(reason_codes=json.dumps({0:'prediction', 1:'insufficient_valid_source_area', 2:'outside_polygon'}))
    with rasterio.open(output/'coverage.tif', 'w', **profile, dtype='float32', nodata=-1) as target:
        target.write(np.where(inside, coverage, -1).astype(np.float32), 1)
        target.set_band_description(1, 'valid_native_area_fraction')
    count = int(np.isfinite(values).sum())
    statistics = ({'min': float(np.nanmin(values)), 'max': float(np.nanmax(values)),
                   'mean': float(np.nanmean(values))} if count else {'min': None, 'max': None, 'mean': None})
    legends = comparison_legends(values) if count else {'prediction': {'min_c': 0., 'max_c': 5., 'cmap': 'inferno'}}
    grid = SimpleNamespace(epsg=request['epsg'], width=width, height=height,
                           transform=out_transform, bounds=(left,bottom,right,top))
    bounds = _overlay(output/'overlay.png', values, grid,
                      legends['prediction']['min_c'], legends['prediction']['max_c'])
    return {'bounds': bounds, 'summary': {'predicted_lst_c': statistics, 'valid_pixels': count},
            'legends': {**legends, 'temperature': legends['prediction']},
            'counts': {'predicted': count, 'inside_polygon': int(inside.sum()), 'total': width*height,
                       'missing': int((inside & ~np.isfinite(values)).sum())},
            'grid': {'epsg': request['epsg'], 'resolution_m': resolution, 'native_resolution_m': 100,
                     'grid_id': request.get('grid_id'), 'grid_note': request.get('grid_note'),
                     'shape': [height,width], 'bounds_m': request['bounds_m'], 'transform': list(out_transform)[:6],
                     'aggregation': 'area-weighted average of available native values; >=80% valid area; pixel centre polygon mask'}}


def render(request, output, tile_cache, source_cache, pipeline_hash):
    from .weather_access import inspect_availability,choose_weather_source
    output, tile_cache = Path(output), Path(tile_cache)
    output.mkdir(parents=True, exist_ok=True)
    tile_cache.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'stage':'checking weather publication','progress':.02}),flush=True)
    weather_plan = choose_weather_source(request['datetime_utc'],inspect_availability(source_cache),
                                         allow_operational=os.environ.get('LST_ENABLE_OPERATIONAL_WEATHER')=='1')
    if not weather_plan['enabled']:
        raise ValueError('Weather for the requested hour is not available through an enabled source.')
    paths, receipts = [], []
    for index, spec in enumerate(request['source_tiles']):
        print(json.dumps({'stage': f'preparing source tile {index+1} of {len(request["source_tiles"])}',
                          'progress': .05+.8*index/len(request['source_tiles'])}), flush=True)
        directory, receipt, cached = prepare_tile(spec, request['datetime_utc'], tile_cache,
                                                   source_cache, pipeline_hash,weather_plan=weather_plan)
        paths.append(directory)
        receipts.append({'tile_id': spec['tile_id'], 'cached': cached,
                         'source_receipt_sha256': sha(directory/'provenance.json'),
                         'predicted_pixels': receipt['predicted_pixels'], 'support_counts': receipt['support_counts'],
                         'source_provenance': receipt['source_provenance']})
    print(json.dumps({'stage': 'assembling raster', 'progress': .9}), flush=True)
    result = assemble(request, paths, output)
    result.update(mode='global_experimental', requested_datetime_utc=request['datetime_utc'],
                  model_sha256=MODEL_SHA256, pipeline_sha256=pipeline_hash,
                  global_accuracy_validated=False, uncertainty_interval_available=False,
                  source_tiles=receipts,weather_plan=weather_plan,
                  sources=['Landsat surface reflectance: causal 32-day composite', 'ESA WorldCover 2020',
                           'Copernicus terrain and Köppen climate classification', 'Historical ERA5 weather with actual requested hour',
                           'Verified nearby station air reports', 'ERA5 downward longwave and snow-water equivalent'],
                  warnings=['The regional day/night accuracy target of MAE ≤3°C has not been met.',
                            'New regions and cloudy conditions lack independent validation; nighttime fitting coverage is limited.',
                            'Transparent cells have missing inputs, unsupported climate/twilight or no verified recent station.',
                            'Static land cover and optical composites may miss recent changes. Coarser exports average the same predictions.'],
                  files={key: name for key,name in {
                      'prediction_tif':'prediction.tif', 'overlay_png':'overlay.png', 'quality_tif':'quality.tif',
                      'coverage_tif':'coverage.tif', 'provenance_json':'provenance.json'}.items()})
    if weather_plan['source']=='experimental_gfs':
        result['sources'] = result['sources'][:3]+['NOAA GFS via Open-Meteo: operational weather, exact requested valid hour',
            'Raw-verified nearby METAR air reports with same-provider GFS background residual',
            'NOAA native GFS downward longwave and snow-water equivalent; explicit run time']
        result['warnings'] += weather_plan['source_limitations']
    elif weather_plan['provisional']:
        result['warnings'].append('Recent ERA5T data are provisional and may be revised by the provider.')
    public = _clean_json(result, output)
    (output/'provenance.json').write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    (output/'result.json').write_text(json.dumps(public, allow_nan=False)+'\n')
    print(json.dumps({'stage':'complete', 'progress':1}), flush=True)
    return public


def main():
    parser = argparse.ArgumentParser()
    for name in ('request','output','tile-cache','source-cache','pipeline-hash'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    render(json.loads(Path(args.request).read_text()), args.output, args.tile_cache,
           Path(args.source_cache), args.pipeline_hash)


if __name__ == '__main__':
    main()
