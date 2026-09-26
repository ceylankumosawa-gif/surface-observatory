"""ASTER AST_08 V004 native-grid engineering adapter; never a serving source.

SKT is TES surface kinetic temperature, not at-sensor brightness temperature.
QA=0 is meaningful. Rotated source grids are retained until native masking.
ASTER L1T precision correction must never be imputed to the AST_08 L2 product.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, box
from shapely.ops import transform as shape_transform

from . import ecostress as eco

VERSION = 'aster-v004-engineering-20260910-v1'
COLLECTION = 'C3306885674-LPCLOUD'
PREFIX = '/lp-prod-protected/AST_08.004/'
LAYERS = ('SKT', 'SKT_QA_DataPlane', 'SKT_QA_DataPlane2')
NAME = re.compile(r'^AST_08_004(?P<time>\d{14})_(?P<processing>\d{14})$')
QA_RULES = {'first_qa_plane_exact_good_value': 0, 'skt_raw_scale_kelvin': .1,
            'skt_raw_fill': 0, 'cloud_and_nodata_buffer_m': 350,
            'minimum_valid_fraction': .9, 'plausibility_kelvin': [150, 400],
            'qa2_uncertainty_not_interpreted_as_kelvin': True,
            'independent_registration_required': True, 'independent_cloud_review_required': True}
SOURCES = {
    'product': 'https://doi.org/10.5067/ASTER/AST_08.004',
    'tutorial': 'https://github.com/nasa/ASTER-Data-Resources/blob/main/python/tutorials/Exploring_AST_08_Surface_Kinetic_Temperature.ipynb',
    'guide': 'https://lpdaac.usgs.gov/documents/2265/ASTER_User_Guide_V4_pcP80n5.pdf',
    'qa_plan': 'https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/ASTER%20QA%20Plan%20v2.0.pdf'}


def identity(title):
    match = NAME.fullmatch(title)
    if not match:
        raise eco.AcquisitionError('Expected an exact AST_08 V004 granule name.')
    stamp = datetime.strptime(match['time'], '%m%d%Y%H%M%S').replace(tzinfo=timezone.utc)
    processing = datetime.strptime(match['processing'], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    return {'acquisition_utc': stamp.isoformat(), 'processing_utc': processing.isoformat(),
            'date_group': 'ASTER:UTC-date:' + stamp.date().isoformat()}


def acquisition_time(umm):
    extent = umm['TemporalExtent']
    if set(extent) == {'SingleDateTime'}:
        value = extent['SingleDateTime']
    elif 'RangeDateTime' in extent and 'SingleDateTime' not in extent:
        value = extent['RangeDateTime']['BeginningDateTime']
    else:
        raise eco.AcquisitionError('Ambiguous ASTER temporal metadata.')
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise eco.AcquisitionError('ASTER acquisition time requires timezone.')
    return stamp.tz_convert('UTC')


def asset_links(envelope, record):
    meta, umm = envelope['meta'], envelope['umm']
    title = record['granule_title']
    parsed = identity(title)
    if (meta['concept-id'] != record['granule_concept_id'] or meta['collection-concept-id'] != COLLECTION
            or umm['CollectionReference'] != {'ShortName': 'AST_08', 'Version': '004'}
            or umm['GranuleUR'] != title or int(meta['revision-id']) < 1 or meta.get('deleted')):
        raise eco.AcquisitionError('ASTER CMR collection/revision/granule identity mismatch.')
    if 'cmr_revision_id' in record and int(record['cmr_revision_id']) != int(meta['revision-id']):
        raise eco.AcquisitionError('ASTER CMR revision changed after plan freeze.')
    stamp = acquisition_time(umm)
    if stamp.year not in (2021, 2022):
        raise eco.AcquisitionError('Engineering ASTER collection permits fitting years2021–2022 only.')
    if abs((stamp - pd.Timestamp(parsed['acquisition_utc'])).total_seconds()) >= 2:
        raise eco.AcquisitionError('ASTER granule name and CMR acquisition time differ.')
    links = {}
    for entry in umm.get('RelatedUrls', []):
        p = urlsplit(entry.get('URL', ''))
        if (entry.get('Type') == 'GET DATA' and p.scheme == 'https' and p.hostname == eco.HOST
                and p.port in (None, 443) and not p.query and not p.fragment and not p.username and not p.password):
            for layer in LAYERS:
                if p.path == PREFIX + title + '/' + title + '_' + layer + '.tif':
                    if layer in links:
                        raise eco.AcquisitionError('Duplicate ASTER asset link.')
                    links[layer] = entry['URL']
    if set(links) != set(LAYERS):
        raise eco.AcquisitionError('ASTER SKT and both native QA planes are required.')
    return links


def native_valid(raw, source_transform):
    if not (np.issubdtype(raw['SKT'].dtype, np.unsignedinteger)
            and all(np.issubdtype(raw[k].dtype, np.integer) for k in LAYERS[1:])):
        raise eco.AcquisitionError('Unsupported ASTER DN/QA types; scaling is never guessed.')
    if any(raw[k].shape != raw['SKT'].shape for k in LAYERS):
        raise eco.AcquisitionError('ASTER native QA shapes differ.')
    col = np.array([source_transform.a, source_transform.d])
    row = np.array([source_transform.b, source_transform.e])
    spacing = (float(np.linalg.norm(row)), float(np.linalg.norm(col)))
    if not all(85 <= v <= 95 for v in spacing) or abs(float(row @ col)) > 1e-5:
        raise eco.AcquisitionError('ASTER native thermal pixels must be orthogonal90m cells.')
    kelvin = raw['SKT'].astype('float32') * .1
    good_qa = raw['SKT_QA_DataPlane'] == 0
    plausible = (raw['SKT'] != 0) & (kelvin >= 150) & (kelvin <= 400)
    # Conservatively buffer all first-plane non-good and missing SKT, not just
    # cloud-bit subsets. Padding makes the native outside-of-scene area invalid.
    clear = good_qa & (raw['SKT'] > 0)
    distances = distance_transform_edt(np.pad(clear, 1, constant_values=False), sampling=spacing)[1:-1, 1:-1]
    checks = {'first_qa_plane_good': good_qa, 'finite_plausible_skt': plausible,
              'clear_after_cloud_nodata_buffer': distances > QA_RULES['cloud_and_nodata_buffer_m']}
    return np.logical_and.reduce(list(checks.values())), kelvin, checks


def aggregate(raw, source_transform, source_crs, bounds, epsg):
    valid, kelvin, checks = native_valid(raw, source_transform)
    left, bottom, right, top = bounds
    if any(v % 100 for v in bounds) or right <= left or top <= bottom:
        raise eco.AcquisitionError('Destination bounds must use the fixed100m grid.')
    dst_transform = from_origin(left, top, 100, 100)
    dst_shape = (round((top - bottom) / 100), round((right - left) / 100))
    def warp(value, nodata=None):
        destination = np.full(dst_shape, np.nan, dtype='float32')
        reproject(np.asarray(value, dtype='float32'), destination, src_transform=source_transform,
                  src_crs=source_crs, dst_transform=dst_transform, dst_crs=f'EPSG:{epsg}',
                  src_nodata=nodata, dst_nodata=np.nan, resampling=Resampling.average,
                  num_threads=1, tolerance=0)
        return destination
    fraction = warp(valid.astype('float32'))
    lst = warp(np.where(valid, kelvin - 273.15, np.nan), np.nan)
    to_src = Transformer.from_crs(epsg, source_crs, always_xy=True)
    rows, cols = np.indices(dst_shape)
    support = np.ones(dst_shape, dtype=bool)
    inverse = ~source_transform
    # Nine points guard partial rotated footprints; the native edge buffer adds
    # a further350m margin. Missing fractional source support is never filled.
    for dx in (0, 50, 100):
        for dy in (0, 50, 100):
            sx, sy = to_src.transform(left + cols * 100 + dx, top - rows * 100 - dy)
            cc, rr = inverse * (sx, sy)
            support &= (cc >= 0) & (cc <= valid.shape[1]) & (rr >= 0) & (rr <= valid.shape[0])
    accepted = support & np.isfinite(fraction) & (fraction >= .9)
    return {'lst_c': np.where(accepted, lst, np.nan), 'valid_fraction': fraction}, dst_transform, checks


def inspect(record, pilot, envelope, client, cache, output):
    links = asset_links(envelope, record)
    title = record['granule_title']
    files = {layer: Path(cache) / title / (layer + '.tif') for layer in LAYERS}
    downloads = {layer: client.download(links[layer], files[layer], kind='aster_cog') for layer in LAYERS}
    with ExitStack() as stack:
        sources = {layer: stack.enter_context(rasterio.open(path)) for layer, path in files.items()}
        base = sources['SKT']
        if base.count != 1 or base.crs is None or not base.crs.is_projected or base.width * base.height > 2_000_000:
            raise eco.AcquisitionError('Unexpected ASTER native raster dimensions/CRS.')
        if base.crs.linear_units != 'metre':
            raise eco.AcquisitionError('ASTER native CRS units must be metres.')
        if any(src.shape != base.shape or src.transform != base.transform or src.crs != base.crs or src.count != 1 for src in sources.values()):
            raise eco.AcquisitionError('ASTER temperature and QA are not on one native grid.')
        if base.scales not in ((1.,), (.1,)) or base.offsets != (0.,):
            raise eco.AcquisitionError('Unrecognized ASTER SKT scale/offset metadata.')
        polygon = Polygon([base.transform * point for point in ((0, 0), (base.width, 0), (base.width, base.height), (0, base.height))])
        projected = shape_transform(Transformer.from_crs(base.crs, pilot['epsg'], always_xy=True).transform, polygon)
        if not projected.intersects(box(*pilot['extent_m'])):
            raise eco.AcquisitionError('Actual rotated ASTER footprint does not intersect pilot.')
        raw = {layer: src.read(1) for layer, src in sources.items()}
        values, target_transform, checks = aggregate(raw, base.transform, base.crs, pilot['extent_m'], pilot['epsg'])
        geometry = {'crs': str(base.crs), 'transform': list(base.transform), 'shape': list(base.shape),
                    'resolution_m': list(base.res), 'bounds': list(base.bounds),
                    'dtypes': {k: s.dtypes[0] for k, s in sources.items()},
                    'actual_overlap_m2': projected.intersection(box(*pilot['extent_m'])).area}
        qa2_values, qa2_counts = np.unique(raw['SKT_QA_DataPlane2'], return_counts=True)
    directory = Path(output) / record['pilot_id'] / title
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'engineering_labels.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=values['lst_c'].shape[1], height=values['lst_c'].shape[0],
                       count=2, dtype='float32', crs=f"EPSG:{pilot['epsg']}", transform=target_transform,
                       nodata=np.nan, compress='deflate') as dest:
        for band, (name, value) in enumerate(values.items(), 1):
            dest.write(value.astype('float32'), band)
            dest.set_band_description(band, name)
    good = np.isfinite(values['lst_c'])
    summary = {'version': VERSION, 'pilot_id': record['pilot_id'], 'sensor': 'ASTER', 'product': 'AST_08',
               'product_version': '004', 'title': title, 'granule_id': record['granule_concept_id'],
               'time_start': acquisition_time(envelope['umm']).isoformat(),
               'cmr_meta': envelope['meta'], 'native_geometry': geometry,
               'qa_rules': QA_RULES, 'official_sources': SOURCES, 'downloads': downloads,
               'native_checks_pass_count': {k: int(v.sum()) for k, v in checks.items()},
               'native_qa2_histogram': {str(v): int(c) for v, c in zip(qa2_values, qa2_counts)},
               'grid_cells': int(good.size), 'qa_pass_cells': int(good.sum()),
               'raster_path': str(path), 'raster_sha256': eco.digest(path),
               'training_eligible': False, 'engineering_only': True, 'source_screen_pass': False,
               'native_qa_pass': bool(good.any()),
               'pending': ['Independent AST_08 registration: AST_L1T correction does not certify thisL2 grid',
                           'Independent cloud screening: QA0 does not prove all residual clouds absent',
                           'Permanent spatial blocks and1km buffers', 'Weather/station/surface joins',
                           'Frozen sensor-calibration protocol'],
               'thermal_values_are_labels_not_predictors': True,
               'source_uncertainty_k': None,
               'uncertainty_note': 'No Kelvin error estimate is inferred from zero-filled or unverified QA2 bits.'}
    eco.save_json(directory / 'summary.json', summary)
    return summary
