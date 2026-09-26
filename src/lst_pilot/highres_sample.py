"""Sample a fixed high-resolution engineering snapshot without viewing errors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from . import ecostress as eco
from .option_b_cohort import choose_night_positions, coordinate_flags


def sample_record(record, planned, area):
    if record.get('status') != 'engineering_qa_complete' or record.get('qa_pass_cells', 0) <= 0:
        return pd.DataFrame(), []
    raster_path = Path(record['raster_path'])
    support = record['native_fit_support']
    if eco.digest(raster_path) != record['raster_sha256'] or eco.digest(support['path']) != support['sha256']:
        raise ValueError('Source raster or native support checksum changed.')
    with rasterio.open(raster_path) as src, rasterio.open(support['path']) as fits:
        expected = from_origin(area['extent_m'][0], area['extent_m'][3], 100, 100)
        if (src.crs.to_epsg() != area['epsg'] or src.shape != tuple(area['grid_shape'])
                or src.transform != expected or fits.transform != expected or fits.shape != src.shape):
            raise ValueError('Unexpected fixed grid.')
        arrays = {name: src.read(i+1) for i, name in enumerate(src.descriptions)}
        safe = fits.read(1) == 1
    valid = np.isfinite(arrays['lst_c']) & (arrays['valid_fraction'] >= .9)
    if planned['product'] == 'ecostress_v2':
        valid &= arrays['max_source_lst_error_k'] <= 2
    rr, cc = np.indices(valid.shape)
    _, _, reserved, _ = coordinate_flags(area['extent_m'][0] + (cc+.5)*100,
                                         area['extent_m'][3] - (rr+.5)*100, area)
    positions, windows = choose_night_positions(valid, record['title'], maximum=400,
                                               safe=valid & safe, reserved=valid & reserved)
    if len(positions) == 0:
        return pd.DataFrame(), windows
    rows, cols = positions.T
    x, y = area['extent_m'][0] + (cols+.5)*100, area['extent_m'][3] - (rows+.5)*100
    lon, lat = Transformer.from_crs(area['epsg'], 4326, always_xy=True).transform(x, y)
    product = 'ecostress_v2' if planned['product'] == 'ecostress_v2' else 'aster_ast08_v004'
    acquisition = planned['identity']['acquisition_group']
    data = pd.DataFrame({'region_id': area['id'], 'datetime_utc': pd.Timestamp(record['time_start']),
        'grid_row': rows, 'grid_col': cols, 'epsg': area['epsg'], 'pixel_epsg': area['epsg'],
        'pixel_x': x, 'pixel_y': y, 'longitude': lon, 'latitude': lat,
        'lst_c': arrays['lst_c'][rows, cols], 'label_valid_fraction': arrays['valid_fraction'][rows, cols],
        'native_fit_support_pass': safe[rows, cols], 'source_screen_pass': bool(record['source_screen_pass']),
        'max_source_lst_error_k': arrays.get('max_source_lst_error_k', np.full(valid.shape, np.nan))[rows, cols]})
    data['pixel_id'] = [f'{area["id"]}:{r}:{c}' for r, c in positions]
    data['acquisition_id'] = acquisition
    data['sample_id'] = [hashlib.sha256(f'{product}:{acquisition}:{p}'.encode()).hexdigest()[:32] for p in data.pixel_id]
    data['scene_id'] = record['title']
    data['label_product'] = product
    data['cohort_origin'] = 'expanded'
    data['day_night'] = planned['actual_phase']
    data['label_condition'] = 'native_qa_clear_' + planned['actual_phase']
    data['label_source'] = record['title']
    data['label_source_sha256'] = record['label_source_sha256']
    data['label_raster_sha256'] = record['raster_sha256']
    data['native_support_mask_sha256'] = support['sha256']
    data['cmr_concept_id'] = record['granule_id']
    data['cmr_revision_id'] = planned['cmr_revision_id']
    data['geolocation_proof'] = json.dumps(record['label_source_geolocation_proof'], sort_keys=True)
    data['cloud_proof'] = json.dumps(record['label_source_cloud_proof'], sort_keys=True)
    data['registration_status'] = ('positive matched Good/Best GEO; independent100m registration uncertainty remains'
                                   if product == 'ecostress_v2' else 'unverified AST_08 registration; not a100m training label')
    return data, windows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('New immutable sampled snapshot required.')
    plan_path, manifest_path = args.source/'plan.json', args.source/'manifest.json'
    plan_bytes, manifest_bytes = plan_path.read_bytes(), manifest_path.read_bytes()
    plan, manifest = json.loads(plan_bytes), json.loads(manifest_bytes)
    if eco.digest(plan_path) != manifest['plan_sha256']:
        raise ValueError('Plan identity mismatch.')
    args.output.mkdir(parents=True)
    (args.output/'plan_snapshot.json').write_bytes(plan_bytes)
    (args.output/'manifest_snapshot.json').write_bytes(manifest_bytes)
    planned = {r['granule_concept_id']: r for r in plan['records']}
    frames, audit = [], []
    for record in manifest['records']:
        frame, windows = sample_record(record, planned[record['granule_id']], plan['pilot_areas'][record['pilot_id']])
        if not frame.empty:
            frames.append(frame)
        audit.append({'granule_id': record['granule_id'], 'pilot_id': record['pilot_id'],
                      'rows': len(frame), 'source_screen_pass': record.get('source_screen_pass', False), 'windows': windows})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not data.empty and data.sample_id.duplicated().any():
        raise ValueError('Overlapping physical sample identities require explicit overlap resolution.')
    outputs = {}
    for name, subset in [('source_screened_samples', data.loc[data.source_screen_pass] if not data.empty else data),
                         ('engineering_only_samples', data.loc[~data.source_screen_pass] if not data.empty else data)]:
        if len(subset):
            path = args.output/(name+'.parquet')
            subset.to_parquet(path, index=False)
            outputs[name] = {'path': str(path), 'sha256': eco.digest(path), 'rows': len(subset),
                             'pilot_dates': len(subset.assign(utc_date=subset.datetime_utc.dt.strftime('%Y-%m-%d'))[['region_id','utc_date']].drop_duplicates())}
    summary = {'source_snapshot_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
               'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'sampler_sha256': eco.digest(__file__),
               'snapshot_collection_complete': manifest.get('complete', False), 'records': audit, 'outputs': outputs,
               'selection_rule': 'Up to4 fixed128-cell tiles, one per quadrant chosen by native-fit-safe valid support; up to80 safe+20 reserved samples/tile with existing deterministic seed. Thermal magnitudes and residuals never rank samples.'}
    eco.save_json(args.output/'sampling_manifest.json', summary)
    print(json.dumps(summary['outputs']), flush=True)


if __name__ == '__main__':
    main()
