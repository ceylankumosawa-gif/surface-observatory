"""Crop a completed canonical tile without fetching data or running the model.

This makes overlapping drawn areas return exactly the same pixel values,
weather, optical composites and station selection. Grid coordinates in cropped
tables are local, with original canonical row/column retained separately.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from .model import MODEL_SHA256, FEATURES
from .planner import sha

FILES = ('lst.tif', 'support.tif', 'inputs.parquet', 'features.parquet', 'pixels.parquet')


def crop_tile(parent, output, row, column, cells):
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (row, column, cells)):
        raise ValueError('Crop coordinates must be integers.')
    if min(row, column) < 0 or not 1 <= cells <= 512 or row + cells > 512 or column + cells > 512:
        raise ValueError('Crop must lie inside its 512×512 parent tile.')
    parent, output = Path(parent).resolve(), Path(output)
    proof = json.loads((parent / 'provenance.json').read_text())
    if (proof.get('status') != 'complete' or proof.get('area', {}).get('grid_shape') != [512, 512]
            or proof.get('crop_only')
            or proof.get('model_sha256', proof.get('inference', {}).get('model_sha256')) != MODEL_SHA256):
        raise ValueError('Expected a complete canonical tile using the pinned F model.')
    for name in FILES:
        if sha(parent / name) != proof['artifacts'][name]['sha256']:
            raise ValueError(f'Parent artifact hash mismatch: {name}')
    tables = {name: pd.read_parquet(parent / name) for name in ('inputs.parquet', 'features.parquet', 'pixels.parquet')}
    original = tables['inputs.parquet']
    rr, cc = np.indices((512, 512))
    if (len(original) != 512**2 or not np.array_equal(original.grid_row, rr.ravel())
            or not np.array_equal(original.grid_col, cc.ravel())):
        raise ValueError('Parent rows do not follow canonical raster order.')
    pd.testing.assert_frame_equal(original[list(FEATURES)], tables['features.parquet'], check_exact=True)
    pixels = tables['pixels.parquet']
    for name in ('sample_id', 'grid_row', 'grid_col', 'longitude', 'latitude'):
        pd.testing.assert_series_equal(original[name], pixels[name], check_exact=True)
    take = ((original.grid_row >= row) & (original.grid_row < row + cells)
            & (original.grid_col >= column) & (original.grid_col < column + cells)).to_numpy()
    cropped = {}
    for name, table in tables.items():
        part = table.loc[take].copy().reset_index(drop=True)
        if 'grid_row' in part:
            part['canonical_grid_row'], part['canonical_grid_col'] = part.grid_row, part.grid_col
            part['grid_row'] -= row
            part['grid_col'] -= column
        part.attrs = {}
        cropped[name] = part
    win = Window(column, row, cells, cells)
    output.mkdir(parents=True, exist_ok=False)
    transform = None
    for name in ('lst.tif', 'support.tif'):
        with rasterio.open(parent / name) as source:
            if source.shape != (512, 512) or source.crs.to_epsg() != proof['area']['epsg']:
                raise ValueError('Parent raster geometry differs from its receipt.')
            values = source.read(1, window=win)
            transform = window_transform(win, source.transform)
            profile = source.profile.copy()
            profile.update(width=cells, height=cells, transform=transform)
            with rasterio.open(output / name, 'w', **profile) as dest:
                dest.write(values, 1)
                dest.update_tags(**source.tags(), parent_tile=proof['area']['tile_id'], crop_only='true')
                if source.descriptions[0]:
                    dest.set_band_description(1, source.descriptions[0])
            if name == 'lst.tif':
                predicted = cropped['pixels.parquet'].predicted_lst_c.to_numpy().reshape(cells, cells)
                np.testing.assert_array_equal(np.where(values == source.nodata, np.nan, values), predicted)
            else:
                np.testing.assert_array_equal(values.ravel(), cropped['pixels.parquet'].support_code)
    for name, table in cropped.items():
        table.to_parquet(output / name, index=False)
    # Keep every source receipt intact. Counts below describe the crop; parent
    # preparation counts and complete canonical source selection remain linked.
    result = copy.deepcopy(proof)
    result['area']['extent_m'] = list(window_bounds(Window(0, 0, cells, cells), transform))
    result['area']['grid_shape'] = [cells, cells]
    result['area']['id'] = proof['area']['id'] + f'-crop-{row}-{column}-{cells}'
    result['canonical_parent'] = {'path': str(parent), 'provenance_sha256': sha(parent / 'provenance.json'),
                                  'artifacts': proof['artifacts'], 'row_offset': row, 'column_offset': column}
    result['crop_only'] = True
    result['source_provenance_scope'] = 'Canonical parent tile; source selection and source receipt counts are unchanged.'
    result['crop_code_sha256'] = sha(__file__)
    result['crop_network_requests'] = 0
    result['crop_model_predictions'] = 0
    result['stage'] = 'complete'
    result['pixel_count'] = cells**2
    codes = cropped['pixels.parquet'].support_code
    result['predicted_pixels'] = int(codes.eq(0).sum())
    from .patch import REASONS
    result['support_counts'] = {label: int(codes.eq(code).sum()) for code, label in REASONS.items()}
    result['air_sources'] = cropped['inputs.parquet'].air_temperature_source.value_counts().to_dict()
    sun = cropped['inputs.parquet'].solar_elevation_deg
    result['phase_counts'] = {'day': int(sun.gt(0).sum()), 'night': int(sun.le(0).sum())}
    result['parent_preparation_elapsed_seconds'] = result.pop('elapsed_seconds')
    if 'inference' in result:
        result['inference']['scope'] = 'Unchanged parent-tile prediction; input_rows and source provenance refer to the complete parent.'
    result['artifacts'] = {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in FILES}
    (output / 'provenance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row', type=int, required=True)
    parser.add_argument('--column', type=int, required=True)
    parser.add_argument('--cells', type=int, required=True)
    args = parser.parse_args()
    crop_tile(args.parent, args.output, args.row, args.column, args.cells)


if __name__ == '__main__':
    main()
