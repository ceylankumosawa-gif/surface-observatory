"""Version 3 four-field ARCO continuation (JSON-safe metadata indexes); true SWE is acquired separately.

Imports the unchanged v1 low-level decoder and bounded transport. Each new plan
binds both source hashes, parent plan/cache evidence and carried transfer charges.
Outputs cannot satisfy the five-field experiment until independently verified
standard-CDS GRIB141 SWE is joined at these exact native cells and valid hours.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).with_name('era5_land.py')
_spec = importlib.util.spec_from_file_location('era5_land_frozen_v1', _BASE_PATH)
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
GROUPS = ('skin', 'air', 'soil_temperature', 'soil_water')
COLUMNS = [b.VARIABLES[g]['output'] for g in GROUPS]


def verify(plan, output):
    checks = {str(Path(__file__)): plan['source_sha256'], str(_BASE_PATH): plan['base_source_sha256'],
              plan['input_path']: plan['input_sha256'], str(Path(output)/'points.parquet'): plan['points_sha256']}
    if any(b.sha(p) != expected for p, expected in checks.items()):
        raise ValueError('Frozen source, input or request points changed.')


def plan_from_parent(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new immutable continuation directory.')
    p = json.loads((parent/'plan.json').read_text())
    parent_version=p.get('version',1)
    parent_source=Path(__file__).with_name('era5_land.py' if parent_version==1 else f'era5_land_v{parent_version}.py')
    if b.sha(parent_source) != p['source_sha256'] or b.sha(_BASE_PATH) != p.get('base_source_sha256',p['source_sha256']) or b.sha(parent/'points.parquet') != p['points_sha256'] or b.sha(p['input_path']) != p['input_sha256']:
        raise ValueError('Original input/source/points changed.')
    old = json.loads((parent/'transfer_ledger.json').read_text()) if (parent/'transfer_ledger.json').exists() else None
    if old and old['plan_sha256'] != b.sha(parent/'plan.json'):
        raise ValueError('Parent ledger plan mismatch.')
    output.mkdir(parents=True)
    shutil.copy2(parent/'points.parquet', output/'points.parquet')
    plan = dict(p)
    plan.update({'version': 3, 'source_sha256': b.sha(__file__), 'base_source_sha256': b.sha(_BASE_PATH),
                 'parent_stop_reason': ('v2 metadata serialization rejected numpy chunk indexes after native cell table completed; no temperature chunks read.' if parent_version==2 else 'Original five-field preflight rejected absent sd141 SWE; ARCO sde3066 is physical snow depth, never substituted.'),
                 'parent_source_sha256': b.sha(parent_source),
                 'parent_live_metadata_audit_sha256': b.sha(parent/'live_metadata_audit.json') if (parent/'live_metadata_audit.json').exists() else None,
                 'parent_path': str(parent), 'parent_plan_sha256': b.sha(parent/'plan.json'),
                 'parent_ledger_sha256': b.sha(parent/'transfer_ledger.json') if old else None,
                 'carried_requests': old['requests'] if old else 0,
                 'carried_charged_bytes': old['charged_bytes'] if old else 0,
                 'parent_started_epoch': old['started_epoch'] if old else None,
                 'stores': {g:b.store_url(g) for g in GROUPS},
                 'variables': {g:b.VARIABLES[g] for g in GROUPS},
                 'snow_delivery': 'Separate official standard CDS ERA5-Land sd GRIB141 at exact native cell and valid time; no snow value read here.',
                 'complete_experiment_inputs': False,
                 'continuation_wall_seconds': p['limits']['wall_seconds'],
                 'continued_transfer_and_request_limits': True,
                 'coordinate_rule': 'Identical full spatial axes; every requested valid time exactly present in each store. Entire historical/latest time-axis endpoints need not match.'})
    b.save(output/'plan.json', plan)
    if old:
        new = dict(old)
        new.update({'plan_sha256': b.sha(output/'plan.json'), 'started_epoch': time.time(),
                    'parent_ledger_sha256': plan['parent_ledger_sha256'],
                    'parent_started_epoch': old['started_epoch']})
        (output/'http_cache').mkdir()
        for url, item in old['objects'].items():
            if item.get('status') != 'complete':
                continue
            key = hashlib.sha256(url.encode()).hexdigest()
            source = parent/'http_cache'/key
            if not source.is_file() or b.sha(source) != item['sha256']:
                raise ValueError('Parent cached source checksum mismatch.')
            shutil.copy2(source, output/'http_cache'/key)
        b.save(output/'transfer_ledger.json', new)
    return plan


def checked_variable(reader, group):
    name, attrs = b.variable_name(reader, group)
    if attrs.get('GRIB_paramId') != b.VARIABLES[group]['param'] or attrs.get('GRIB_stepType') != 'instant':
        raise ValueError('Require explicit expected GRIB identity and instantaneous valid-time field.')
    return name, attrs


def validate_values(values):
    # Conservative representable physical bounds: flag rather than drop/impute.
    valid = np.isfinite(values[COLUMNS].to_numpy()).all(axis=1)
    for column in COLUMNS:
        x = values[column].to_numpy(float)
        if column.endswith('_c'):
            valid &= (x >= -173.15) & (x <= 126.85)
        else:
            valid &= (x >= 0) & (x <= 1)
    return valid


def run(output, credential_file=None, metadata_only=False):
    output = Path(output)
    plan = json.loads((output/'plan.json').read_text())
    verify(plan, output)
    if (output/'completion_4fields.json').exists():
        raise FileExistsError('Completed four-field extraction is immutable.')
    points = pd.read_parquet(output/'points.parquet')
    client = b.BoundedHTTP(output, b.load_token(credential_file), b.sha(output/'plan.json'), plan['limits'])
    grid = None
    records, readers = {}, {}
    for group in GROUPS:
        reader = b.ZarrReader(client, group)
        name, attrs = checked_variable(reader, group)
        lat, la = reader.coordinate('latitude')
        lon, lo = reader.coordinate('longitude')
        tv, ta = reader.coordinate('time')
        times = b.decode_times(tv, ta)
        if la.get('units') not in ('degrees_north','degree_north') or lo.get('units') not in ('degrees_east','degree_east'):
            raise ValueError('Unverified coordinate units.')
        if grid is None:
            grid = (lat,lon)
        elif not all(np.array_equal(a,c) for a,c in zip(grid,(lat,lon))):
            raise ValueError('ARCO fields do not share the identical native spatial grid.')
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError('Ambiguous valid-time axis.')
        ilat=b.nearest_indexes(lat,points.latitude)
        ilon=b.nearest_indexes(lon,points.longitude,longitude=True)
        wanted=pd.DatetimeIndex(points.era5_land_valid_time_utc)
        itime=times.get_indexer(wanted)
        if np.any(itime<0) or not np.array_equal(times[itime].asi8,wanted.asi8):
            raise ValueError('Exact requested valid hour absent.')
        positions=np.column_stack([itime,ilat,ilon])
        meta,_=reader.array(name)
        chunks=sorted(set(map(tuple,positions//np.asarray(meta['chunks']))))
        records[group]={'store':reader.url,'variable':name,'metadata_sha256':reader.metadata_sha,
                        'attributes':attrs,'global_attributes':reader.meta.get('.zattrs',{}),'array':meta,
                        'coordinate_attributes':{'latitude':la,'longitude':lo,'time':ta},
                        'coordinate_decoded_sha256':{'latitude':hashlib.sha256(lat.tobytes()).hexdigest(),
                          'longitude':hashlib.sha256(lon.tobytes()).hexdigest(),
                          'selected_valid_time':hashlib.sha256(times[itime].asi8.tobytes()).hexdigest()},
                        'requested_chunk_count':len(chunks),'chunk_indexes':[[int(z) for z in c] for c in chunks],
                        'decoded_bytes_upper_bound':int(len(chunks)*np.prod(meta['chunks'])*np.dtype(meta['dtype']).itemsize),
                        'chunk_time_coverage':[{'first':times[c*meta['chunks'][0]].isoformat(),
                          'last':times[min((c+1)*meta['chunks'][0],len(times))-1].isoformat()}
                          for c in sorted(set(x[0] for x in chunks))]}
        readers[group]=(reader,name,positions)
    native=points.copy()
    native['era5_land_grid_latitude']=grid[0][ilat]
    native['era5_land_grid_longitude']=grid[1][ilon]
    native['era5_land_latitude_delta_degrees']=native.era5_land_grid_latitude-native.latitude
    native['era5_land_longitude_delta_degrees']=(native.era5_land_grid_longitude-native.longitude+180)%360-180
    cellcols=['region_id','era5_land_valid_time_utc','era5_land_grid_latitude','era5_land_grid_longitude']
    cells=native[cellcols].drop_duplicates().sort_values(cellcols).reset_index(drop=True)
    cellpath=output/'native_cells.parquet'
    if cellpath.exists():
        if not pd.read_parquet(cellpath).equals(cells):
            raise ValueError('Frozen native cell requests changed.')
    else:
        cells.to_parquet(cellpath,index=False)
    preflight={'plan_sha256':b.sha(output/'plan.json'),'stores':records,'metadata_only':True,
               'all_four_variable_native_spatial_axes_identical':True,'all_selected_valid_times_identical':True,
               'native_cells_sha256':b.sha(cellpath),'unique_pilot_native_cell_hours':len(cells),
               'retrospective_reanalysis':True,'snow_water_equivalent_unavailable_in_arco':True,
               'full_five_field_contract_complete':False}
    pp=output/'preflight_4fields.json'
    if pp.exists():
        if json.loads(pp.read_text()) != preflight:
            raise ValueError('Live preflight changed since freezing.')
    else:
        b.save(pp,preflight)
    if metadata_only:
        return {'status':'four_field_preflight_complete','rows':len(points),'native_cell_hours':len(cells),
                'requested_chunks':sum(v['requested_chunk_count'] for v in records.values()),
                'decoded_upper_bound_bytes':sum(v['decoded_bytes_upper_bound'] for v in records.values()),
                'charged_bytes':client.state['charged_bytes'],'requests':client.state['requests'],
                'preflight_sha256':b.sha(pp),'native_cells_sha256':b.sha(cellpath)}
    for group,(reader,name,positions) in readers.items():
        vals=reader.read(name,positions)
        attrs=records[group]['attributes']
        if 'GRIB_missingValue' in attrs:
            vals[np.isclose(vals,float(attrs['GRIB_missingValue']),rtol=1e-6,atol=0)]=np.nan
        native[b.VARIABLES[group]['output']]=vals+b.VARIABLES[group]['offset']
    native['era5_land_four_fields_complete']=validate_values(native)
    native['era5_land_complete']=False
    native['era5_land_status']=np.where(native.era5_land_four_fields_complete,'four_fields_available_SWE_pending','native_cell_missing_no_land_substitution')
    native['era5_land_preflight_sha256']=b.sha(pp)
    native.to_parquet(output/'values_4fields.parquet',index=False)
    verify(plan,output)
    if b.sha(output/'plan.json') != client.state['plan_sha256']:
        raise ValueError('Plan changed during extraction.')
    result={'status':'four_fields_complete_SWE_pending','rows':len(native),
            'available_four_field_rows':int(native.era5_land_four_fields_complete.sum()),
            'available_five_field_rows':0,'plan_sha256':b.sha(output/'plan.json'),
            'preflight_sha256':b.sha(pp),'output_sha256':b.sha(output/'values_4fields.parquet'),
            'native_cells_sha256':b.sha(cellpath),'ledger_sha256':b.sha(client.path),
            'source_sha256':b.sha(__file__),'base_source_sha256':b.sha(_BASE_PATH),
            'requests':client.state['requests'],'charged_transfer_bytes':client.state['charged_bytes'],
            'no_new_thermal_labels':True,'no_training':True,'no_2025_requested_rows':True}
    b.save(output/'completion_4fields.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    for command in ['preflight','extract']:
        p=sub.add_parser(command);p.add_argument('--output',required=True);p.add_argument('--credential-file')
    args=ap.parse_args()
    if args.command=='plan':
        result=plan_from_parent(args.parent,args.output)
        print(json.dumps({k:result[k] for k in ['rows','years','pilot_count','carried_requests','carried_charged_bytes']}))
    else:print(json.dumps(run(args.output,args.credential_file,args.command=='preflight')))

if __name__=='__main__':main()
