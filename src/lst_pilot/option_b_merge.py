"""Audit source admissibility and freeze a paired exploratory fitting cohort."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .ecostress import digest,save_json
from .option_b_cohort import spatial_flags


def merge_cohort(legacy, expanded, areas_path, output):
    paths=[Path(legacy),*map(Path,expanded)]
    areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']}
    frames=[];sources=[]
    for i,path in enumerate(paths):
        dates=pd.to_datetime(pd.read_parquet(path,columns=['datetime_utc']).datetime_utc,utc=True)
        if dates.isna().any() or not dates.dt.year.isin([2021,2022,2023]).all():
            raise ValueError('Reserved test years cannot enter fitting cohort; thermal columns were not loaded.')
        frame=pd.read_parquet(path)
        origin='legacy' if i==0 else 'expanded'
        if not frame.cohort_origin.eq(origin).all():raise ValueError('Input source origin differs from declared cohort.')
        if frame.sample_id.duplicated().any():raise ValueError('Duplicated input sample IDs.')
        sources.append({'path':str(path.resolve()),'sha256':digest(path),'rows':len(frame),'origin':origin})
        frame['paired_input_path']=str(path.resolve());frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    stamp=pd.to_datetime(data.datetime_utc,utc=True)
    if not stamp.dt.year.isin([2021,2022,2023]).all():raise ValueError('Reserved test years cannot enter fitting cohort.')
    if data.sample_id.isna().any():raise ValueError('Missing sample identity.')
    # Input order makes legacy the explicit tie winner; each physical sample appears once.
    duplicate=data.sample_id.duplicated(keep='first')
    reason=pd.Series('',index=data.index,dtype='object')
    reason.loc[duplicate]='duplicate_sample_prefer_legacy_then_earlier_batch'
    def reject(mask,label):reason.loc[reason.eq('') & mask]=label
    reject(~np.isfinite(data.lst_c.to_numpy(float)),'nonfinite_label')
    reject(data.climate_class.isna() | data.climate_class.astype(str).isin(['unknown','__unknown__','']),'unknown_climate')
    new=data.cohort_origin.eq('expanded')
    reject(new & ~data.worldcover_land_fraction.ge(.8),'missing_or_insufficient_independent_land_fraction')
    reject(new & ~data.worldcover_water_fraction.le(.05),'independent_water_fraction_exceeds_five_percent')
    reject(new & ~data.station_pair_available.eq(True).fillna(False).astype(bool),'no_actual_station_pair')
    reject(new & ~data.station_age_minutes.between(0,90),'station_time_mismatch')
    reject(new & ~data.station_distance_km.between(0,100),'station_distance_mismatch')
    night=new & data.label_product.eq('ecostress_v2')
    source_pass=data.get('source_screen_pass',pd.Series(False,index=data.index))
    reject(night & ~source_pass.eq(True).fillna(False).astype(bool),'night_source_screen_incomplete')
    reject(night & ~data.solar_elevation_deg.le(-6),'night_solar_geometry_not_supported')
    # Causal optics were already validated during pairing; assert the per-row valid times again.
    for column in ('optical_earliest_source_utc','optical_latest_source_utc'):
        t=pd.to_datetime(data[column],utc=True)
        reject(new & (t.isna() | t.gt(stamp) | t.lt(stamp-pd.Timedelta(days=32))),'missing_future_or_stale_optical')
    reject(new & pd.to_datetime(data.optical_earliest_source_utc,utc=True).gt(
        pd.to_datetime(data.optical_latest_source_utc,utc=True)),'inverted_optical_time_interval')
    data=spatial_flags(data,areas)
    data['research_admissibility_reason']=reason.to_numpy()
    admitted=data.loc[reason.eq('')].copy()
    if admitted.sample_id.duplicated().any():raise ValueError('Cohort duplicates remain.')
    # Same physical cell/acquisition is still forbidden if an external sample ID changed.
    if admitted.duplicated(['region_id','acquisition_id','grid_row','grid_col']).any():
        raise ValueError('Duplicate physical observations have different sample IDs.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'paired_input.parquet').exists():raise ValueError('Frozen cohort exists; choose a new output directory.')
    admitted.to_parquet(out/'paired_input.parquet',index=False)
    data.loc[reason.ne('')].to_parquet(out/'excluded_rows.parquet',index=False)
    counts=[]
    for (region,origin,product),group in admitted.groupby(['region_id','cohort_origin','label_product']):
        counts.append({'region_id':region,'origin':origin,'product':product,'rows':len(group),
                       'dates':pd.to_datetime(group.datetime_utc,utc=True).dt.floor('D').nunique(),
                       'acquisitions':group.acquisition_id.nunique(),
                       'reserved_rows':int(group.spatial_holdout.sum()),'buffer_rows':int(group.in_holdout_buffer.sum()),
                       'complete_A':int(group.features_A_complete.sum()),'complete_D':int(group.features_D_complete.sum())})
    report={'input_sources':sources,'code_sha256':digest(__file__),'cohort_code_sha256':digest(Path(__file__).with_name('option_b_cohort.py')),
            'input_rows':len(data),'admitted_rows':len(admitted),'exclusion_counts':reason[reason.ne('')].value_counts().to_dict(),
            'counts':counts,'paired_sha256':digest(out/'paired_input.parquet'),
            'scope':'Exploratory clear-sky research; model split/feature completeness applied separately by trainer.',
            'sampling_note':'Early sampling used cell-centre buffers; all final rows were reclassified using whole-cell footprint buffers before fitting.'}
    save_json(out/'cohort_manifest.json',report)
    print(json.dumps({k:report[k] for k in ('input_rows','admitted_rows','exclusion_counts','paired_sha256')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--legacy',required=True)
    p.add_argument('--expanded',nargs='+',required=True);p.add_argument('--areas',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();merge_cohort(a.legacy,a.expanded,a.areas,a.output)


if __name__=='__main__':main()
