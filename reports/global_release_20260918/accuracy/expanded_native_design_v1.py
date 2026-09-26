"""Prospective, metadata-only expanded native cohort and split contract.

No temperatures, predictors, estimator, network operation or fitting lives here.
Old and new source masks are inputs and are never edited. Physical overpasses,
not catalogue revisions/concepts, define exclusion groups.
"""
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd

MONTHS=('2021-01','2021-04','2021-07','2021-10')
PHYSICAL=re.compile(r'^VNP21:A2021\d{3}\.\d{4}$')
GROUP=['region_id','date','phase']
SOURCE=['region_id','date','phase','physical_acquisition_key','granule_start_utc','granule_end_utc']
ROLES={'original_development','original_reference','new_development','reserved_new_geography'}


def require(condition,message):
    if not condition:raise ValueError(message)


def combine_registry(original,global_registry):
    """Only metadata: original 384 + unchanged global 576; no source filtering."""
    require(len(original)==384 and len(global_registry)==576,'Original384/global576 required')
    old,new=original.copy(),global_registry.copy()
    require(old.region_id.nunique()==12 and new.region_id.nunique()==36,'Catalogue area count changed')
    require(set(old.region_id).isdisjoint(new.region_id),'Old/new region identity collision')
    require(not old.duplicated(GROUP).any() and not new.duplicated(GROUP).any(),'Duplicate requested group')
    old['cohort_family']='original_weekly'
    old['geography_role']=np.where(old.region_id.eq('cabauw'),'original_reference','original_development')
    require(set(new.role)=={'development_geography','reserved_new_geography'},'Unexpected global roles')
    require(new.loc[new.role.eq('development_geography'),'region_id'].nunique()==28,'Require 28 new development areas')
    require(new.loc[new.role.eq('reserved_new_geography'),'region_id'].nunique()==8,'Require eight reserved areas')
    new['cohort_family']='global_source_only_catalogue'
    new['geography_role']=new.role.map({'development_geography':'new_development','reserved_new_geography':'reserved_new_geography'})
    result=pd.concat([old,new],ignore_index=True,sort=False)
    require(len(result)==960 and not result.duplicated(GROUP).any(),'Full requested denominator changed')
    require(result.phase.isin(['day','night']).all(),'Unsupported phase')
    require(result.groupby('region_id').geography_role.nunique().eq(1).all(),'A geography has multiple roles')
    require(result.geography_role.value_counts().to_dict()=={'new_development':448,'original_development':352,'reserved_new_geography':128,'original_reference':32},'Role/group count changed')
    result['group_id']=result.region_id+'|'+result.date+'|'+result.phase
    return result


def associations_from_selected_plan(plan):
    """Retain selected metadata even when an acquisition later has no geometry."""
    rows=[]
    for source in plan['sources']:
        rec=source['record'];physical=rec['acquisition_key']
        require(PHYSICAL.fullmatch(physical) is not None,'Invalid canonical physical identity')
        for g in source['groups']:
            rows.append(dict(region_id=g['region_id'],date=g['date'],phase=g['phase'],
                physical_acquisition_key=physical,granule_start_utc=rec['granule_start_utc'],granule_end_utc=rec['granule_end_utc'],
                exact_stem=rec['stem'],cmr_concept=rec['granule_id'],cmr_revision=rec['cmr_revision']))
    # Multiple revisions may coexist in evidence. Do not collapse them or use
    # revision identity for split exclusions.
    return pd.DataFrame(rows,columns=SOURCE+['exact_stem','cmr_concept','cmr_revision'])


def validate_metadata(rows,registry,associations,reserved_keys):
    required=['native_cell_id','native_row','native_col','region_id','phase','utc_date','physical_acquisition_key',
              'granule_start_utc','granule_end_utc','native_label_admitted','native_fit_admitted','feature_complete']
    require(set(required)<=set(rows),'Missing metadata/mask fields')
    require(set(SOURCE)<=set(associations),'Missing source-association fields')
    require(not registry.duplicated(GROUP).any(),'Duplicate group registry')
    require(registry.geography_role.isin(ROLES).all(),'Unknown geography role')
    require(registry.groupby('region_id').geography_role.nunique().eq(1).all(),'Conflicting geography roles')
    require(all(PHYSICAL.fullmatch(k) for k in reserved_keys),'Invalid reserved physical key')
    require(not rows.native_cell_id.duplicated().any(),'Duplicate exact native cell')
    require(not rows.duplicated(['physical_acquisition_key','native_row','native_col']).any(),
            'Duplicate physical native cell across revisions/geographies; metadata-first resolution required')
    for flag in ['native_label_admitted','native_fit_admitted','feature_complete']:
        require(rows[flag].dtype==bool and not rows[flag].isna().any(),'Final cohort masks must be explicit booleans: '+flag)
    require(not (rows.native_fit_admitted&~rows.native_label_admitted).any(),'Fit mask admits rejected source label')
    allowed=set(map(tuple,registry[GROUP].itertuples(index=False,name=None)))
    require(set(zip(rows.region_id,rows.utc_date,rows.phase))<=allowed,'Rows outside requested groups')
    require(set(map(tuple,associations[GROUP].itertuples(index=False,name=None)))<=allowed,'Source associations outside requested groups')
    roles=registry.drop_duplicates('region_id').set_index('region_id').geography_role
    require(not rows.region_id.map(roles).eq('reserved_new_geography').any(),'Reserved arrays/labels remain unopened at development assembly')
    require(not rows.loc[rows.region_id.map(roles).eq('original_reference'),'native_fit_admitted'].any(),'Cabauw entered source fitting mask')
    require(rows.physical_acquisition_key.map(lambda k:bool(PHYSICAL.fullmatch(k))).all(),'Invalid row physical key')
    if 'acquisition_id' in rows:require(rows.acquisition_id.eq(rows.physical_acquisition_key).all(),'Old/new physical identity mismatch')
    source_keys=set(zip(associations.region_id,associations.date,associations.phase,associations.physical_acquisition_key))
    require(set(zip(rows.region_id,rows.utc_date,rows.phase,rows.physical_acquisition_key))<=source_keys,'Rows lack selected source metadata')
    combined=pd.concat([associations[['physical_acquisition_key','granule_start_utc','granule_end_utc']],rows[['physical_acquisition_key','granule_start_utc','granule_end_utc']]],ignore_index=True)
    start=pd.to_datetime(combined.granule_start_utc,utc=True,format='mixed');end=pd.to_datetime(combined.granule_end_utc,utc=True,format='mixed')
    require(not combined[['granule_start_utc','granule_end_utc']].isna().any().any(),'Missing source interval')
    require(((end-start).dt.total_seconds().gt(0)&(end-start).dt.total_seconds().le(361)).all(),'Unsupported VNP21 interval')
    require(start.dt.year.eq(2021).all()&end.dt.year.eq(2021).all(),'Only original 2021 interval scope')
    compact=combined.assign(_start=start,_end=end).groupby('physical_acquisition_key')[['_start','_end']].nunique()
    require(compact.eq(1).all().all(),'Same physical overpass has inconsistent interval across revisions')
    for r in combined.drop_duplicates().itertuples(index=False):
        require(pd.Timestamp(r.granule_start_utc).tzinfo is not None and pd.Timestamp(r.granule_end_utc).tzinfo is not None,'Explicit UTC offsets required')
        token=r.physical_acquisition_key.split(':A')[1].replace('.','')
        timestamp=pd.to_datetime(token,format='%Y%j%H%M',utc=True)
        require(timestamp.strftime('%Y%j%H%M')==token and abs((timestamp-pd.Timestamp(r.granule_start_utc)).total_seconds())<=1,'Physical key and observation interval disagree')
    row_start=pd.to_datetime(rows.granule_start_utc,utc=True,format='mixed')
    require(row_start.dt.strftime('%Y-%m-%d').eq(rows.utc_date).all(),'Source-start date changed')
    return roles


def fitting_mask(rows,registry,reserved_keys):
    roles=registry.drop_duplicates('region_id').set_index('region_id').geography_role
    return (rows.native_fit_admitted.to_numpy()&rows.feature_complete.to_numpy()&
        rows.region_id.map(roles).isin(['original_development','new_development']).to_numpy()&
        ~rows.physical_acquisition_key.isin(reserved_keys).to_numpy())


@dataclass
class Split:
    mode:str
    key:str
    train_positions:np.ndarray
    held_positions:np.ndarray
    held_group_ids:tuple
    excluded_physical_keys:tuple
    reference_only:bool=False


@dataclass
class PairedSplit:
    mode:str
    key:str
    old_only_train_positions:np.ndarray
    expanded_train_positions:np.ndarray
    held_positions:np.ndarray
    held_group_ids:tuple
    excluded_physical_keys:tuple
    reference_only:bool=False


def split_definitions(rows,registry,associations,reserved_keys,*,include_months=True,include_full=True):
    """All development/reference areas, including zero-row requested areas.

    Held predictions retain all source rows; downstream scoring uses unchanged
    label and finite-feature masks. Rows with reserved physical keys never enter
    fitting; they may remain explicitly excluded diagnostics on development.
    """
    roles=validate_metadata(rows,registry,associations,reserved_keys)
    base=fitting_mask(rows,registry,reserved_keys)
    group_ids=registry.assign(_id=registry.region_id+'|'+registry.date+'|'+registry.phase)
    for region in sorted(roles.index):
        role=roles[region]
        if role=='reserved_new_geography':continue
        held_sources=set(associations.loc[associations.region_id.eq(region),'physical_acquisition_key'])
        excluded=set(reserved_keys)|held_sources
        train=base&~rows.physical_acquisition_key.isin(held_sources).to_numpy()
        held=rows.region_id.eq(region).to_numpy()
        mode='reference' if role=='original_reference' else 'original_pilot' if role=='original_development' else 'new_pilot'
        yield Split(mode,region,np.flatnonzero(train),np.flatnonzero(held),tuple(group_ids.loc[group_ids.region_id.eq(region),'_id']),tuple(sorted(excluded)),role=='original_reference')
    if include_months:
        start=pd.to_datetime(associations.granule_start_utc,utc=True,format='mixed');end=pd.to_datetime(associations.granule_end_utc,utc=True,format='mixed')
        row_start=pd.to_datetime(rows.granule_start_utc,utc=True,format='mixed');row_end=pd.to_datetime(rows.granule_end_utc,utc=True,format='mixed')
        midpoint_month=(row_start+(row_end-row_start)/2).dt.strftime('%Y-%m')
        development=rows.region_id.map(roles).isin(['original_development','new_development']).to_numpy()
        for month in MONTHS:
            touches=start.dt.strftime('%Y-%m').eq(month)|end.dt.strftime('%Y-%m').eq(month)
            excluded=set(reserved_keys)|set(associations.loc[touches,'physical_acquisition_key'])
            train=base&~rows.physical_acquisition_key.isin(excluded).to_numpy()
            held=midpoint_month.eq(month).to_numpy()&development
            groups=group_ids.loc[group_ids.date.str.startswith(month)&group_ids.geography_role.isin(['original_development','new_development']),'_id']
            yield Split('month',month,np.flatnonzero(train),np.flatnonzero(held),tuple(groups),tuple(sorted(excluded)))
    if include_full:yield Split('full','all',np.flatnonzero(base),np.empty(0,dtype=int),tuple(),tuple(sorted(reserved_keys)))


def paired_split_definitions(rows,registry,associations,reserved_keys,*,include_months=True,include_full=True):
    """Same held population and source exclusions, two training populations.

    This emits at most 45 pairs / 90 prospective fit recipes for the complete
    960-group registry. It does not fit, assign weights, or declare old saved
    controls reusable. Weights must be recomputed within each arm and exact
    ordered membership/target/weight/feature hashes must prove any reuse.
    """
    roles=registry.drop_duplicates('region_id').set_index('region_id').geography_role
    is_old=rows.region_id.map(roles).eq('original_development').to_numpy()
    for split in split_definitions(rows,registry,associations,reserved_keys,
            include_months=include_months,include_full=include_full):
        expanded=split.train_positions
        old=expanded[is_old[expanded]]
        yield PairedSplit(split.mode,split.key,old,expanded,split.held_positions,
            split.held_group_ids,split.excluded_physical_keys,split.reference_only)


def coverage_ledger(rows,registry,reserved_keys):
    """Zero rows remain zero availability, never a zero error or a dropped group."""
    fit=fitting_mask(rows,registry,reserved_keys)
    q=rows.assign(date=rows.utc_date,source_rows=1,final_fit_eligible=fit,
                  excluded_reserved_physical=rows.physical_acquisition_key.isin(reserved_keys))
    names={'source_rows':'collected_row_count','native_label_admitted':'source_label_admitted_count','native_fit_admitted':'source_fit_admitted_count','feature_complete':'feature_complete_count','final_fit_eligible':'final_fit_eligible_count','excluded_reserved_physical':'reserved_acquisition_excluded_count'}
    counts=q.groupby(GROUP,observed=True)[list(names)].sum().rename(columns=names)
    require(not set(counts.columns)&set(registry.columns),'Coverage count columns already present; do not overwrite previous evidence')
    out=registry.merge(counts.reset_index(),on=GROUP,how='left',validate='one_to_one')
    for name in counts.columns:out[name]=out[name].fillna(0).astype(int)
    require(len(out)==len(registry),'Coverage join changed denominator')
    out['has_prediction_support']=out.feature_complete_count.gt(0)
    return out
