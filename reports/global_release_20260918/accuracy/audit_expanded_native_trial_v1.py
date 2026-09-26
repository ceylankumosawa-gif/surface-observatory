"""Independent saved-model replay of the paired geographic-data experiment.

No producer recipe, splitter, weight or score function is imported. No fitting
or source retrieval is implemented. Actual execution requires a sealed trial.
"""
from collections import defaultdict
from pathlib import Path
import argparse,gc,hashlib,json,resource,sys,time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT=Path('/opt/lst-pilot')
AIR='era5_land_air_temperature_c';SKIN='era5_land_skin_temperature_c'
FIELDS=[AIR,SKIN,'era5_land_soil_temperature_0_7cm_c','era5_land_soil_moisture_0_7cm_m3_m3',
    'era5_land_snow_water_equivalent_m','dewpoint_c','wind_speed_m_s','wind_direction_sin','wind_direction_cos',
    'cloud_cover_fraction','shortwave_down_w_m2','era5_longwave_down_w_m2','precipitation_mm_h',
    'rain_mm_24h','rain_mm_72h','air_temperature_lag1_c','air_temperature_lag24_c','shortwave_down_mean3_w_m2',
    'elevation','slope','worldcover_tree_class_fraction','worldcover_grass_class_fraction',
    'worldcover_crop_class_fraction','worldcover_built_class_fraction','worldcover_bare_class_fraction',
    'worldcover_water_fraction','solar_elevation_deg','hour_sin','hour_cos','day_of_year_sin','day_of_year_cos']
PARAMS=dict(loss='squared_error',learning_rate=.05,max_iter=100,max_leaf_nodes=7,max_depth=None,
    min_samples_leaf=20,l2_regularization=10.,max_bins=255,early_stopping=False,random_state=20260919)
IDENTITY=['native_cell_id','region_id','phase','utc_date','acquisition_id','granule_id',
    'granule_start_utc','granule_end_utc','native_footprint_area_m2']
EXTRA=['native_row','native_col','physical_acquisition_key']
MASKS=['native_label_admitted','native_fit_admitted','feature_complete']
ARMS=['old_only','expanded'];PREDICTORS=ARMS+['raw_air','raw_skin']
MONTHS=['2021-01','2021-04','2021-07','2021-10']
ROLES=['original_development','new_development','original_reference','reserved_new_geography']


def need(ok,message):
    if not ok:raise AssertionError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def check(binding):
    p=Path(binding['path']).resolve();need(p.is_relative_to(ROOT),'Evidence outside project')
    need(sha(p)==binding['sha256'],'Changed evidence: '+str(p));return p
def frame_hash(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
def deny_network(event,args):
    if event=='socket.connect':raise RuntimeError('Audit is offline')


def weights(frame):
    """Independent nested area/date/acquisition hierarchy, not group transforms."""
    if not len(frame):return np.empty(0,dtype=float)
    dates=defaultdict(set);passes=defaultdict(set);areas=defaultdict(float)
    keys=list(zip(frame.region_id,frame.phase,frame.utc_date,frame.acquisition_id))
    area=frame.native_footprint_area_m2.to_numpy(float)
    need(np.isfinite(area).all() and (area>0).all(),'Invalid weighted native area')
    for k,value in zip(keys,area):
        dates[k[:2]].add(k[2]);passes[k[:3]].add(k[3]);areas[k]+=value
    result=np.array([value/areas[k]/len(passes[k[:3]])/len(dates[k[:2]])/len(dates) for k,value in zip(keys,area)])
    need(abs(result.sum()-1)<1e-12 and np.isfinite(result).all(),'Independent weights fail closure')
    return result


def expected_splits(frame,registry,associations,reserved):
    roles=dict(registry[['region_id','geography_role']].drop_duplicates().itertuples(index=False,name=None))
    need(len(roles)==registry.region_id.nunique(),'Conflicting geographic roles')
    source_sets=defaultdict(set);intervals={}
    for r in associations.itertuples(index=False):
        source_sets[r.region_id].add(r.physical_acquisition_key)
        interval=(pd.Timestamp(r.granule_start_utc),pd.Timestamp(r.granule_end_utc))
        need(all(t.tz is not None for t in interval) and 0<(interval[1]-interval[0]).total_seconds()<=361,'Invalid source interval')
        need(r.physical_acquisition_key not in intervals or intervals[r.physical_acquisition_key]==interval,'Revision changes physical interval')
        intervals[r.physical_acquisition_key]=interval
    role=frame.region_id.map(roles)
    base=frame.native_fit_admitted.to_numpy()&frame.feature_complete.to_numpy()&role.isin(ROLES[:2]).to_numpy()
    base&=~frame.physical_acquisition_key.isin(reserved).to_numpy()
    def emit(mode,key,held,excluded,groups):
        expanded=np.flatnonzero(base&~frame.physical_acquisition_key.isin(excluded).to_numpy())
        old=expanded[role.iloc[expanded].eq('original_development').to_numpy()]
        return dict(mode=mode,key=key,held=np.flatnonzero(held),excluded=sorted(excluded),
            held_groups=tuple(groups),old_only=old,expanded=expanded)
    for region in sorted(roles):
        if roles[region]=='reserved_new_geography':continue
        g=registry.loc[registry.region_id.eq(region)]
        group_ids=[f'{r.region_id}|{r.date}|{r.phase}' for r in g.itertuples(index=False)]
        mode='reference' if roles[region]=='original_reference' else 'original_pilot' if roles[region]=='original_development' else 'new_pilot'
        yield emit(mode,region,frame.region_id.eq(region).to_numpy(),set(reserved)|source_sets[region],group_ids)
    starts=pd.to_datetime(frame.granule_start_utc,utc=True,format='mixed');ends=pd.to_datetime(frame.granule_end_utc,utc=True,format='mixed')
    middle=(starts+(ends-starts)/2).dt.strftime('%Y-%m')
    for month in MONTHS:
        excluded=set(reserved)|{k for k,(s,e) in intervals.items() if s.strftime('%Y-%m')==month or e.strftime('%Y-%m')==month}
        selected=registry.loc[registry.date.str.startswith(month)&registry.geography_role.isin(ROLES[:2])]
        groups=[f'{r.region_id}|{r.date}|{r.phase}' for r in selected.itertuples(index=False)]
        yield emit('month',month,middle.eq(month).to_numpy()&role.isin(ROLES[:2]).to_numpy(),excluded,groups)
    yield emit('full','all',np.zeros(len(frame),bool),set(reserved),[])


def validate_membership(saved,train):
    columns=IDENTITY+EXTRA+['weight','baseline_c','target_residual_c']
    need(saved.columns.tolist()==columns,'Fitting membership columns changed')
    pd.testing.assert_frame_equal(saved[IDENTITY+EXTRA].reset_index(drop=True),train[IDENTITY+EXTRA].reset_index(drop=True),check_dtype=False,check_exact=True)
    np.testing.assert_allclose(saved.weight,weights(train),rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(saved.baseline_c,train[AIR])
    np.testing.assert_array_equal(saved.target_residual_c,train.lst_c.to_numpy()-train[AIR].to_numpy())
    need(np.isfinite(saved[['weight','baseline_c','target_residual_c']].to_numpy()).all(),'Nonfinite fitting recipe')


def expected_fingerprint(membership,feature_frame):
    need(feature_frame.columns.tolist()==FIELDS,'Ordered feature columns changed')
    record={'recipe':frame_hash(membership),'features':frame_hash(feature_frame),'parameters':PARAMS,'fields':FIELDS}
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def population_indices(frame,registry):
    yield 'overall','all',np.arange(len(frame))
    for role in ROLES:yield 'geography_role',role,np.flatnonzero(frame.geography_role.eq(role))
    phase_indices=frame.groupby(['region_id','phase'],observed=True).indices
    for region,phase in sorted(set(zip(registry.region_id,registry.phase))):
        yield 'pilot_phase',region+'|'+phase,phase_indices.get((region,phase),np.empty(0,int))
    dates=frame.groupby(['region_id','utc_date','phase'],observed=True).indices
    for region,date,phase in registry[['region_id','date','phase']].itertuples(index=False,name=None):
        yield 'pilot_date_phase','|'.join((region,date,phase)),dates.get((region,date,phase),np.empty(0,int))
    for month in MONTHS:yield 'month',month,np.flatnonzero(frame.utc_date.str.startswith(month))


def independent_metrics(frame,registry):
    output=[]
    for kind,key,positions in population_indices(frame,registry):
        part=frame.iloc[positions]
        eligible=part.native_label_admitted.to_numpy()
        common=eligible&np.isfinite(part[PREDICTORS].to_numpy(float)).all(axis=1)
        for view in ['all_four_matched','own_available']:
            for model in PREDICTORS:
                mask=common if view=='all_four_matched' else eligible&np.isfinite(part[model].to_numpy(float))
                selected=part.iloc[np.flatnonzero(mask)]
                error=selected[model].to_numpy(float)-selected.lst_c.to_numpy(float)
                absolute=np.abs(error);w=weights(selected);n=len(selected)
                row=dict(segment_type=kind,segment=key,view=view,model=model,collected_rows=len(part),
                    qa_admitted=int(eligible.sum()),feature_complete=int(part.feature_complete.sum()),
                    predicted=int(np.isfinite(part[model]).sum()),paired=n,unscored=len(part)-n,
                    sites=selected.region_id.nunique(),dates=selected.utc_date.nunique(),acquisitions=selected.acquisition_id.nunique(),
                    reserved_physical_rows=int(selected.reserved_acquisition_excluded.sum()),
                    mae=float(absolute.sum()/n) if n else None,balanced_mae=float(sum(w*absolute)) if n else None,
                    bias=float(error.sum()/n) if n else None,balanced_bias=float(sum(w*error)) if n else None)
                for limit in [3,5,7]:
                    above=absolute>limit
                    row[f'above_{limit}_fraction']=float(above.sum()/n) if n else None
                    row[f'balanced_above_{limit}_fraction']=float(sum(w*above)) if n else None
                output.append(row)
    return pd.DataFrame(output)


def freeze(trial,out):
    trial=trial.resolve()
    need(not out.exists(),'New immutable audit directory required')
    completion=read(trial/'completion.json')
    need(completion['status']=='complete_exploratory_data_comparison','Sealed paired trial required')
    plan=read(check(completion['plan']))
    need(plan['features']==FIELDS and plan['parameters']==PARAMS and len(plan['fits'])==90,'Unexpected study recipe')
    bindings=[bind(trial/'completion.json'),completion['plan'],bind(Path(__file__))]+list(plan['inputs'].values())
    for name in ['plan.json','fits.json','geography_predictions.parquet','month_predictions.parquet','reference_predictions.parquet','metrics.csv']:
        need(name in completion['artifacts'] and Path(completion['artifacts'][name]['path']).resolve()==trial/name,'Named output binding differs: '+name)
    bindings.extend(completion['artifacts'].values())
    fits=read(trial/'fits.json')
    for f in fits:
        if f['eligible']:bindings.extend([f['model'],f['fitting_rows']])
    for b in bindings:check(b)
    out.mkdir(parents=True)
    save(out/'plan.json',dict(status='frozen_before_independent_replay',trial=str(trial),
        trial_completion=bind(trial/'completion.json'),bindings=list({(b['path'],b['sha256']):b for b in bindings}.values()),
        limits={'cpu_threads':2,'memory_bytes':6*2**30,'seconds':900,'network':False},
        scope='All 90 declared recipes, exact saved model reuse, all held prediction ledger rows and all 96 area-phase/960 requested-case metrics',
        fitting_allowed=False,requests_allowed=False))


def run(out):
    started=time.monotonic();cpu=time.process_time();sys.addaudithook(deny_network)
    audit_plan=read(out/'plan.json');trial=Path(audit_plan['trial'])
    need(not (out/'audit.json').exists(),'Audit cannot overwrite completion')
    for b in audit_plan['bindings']:check(b)
    done=read(check(audit_plan['trial_completion']));plan=read(check(done['plan']))
    receipt=read(check(plan['inputs']['cohort_receipt']));proof=read(check(plan['inputs']['cohort_check']))
    need(proof['status']=='passed' and proof['cohort_receipt']==plan['inputs']['cohort_receipt'],'Wrong cohort proof')
    need(plan['features']==FIELDS and plan['parameters']==PARAMS and not plan['saved_previous_model_reuse'],'Unexpected model contrast')
    for key in ['cohort','registry','source_associations','reserved_physical_acquisitions']:
        need(plan['inputs'][key]==receipt[key],'Plan/receipt input role mismatch: '+key)
    columns=list(dict.fromkeys(IDENTITY+EXTRA+MASKS+['lst_c']+FIELDS))
    d=pd.read_parquet(check(receipt['cohort']),columns=columns).reset_index(drop=True)
    need(len(d)==plan['row_count']==done['rows'],'Completion/preparation row denominator differs')
    registry=pd.read_csv(check(receipt['registry']),dtype=str,keep_default_na=False)
    sources=pd.read_csv(check(receipt['source_associations']),dtype=str,keep_default_na=False)
    reserved_frame=pd.read_csv(check(receipt['reserved_physical_acquisitions']),dtype=str,keep_default_na=False)
    reserved=set(reserved_frame.physical_acquisition_key)
    need(len(registry)==960 and registry.region_id.nunique()==48 and len(set(zip(registry.region_id,registry.phase)))==96,'Requested denominator changed')
    need(len(reserved)==len(reserved_frame)==171,'Reserved pass set changed')
    need(not d.native_cell_id.duplicated().any() and not d.duplicated(['physical_acquisition_key','native_row','native_col']).any(),'Duplicate physical observation')
    need(frame_hash(d)==plan['cohort_frame_sha256'],'Frozen cohort differs')
    need(np.isfinite(d.loc[d.native_label_admitted,'lst_c']).all() and np.isfinite(d.loc[d.feature_complete,FIELDS].to_numpy()).all(),'Nonfinite admitted inputs')
    fits=read(trial/'fits.json');need(len(fits)==len(plan['fits'])==90,'Fit ledger incomplete')
    specs={(x['mode'],x['key'],x['arm']):x for x in plan['fits']};need(len(specs)==90,'Duplicate fit recipe')
    predicted={m:np.full((len(d),2),np.nan) for m in ['geography','month','reference']}
    fit_index={m:np.full((len(d),2),-1,dtype=np.int16) for m in predicted}
    seen={};fit_checks=[];number=0
    for split in expected_splits(d,registry,sources,reserved):
        for arm in ARMS:
            need(time.monotonic()-started<audit_plan['limits']['seconds']-20,'Audit deadline')
            positions=split[arm];train=d.iloc[positions];spec=specs[split['mode'],split['key'],arm];actual=fits[number]
            need((actual['mode'],actual['key'],actual['arm'])==(split['mode'],split['key'],arm),'Fit order changed')
            for key,value in spec.items():need(actual[key]==value,'Fit receipt changes frozen '+key)
            eligible=len(train)>=80 and train.region_id.nunique()>=2 and train.utc_date.nunique()>=2
            need(spec['eligible']==eligible and spec['rows']==len(train),'Fitting support differs')
            need(spec['held_rows']==len(split['held']) and tuple(spec['held_group_ids'])==split['held_groups'],'Held denominator differs')
            need(spec['excluded_physical_keys']==split['excluded'],'Physical exclusion set differs')
            need(spec['held_positions_sha256']==hashlib.sha256(split['held'].astype('<i8').tobytes()).hexdigest(),'Held row positions differ')
            need(spec['feature_sha256']==frame_hash(train[FIELDS]),'Ordered fitting features differ')
            if not eligible:
                need(actual['status']=='insufficient_fitting_support','Unsupported fit silently substituted')
                fit_checks.append(dict(mode=split['mode'],key=split['key'],arm=arm,status=actual['status'],rows=len(train)))
                number+=1;continue
            membership=pd.read_parquet(check(actual['fitting_rows']));validate_membership(membership,train)
            need(frame_hash(membership)==spec['recipe_sha256'],'Saved target/weight/identity recipe changed')
            fingerprint=expected_fingerprint(membership,train[FIELDS]);need(fingerprint==spec['fingerprint'],'Recipe fingerprint changed')
            pair=(actual['model'],actual['fitting_rows'],actual['origin'])
            if actual['status']=='reused_exact_recipe':need(fingerprint in seen and seen[fingerprint]==pair,'Reuse is not an earlier exact recipe')
            else:
                need(actual['status']=='fitted' and fingerprint not in seen,'Unexplained refit/reuse')
                need(actual['origin']=='/'.join((split['mode'],split['key'],arm)),'Fitted origin changed')
                seen[fingerprint]=pair
            model_path=check(actual['model']);need(model_path.is_relative_to(trial/'models'),'Previous-study model reused')
            model=joblib.load(model_path);need(isinstance(model,HistGradientBoostingRegressor),'Unexpected estimator')
            need(all(model.get_params()[k]==v for k,v in PARAMS.items()),'Estimator parameters changed')
            need(list(model.feature_names_in_)==FIELDS and model.n_features_in_==31 and model.n_iter_==100,'Fitted feature/iteration contract changed')
            held=split['held'][d.feature_complete.to_numpy()[split['held']]]
            if split['mode']!='full' and len(held):
                mode='geography' if split['mode'] in ['original_pilot','new_pilot'] else split['mode'];column=ARMS.index(arm)
                need((fit_index[mode][held,column]==-1).all(),'Held prediction overwritten')
                predicted[mode][held,column]=d.iloc[held][AIR].to_numpy()+model.predict(d.iloc[held][FIELDS])
                fit_index[mode][held,column]=number
            fit_checks.append(dict(mode=split['mode'],key=split['key'],arm=arm,status=actual['status'],rows=len(train),
                held_complete=len(held),model_sha256=actual['model']['sha256'],fingerprint=fingerprint))
            number+=1;del model,membership,train
    need(number==done['fit_recipes']==90 and len(seen)==done['unique_fits'],'Fit/reuse total differs')
    need(done['requested_groups']==960 and done['exact_recipe_reuses']==sum(x['status']=='reused_exact_recipe' for x in fits),'Completion summary differs')
    pd.DataFrame(fit_checks).to_csv(out/'fit_checks.csv',index=False)
    roles=registry.drop_duplicates('region_id').set_index('region_id').geography_role
    base=d[IDENTITY+EXTRA+MASKS+['lst_c']].copy();base['raw_air']=d[AIR].to_numpy();base['raw_skin']=d[SKIN].to_numpy()
    base['geography_role']=base.region_id.map(roles);base['reserved_acquisition_excluded']=base.physical_acquisition_key.isin(reserved)
    del d;gc.collect();metrics=[];prediction_checks=[]
    for mode in ['geography','month','reference']:
        p=pd.read_parquet(trial/(mode+'_predictions.parquet'))
        keep=base.region_id.eq('cabauw').to_numpy() if mode=='reference' else base.region_id.ne('cabauw').to_numpy()
        expected=base.loc[keep].reset_index(drop=True)
        pd.testing.assert_frame_equal(p[expected.columns].reset_index(drop=True),expected,check_dtype=False,check_exact=True)
        for col,arm in enumerate(ARMS):
            np.testing.assert_allclose(p[arm],predicted[mode][keep,col],rtol=0,atol=1e-10,equal_nan=True)
            np.testing.assert_array_equal(p[arm+'_fit_index'],fit_index[mode][keep,col])
        prediction_checks.append(dict(mode=mode,rows=len(p),old_only_predicted=int(p.old_only.notna().sum()),expanded_predicted=int(p.expanded.notna().sum())))
        score=independent_metrics(p,registry);score['test_mode']=mode;metrics.append(score)
        del p,expected;gc.collect()
    del base,predicted,fit_index;gc.collect()
    calculated=pd.concat(metrics,ignore_index=True);saved=pd.read_csv(trial/'metrics.csv')
    keys=['test_mode','segment_type','segment','view','model'];need(not saved.duplicated(keys).any(),'Duplicate metric key')
    left=calculated.set_index(keys).sort_index();right=saved.set_index(keys).sort_index()
    need(left.index.equals(right.index),'Metric populations/denominators differ')
    need(set(left.columns)==set(right.columns),'Metric schema differs')
    for c in left:np.testing.assert_allclose(left[c].to_numpy(float),right[c].to_numpy(float),rtol=1e-11,atol=1e-10,equal_nan=True)
    calculated.to_csv(out/'independent_metrics.csv',index=False)
    pd.DataFrame(prediction_checks).to_csv(out/'prediction_checks.csv',index=False)
    for b in audit_plan['bindings']:check(b)
    save(out/'audit.json',dict(status='passed',trial_completion=audit_plan['trial_completion'],plan=bind(out/'plan.json'),
        fitting_recipes=90,unique_models=len(seen),reused_recipes=sum(x['status']=='reused_exact_recipe' for x in fits),
        prediction_ledger_rows=sum(x['rows'] for x in prediction_checks),metric_rows=len(calculated),
        requested_groups=960,regional_phase_groups=96,reserved_physical_keys=171,
        independent_memberships_weights_metrics=True,geometry_empty_sources_excluded=True,
        raw_sources_redecoded=False,source_scope='Bound independent cohort/source/feature proofs; this audit replays trial mathematics',
        fits=0,requests=0,production_changed=False,reserved_thermal_opened=False,
        wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        artifacts={p.name:bind(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--trial',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(limits=2):
        if args.action=='freeze':freeze(args.trial,args.output)
        else:run(args.output)
