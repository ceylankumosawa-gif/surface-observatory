"""Independent replay of the prospective two-field seasonal representation.

No candidate/baseline producer loader, transformer, splitter or scorer is
imported. This reuses only the independent geographic auditor's math. No fit,
source request or execution against an unfinished candidate is implemented.
"""
from pathlib import Path
import argparse,gc,hashlib,json,resource,sys,time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import audit_expanded_native_trial_v1 as independent

need,sha,bind,read,save,check,frame_hash=(getattr(independent,k) for k in
    ['need','sha','bind','read','save','check','frame_hash'])
BASE=list(independent.FIELDS)
SOLAR=['toa_daily_mean_horizontal_w_m2','toa_daily_change_w_m2_per_day']
FIELDS=BASE[:-2]+SOLAR
KEY=['native_cell_id','region_id']
MODES=['geography','month','reference']
IDENTITY,EXTRA,MASKS=independent.IDENTITY,independent.EXTRA,independent.MASKS
AIR,SKIN,PARAMS=independent.AIR,independent.SKIN,independent.PARAMS


def load_solar(proof_binding):
    proof=read(check(proof_binding))
    need(proof['status']=='passed' and proof['total_rows']==180253,'Checked full solar population required')
    frames=[];bindings=[proof_binding,proof['plan']]
    check(proof['plan'])
    for completion in proof['sidecar_completions']:
        done=read(check(completion))
        need(done['status']=='complete_source_only_solar_season' and done['fields']==SOLAR,'Different solar sidecar')
        artifact=done['artifacts']['features_midpoint.parquet'];check(done['plan'])
        f=pd.read_parquet(check(artifact),columns=KEY+['native_footprint_area_m2','datetime_utc']+SOLAR+[x+'__valid_area_fraction' for x in SOLAR])
        need(len(f)==done['rows'] and done['support_deficit_rows']==0,'Solar row/support count changed')
        need(np.isfinite(f[SOLAR].to_numpy()).all(),'Nonfinite solar geometry')
        for field in SOLAR:need(f[field+'__valid_area_fraction'].ge(1-1e-8).all(),'Incomplete solar area')
        frames.append(f);bindings.extend([completion,done['plan'],artifact])
    result=pd.concat(frames,ignore_index=True)
    need(len(result)==180253 and not result.duplicated(KEY).any(),'Solar population/identity changed')
    return result.set_index(KEY),bindings


def transform(frame,solar):
    """Independent identity/area/midpoint join; original29 remain bit-exact."""
    index=pd.MultiIndex.from_frame(frame[KEY])
    need(index.is_unique and solar.index.is_unique,'Duplicate seasonal identity')
    positions=solar.index.get_indexer(index)
    need((positions>=0).all(),'Previously complete row lacks seasonal support')
    matched=solar.iloc[positions]
    np.testing.assert_array_equal(matched.native_footprint_area_m2,frame.native_footprint_area_m2)
    start=pd.to_datetime(frame.granule_start_utc,utc=True,format='mixed')
    end=pd.to_datetime(frame.granule_end_utc,utc=True,format='mixed')
    need((end>start).all(),'Invalid granule interval')
    actual=pd.to_datetime(matched.datetime_utc,utc=True,format='mixed')
    np.testing.assert_array_equal(actual.to_numpy(),(start+(end-start)/2).to_numpy())
    result=frame[BASE[:-2]].copy(deep=True)
    for field in SOLAR:
        need(np.isfinite(matched[field]).all(),'Nonfinite seasonal field')
        need(matched[field+'__valid_area_fraction'].ge(1-1e-8).all(),'Partial seasonal support')
        result[field]=matched[field].to_numpy()
    need(result.columns.tolist()==FIELDS and np.isfinite(result.to_numpy()).all(),'Invalid transformed matrix')
    for field in BASE[:-2]:pd.testing.assert_series_equal(result[field],frame[field],check_exact=True)
    return result


def fingerprint(membership,values):
    need(values.columns.tolist()==FIELDS,'Different ordered candidate fields')
    return hashlib.sha256(json.dumps(dict(recipe=frame_hash(membership),features=frame_hash(values),parameters=PARAMS,fields=FIELDS),sort_keys=True).encode()).hexdigest()


def membership(frame):
    result=frame[IDENTITY+EXTRA].copy()
    result['weight']=independent.weights(frame)
    result['baseline_c']=frame[AIR].to_numpy()
    result['target_residual_c']=frame.lst_c.to_numpy()-frame[AIR].to_numpy()
    return result


def compare_recipe_table(saved,recipe):
    # Parquet is intentionally written index=False. Ordered identity columns,
    # target, anchor and weight remain checked; pandas source positions do not.
    pd.testing.assert_frame_equal(saved,recipe.reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-14)


def compare_metrics(calculated,saved,keys):
    need(not calculated.duplicated(keys).any() and not saved.duplicated(keys).any(),'Duplicate metric population')
    a,b=calculated.set_index(keys).sort_index(),saved.set_index(keys).sort_index()
    need(a.index.equals(b.index) and set(a.columns)==set(b.columns),'Metric denominator/schema differs')
    for field in a:np.testing.assert_allclose(a[field].to_numpy(float),b[field].to_numpy(float),rtol=1e-11,atol=1e-10,equal_nan=True)


def compare_saved_columns(path,expected,columns,check_dtype=True):
    """Bound memory while preserving every exact identity/control check."""
    for field in columns:
        actual=pd.read_parquet(path,columns=[field])[field]
        pd.testing.assert_series_equal(actual,expected[field],check_exact=True,check_dtype=check_dtype)


def metrics(frame,registry):
    # Only column labels are remapped; the independent metric implementation
    # uses its own split/area hierarchy and never calls the producer scorer.
    renamed=frame.rename(columns={'expanded':'old_only','solar_season':'expanded'})
    result=independent.independent_metrics(renamed,registry)
    result['model']=result.model.map({'old_only':'expanded','expanded':'solar_season','raw_air':'raw_air','raw_skin':'raw_skin'})
    return result


def find_baseline_proof(plan):
    root=check(plan['baseline_plan']).parent;completion=bind(root/'completion.json')
    done=read(completion['path'])
    need(done['status']=='complete_exploratory_data_comparison' and done['plan']==plan['baseline_plan'],'Wrong geographic baseline')
    matches=[]
    for b in plan['bindings']:
        if Path(b['path']).name!='audit.json':continue
        proof=read(check(b))
        if proof.get('status')=='passed' and proof.get('trial_completion')==completion:matches.append(b)
    need(len(matches)==1,'Exactly one passed exact-baseline audit required')
    need(completion in plan['bindings'],'Unbound baseline completion')
    return completion,matches[0]


def freeze(trial,out):
    sys.addaudithook(independent.deny_network)
    trial=trial.resolve();need(not out.exists(),'New immutable audit directory required')
    done=read(trial/'completion.json');need(done['status']=='complete_exploratory_solar_season_comparison','Candidate must be sealed first')
    p=read(check(done['plan']))
    need(p['features']==FIELDS and p['parameters']==PARAMS and len(p['fits'])==45,'Unexpected seasonal experiment')
    need(p['baseline']==AIR and p['requested_groups']==960 and not p['network_allowed'] and not p['production_changed'],'Changed anchor/scope')
    baseline_done,baseline_audit=find_baseline_proof(p)
    bp=read(check(p['baseline_plan']));bd=read(check(baseline_done))
    need(bp['features']==BASE and bp['parameters']==PARAMS and len(bp['fits'])==90,'Different baseline recipe')
    need(p['cohort_receipt']==bp['inputs']['cohort_receipt'] and p['cohort_frame_sha256']==bp['cohort_frame_sha256'],'Cohort/anchor changed')
    solar,solar_bindings=load_solar(p['geometry_check']);del solar
    bindings=[bind(trial/'completion.json'),done['plan'],baseline_done,baseline_audit,*p['bindings'],*solar_bindings,bind(__file__),bind(independent.__file__)]
    bindings.extend(bp['inputs'].values());bindings.extend(bd['artifacts'].values());bindings.extend(done['artifacts'].values())
    for mode in MODES:
        need(p['baseline_predictions'][mode]==bd['artifacts'][mode+'_predictions.parquet'],'Control prediction binding differs')
        for name in [mode+'_candidate_predictions.parquet',mode+'_predictions.parquet']:
            need(Path(done['artifacts'][name]['path']).resolve()==trial/name,'Wrong named candidate output')
    for name in ['fits.json','metrics.csv','training_metrics.csv']:
        need(Path(done['artifacts'][name]['path']).resolve()==trial/name,'Wrong named trial output')
    candidate_fits=read(trial/'fits.json');baseline_fits=read(check(bd['artifacts']['fits.json']))
    need(len(candidate_fits)==45 and len(baseline_fits)==90,'Incomplete fitting ledgers')
    for f in [*candidate_fits,*[x for x in baseline_fits if x['arm']=='expanded']]:
        if f['eligible']:bindings.extend([f['model'],f['fitting_rows']])
    for b in bindings:check(b)
    out.mkdir(parents=True)
    save(out/'plan.json',dict(status='frozen_before_independent_replay',trial=str(trial),trial_completion=bind(trial/'completion.json'),baseline_completion=baseline_done,baseline_audit=baseline_audit,
        bindings=list({(b['path'],b['sha256']):b for b in bindings}.values()),
        limits=dict(cpu_threads=2,memory_bytes=6*2**30,seconds=880,network=False),
        scope='All45 candidate recipes and expanded-baseline held model predictions, exact two-field join, all96 area-phase/960 case metrics and training metrics',fits_allowed=False,requests_allowed=False))
    print(json.dumps(bind(out/'plan.json')),flush=True)


def estimator(binding,fields,root):
    p=check(binding);need(p.is_relative_to(root/'models'),'Model from another experiment')
    model=joblib.load(p)
    need(isinstance(model,HistGradientBoostingRegressor),'Different estimator')
    need(all(model.get_params()[k]==v for k,v in PARAMS.items()),'Estimator parameters changed')
    need(list(model.feature_names_in_)==fields and model.n_features_in_==31 and model.n_iter_==100,'Feature/iteration contract changed')
    return model


def run(out):
    sys.addaudithook(independent.deny_network);started,cpu=time.monotonic(),time.process_time()
    ap=read(out/'plan.json');need(not(out/'audit.json').exists(),'Cannot overwrite audit')
    for b in ap['bindings']:check(b)
    done=read(check(ap['trial_completion']));trial=Path(ap['trial']);p=read(check(done['plan']))
    bd=read(check(ap['baseline_completion']));bp=read(check(p['baseline_plan']));broot=Path(ap['baseline_completion']['path']).parent
    need(read(check(ap['baseline_audit']))['trial_completion']==ap['baseline_completion'],'Baseline audit changed')
    receipt=read(check(p['cohort_receipt']));proof=read(check(bp['inputs']['cohort_check']))
    need(proof['status']=='passed' and proof['cohort_receipt']==p['cohort_receipt'],'Wrong cohort proof')
    for key in ['cohort','registry','source_associations','reserved_physical_acquisitions']:need(bp['inputs'][key]==receipt[key],'Different cohort input '+key)
    columns=list(dict.fromkeys(IDENTITY+EXTRA+MASKS+['lst_c']+BASE))
    d=pd.read_parquet(check(receipt['cohort']),columns=columns).reset_index(drop=True)
    need(len(d)==p['row_count']==done['row_count']==bp['row_count'] and frame_hash(d)==p['cohort_frame_sha256']==bp['cohort_frame_sha256'],'Changed cohort')
    registry=pd.read_csv(check(receipt['registry']),dtype=str,keep_default_na=False)
    sources=pd.read_csv(check(receipt['source_associations']),dtype=str,keep_default_na=False)
    reserved_table=pd.read_csv(check(receipt['reserved_physical_acquisitions']),dtype=str,keep_default_na=False);reserved=set(reserved_table.physical_acquisition_key)
    need(len(registry)==960 and registry.region_id.nunique()==48 and len(set(zip(registry.region_id,registry.phase)))==96,'Changed reporting denominator')
    need(len(reserved)==len(reserved_table)==171,'Changed reserved physical passes')
    need(not d.native_cell_id.duplicated().any() and not d.duplicated(['physical_acquisition_key','native_row','native_col']).any(),'Duplicate physical observation')
    solar,_=load_solar(p['geometry_check']);transform(d.loc[d.feature_complete],solar)
    specs={(x['mode'],x['key']):x for x in p['fits']};bspecs={(x['mode'],x['key']):x for x in bp['fits'] if x['arm']=='expanded'}
    fits=read(check(done['artifacts']['fits.json']));bfits_all=read(check(bd['artifacts']['fits.json']))
    bfits={(x['mode'],x['key']):(i,x) for i,x in enumerate(bfits_all) if x['arm']=='expanded'}
    need(len(specs)==len(bspecs)==len(fits)==45,'Changed recipe denominator')
    predicted={m:np.full((len(d),2),np.nan) for m in MODES};indices={m:np.full((len(d),2),-1,dtype=np.int16) for m in MODES}
    seen={};checks=[];training=[];complete=d.feature_complete.to_numpy()
    for number,split in enumerate(independent.expected_splits(d,registry,sources,reserved)):
        need(time.monotonic()-started<ap['limits']['seconds']-20,'Audit deadline')
        key=(split['mode'],split['key']);spec,bspec,actual=specs[key],bspecs[key],fits[number];bindex,bactual=bfits[key]
        need((actual['mode'],actual['key'],actual['arm'])==(*key,'expanded'),'Candidate fit order changed')
        for k,v in spec.items():need(actual[k]==v,'Frozen candidate recipe changed: '+k)
        for k,v in bspec.items():
            need(bactual[k]==v,'Baseline frozen recipe changed: '+k)
            if k!='fingerprint':need(spec[k]==v,'Non-seasonal candidate recipe change: '+k)
        need(spec['baseline_fingerprint']==bspec['fingerprint'],'Different paired baseline')
        train=d.iloc[split['expanded']];values=transform(train,solar);recipe=membership(train)
        eligible=len(train)>=80 and train.region_id.nunique()>=2 and train.utc_date.nunique()>=2
        need(spec['eligible']==eligible and spec['rows']==len(train),'Fitting support differs')
        need(spec['held_rows']==len(split['held']) and tuple(spec['held_group_ids'])==split['held_groups'],'Held denominator differs')
        need(spec['excluded_physical_keys']==split['excluded'],'Physical-pass exclusion differs')
        need(spec['held_positions_sha256']==hashlib.sha256(split['held'].astype('<i8').tobytes()).hexdigest(),'Held positions differ')
        need(frame_hash(train[BASE])==spec['feature_sha256'],'Original fitting features differ')
        need(frame_hash(values)==spec['transformed_feature_sha256'],'Seasonal fitting features differ')
        if not eligible:
            need(actual['status']==bactual['status']=='insufficient_fitting_support','Unsupported fit substituted')
            checks.append(dict(mode=key[0],key=key[1],rows=len(train),status=actual['status']));continue
        saved=pd.read_parquet(check(actual['fitting_rows']));independent.validate_membership(saved,train)
        compare_recipe_table(saved,recipe)
        need(frame_hash(saved)==spec['recipe_sha256'],'Saved identity/target/weight recipe differs')
        need(independent.expected_fingerprint(saved,train[BASE])==bspec['fingerprint'],'Baseline fingerprint differs')
        need(fingerprint(saved,values)==spec['fingerprint'],'Candidate fingerprint differs')
        baseline_saved=pd.read_parquet(check(bactual['fitting_rows']))
        pd.testing.assert_frame_equal(saved,baseline_saved,check_exact=True)
        pair=(actual['model'],actual['fitting_rows'],actual['origin']);fp=spec['fingerprint']
        if actual['status']=='reused_exact_recipe':need(fp in seen and seen[fp]==pair,'Reuse lacks earlier exact recipe')
        else:
            need(actual['status']=='fitted' and fp not in seen and actual['origin']=='/'.join(key),'Unexplained fit/reuse')
            seen[fp]=pair
        model=estimator(actual['model'],FIELDS,trial);control=estimator(bactual['model'],BASE,broot)
        held=split['held'][complete[split['held']]]
        if key[0]!='full' and len(held):
            mode='geography' if key[0] in ['original_pilot','new_pilot'] else key[0]
            need((indices[mode][held]==-1).all(),'Held predictions overwritten')
            part=d.iloc[held];anchor=part[AIR].to_numpy()
            predicted[mode][held,0]=anchor+control.predict(part[BASE]);indices[mode][held,0]=bindex
            predicted[mode][held,1]=anchor+model.predict(transform(part,solar));indices[mode][held,1]=number
            del part
        error=model.predict(values)-recipe.target_residual_c.to_numpy()
        training.append(dict(mode=key[0],key=key[1],rows=len(train),mae=float(np.abs(error).mean()),balanced_mae=float(independent.weights(train)@np.abs(error))))
        checks.append(dict(mode=key[0],key=key[1],rows=len(train),held_complete=len(held),status=actual['status'],fingerprint=fp,model_sha256=actual['model']['sha256'],baseline_model_sha256=bactual['model']['sha256']))
        del model,control,train,values,recipe,saved,baseline_saved
    need(len(checks)==done['fit_recipes']==45 and len(seen)==done['unique_fits'],'Fit totals differ')
    need(done['exact_recipe_reuses']==sum(x['status']=='reused_exact_recipe' for x in fits),'Reuse count differs')
    compare_metrics(pd.DataFrame(training),pd.read_csv(check(done['artifacts']['training_metrics.csv'])),['mode','key'])
    pd.DataFrame(training).to_csv(out/'independent_training_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'fit_checks.csv',index=False)
    roles=registry.drop_duplicates('region_id').set_index('region_id').geography_role
    base=d[IDENTITY+EXTRA+MASKS+['lst_c']].copy();base['raw_air']=d[AIR].to_numpy();base['raw_skin']=d[SKIN].to_numpy()
    base['geography_role']=base.region_id.map(roles);base['reserved_acquisition_excluded']=base.physical_acquisition_key.isin(reserved)
    del d,solar;gc.collect();all_metrics=[];prediction_checks=[]
    for mode in MODES:
        keep=base.region_id.eq('cabauw').to_numpy() if mode=='reference' else base.region_id.ne('cabauw').to_numpy()
        expected=base.loc[keep].reset_index(drop=True)
        paired=pd.read_parquet(check(done['artifacts'][mode+'_predictions.parquet']))
        for c in expected:pd.testing.assert_series_equal(paired[c],expected[c],check_exact=True,check_dtype=False)
        candidate_path=check(done['artifacts'][mode+'_candidate_predictions.parquet'])
        compare_saved_columns(candidate_path,expected,IDENTITY+EXTRA+MASKS+['lst_c'],check_dtype=False)
        for column,arm in enumerate(['expanded','solar_season']):
            np.testing.assert_allclose(paired[arm],predicted[mode][keep,column],rtol=0,atol=1e-10,equal_nan=True)
            np.testing.assert_array_equal(paired[arm+'_fit_index'],indices[mode][keep,column])
        np.testing.assert_array_equal(np.isfinite(paired.expanded),np.isfinite(paired.solar_season))
        compare_saved_columns(candidate_path,paired,['solar_season','solar_season_fit_index'])
        control_columns=list(expected.columns)+['expanded','expanded_fit_index']
        compare_saved_columns(check(p['baseline_predictions'][mode]),paired,control_columns)
        score=metrics(paired,registry);score['test_mode']=mode;all_metrics.append(score)
        prediction_checks.append(dict(mode=mode,rows=len(paired),predicted=int(paired.solar_season.notna().sum())))
        del paired,expected;gc.collect()
    del base,predicted,indices;gc.collect()
    calculated=pd.concat(all_metrics,ignore_index=True)
    compare_metrics(calculated,pd.read_csv(check(done['artifacts']['metrics.csv'])),['test_mode','segment_type','segment','view','model'])
    calculated.to_csv(out/'independent_metrics.csv',index=False);pd.DataFrame(prediction_checks).to_csv(out/'prediction_checks.csv',index=False)
    for b in ap['bindings']:check(b)
    save(out/'audit.json',dict(status='passed',trial_completion=ap['trial_completion'],baseline_completion=ap['baseline_completion'],baseline_audit=ap['baseline_audit'],geometry_check=p['geometry_check'],plan=bind(out/'plan.json'),
        fitting_recipes=45,unique_models=len(seen),reused_recipes=done['exact_recipe_reuses'],prediction_ledger_rows=sum(x['rows'] for x in prediction_checks),metric_rows=len(calculated),training_metric_rows=len(training),
        requested_groups=960,regional_phase_groups=96,reserved_physical_keys=171,independent_solar_identity_area_midpoint_join=True,unchanged_original_fields=29,independent_memberships_weights_metrics=True,paired_baseline_models_replayed=True,
        source_scope='Previously checked solar geometry and cohort/source proofs; exact joins, saved models and experiment mathematics independently replayed',raw_sources_redecoded=False,
        fits=0,requests=0,production_changed=False,reserved_thermal_opened=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        artifacts={f.name:bind(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(bind(out/'audit.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--trial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.action=='freeze':freeze(a.trial,a.output)
        else:run(a.output)
