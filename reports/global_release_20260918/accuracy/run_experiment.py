"""One frozen phase/snow convex blend, using existing nested held-month F.

Execute only on Hetzner. No source retrieval, tree fitting or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lst_global.accuracy import evaluate
from lst_pilot import option_b_train as old

ROOT = Path('/opt/lst-pilot')
PARENT = ROOT/'runs/weather_alignment_20260911_v1/stage_a_model_v1'
PARENT_SHA = 'ec5ab8a5eb928d3b0cb5f3cac3f6dc036e88aabafd83fe812a610d54a448ace7'
FIT_SHA = 'a7b99629e08bcf848ce2adb26615012f2a1cfbb0645dc68f24e4fbb4ea26bdc6'
MODEL_SHA = 'c955f3a69e393eef29dd95e291d751b73e845a43e7ab2067289c6f1cc394f447'
HERE = Path(__file__).resolve().parent
REGIONS = ('boulder','cabauw','cape_town','darwin_howard_springs','gobabeb','greater_london',
           'lhasa','manaus_zf2','singapore_johor','sioux_falls','sodankyla','utqiagvik')
STRATA = ('day|no_snow','day|snow','night|no_snow','night|snow')
META = ['sample_id','region_id','phase','datetime_utc','utc_day','split','season','air_group',
        'label_product','acquisition_id','weight_surface_group','lst_c','air_temperature_c']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def binding(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(old.json_ready(value),indent=2,allow_nan=False)+'\n')
def require(ok, message):
    if not ok: raise ValueError(message)


def safe_frame(path, years):
    metadata = pd.read_parquet(path, columns=['datetime_utc'])
    stamps = pd.to_datetime(metadata.datetime_utc, utc=True)
    require(stamps.notna().all() and stamps.dt.year.isin(years).all(), 'Thermal year boundary violated')
    result = pd.read_parquet(path)
    result['datetime_utc'] = pd.to_datetime(result.datetime_utc, utc=True)
    return result


def strata(frame):
    require(frame.phase.isin(['day','night']).all(), 'Unknown phase')
    require(np.isfinite(frame.aligned_swe_m.to_numpy(float)).all(), 'Missing independent SWE')
    return frame.phase.astype(str) + '|' + np.where(frame.aligned_swe_m.to_numpy(float)>=.001,'snow','no_snow')


def fit_coefficients(frame, f_pred):
    require(pd.to_datetime(frame.datetime_utc,utc=True).dt.year.isin([2021,2022]).all(), 'No later labels in coefficient fitting')
    f_pred=np.asarray(f_pred,float)
    require(f_pred.shape==(len(frame),) and np.isfinite(f_pred).all(), 'Invalid calibration predictions')
    weight = old.balanced_weights(frame)
    tags = strata(frame)
    output={}
    for key in STRATA:
        mask=tags.eq(key).to_numpy(); group=frame.loc[mask]; w=weight[mask]
        if len(group): w=w/w.sum()
        stamps=pd.to_datetime(group.datetime_utc,utc=True)
        dates=stamps.dt.strftime('%Y-%m-%d');months=stamps.dt.strftime('%Y-%m')
        supported=dates.nunique()>=6 and months.nunique()>=2
        if len(group):
            mass=pd.Series(w).groupby(dates.to_numpy()).sum().to_numpy()
            neff=float(1/np.sum(mass**2))
            delta=group.aligned_skin_c.to_numpy(float)-f_pred[mask]
            target=group.lst_c.to_numpy(float)-f_pred[mask]
            require(np.isfinite(delta).all() and np.isfinite(target).all(), 'Missing calibration values')
            numerator=float(neff*np.sum(w*delta*target));denominator=float(neff*np.sum(w*delta**2)+20.)
            raw=numerator/denominator;lam=float(np.clip(raw,0,1)) if supported else 0.
        else: neff=0.;numerator=0.;denominator=20.;raw=0.;lam=0.
        output[key]={'rows':len(group),'dates':int(dates.nunique()),'months':int(months.nunique()),
                     'effective_global_dates':neff,'supported':bool(supported),'lambda':lam,'raw_lambda':raw,
                     'numerator':numerator,'denominator':denominator,'regularization':20.,
                     'row_sha256':old.row_hash(group),'normalised_weight_sha256':hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()}
    return output


def apply_coefficients(frame, f_pred, coefficients):
    tags=strata(frame);lam=tags.map({key:value['lambda'] for key,value in coefficients.items()}).to_numpy(float)
    f=np.asarray(f_pred,float);skin=frame.aligned_skin_c.to_numpy(float)
    require(np.isfinite(skin).all() and np.isfinite(lam).all(), 'Incomplete candidate input')
    result=f+lam*(skin-f)
    require(np.all(result>=np.minimum(f,skin)-1e-12) and np.all(result<=np.maximum(f,skin)+1e-12), 'Convex bound violated')
    require(np.array_equal(result[lam==0],f[lam==0]), 'Unsupported fallback differs from F')
    return result,lam


def metrics(frame, values, cohort):
    groups=[('overall','overall',np.arange(len(frame)))]
    for kind,keys in [('region_phase',['region_id','phase']),('region_phase_season',['region_id','phase','season'])]:
        for key,index in frame.groupby(keys,observed=True).indices.items():
            groups.append((kind,'|'.join(map(str,key)),index))
    result=[]
    for kind,key,index in groups:
        group=frame.iloc[index]
        for model,pred in values.items():
            p=np.asarray(pred)[index];error=p-group.lst_c.to_numpy(float)
            result.append({'cohort':cohort,'segment_type':kind,'segment':key,'model':model,
                           **old.metrics(group,p),'raw_gt5_fraction':float((abs(error)>5).mean()),
                           'raw_gt7_fraction':float((abs(error)>7).mean())})
    return result


def select(rows):
    records={(r['segment_type'],r['segment'],r['model']):r for r in rows}
    f=records['overall','overall','F'];c=records['overall','overall','skin_blend']
    failures=[]
    if f['mae_c']-c['mae_c']<.10: failures.append('overall_improvement_below_0.10c')
    if c['unweighted_pixel_mae_c']>f['unweighted_pixel_mae_c']+.10: failures.append('ordinary_mae_regression')
    if c['centered_contrast_mae_c']>f['centered_contrast_mae_c']+.10: failures.append('contrast_regression')
    for (kind,key,model),base in records.items():
        if model!='F' or kind not in ('overall','region_phase') or (kind!='overall' and base['utc_date_count']<6): continue
        candidate=records[kind,key,'skin_blend']
        if kind!='overall' and candidate['mae_c']>base['mae_c']+.20: failures.append(key+':mae_regression')
        for n in (5,7):
            if candidate[f'fraction_abs_error_gt_{n}c']>base[f'fraction_abs_error_gt_{n}c']+.02:
                failures.append(key+f':gt{n}_regression')
    return {'eligible':not failures,'selected_candidate':'skin_blend' if not failures else None,
            'failures':failures,'global_target_met':False,'automatic_promotion':False,
            'overall_F':f,'overall_candidate':c}


def requirements(regions=REGIONS, kind='date_holdout'):
    return [{'region_id':r,'phase':p,'resolution_m':100,'validation_kind':kind} for r in regions for p in ('day','night')]


def acceptance(frame, pred, training, groups, model_id, kind='date_holdout'):
    reference=frame[['sample_id','region_id','phase','datetime_utc','lst_c']].assign(resolution_m=100,validation_kind=kind)
    return evaluate(reference,pred,training,groups,model_id=model_id,evidence_status='repeated_diagnostic')


def run(output):
    start=time.monotonic();cpu=time.process_time()
    require(not output.exists(),'Use a new immutable experiment directory')
    require(sha(PARENT/'completion.json')==PARENT_SHA,'Parent completion changed')
    completion=read(PARENT/'completion.json')
    for name,digest in completion['artifacts'].items(): require(sha(PARENT/name)==digest,'Parent output changed: '+name)
    for file in ('selection_freeze.json','full_fit_freeze.json'):
        for name,digest in read(PARENT/file)['files'].items(): require(sha(PARENT/name)==digest,'Parent fitting changed: '+name)
    require(sha(PARENT/'fit_frame.parquet')==FIT_SHA and sha(PARENT/'full/F.joblib')==MODEL_SHA,'Frozen F changed')
    output.mkdir(parents=True)
    save(output/'manifest.json',{'protocol':binding(HERE/'PROTOCOL.md'),'runner':binding(__file__),
        'evaluator':binding(ROOT/'src/lst_global/accuracy.py'),'metrics_source':binding(ROOT/'src/lst_pilot/option_b_train.py'),
        'parent_completion':binding(PARENT/'completion.json'),'parent_fit':binding(PARENT/'fit_frame.parquet'),
        'parent_selection_freeze':binding(PARENT/'selection_freeze.json'),'parent_full_freeze':binding(PARENT/'full_fit_freeze.json'),
        'model':binding(PARENT/'full/F.joblib'),'reserved_2025_opened':False,'candidate_count':1})
    frame=safe_frame(PARENT/'fit_frame.parquet',[2021,2022]).reset_index(drop=True)
    saved=safe_frame(PARENT/'oof_predictions.parquet',[2021,2022]).set_index('sample_id').loc[frame.sample_id]
    require(np.array_equal(saved.lst_c.to_numpy(),frame.lst_c.to_numpy()),'OOF label alignment changed')
    folds=saved.fold.to_numpy(int);f=saved.F_lst_c.to_numpy(float);candidate=np.full(len(frame),np.nan);lambdas=candidate.copy()
    require(np.array_equal(folds,((frame.datetime_utc.dt.month-1)%3).to_numpy()),'Frozen global-month folds changed')
    all_coefficients={};members=[];lineage=[]
    for fold in range(3):
        mask=folds!=fold;train=frame.loc[mask].reset_index(drop=True);held=frame.loc[~mask]
        inner=[]
        for number in sorted(set(folds[mask])):
            path=PARENT/f'fold_{fold}/inner_held_{number}/held_predictions.parquet'
            part=safe_frame(path,[2021,2022])
            expected=train.loc[((train.datetime_utc.dt.month-1)%3).eq(number)]
            require(set(part.sample_id)==set(expected.sample_id),'Inner held identities changed')
            fit_path=path.with_name('F_training_rows.parquet')
            membership=safe_frame(fit_path,[2021,2022])
            allowed=train.loc[~train.sample_id.isin(expected.sample_id)]
            require(set(membership.sample_id)==set(allowed.sample_id),'Inner fitting identity changed')
            require(set(membership.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(part.datetime_utc.dt.strftime('%Y-%m')),'Inner month leakage')
            require(set(membership.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(held.datetime_utc.dt.strftime('%Y-%m')),'Outer month leakage')
            inner.append(part);lineage.append({'outer_fold':fold,'inner_held':int(number),'held':binding(path),'fit':binding(fit_path)})
        inner=pd.concat(inner).set_index('sample_id').loc[train.sample_id]
        require(np.array_equal(inner.lst_c.to_numpy(),train.lst_c.to_numpy()),'Inner labels changed')
        coefficients=fit_coefficients(train,inner.F_lst_c.to_numpy(float));all_coefficients[str(fold)]=coefficients
        candidate[~mask],lambdas[~mask]=apply_coefficients(held,f[~mask],coefficients)
        members.append(train[['sample_id','region_id','datetime_utc']].assign(model_fit_id=f'outer_{fold}'))
    require(np.isfinite(candidate).all(),'Incomplete OOF candidate')
    membership=pd.concat(members,ignore_index=True);membership.to_parquet(output/'fitting_membership.parquet',index=False)
    all_coefficients['full']=fit_coefficients(frame,f)
    save(output/'coefficients.json',all_coefficients);save(output/'nested_lineage.json',lineage)
    prediction=frame[[*META,'aligned_skin_c','aligned_swe_m']].assign(fold=folds,F_lst_c=f,skin_blend_lst_c=candidate,blend_lambda=lambdas)
    prediction.to_parquet(output/'oof_predictions.parquet',index=False)
    rows=metrics(frame,{'F':f,'skin_blend':candidate},'original_oof')
    pd.DataFrame(rows).to_csv(output/'oof_metrics.csv',index=False)
    selection=select(rows);save(output/'selection.json',selection)
    # Explicitly freeze candidate choices before loading any external evaluation values.
    save(output/'selection_freeze.json',{'frozen_before_external_values':True,
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'reserved_2025_opened':False})
    # Preserve the original 44,316 source-screened rows, including 144 unavailable-input rows.
    original_path=ROOT/'runs/shared_model_remedies_20260910_v1/fit_frame.parquet'
    original=safe_frame(original_path,[2021,2022])
    missing=original.loc[~original.sample_id.isin(frame.sample_id),'sample_id'].tolist()
    require(len(original)==44316 and len(missing)==144,'Original reference coverage changed')
    panels={}
    for name,values in [('F',f),('skin_blend',candidate)]:
        pred=frame[['sample_id']].assign(predicted_lst_c=values,model_fit_id=[f'outer_{v}' for v in folds])
        panels[name]=acceptance(original,pred,membership,requirements(),name)
        save(output/f'{name}_oof_acceptance.json',panels[name])
    save(output/'original_coverage.json',{'input':binding(original_path),'reference_rows':44316,'predicted_rows':44172,'missing_sample_ids':missing})
    external=[];external_panels=[]
    full_membership=frame[['sample_id','region_id','datetime_utc']].assign(model_fit_id='full')
    for family in ('old','newly_collected_2023','legacy_2024'):
        source=PARENT/(family+'_predictions.parquet')
        data=safe_frame(source,[2021,2022,2023,2024]);base=data.F_lst_c.to_numpy(float)
        values,lam=apply_coefficients(data,base,all_coefficients['full'])
        data=data.assign(skin_blend_lst_c=values,blend_lambda=lam)
        data.to_parquet(output/(family+'_predictions.parquet'),index=False)
        for split,index in data.groupby('split',observed=True).indices.items():
            part=data.iloc[index].reset_index(drop=True)
            label=family+'_'+str(split)
            external.extend(metrics(part,{'F':base[index],'skin_blend':values[index]},label))
            kind='region_holdout' if split=='heldout_region' else 'spatial_diagnostic' if split=='heldout_spatial' else 'date_holdout'
            for model,column in [('F','F_lst_c'),('skin_blend','skin_blend_lst_c')]:
                pred=part[['sample_id']].assign(predicted_lst_c=part[column].to_numpy(float),model_fit_id='full')
                panel=acceptance(part,pred,full_membership,requirements(kind=kind),model,kind)
                save(output/(label+'_'+model+'_acceptance.json'),panel)
                external_panels.append({'cohort':label,'model':model,'required_groups':panel['required_groups'],'observed_groups':panel['observed_groups'],
                                        'passed_groups':panel['passed_groups'],'point_target_met_groups':panel['point_target_met_groups'],'missing_groups':panel['missing_groups']})
    pd.DataFrame(external).to_csv(output/'external_metrics.csv',index=False)
    save(output/'external_acceptance_inventory.json',external_panels)
    status=panels['F'].copy()
    status.update({'active_model':'F_complete_native_44172','model_sha256':MODEL_SHA,'status':'target_unmet',
        'global_target_met':False,'release_qualified':False,'production_unchanged':True,'training_changes_deployed':False,
        'experiment':{'candidate':'phase_snow_skin_blend','eligible_relative_improvement':selection['eligible'],
                      'selection_failures':selection['failures'],'run':str(output),'selection_sha256':sha(output/'selection.json')},
        'validation_gaps':['No independent all-region day/night confirmation panel','No unopened final confirmation used',
            'Existing 2023–24 evidence was previously inspected','Other output resolutions have no matched accuracy panel',
            'Clear-sky thermal sources do not establish all-weather accuracy','Present-time forecast inputs are not validated'],
        'source_completion_sha256':PARENT_SHA,'reference_scope':'12-pilot panel; original source-screened 2021–22 OOF rows',
        'experiment_manifest_sha256':sha(output/'manifest.json')})
    save(output/'status.json',status)
    save(output/'completion.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'reserved_2025_opened':False,'production_unchanged':True,'new_tree_fits':0,'new_coefficient_fits':16})
    print(json.dumps({'output':str(output),'selection':selection,'status_groups':{k:status[k] for k in ['required_groups','observed_groups','passed_groups','missing_groups','point_target_met_groups']},
                      'seconds':time.monotonic()-start},default=str),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=4): run(args.output.resolve())
