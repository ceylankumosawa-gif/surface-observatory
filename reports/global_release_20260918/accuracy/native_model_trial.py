"""Bounded exploratory native-footprint trial, after a checked cohort is frozen.

The separate cohort assembler must retain every collected native identity and
the complete requested-group registry, including zero-row/missing groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ground_native_trial import FEATURES, PARAMS, AIR, SKIN, ARMS
from native_trial_design import balanced_weights, split_definitions, design_checks

IDENTITY = ['native_cell_id', 'region_id', 'phase', 'utc_date', 'acquisition_id',
            'granule_id', 'granule_start_utc', 'granule_end_utc', 'native_footprint_area_m2']
MASKS = ['native_label_admitted', 'native_fit_admitted', 'feature_complete']
CONTROLS = ['raw_air', 'raw_skin']
MODELS = list(ARMS) + CONTROLS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def load(cohort, registry):
    d = pd.read_parquet(cohort, columns=IDENTITY+MASKS+['lst_c']+FEATURES).reset_index(drop=True)
    if d.native_cell_id.duplicated().any():
        raise ValueError('Deduplicate native source/cell identity before trial')
    if d[IDENTITY].isna().any().any():
        raise ValueError('Incomplete native identity')
    for flag in MASKS:
        if d[flag].dtype != bool or d[flag].isna().any():
            raise ValueError('Admission flags require explicit boolean values: '+flag)
    if (d.native_fit_admitted & ~d.native_label_admitted).any():
        raise ValueError('Fitting admitted a rejected native label')
    if d.loc[d.region_id.eq('cabauw'), 'native_fit_admitted'].any():
        raise ValueError('Reference-only Cabauw admitted to fitting')
    if not np.isfinite(d.loc[d.native_label_admitted,'lst_c']).all():
        raise ValueError('Admitted label is nonfinite')
    if not np.isfinite(d.loc[d.feature_complete, FEATURES].to_numpy(float)).all():
        raise ValueError('Complete feature row contains a nonfinite value')
    if not (np.isfinite(d.native_footprint_area_m2) & d.native_footprint_area_m2.gt(0)).all():
        raise ValueError('Invalid footprint areas')
    g = pd.read_csv(registry, dtype=str, keep_default_na=False)
    if len(g) != 384 or g.duplicated(['region_id','date','phase']).any():
        raise ValueError('Full 384-group registry required')
    if not set(d.region_id).issubset(g.region_id) or not set(d.phase).issubset({'day','night'}):
        raise ValueError('Cohort outside requested pilot/phase registry')
    start = pd.to_datetime(d.granule_start_utc, utc=True)
    if not start.dt.strftime('%Y-%m-%d').eq(d.utc_date).all():
        raise ValueError('UTC acquisition-date definition changed')
    requested = set(zip(g.region_id,g.date,g.phase))
    actual = set(zip(d.region_id,d.utc_date,d.phase))
    if not actual.issubset(requested):
        raise ValueError('Native rows outside fixed source-start dates')
    d['fit_complete'] = d.native_fit_admitted & d.feature_complete
    return d, g


def recipe(d, train_mask, baseline):
    train = d.loc[train_mask & d.fit_complete.to_numpy()]
    identifiers = train[IDENTITY].copy()
    identifiers['weight'] = balanced_weights(train)
    identifiers['baseline_c'] = train[baseline]
    identifiers['target_residual_c'] = train.lst_c - train[baseline]
    eligible = len(train) >= 80 and train.region_id.nunique() >= 2 and train.utc_date.nunique() >= 2
    return train, identifiers, eligible


def prepare(cohort, registry, cohort_receipt, out):
    started=time.monotonic()
    if out.exists():
        raise ValueError('Trial output must be new')
    receipt = read(cohort_receipt)
    if receipt.get('status') != 'ready_for_exploratory_native_trial':
        raise ValueError('Checked cohort receipt required')
    for name, expected in [('cohort',cohort),('registry',registry)]:
        b = receipt[name]
        if Path(b['path']).resolve() != expected.resolve() or b['sha256'] != sha(expected):
            raise ValueError('Cohort receipt binding mismatch')
    design_checks()
    d, g = load(cohort, registry)
    fits = []
    for mode, key, training, held in split_definitions(d):
        for arm, baseline in ARMS.items():
            train, membership, eligible = recipe(d, training, baseline)
            fits.append(dict(mode=mode, key=key, arm=arm, eligible=eligible,
                rows=len(train), recipe_sha256=frame_sha(membership),
                feature_sha256=frame_sha(train[FEATURES]), held_rows=int(held.sum())))
    if len(fits) > 34:
        raise ValueError('Declared fit budget exceeded')
    here = Path(__file__).resolve().parent
    import ground_native_trial, native_trial_design
    out.mkdir(parents=True)
    save(out/'plan.json', dict(status='frozen_before_fits',
        bindings={name:bind(path) for name,path in [
            ('source',Path(__file__)), ('design',Path(native_trial_design.__file__)),
            ('common_model',Path(ground_native_trial.__file__)),
            ('protocol',here/'NATIVE_VIIRS_TRIAL_PROTOCOL.md'), ('cohort',cohort),
            ('registry',registry), ('cohort_receipt',cohort_receipt)]},
        features=FEATURES, parameters=PARAMS, fits=fits, row_count=len(d),
        cohort_frame_sha256=frame_sha(d), requested_groups=len(g),
        preparation_wall_seconds=time.monotonic()-started,
        preparation_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        network_allowed=False, production_changed=False, automatic_promotion=False,
        limits={'maximum_fits':34,'cpu_threads':2,'memory_bytes':4*2**30,'seconds':900}))
    print(json.dumps(bind(out/'plan.json')), flush=True)


def score(predictions, registry):
    """Score shared support; retain zero-observation requested groups."""
    p = predictions
    def segments():
        # Keep only one row-sized mask alive under the research memory budget.
        yield 'overall','all',np.ones(len(p),bool)
        for region,phase in sorted(set(zip(registry.region_id,registry.phase))):
            yield 'pilot_phase',region+'|'+phase,(p.region_id.eq(region)&p.phase.eq(phase)).to_numpy()
        for region,date,phase in registry[['region_id','date','phase']].itertuples(index=False,name=None):
            yield 'pilot_date_phase',region+'|'+date+'|'+phase,(p.region_id.eq(region)&p.utc_date.eq(date)&p.phase.eq(phase)).to_numpy()
        for month in ['2021-01','2021-04','2021-07','2021-10']:
            yield 'month',month,p.utc_date.str.startswith(month).to_numpy()
    rows=[]
    for kind,key,mask in segments():
        part=p.loc[mask]
        common=part.native_label_admitted & np.isfinite(part[MODELS]).all(axis=1)
        for view in ['all_four_matched','own_available']:
            for name in MODELS:
                admitted=common if view=='all_four_matched' else part.native_label_admitted & np.isfinite(part[name])
                values=part.loc[admitted]
                weights=balanced_weights(values)
                delta=(values[name]-values.lst_c).to_numpy(float)
                absolute=abs(delta)
                row=dict(segment_type=kind,segment=key,view=view,model=name,
                    collected_rows=len(part),qa_admitted=int(part.native_label_admitted.sum()),
                    feature_complete=int(part.feature_complete.sum()),predicted=int(np.isfinite(part[name]).sum()),
                    paired=len(values),unscored=int(len(part)-len(values)),
                    sites=values.region_id.nunique(),dates=values.utc_date.nunique(),
                    acquisitions=values.acquisition_id.nunique(),
                    mae=float(absolute.mean()) if len(values) else None,
                    balanced_mae=float(weights@absolute) if len(values) else None,
                    bias=float(delta.mean()) if len(values) else None,
                    balanced_bias=float(weights@delta) if len(values) else None)
                for threshold in [3,5,7]:
                    row[f'above_{threshold}_fraction']=float(np.mean(absolute>threshold)) if len(values) else None
                    row[f'balanced_above_{threshold}_fraction']=float(weights@(absolute>threshold)) if len(values) else None
                rows.append(row)
    return rows


def run(out):
    started=time.monotonic(); cpu_started=time.process_time(); plan=read(out/'plan.json')
    if (out/'models').exists():
        raise ValueError('Frozen trial cannot be restarted')
    for name,b in plan['bindings'].items():
        if sha(b['path']) != b['sha256']:
            raise ValueError('Changed frozen artifact: '+name)
    if plan['features'] != FEATURES or plan['parameters'] != PARAMS:
        raise ValueError('Changed model recipe')
    d,g=load(plan['bindings']['cohort']['path'],plan['bindings']['registry']['path'])
    if frame_sha(d) != plan['cohort_frame_sha256']:
        raise ValueError('Prepared cohort changed')
    # The OS service must also deny network access.
    import requests
    def denied(*args,**kwargs):
        raise RuntimeError('Network prohibited in model trial')
    requests.sessions.Session.request=denied
    base=d[IDENTITY+MASKS+['lst_c']].copy()
    base['raw_air']=d[AIR];base['raw_skin']=d[SKIN]
    # Baselines remain on their own finite available support; comparisons use
    # explicit paired masks rather than removing missing feature rows.
    ordinary=base.loc[base.region_id.ne('cabauw')]
    predictions={mode:ordinary.copy().assign(air_residual=np.nan,skin_residual=np.nan,fit_id='')
                 for mode in ['pilot','month']}
    predictions['reference']=base.loc[base.region_id.eq('cabauw')].copy().assign(
        air_residual=np.nan,skin_residual=np.nan,fit_id='')
    recipes={(x['mode'],x['key'],x['arm']):x for x in plan['fits']};completed=[]
    for mode,key,training,held_mask in split_definitions(d):
        held=d.loc[held_mask & d.feature_complete.to_numpy()]
        for arm,baseline in ARMS.items():
            expected=recipes[mode,key,arm]
            train,membership,eligible=recipe(d,training,baseline)
            if eligible != expected['eligible'] or frame_sha(membership)!=expected['recipe_sha256'] or frame_sha(train[FEATURES])!=expected['feature_sha256']:
                raise ValueError('Fit recipe changed')
            if not eligible:
                completed.append(dict(**expected,status='insufficient_fitting_support'))
                save(out/'fits.json',completed)
                continue
            target=out/'models'/mode/key/arm;target.mkdir(parents=True)
            membership.to_parquet(target/'fitting_rows.parquet',index=False)
            model=HistGradientBoostingRegressor(**PARAMS)
            model.fit(train[FEATURES],membership.target_residual_c,
                      sample_weight=membership.weight.to_numpy()*len(train))
            joblib.dump(model,target/'model.joblib',compress=3)
            if mode!='full' and len(held):
                predictions[mode].loc[held.index,arm]=held[baseline].to_numpy()+model.predict(held[FEATURES])
                predictions[mode].loc[held.index,'fit_id']=mode+'/'+key
            completed.append(dict(**expected,status='fitted',model=bind(target/'model.joblib')))
            save(out/'fits.json',completed)
            print(json.dumps({'mode':mode,'key':key,'arm':arm,'rows':len(train)}),flush=True)
    # Seal every prediction ledger before metric aggregation. If scoring hits
    # the external deadline, completed fits and predictions remain inspectable;
    # that is not permission to restart fits or silently extend the budget.
    for mode,frame in predictions.items():
        frame.to_parquet(out/(mode+'_predictions.parquet'),index=False)
    metrics=[]
    for mode,frame in predictions.items():
        registry=g.loc[g.region_id.eq('cabauw')] if mode=='reference' else g
        metrics.extend(dict(test_mode=mode,**row) for row in score(frame,registry))
        print(json.dumps({'scored_mode':mode,'wall_seconds':time.monotonic()-started}),flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    save(out/'fits.json',completed)
    save(out/'completion.json',dict(status='complete_exploratory_comparison',plan=bind(out/'plan.json'),
        fits=sum(x['status']=='fitted' for x in completed),rows=len(d),requested_groups=len(g),
        wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        production_changed=False,globally_qualified=False,
        artifacts={p.name:bind(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(bind(out/'completion.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--cohort',type=Path);parser.add_argument('--registry',type=Path)
    parser.add_argument('--cohort-receipt',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.mode=='prepare':prepare(args.cohort,args.registry,args.cohort_receipt,args.output)
        else:run(args.output)
