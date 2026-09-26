"""Fixed, cache-only exploratory ground-reference models; no release promotion."""
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

ROOT = Path('/opt/lst-pilot')
BASE = ROOT/'runs/global_release_20260918/accuracy'
HERE = ROOT/'reports/global_release_20260918/accuracy'
GROUND = BASE/'surfrad_diagnostic_v2/hourly_diagnostics.parquet'
GROUND_SHA = '5a467ea63d54fb7be51cdc942678808b7f5599a4dc647e29359ff9f5049d5a6e'
AIR = 'era5_land_air_temperature_c'
SKIN = 'era5_land_skin_temperature_c'
FEATURES = [
    AIR, SKIN, 'era5_land_soil_temperature_0_7cm_c',
    'era5_land_soil_moisture_0_7cm_m3_m3', 'era5_land_snow_water_equivalent_m',
    'dewpoint_c', 'wind_speed_m_s', 'wind_direction_sin', 'wind_direction_cos',
    'cloud_cover_fraction', 'shortwave_down_w_m2', 'era5_longwave_down_w_m2',
    'precipitation_mm_h', 'rain_mm_24h', 'rain_mm_72h',
    'air_temperature_lag1_c', 'air_temperature_lag24_c', 'shortwave_down_mean3_w_m2',
    'elevation', 'slope', 'worldcover_tree_class_fraction',
    'worldcover_grass_class_fraction', 'worldcover_crop_class_fraction',
    'worldcover_built_class_fraction', 'worldcover_bare_class_fraction',
    'worldcover_water_fraction', 'solar_elevation_deg', 'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos',
]
PARAMS = dict(loss='squared_error', learning_rate=.05, max_iter=100,
              max_leaf_nodes=7, max_depth=None, min_samples_leaf=20,
              l2_regularization=10., max_bins=255, early_stopping=False,
              random_state=20260919)
ARMS = {'air_residual': AIR, 'skin_residual': SKIN}
PHASES = ['day', 'night', 'twilight']
EPS = ['e0p95', 'e0p97', 'e0p99']


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def binding(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def weights(frame):
    require(len(frame) > 0, 'Cannot weight empty population')
    a = frame.groupby('station_id').utc_date.transform('nunique')
    b = frame.groupby(['station_id', 'utc_date']).phase.transform('nunique')
    c = frame.groupby(['station_id', 'utc_date', 'phase']).sample_id.transform('size')
    w = 1. / frame.station_id.nunique() / a.to_numpy() / b.to_numpy() / c.to_numpy()
    require(np.isfinite(w).all() and (w > 0).all() and abs(w.sum()-1.) < 1e-12,
            'Invalid site/date/phase weights')
    return w


def load(features):
    require(sha(GROUND) == GROUND_SHA, 'Ground input changed')
    keys = ['sample_id', 'station_id', 'datetime_utc', 'latitude', 'longitude', 'phase']
    f = pd.read_parquet(features, columns=list(dict.fromkeys(keys+FEATURES)))
    require(len(f) == 672 and f.sample_id.is_unique, 'Require all 672 unique requested feature rows')
    require(isinstance(f.datetime_utc.dtype, pd.DatetimeTZDtype), 'Explicit UTC feature timestamps required')
    f['datetime_utc'] = pd.to_datetime(f.datetime_utc, utc=True)
    require(f.datetime_utc.eq(f.datetime_utc.dt.floor('h')).all(), 'Targets must be exact hours')
    f['utc_date'] = f.datetime_utc.dt.strftime('%Y-%m-%d')
    dates = ['2021-01-15', '2021-04-15', '2021-07-15', '2021-10-15']
    require(sorted(f.utc_date.unique()) == dates, 'Original four dates changed')
    require(f.station_id.nunique() == 7 and f.groupby(['station_id', 'utc_date']).size().eq(24).all(),
            'Original seven complete station calendars changed')
    require(f.sample_id.tolist() == [s+':'+t.isoformat() for s,t in zip(f.station_id,f.datetime_utc)],
            'Unexpected source identity format')
    g = pd.read_parquet(GROUND, columns=['station_id','timestamp_utc',
        'longwave_up_w_m2_qc','longwave_down_w_m2_qc']+[f'radiometric_{e}_c' for e in EPS])
    g = g.rename(columns={'timestamp_utc':'datetime_utc'})
    require(len(g) == 672 and not g.duplicated(['station_id','datetime_utc']).any(), 'Ground identity drift')
    d = f.merge(g, on=['station_id','datetime_utc'], how='left', validate='one_to_one', indicator=True)
    require(d['_merge'].eq('both').all() and d.sample_id.tolist() == f.sample_id.tolist(), 'Ground join changed population')
    d = d.drop(columns='_merge')
    solar = pd.to_numeric(d.solar_elevation_deg, errors='raise').to_numpy(float)
    require(np.isfinite(solar).all(), 'Every planned target needs solar geometry')
    expected = np.where(solar >= 10, 'day', np.where(solar <= -6, 'night', 'twilight'))
    require(np.array_equal(d.phase, expected), 'Phase does not follow frozen solar thresholds')
    for col in FEATURES: d[col] = pd.to_numeric(d[col], errors='raise')
    d['feature_complete'] = np.isfinite(d[FEATURES].to_numpy(float)).all(axis=1)
    d['ground_complete'] = (d.longwave_up_w_m2_qc.eq(0) & d.longwave_down_w_m2_qc.eq(0)
                            & np.isfinite(d.radiometric_e0p97_c))
    d['fit_complete'] = d.feature_complete & d.ground_complete
    d['missing_features'] = [','.join(np.asarray(FEATURES)[~np.isfinite(x)]) for x in d[FEATURES].to_numpy(float)]
    return d


def splits(d):
    out = []
    for site in sorted(d.station_id.unique()):
        out.append(('site', site, d.station_id.ne(site).to_numpy(), d.station_id.eq(site).to_numpy()))
    for day in sorted(d.utc_date.unique()):
        out.append(('date', day, d.utc_date.ne(day).to_numpy(), d.utc_date.eq(day).to_numpy()))
    out.append(('full', 'all', np.ones(len(d),bool), np.zeros(len(d),bool)))
    require(len(out) == 12, 'Expected eleven held folds plus full')
    return out


def prepare(features, source_completion, out, source_proof=None):
    require(not out.exists(), 'Use a new immutable run')
    done = read(source_completion)
    require(done.get('status') in ['complete','complete_with_source_gaps'], 'Source preparation is unfinished')
    require(done.get('requested_rows')==done.get('emitted_rows')==672, 'Source preparation must retain all requested rows')
    require(features.resolve()==source_completion.parent.resolve()/'features.parquet', 'Unexpected feature artifact path')
    digest = sha(features)
    artifact=done.get('artifacts',{}).get('features.parquet')
    if isinstance(artifact,dict):
        require(Path(artifact.get('path','')).resolve()==features.resolve(), 'Feature binding path changed')
        artifact=artifact.get('sha256')
    require(artifact==digest, 'Feature table not explicitly bound by source completion')
    require(source_proof is not None, 'Completed recursive source proof required')
    proof=read(source_proof)
    require(proof.get('status')=='passed', 'Source proof did not pass')
    for key,expected in [('feature_binding',features),('preparation_completion',source_completion)]:
        item=proof[key]
        require(Path(item['path']).resolve()==expected.resolve() and item['sha256']==sha(expected),
                'Source proof is for a different delivery')
    require(isinstance(proof.get('bindings'),list) and len(proof['bindings'])>0, 'Empty recursive source proof')
    for item in proof['bindings']:
        require(sha(item['path'])==item['sha256'], 'Source proof artifact changed')
    d = load(features)
    recipes = []
    for mode,key,train_mask,held_mask in splits(d):
        train = d.loc[train_mask & d.fit_complete.to_numpy()].reset_index(drop=True)
        eligible = len(train)>=80 and train.station_id.nunique()>=2 and train.utc_date.nunique()>=2
        for arm,baseline in ARMS.items():
            rows = train[['sample_id','station_id','utc_date','phase']].copy()
            if eligible:
                rows['normalized_weight'] = weights(train)
                rows['baseline_c'] = train[baseline]
                rows['target_offset_c'] = train.radiometric_e0p97_c - train[baseline]
            recipes.append(dict(mode=mode, key=key, arm=arm, eligible=eligible,
                                rows=len(train), recipe_sha256=frame_sha(rows),
                                feature_sha256=frame_sha(train[FEATURES])))
    out.mkdir(parents=True)
    save(out/'plan.json', dict(status='frozen_before_fits', source=binding(__file__),
        protocol=binding(HERE/'GROUND_NATIVE_BASELINE_PROTOCOL.md'),
        features=binding(features), preparation_completion=binding(source_completion),
        source_proof=binding(source_proof), recursive_source_bindings=proof['bindings'],
        ground=binding(GROUND), feature_order=FEATURES, parameters=PARAMS, recipes=recipes,
        requested_rows=672, complete_rows=int(d.fit_complete.sum()), maximum_fits=24,
        frame_sha256=frame_sha(d), scopes=['whole_site','whole_global_date'],
        model_selection=False, production_changed=False, global_target_met=False,
        network_allowed=False, limits=dict(cpu_threads=2,memory_bytes=2*2**30,seconds=300)))
    print(json.dumps(dict(plan=str(out/'plan.json'),sha256=sha(out/'plan.json'))),flush=True)


def metrics(frame):
    rows=[]
    groups=[('overall','all',frame)]
    for site in sorted(frame.station_id.unique()):
        for phase in PHASES:
            groups.append(('site_phase',site+'|'+phase,frame.loc[frame.station_id.eq(site)&frame.phase.eq(phase)]))
    for day in sorted(frame.utc_date.unique()):
        groups.append(('date',day,frame.loc[frame.utc_date.eq(day)]))
        for site in sorted(frame.station_id.unique()):
            for phase in PHASES:
                groups.append(('site_date_phase',site+'|'+day+'|'+phase,
                    frame.loc[frame.station_id.eq(site)&frame.utc_date.eq(day)&frame.phase.eq(phase)]))
    for phase in PHASES:
        groups.append(('phase',phase,frame.loc[frame.phase.eq(phase)]))
    names = list(ARMS)+['raw_air','raw_skin']
    for kind,key,g in groups:
        common=g.fit_complete & np.isfinite(g[names]).all(axis=1)
        for view in ['all_four_matched','own_available']:
            for name in names:
                mask=common if view=='all_four_matched' else g.ground_complete & np.isfinite(g[name])
                a=g.loc[mask]
                row=dict(mode=frame.test_mode.iloc[0],segment_type=kind,segment=key,view=view,model=name,
                    requested=len(g),predicted=int(np.isfinite(g[name]).sum()),paired=len(a),
                    missing=int(len(g)-len(a)),sites=a.station_id.nunique(),dates=a.utc_date.nunique())
                for e in EPS:
                    finite=a.loc[np.isfinite(a[f'radiometric_{e}_c'])]
                    w=weights(finite) if len(finite) else np.array([])
                    delta=(finite[name]-finite[f'radiometric_{e}_c']).to_numpy(float);err=abs(delta)
                    row.update({f'{e}_paired':len(finite),f'{e}_missing':len(g)-len(finite),
                        f'{e}_mae':float(err.mean()) if len(finite) else None,
                        f'{e}_bias':float(delta.mean()) if len(finite) else None,
                        f'{e}_balanced_bias':float(w@delta) if len(finite) else None,
                        f'{e}_balanced_mae':float(w@err) if len(finite) else None})
                    for threshold in [3,5,7]:
                        row[f'{e}_above_{threshold}_fraction']=float(np.mean(err>threshold)) if len(finite) else None
                        row[f'{e}_balanced_above_{threshold}_fraction']=float(w@(err>threshold)) if len(finite) else None
                rows.append(row)
    return rows


def run(out):
    start=time.monotonic();cpu=time.process_time();p=read(out/'plan.json')
    require(not (out/'models').exists(), 'Cannot rerun frozen trial')
    for key in ['source','protocol','features','preparation_completion','source_proof','ground']:
        require(sha(p[key]['path'])==p[key]['sha256'], 'Frozen binding changed: '+key)
    for item in p['recursive_source_bindings']:
        require(sha(item['path'])==item['sha256'], 'Frozen raw-source binding changed')
    require(p['feature_order']==FEATURES and p['parameters']==PARAMS, 'Recipe changed')
    import requests
    def blocked(*a,**k):raise RuntimeError('Network prohibited during trial')
    requests.sessions.Session.request=blocked
    d=load(p['features']['path']);require(frame_sha(d)==p['frame_sha256'], 'Prepared values changed')
    frames={mode:d.copy().assign(test_mode=mode,air_residual=np.nan,skin_residual=np.nan,
                 raw_air=d[AIR],raw_skin=d[SKIN],fit_id='') for mode in ['site','date']}
    recipes={(r['mode'],r['key'],r['arm']):r for r in p['recipes']};fits=[]
    for mode,key,tr,he in splits(d):
        train=d.loc[tr & d.fit_complete.to_numpy()].reset_index(drop=True)
        held=d.loc[he & d.feature_complete.to_numpy()]
        if mode=='site':require(set(train.station_id).isdisjoint(d.loc[he,'station_id']), 'Site leakage')
        if mode=='date':require(set(train.utc_date).isdisjoint(d.loc[he,'utc_date']), 'Date leakage')
        for arm,baseline in ARMS.items():
            expected=recipes[mode,key,arm]
            if not expected['eligible']:
                fits.append(dict(**expected,status='insufficient_training_support'));continue
            membership=train[['sample_id','station_id','utc_date','phase']].copy()
            membership['normalized_weight']=weights(train)
            membership['baseline_c']=train[baseline]
            membership['target_offset_c']=train.radiometric_e0p97_c-train[baseline]
            require(frame_sha(membership)==expected['recipe_sha256'] and
                    frame_sha(train[FEATURES])==expected['feature_sha256'], 'Fit recipe drift')
            path=out/'models'/mode/key/arm;path.mkdir(parents=True)
            membership.to_parquet(path/'fitting_rows.parquet',index=False)
            m=HistGradientBoostingRegressor(**PARAMS)
            m.fit(train[FEATURES],membership.target_offset_c,
                  sample_weight=membership.normalized_weight.to_numpy()*len(train))
            joblib.dump(m,path/'model.joblib',compress=3)
            if mode!='full' and len(held):
                values=held[baseline].to_numpy()+m.predict(held[FEATURES])
                frames[mode].loc[held.index,arm]=values
                frames[mode].loc[held.index,'fit_id']=mode+'/'+key
            fits.append(dict(**expected,status='fitted',model=binding(path/'model.joblib')))
            print(json.dumps(dict(mode=mode,key=key,arm=arm,rows=len(train))),flush=True)
    rows=[]
    for mode,f in frames.items():
        f.to_parquet(out/(mode+'_predictions.parquet'),index=False);rows+=metrics(f)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    save(out/'fits.json',fits)
    save(out/'coverage.json',dict(requested=672,feature_complete=int(d.feature_complete.sum()),
        ground_complete=int(d.ground_complete.sum()),common_complete=int(d.fit_complete.sum()),
        by_site_phase=d.groupby(['station_id','phase']).fit_complete.agg(['size','sum']).reset_index().to_dict('records'),
        omitted_rows=0,solar_phases=PHASES,accuracy_qualified=False))
    save(out/'completion.json',dict(status='complete_exploratory',plan_sha256=sha(out/'plan.json'),
        wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        models_fitted=sum(x['status']=='fitted' for x in fits),network_requests=0,
        production_changed=False,global_target_met=False,automatic_candidate_selected=None,
        artifacts={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}))
    print(json.dumps(dict(completion=str(out/'completion.json'),sha256=sha(out/'completion.json'))),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--features',type=Path)
    a.add_argument('--source-completion',type=Path);a.add_argument('--source-proof',type=Path)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    with threadpool_limits(limits=2):
        if args.action=='prepare':prepare(args.features,args.source_completion,args.output,args.source_proof)
        else:run(args.output)
