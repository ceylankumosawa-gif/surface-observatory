"""Frozen five-arm alignment experiment, with separate fitting/evaluation commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

import models as m
from lst_pilot import option_b_train as old
from lst_pilot import multisensor_train as multi

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'weather_baseline_20260911'
PRIOR_RUN = Path('/opt/lst-pilot/runs/weather_baseline_20260911_v1/secondary_complete_native_model_v1')
PRIOR_COMPLETION_SHA = 'e031d3dbddc11c26a159dd59229d79fccf310b9937025f7f84fa9e374823bdad'
ORIGINAL = Path('/opt/lst-pilot/runs/shared_model_remedies_20260910_v1/fit_frame.parquet')
spec = importlib.util.spec_from_file_location('previous_frozen_land_comparison', PREVIOUS / 'run_secondary_complete_native.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
PHYSICAL = dict(zip(m.LAND_QUANTITIES, previous.LAND_VALUES))
META = list(dict.fromkeys([*previous.META, *previous.IDENTITY]))
FAMILIES = previous.FAMILIES


def read(path): return json.loads(Path(path).read_text())
def log(**items): print(json.dumps(old.json_ready(items)), flush=True)
def binding(path): return {'path': str(Path(path).resolve()), 'sha256': old.sha(path)}


def verify_bindings(value):
    """Recursively verify explicit path/hash pairs; ignore descriptive strings."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            m.require(old.sha(value['path']) == value['sha256'], 'Evidence binding changed: '+value['path'])
        for child in value.values(): verify_bindings(child)
    elif isinstance(value, list):
        for child in value: verify_bindings(child)


def verify_parent():
    m.require(old.sha(PRIOR_RUN/'completion.json') == PRIOR_COMPLETION_SHA, 'Previous completion changed')
    manifest = previous.verify_new_freeze(PRIOR_RUN)
    references = manifest['bindings']
    m.require(previous.reference_bindings(Path(references['reference']), Path(references['blends_reference'])) == references,
              'Transitive reference bindings changed')
    complete = read(PRIOR_RUN/'completion.json')
    for name in ('manifest.json', 'fit_frame.parquet', 'oof_predictions.parquet'):
        m.require(old.sha(PRIOR_RUN/name) == complete['artifacts'][name], 'Previous research artifact changed')
    original_manifest = read(ORIGINAL.parent/'manifest.json')
    m.require(old.sha(ORIGINAL) == read(ORIGINAL.parent/'completion.json')['artifacts']['fit_frame.parquet'], 'Original input changed')
    return manifest, original_manifest


def source_paths():
    return [Path(__file__), Path(m.__file__), Path(previous.__file__), Path(old.__file__),
            Path(multi.__file__), Path(previous.physics.__file__),
            Path(old.__file__).with_name('model.py'), HERE/'PROTOCOL.md',
            previous.PRIOR_CODE/'run_comparison.py', previous.PRIOR_CODE/'candidates.py',
            HERE/'aligned_land_v1.py', HERE/'cds_swe_batched_v1.py', HERE/'cds_swe_batched_v2.py']


ENDPOINT_KEYS = ['region_id','era5_land_valid_time_utc','era5_land_grid_latitude','era5_land_grid_longitude']


def check_delivery_metadata(frame, metadata, endpoint_metadata, period):
    """Reject out-of-scope identities/hours before any weather values are decoded."""
    m.require(period in ('fit','evaluation'), 'Unknown study period')
    m.require(len(metadata)==len(frame) and metadata.sample_id.notna().all()
              and not metadata.sample_id.duplicated().any() and set(metadata.sample_id)==set(frame.sample_id), 'Alignment row identities differ')
    metadata = metadata.set_index('sample_id').loc[frame.sample_id].reset_index()
    for field in previous.IDENTITY:
        pd.testing.assert_series_equal(metadata[field], frame[field].reset_index(drop=True), check_names=False,
            check_dtype=False, check_exact=True, check_categorical=False)
    stamp = pd.to_datetime(metadata.datetime_utc, utc=True)
    allowed = [2021,2022] if period=='fit' else [2021,2022,2023,2024]
    m.require(stamp.notna().all() and stamp.dt.year.isin(allowed).all(), 'Target dates outside study scope')
    for side, expected in [('floor',stamp.dt.floor('h')),('ceil',stamp.dt.ceil('h'))]:
        pd.testing.assert_series_equal(pd.to_datetime(metadata[side+'_utc'],utc=True), expected, check_names=False, check_exact=True)
    m.require(endpoint_metadata[ENDPOINT_KEYS].notna().all().all() and not endpoint_metadata.duplicated(ENDPOINT_KEYS).any(),
              'Missing or duplicated canonical endpoint identity')
    times = endpoint_metadata.era5_land_valid_time_utc
    m.require(isinstance(times.dtype,pd.DatetimeTZDtype) and times.eq(times.dt.floor('h')).all(), 'Canonical endpoint is not a UTC native hour')
    requested = pd.concat([metadata[['region_id',side+'_utc',*ENDPOINT_KEYS[2:]]].rename(columns={side+'_utc':ENDPOINT_KEYS[1]})
                           for side in ('floor','ceil')],ignore_index=True).drop_duplicates()
    joined = requested.merge(endpoint_metadata, on=ENDPOINT_KEYS, how='outer',validate='one_to_one',indicator=True)
    m.require(joined['_merge'].eq('both').all(), 'Canonical endpoint keys differ from exact target brackets')
    return metadata


def check_canonical_endpoints(values, endpoints):
    """Tie every duplicated target endpoint to the canonical cell/hour table."""
    for side in ('floor','ceil'):
        keys = values[['region_id',side+'_utc',*ENDPOINT_KEYS[2:]]].rename(columns={side+'_utc':ENDPOINT_KEYS[1]})
        canonical = keys.merge(endpoints,on=ENDPOINT_KEYS,how='left',validate='many_to_one',sort=False)
        for field in PHYSICAL.values():
            for suffix in ('','_raw'):
                np.testing.assert_allclose(values[side+'__'+field+suffix],canonical[field+suffix],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(old.strict_bool(values[side+'__'+field+'_zero_normalized'],'target repair flag'),
                                          old.strict_bool(canonical[field+'_zero_normalized'],'canonical repair flag'))
        np.testing.assert_array_equal(old.strict_bool(values[side+'__endpoint_complete'],'target endpoint availability'),
                                      old.strict_bool(canonical.endpoint_complete,'canonical endpoint availability'))


def attach(frame, delivery, period):
    """Independently reconstruct the common mask/interpolation from saved endpoints."""
    delivery = Path(delivery).resolve()
    done, plan = read(delivery/'completion.json'), read(delivery/'plan.json')
    m.require(done['status'] == 'complete' and done['format'] == 'bracketed_instantaneous_land_v1', 'Unsupported aligned delivery')
    for name, field in [('plan.json','plan_sha256'), ('values.parquet','output_sha256'),
                        ('native_endpoints.parquet','native_endpoints_sha256')]:
        m.require(old.sha(delivery/name) == done[field], 'Aligned artifact changed: '+name)
    m.require(done['source_sha256'] == old.sha(HERE/'aligned_land_v1.py'), 'Alignment source changed')
    m.require(done['protocol']['sha256'] == old.sha(HERE/'PROTOCOL.md'), 'Alignment protocol differs')
    verify_bindings(plan); verify_bindings(done)
    metadata = pd.read_parquet(delivery/'values.parquet',columns=[*previous.IDENTITY,'floor_utc','ceil_utc',*ENDPOINT_KEYS[2:]])
    endpoint_metadata = pd.read_parquet(delivery/'native_endpoints.parquet',columns=ENDPOINT_KEYS)
    check_delivery_metadata(frame,metadata,endpoint_metadata,period)
    values = pd.read_parquet(delivery/'values.parquet')
    m.require(len(values) == len(frame) and values.sample_id.notna().all()
              and not values.sample_id.duplicated().any() and set(values.sample_id) == set(frame.sample_id), 'Alignment row identities differ')
    values = values.set_index('sample_id').loc[frame.sample_id].reset_index()
    endpoints = pd.read_parquet(delivery/'native_endpoints.parquet')
    check_canonical_endpoints(values,endpoints)
    for field in previous.IDENTITY:
        pd.testing.assert_series_equal(values[field], frame[field].reset_index(drop=True), check_names=False,
            check_dtype=False, check_exact=True, check_categorical=False)
    stamp = pd.to_datetime(values.datetime_utc, utc=True)
    allowed = [2021,2022] if period == 'fit' else [2021,2022,2023,2024]
    m.require(stamp.dt.year.isin(allowed).all(), 'Target dates outside study scope')
    floor, ceiling = stamp.dt.floor('h'), stamp.dt.ceil('h')
    pd.testing.assert_series_equal(values.floor_utc, floor, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(values.ceil_utc, ceiling, check_names=False, check_exact=True)
    alpha = (stamp-floor).dt.total_seconds().to_numpy()/3600
    np.testing.assert_array_equal(values.interpolation_alpha.to_numpy(float), alpha)
    np.testing.assert_array_equal(old.strict_bool(values.later_valid_time_used, 'later_valid_time_used'), alpha > 0)
    latitude_delta = values.era5_land_grid_latitude.to_numpy(float)-values.latitude.to_numpy(float)
    longitude_delta = (values.era5_land_grid_longitude.to_numpy(float)-values.longitude.to_numpy(float)+180)%360-180
    m.require(np.isfinite(latitude_delta).all() and np.isfinite(longitude_delta).all()
              and (np.abs(latitude_delta)<=.050001).all() and (np.abs(longitude_delta)<=.050001).all(), 'Wrong native cell')
    complete = np.ones(len(values), bool)
    reconstructed = {}
    for quantity, field in PHYSICAL.items():
        ends = []
        for side in ('floor', 'ceil'):
            raw = values[f'{side}__{field}_raw'].to_numpy(float)
            value = raw.copy(); repair = np.zeros(len(value), bool)
            if quantity in ('swe_m', 'moisture_m3_m3'):
                repair = (raw >= -1e-12) & (raw < 0); value[repair] = 0
            np.testing.assert_allclose(values[f'{side}__{field}'], value, rtol=0, atol=0, equal_nan=True)
            saved_flag = values[f'{side}__{field}_zero_normalized']
            # An unused exact-hour ceiling can be absent; only required flags enter the contract.
            needed = np.ones(len(values), bool) if side == 'floor' else alpha > 0
            np.testing.assert_array_equal(old.strict_bool(saved_flag.loc[needed], 'repair_flag'), repair[needed])
            low, high = (-173.15,126.85) if quantity.endswith('_c') else ((0,10) if quantity == 'swe_m' else (0,1))
            valid = np.isfinite(value) & (value>=low) & (value<=high)
            complete &= valid | ~needed
            ends.append(value)
        expected = ends[0].copy(); between = alpha > 0
        expected[between] += alpha[between]*(ends[1][between]-ends[0][between])
        reconstructed[quantity] = (ends[0], expected)
    flag = old.strict_bool(values.era5_land_complete, 'era5_land_complete').to_numpy()
    np.testing.assert_array_equal(flag, complete)
    data = frame.copy().reset_index(drop=True)
    for quantity, field in PHYSICAL.items():
        lo, expected = reconstructed[quantity]
        expected[~complete] = np.nan
        np.testing.assert_allclose(values[field], expected, rtol=0, atol=1e-12, equal_nan=True)
        data['floor_'+quantity] = lo
        data['aligned_'+quantity] = expected
    for field in ['floor_utc','ceil_utc','interpolation_alpha','later_valid_time_used',
                  'era5_land_grid_latitude','era5_land_grid_longitude','era5_land_complete','era5_land_status']:
        m.require(field not in data, 'Alignment field overwrites original input')
        data[field] = values[field].to_numpy()
    pd.testing.assert_frame_equal(data[list(frame)], frame.reset_index(drop=True), check_exact=True)
    audit = {'directory': str(delivery), 'period': period, 'completion': binding(delivery/'completion.json'),
             'plan': binding(delivery/'plan.json'), 'values': binding(delivery/'values.parquet'),
             'endpoints': binding(delivery/'native_endpoints.parquet'), 'source': binding(HERE/'aligned_land_v1.py'),
             'rows': len(data), 'available_rows': int(complete.sum()), 'target_air_A_unchanged': True,
             'independent_endpoint_reconstruction': True, 'retrospective': True}
    return data, audit


def freeze_mask(frame, audit, directory, family):
    directory.mkdir(parents=True, exist_ok=True)
    keep = old.strict_bool(frame.era5_land_complete, 'era5_land_complete').to_numpy()
    fields = [*previous.IDENTITY, 'phase','utc_day','climate_class','split','acquisition_id',
              'floor_utc','ceil_utc','interpolation_alpha','era5_land_grid_latitude','era5_land_grid_longitude']
    table = frame[fields].copy(); table['included'] = keep
    inventory = []
    dimensions = {'pilot_phase':['region_id','phase'], 'pilot_date':['region_id','phase','utc_day'],
        'acquisition':['region_id','acquisition_id'], 'climate':['climate_class'],
        'native_cell':['region_id','era5_land_grid_latitude','era5_land_grid_longitude']}
    for kind, keys in dimensions.items():
        for key, group in table.groupby(keys, observed=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            retained = group.loc[group.included]
            inventory.append({'dimension':kind, **dict(zip(keys,key)), 'original_rows':len(group),
                'retained_rows':len(retained), 'excluded_rows':len(group)-len(retained),
                'original_utc_dates':group.utc_day.nunique(), 'retained_utc_dates':retained.utc_day.nunique()})
    table.to_parquet(directory/(family+'_mask.parquet'), index=False)
    pd.DataFrame(inventory).to_csv(directory/(family+'_support.csv'), index=False)
    losses = [x for x in inventory if x['dimension']=='pilot_phase' and x['original_utc_dates']>=6 and x['retained_utc_dates']<6]
    proof = {'family':family, 'original_rows':len(frame), 'retained_rows':int(keep.sum()), 'excluded_rows':int((~keep).sum()),
        'retained_id_sha256':old.row_hash(frame.loc[keep]), 'lost_supported_groups':losses,
        'mask':binding(directory/(family+'_mask.parquet')), 'support':binding(directory/(family+'_support.csv')),
        'source':audit, 'mask_uses_labels_or_errors':False, 'frozen_before_new_scoring':True}
    m.save(directory/(family+'_mask_freeze.json'), proof)
    return keep, proof


def score(frame, values, cohort):
    prior = previous.previous_runner()
    saved = prior.LABELS
    try:
        prior.LABELS = m.LABELS
        return prior.score(frame, values, cohort)
    finally: prior.LABELS = saved


def select(metrics, coverage):
    saved_labels, saved_order = previous.LABELS, previous.TIE_ORDER
    try:
        previous.LABELS, previous.TIE_ORDER = m.LABELS, m.ELIGIBLE
        result = previous.select(metrics)
    finally: previous.LABELS, previous.TIE_ORDER = saved_labels, saved_order
    if coverage['lost_supported_groups']:
        for decision in result['decisions']:
            decision['eligible'] = False
            decision['failures'].append('Weather masking loses previously supported pilot/phase groups')
        result.update(selected_candidate=None, status='no_qualifying_remedy_support_loss')
    result.update(reference_model='F_matched_bracket_complete_refit',
        conditional_on_weather_availability=True, lost_supported_groups=coverage['lost_supported_groups'])
    return result


def export_predictions(frame, values, path, fold=None):
    fields = [*META, 'floor_utc','ceil_utc','interpolation_alpha','later_valid_time_used',
              'era5_land_grid_latitude','era5_land_grid_longitude',
              *[f'{timing}_{quantity}' for timing in m.TIMINGS for quantity in m.LAND_QUANTITIES]]
    output = frame[list(dict.fromkeys(fields))].copy()
    if fold is not None: output['fold'] = fold
    for name, value in values.items(): output[name+'_lst_c'] = value
    output.to_parquet(path, index=False)


def training_guard(frame):
    stamp = pd.to_datetime(frame.datetime_utc, utc=True)
    m.require(stamp.dt.year.isin([2021,2022]).all() and frame.split.eq('fit').all(), 'Training years/split changed')
    m.require(not frame.sample_id.duplicated().any() and not frame.region_id.eq('cabauw').any(), 'Reserved training identity')
    for field in ('spatial_holdout','in_holdout_buffer'):
        m.require(not old.strict_bool(frame[field],field).any(), 'Reserved spatial training rows')
    for name in ('F','land_features_floor','land_features_aligned'):
        m.require(old.complete_rows(frame, m.features(name)).all(), 'Incomplete model inputs')


def fit_stage(frame, path, name, calibration_f):
    training_guard(frame)
    estimators = {key:m.fit_hgb(frame,key,path,name) for key in ('F','land_features_floor','land_features_aligned')}
    corrections = {timing:m.fit_air_correction(frame,calibration_f,timing) for timing in m.TIMINGS}
    m.save(path/'air_corrections.json', corrections)
    frame[[*META, 'floor_air_c','aligned_air_c']].assign(
        calibration_held_F_c=calibration_f, calibration_weight=old.balanced_weights(frame)
    ).to_parquet(path/'air_calibration_rows.parquet', index=False)
    return estimators, corrections


def verify_run(run, full=True):
    manifest = read(run/'manifest.json')
    verify_bindings(manifest)
    verify_parent()
    for name, sha in read(run/'selection_freeze.json')['files'].items():
        m.require(old.sha(run/name)==sha, 'Frozen training artifact changed: '+name)
    if full:
        frozen = read(run/'full_fit_freeze.json')
        m.require(frozen['selection_freeze_sha256']==old.sha(run/'selection_freeze.json'), 'Selection binding changed')
        for name, sha in frozen['files'].items(): m.require(old.sha(run/name)==sha, 'Frozen full fit changed')
    return manifest


def fit(args):
    start = time.monotonic(); out = args.output.resolve()
    m.require(not out.exists(), 'Use a new immutable run directory')
    parent, _ = verify_parent()
    previous.check_years(ORIGINAL,(2021,2022))
    original = pd.read_parquet(ORIGINAL)
    m.require(len(original)==44316 and not original.sample_id.duplicated().any(), 'Original cohort changed')
    attached, audit = attach(original,args.land,'fit')
    out.mkdir(parents=True)
    included, coverage = freeze_mask(attached,audit,out/'coverage','fit')
    frame = m.prepare_features(attached.loc[included].reset_index(drop=True)); training_guard(frame)
    folds = multi.month_folds(frame)
    m.require(set(folds)=={0,1,2}, 'Missing outer folds')
    manifest = {'study':'F_preserving_alignment_stage_A','protocol':binding(HERE/'PROTOCOL.md'),
        'sources':[binding(p) for p in source_paths()], 'input':binding(ORIGINAL), 'land':audit,
        'parent_completion':binding(PRIOR_RUN/'completion.json'),
        'protected_baseline':binding(parent['protected_baseline_path']),
        'labels':m.LABELS,'eligible':list(m.ELIGIBLE),'gates':previous.GATES,
        'coefficient_regularization_c2':m.REGULARIZATION_C2,'coefficient_minimum_dates':m.MIN_DATES,
        'config':asdict(old.CONFIG),'features':{key:list(m.features(key)) for key in ('F','land_features_floor','land_features_aligned')},
        'rows':len(frame),'sample_id_sha256':old.row_hash(frame),'coverage':coverage,
        'original_air_sha256':m.digest(frame.air_temperature_c),
        'python':sys.version,'sklearn':sklearn.__version__,'research_only':True,'reserved_2025_opened':False}
    m.save(out/'manifest.json',manifest); frame.to_parquet(out/'fit_frame.parquet',index=False)
    values = {name:np.full(len(frame),np.nan) for name in m.LABELS}
    with threadpool_limits(limits=4):
        for number in range(3):
            outer_train = folds!=number; outer_held = ~outer_train
            train, held = frame.loc[outer_train].reset_index(drop=True), frame.loc[outer_held].reset_index(drop=True)
            train_folds = folds[outer_train]
            calibration = np.full(len(train),np.nan)
            path = out/f'fold_{number}'
            for inner_held in sorted(set(train_folds)):
                inner_train_mask = train_folds!=inner_held; inner_held_mask = ~inner_train_mask
                inner_train, inner_test = train.loc[inner_train_mask].reset_index(drop=True), train.loc[inner_held_mask].reset_index(drop=True)
                m.require(set(inner_train.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(inner_test.datetime_utc.dt.strftime('%Y-%m')), 'Inner month leakage')
                m.require(set(train.datetime_utc.dt.strftime('%Y-%m')).isdisjoint(held.datetime_utc.dt.strftime('%Y-%m')), 'Outer month leakage')
                inner_path = path/f'inner_held_{inner_held}'
                estimator = m.fit_hgb(inner_train,'F',inner_path,f'outer_{number}_inner_held_{inner_held}')
                prediction = m.predict_hgb(inner_test,estimator,'F')
                calibration[inner_held_mask] = prediction
                inner_test[META].assign(F_lst_c=prediction, original_fold=inner_held).to_parquet(inner_path/'held_predictions.parquet',index=False)
                del estimator
            m.require(np.isfinite(calibration).all(), 'Inner calibration has uncovered rows')
            estimators, corrections = fit_stage(train,path,f'fold_{number}',calibration)
            predictions = m.predict_all(held,estimators,corrections)
            for name,prediction in predictions.items(): values[name][outer_held]=prediction
            export_predictions(held,predictions,path/'held_predictions.parquet',fold=folds[outer_held])
            log(stage='outer_fold_complete',fold=number,rows=len(held),gammas={t:{p:x['gamma'] for p,x in corrections[t]['phases'].items()} for t in m.TIMINGS})
            del estimators
    m.require(all(np.isfinite(value).all() for value in values.values()), 'Missing OOF predictions')
    export_predictions(frame,values,out/'oof_predictions.parquet',fold=folds)
    # A matching cohort must reproduce both previous shared controls exactly.
    previous_frame = pd.read_parquet(PRIOR_RUN/'fit_frame.parquet')
    reproduction = {'same_previous_complete_native_cohort':set(frame.sample_id)==set(previous_frame.sample_id)}
    if reproduction['same_previous_complete_native_cohort']:
        saved = pd.read_parquet(PRIOR_RUN/'oof_predictions.parquet').set_index('sample_id').loc[frame.sample_id]
        for name,old_name in [('F','F'),('land_features_floor','land_features')]:
            delta = float(np.max(np.abs(values[name]-saved[old_name+'_lst_c'].to_numpy(float))))
            m.require(delta<=1e-9,'Previous matched control does not reproduce: '+name)
            reproduction[name+'_maximum_difference_c']=delta
    m.save(out/'control_reproduction.json',reproduction)
    metrics = pd.DataFrame(score(frame,values,'training_2021_22_oof'));metrics.to_csv(out/'oof_metrics.csv',index=False)
    selection = select(metrics,coverage);m.save(out/'selection.json',selection)
    m.save(out/'selection_freeze.json',{'files':{str(p.relative_to(out)):old.sha(p) for p in out.rglob('*') if p.is_file()},
        'evaluation_labels_decoded_before_selection':False})
    log(stage='selection_frozen',**selection)
    with threadpool_limits(limits=4): fit_stage(frame,out/'full','full',values['F'])
    m.save(out/'full_fit_freeze.json',{'files':{str(p.relative_to(out)):old.sha(p) for p in (out/'full').rglob('*') if p.is_file()},
        'selection_freeze_sha256':old.sha(out/'selection_freeze.json'),'frozen_before_evaluation':True})
    verify_run(out)
    m.save(out/'fitting_complete.json',{'status':'complete','seconds':time.monotonic()-start,'rows':len(frame),
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'selection_freeze':binding(out/'selection_freeze.json'),'full_fit_freeze':binding(out/'full_fit_freeze.json'),
        'evaluation_labels_decoded':False,'production_unchanged':True,'reserved_2025_opened':False})
    log(stage='fitting_complete',seconds=time.monotonic()-start)


def evaluate(args):
    start = time.monotonic(); run = args.run.resolve()
    m.require(not (run/'completion.json').exists(), 'Completed comparison is immutable')
    ready = read(run/'fitting_complete.json');m.require(ready['status']=='complete' and ready['evaluation_labels_decoded'] is False,'Fitting incomplete')
    manifest = verify_run(run); verify_bindings(ready)
    prior = previous.previous_runner()
    frames = previous.evaluation_frames(prior,pd.read_parquet(ORIGINAL))
    deliveries = dict(zip(FAMILIES,(args.land_old,args.land_new,args.land_legacy)))
    joined, audits, coverages = {},{},{}
    for family,frame in frames.items():
        attached,audits[family] = attach(frame,deliveries[family],'evaluation')
        keep,coverages[family] = freeze_mask(attached,audits[family],run/'coverage',family)
        joined[family] = m.prepare_features(attached.loc[keep].reset_index(drop=True))
    m.save(run/'evaluation_mask_freeze.json',{'files':{str(p.relative_to(run)):old.sha(p) for p in (run/'coverage').iterdir() if p.is_file()},
        'created_before_new_error_scoring':True})
    estimators = {name:joblib.load(run/'full'/(name+'.joblib')) for name in ('F','land_features_floor','land_features_aligned')}
    corrections = read(run/'full/air_corrections.json')
    metrics,acquisitions = [],[]
    for family,frame in joined.items():
        frame.to_parquet(run/(family+'_evaluation_frame.parquet'),index=False)
        with threadpool_limits(limits=4): predictions = m.predict_all(frame,estimators,corrections)
        export_predictions(frame,predictions,run/(family+'_predictions.parquet'))
        for split,ids in frame.groupby('split',observed=True).indices.items():
            cohort = 'legacy_2024' if family=='legacy_2024' else family+'_'+split
            group = frame.iloc[ids].reset_index(drop=True);px={name:value[ids] for name,value in predictions.items()}
            metrics.extend(score(group,px,cohort))
            focal=group.loc[group.region_id.isin(['greater_london','sioux_falls'])]
            for acquisition,positions in focal.groupby('acquisition_id',observed=True).groups.items():
                data=group.loc[positions].reset_index(drop=True);weights=old.balanced_weights(data)
                for name,value in px.items():
                    predicted=value[positions];error=np.abs(predicted-data.lst_c.to_numpy(float))
                    acquisitions.append({'cohort':cohort,'region_id':data.region_id.iloc[0],'phase':data.phase.iloc[0],
                        'acquisition_id':acquisition,'date':str(data.utc_day.iloc[0].date()),'timestamp_utc':data.datetime_utc.iloc[0].isoformat(),
                        'model':name,'model_label':m.LABELS[name],**old.metrics(data,predicted),
                        'raw_gt5_count':int((error>5).sum()),'raw_gt7_count':int((error>7).sum()),
                        'raw_gt5_fraction':float((error>5).mean()),'raw_gt7_fraction':float((error>7).mean()),
                        'observed_mean_c':float(weights@data.lst_c.to_numpy(float)), 'predicted_mean_c':float(weights@predicted),
                        'air_mean_c':float(weights@data.air_temperature_c.to_numpy(float)),
                        'floor_land_air_mean_c':float(weights@data.floor_air_c.to_numpy(float)),
                        'aligned_land_air_mean_c':float(weights@data.aligned_air_c.to_numpy(float)),
                        'interpolation_alpha_mean':float(weights@data.interpolation_alpha.to_numpy(float))})
        log(stage='evaluation_scored',family=family,rows=len(frame))
    pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False);pd.DataFrame(acquisitions).to_csv(run/'acquisition_metrics.csv',index=False)
    m.save(run/'evaluation_sources.json',audits);m.save(run/'evaluation_coverage.json',coverages)
    verify_run(run);verify_bindings(audits)
    m.save(run/'completion.json',{'status':'complete','study':manifest['study'],'fitting_seconds':ready['seconds'],
        'evaluation_seconds':time.monotonic()-start,'peak_rss_bytes':max(ready['peak_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        'reserved_2025_opened':False,'production_unchanged':True,'research_only':True,
        'protected_baseline_sha256':old.sha(manifest['protected_baseline']['path']),
        'artifacts':{str(p.relative_to(run)):old.sha(p) for p in run.rglob('*') if p.is_file()}})
    log(stage='complete',seconds=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('--land',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--run',type=Path,required=True)
    for family in ('old','new','legacy'):p.add_argument('--land-'+family,type=Path,required=True)
    args=parser.parse_args();fit(args) if args.command=='fit' else evaluate(args)


if __name__=='__main__': main()
