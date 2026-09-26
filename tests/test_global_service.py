from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from pyproj import Transformer
import pytest
import rasterio
from rasterio.transform import from_origin

from lst_global.service_contract import GlobalInput, validate
from lst_global.api import GlobalManager, create_app
from lst_global.render import assemble
from lst_pilot.api import Settings


def payload(**extra):
    return GlobalInput.model_validate({
        'polygon': {'type':'Polygon','coordinates':[[[-1.26,51.75],[-1.24,51.75],[-1.24,51.77],[-1.26,51.77],[-1.26,51.75]]]},
        'datetime_utc':'2023-06-21T12:00:00Z','resolution_m':100,**extra})


def test_global_request_is_not_restricted_to_pilots():
    request = validate(payload())
    assert request['source_tiles'] and request['epsg']==32630
    assert request['shape'][0]*request['shape'][1] <= 800000
    assert all(s['tile_id'].startswith('g100-utm-30n') for s in request['source_tiles'])


@pytest.mark.parametrize('extra',[
    {'resolution_m':30}, {'resolution_m':5000},
    {'datetime_utc':'2023-06-21T12:15:00Z'}, {'datetime_utc':'2023-06-21T12:00:00'},
    {'datetime_utc':'2099-01-01T00:00:00Z'}, {'datetime_utc':'2020-01-01T00:00:00Z'},
    {'polygon':{'type':'Polygon','coordinates':[[[0,0],[30,0],[30,10],[0,10],[0,0]]]}},
    {'polygon':{'type':'Polygon','coordinates':[[[179.5,0],[179.6,0],[179.6,.1],[179.5,.1],[179.5,0]]]}},
    {'polygon':{'type':'Polygon','coordinates':[[[0,85],[.1,85],[.1,85.1],[0,85.1],[0,85]]]}}
])
def test_invalid_requests_fail_before_work(extra):
    with pytest.raises(ValueError):
        validate(payload(**extra))


def test_small_cross_zone_selection_includes_both_sources():
    request = validate(payload(polygon={'type':'Polygon','coordinates':[[[-.01,51.5],[.01,51.5],[.01,51.51],[-.01,51.51],[-.01,51.5]]]}))
    assert {s['epsg'] for s in request['source_tiles']} == {32630,32631}


@pytest.fixture
def settings(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('lst_global.api.shutil.disk_usage',lambda _:SimpleNamespace(free=30*1024**3))
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps({'pilots':[]}))
    cache = tmp_path/'cache'
    cache.mkdir()
    return Settings(root=tmp_path,catalog_path=path,cache_dir=cache,model_path=tmp_path/'model')


def finish(manager, job):
    until = time.monotonic()+5
    while time.monotonic()<until:
        record = manager.get(job['id'])
        if record['status'] not in ('queued','running'):
            return record
        time.sleep(.01)
    raise AssertionError('Job remained active')


def fake_runner(job_id, request, directory):
    (directory/'prediction.tif').write_bytes(b'test fixture')
    return {'files':{'prediction_tif':'prediction.tif'},'global_accuracy_validated':False}


def test_persisted_cache_and_file_containment(settings):
    manager = GlobalManager(settings, fake_runner)
    try:
        first = manager.submit(payload(),'test')
        result = finish(manager,first)
        assert result['status']=='succeeded'
        assert result['result']['files']['prediction_tif'].startswith('/api/global/jobs/')
        assert manager.submit(payload(),'test')['id']==first['id']
        for name in ('worker.log','request.json','../prediction.tif'):
            with pytest.raises(HTTPException):
                manager.file(first['id'],name)
        path = manager.file(first['id'],'prediction.tif')
        path.unlink()
        path.symlink_to(settings.catalog_path)
        with pytest.raises(HTTPException):
            manager.file(first['id'],'prediction.tif')
    finally:
        manager.close()


def test_queue_budget_rejection_does_not_create_work(settings):
    release = threading.Event()
    def blocked(*args):
        release.wait(3)
        return fake_runner(*args)
    manager = GlobalManager(replace(settings,max_pending=1),blocked)
    try:
        job = manager.submit(payload(),'test')
        with pytest.raises(HTTPException) as error:
            manager.submit(payload(resolution_m=250),'test')
        assert error.value.status_code==429
        assert manager.pending()==1
        release.set()
        finish(manager,job)
    finally:
        release.set()
        manager.close()


def test_failed_jobs_and_expired_files_are_explicit(settings):
    clock = [time.time()]
    manager = GlobalManager(settings,fake_runner,clock=lambda:clock[0])
    try:
        job = manager.submit(payload(),'test')
        finish(manager,job)
        clock[0] += 31*86400
        manager._maintain()
        assert manager.get(job['id'])['status']=='expired'
        assert not (settings.jobs_dir/job['id']).exists()
        assert settings.catalog_path.exists()
    finally:
        manager.close()


def test_low_disk_rejects_without_starting_worker(settings,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('lst_global.api.shutil.disk_usage',lambda _:SimpleNamespace(free=1024))
    manager=GlobalManager(settings,fake_runner)
    try:
        with pytest.raises(HTTPException) as error:
            manager.submit(payload(),'test')
        assert error.value.status_code==503
        assert manager.pending()==0
    finally:
        manager.close()


def test_api_boundaries_and_pixel_unfinished(settings):
    with TestClient(create_app(settings,fake_runner)) as client:
        caps = client.get('/api/global/capabilities').json()
        assert caps['accuracy']['qualified'] is False
        assert caps['resolutions_m']==[100,250,500,1000]
        assert client.post('/api/global/jobs',content='x'*65537).status_code==413
        assert client.post('/api/global/jobs',json={'resolution_m':100}).status_code==422
        assert client.get('/api/global/jobs/invalid/pixel?lon=0&lat=0').status_code==404


def test_job_deadline_kills_actual_child_process(settings,monkeypatch):
    import subprocess,sys
    original = subprocess.Popen
    children=[]
    def sleeper(command,**kwargs):
        child=original([sys.executable,'-c','import time; time.sleep(20)'],**kwargs)
        children.append(child)
        return child
    monkeypatch.setattr('lst_global.api.subprocess.Popen',sleeper)
    monkeypatch.setattr('lst_global.api.JOB_TIMEOUT_SECONDS',.05)
    manager=GlobalManager(settings)
    directory=settings.jobs_dir/'deadline-fixture';directory.mkdir()
    try:
        with pytest.raises(TimeoutError):
            manager._subprocess('test',validate(payload()),directory)
        assert len(children)==1 and children[0].poll() is not None
        assert manager.process is None
    finally:
        manager.close()


def synthetic_request(bounds, resolution):
    inverse = Transformer.from_crs(32630,4326,always_xy=True)
    left,bottom,right,top = bounds
    ring = [list(inverse.transform(x,y)) for x,y in [(left,bottom),(right,bottom),(right,top),(left,top),(left,bottom)]]
    return {'epsg':32630,'bounds_m':bounds,'resolution_m':resolution,
            'shape':[round((top-bottom)/resolution),round((right-left)/resolution)],
            'polygon':{'type':'Polygon','coordinates':[ring]},'datetime_utc':'2023-06-21T12:00:00Z'}


def write_native(directory,values,left=600000,top=5701000):
    directory.mkdir()
    with rasterio.open(directory/'lst.tif','w',driver='GTiff',height=values.shape[0],width=values.shape[1],
                       count=1,dtype='float32',crs='EPSG:32630',transform=from_origin(left,top,100,100),nodata=-9999) as dst:
        dst.write(values.astype(np.float32),1)


def test_coarse_aggregation_and_nodata_fraction(tmp_path):
    source = tmp_path/'tile'
    values = np.arange(100,dtype=np.float32).reshape(10,10)
    values[:3,:3]=-9999
    write_native(source,values)
    output=tmp_path/'output'; output.mkdir()
    result = assemble(synthetic_request([600000,5700000,601000,5701000],500),[source],output)
    with rasterio.open(output/'prediction.tif') as raster:
        coarse=raster.read(1,masked=True)
        assert coarse.mask[0,0]  # 36% unavailable: do not fill from neighbouring data.
        assert coarse[1,1]==pytest.approx(values[5:,5:].mean())
        assert raster.descriptions==('predicted_lst_c',)
    assert result['counts']['predicted']==3


@pytest.mark.parametrize('resolution',[100,250,500,1000])
def test_resolution_overlap_is_invariant(tmp_path,resolution):
    source=tmp_path/'tile'
    values=np.arange(400,dtype=np.float32).reshape(20,20)
    write_native(source,values,top=5702000)
    one=tmp_path/'one'; one.mkdir()
    two=tmp_path/'two'; two.mkdir()
    assemble(synthetic_request([600000,5700000,602000,5702000],resolution),[source],one)
    assemble(synthetic_request([601000,5700000,602000,5701000],resolution),[source],two)
    with rasterio.open(one/'prediction.tif') as a,rasterio.open(two/'prediction.tif') as b:
        sub=a.read(1)[1000//resolution:,1000//resolution:]
        np.testing.assert_array_equal(sub,b.read(1))


def test_all_masked_is_transparent_not_fake_temperature(tmp_path):
    source=tmp_path/'tile'; write_native(source,np.full((10,10),-9999))
    output=tmp_path/'output'; output.mkdir()
    result=assemble(synthetic_request([600000,5700000,601000,5701000],100),[source],output)
    assert result['summary']['predicted_lst_c']['mean'] is None
    assert result['counts']['predicted']==0
    from PIL import Image
    assert np.array(Image.open(output/'overlay.png'))[...,3].max()==0
