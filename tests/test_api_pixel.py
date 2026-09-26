"""Pixel route isolation: every manager and database lives under pytest tmp_path."""
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lst_pilot.api import Settings, create_app

JOB = "a"*32


@pytest.fixture
def client(tmp_path):
    catalog=tmp_path/"catalog.json"
    catalog.write_text(json.dumps({"pilots":[]}))
    settings=Settings(root=tmp_path,catalog_path=catalog)

    def forbidden(*args,**kwargs):
        raise AssertionError("Pixel inspection must never render or fetch satellite data")

    with TestClient(create_app(settings,forbidden,forbidden)) as client:
        manager=client.app.state.jobs
        directory=settings.jobs_dir/JOB
        directory.mkdir()
        with rasterio.open(directory/"prediction.tif","w",driver="GTiff",height=1,width=2,count=1,
                           dtype="float32",crs="EPSG:4326",transform=from_origin(10,50,1,1),nodata=-9999) as out:
            out.write(np.array([[[0,-9999]]],dtype="float32"))
            out.set_band_description(1,"predicted_lst_c")
        result={"prediction_method":"daytime_ml","available_files":{"prediction.tif":f"/api/jobs/{JOB}/files/prediction.tif"}}
        with manager._db() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (JOB,"fixture","test",1.,1.,"succeeded","complete",1.,"{}",json.dumps(result),None))
        yield client


def lookup(client,**params):
    return client.get(f"/api/jobs/{JOB}/pixel",params={"lon":10.5,"lat":49.5,**params})


def test_pixel_route_contract_and_read_only_job_state(client):
    manager=client.app.state.jobs
    before=manager.get(JOB)
    response=lookup(client)
    assert response.status_code == 200
    result=response.json()
    assert set(result) == {"job_id","query","status","pixel","values","units"}
    assert result["job_id"] == JOB and result["query"] == {"lon":10.5,"lat":49.5}
    assert result["values"] == {"prediction":0.,"observed":None,"residual":None}
    assert result["status"] == "ok" and result["pixel"]["column"] == 0
    assert lookup(client,lon=11.5).json()["status"] == "nodata"
    assert lookup(client,lon=12).json()["status"] == "outside"
    assert manager.get(JOB) == before
    with manager._db() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("field,value",[("lon","nan"),("lat","inf"),("lon",181),("lat",-91),("lon","abc")])
def test_pixel_http_coordinate_validation(client,field,value):
    assert lookup(client,**{field:value}).status_code == 422


@pytest.mark.parametrize("status",["queued","running","failed"])
def test_unfinished_raster_returns_conflict(client,status):
    client.app.state.jobs._update(JOB,status=status)
    assert lookup(client).status_code == 409


@pytest.mark.parametrize("changes",[{"withdrawn":True},{"prediction_method":"nighttime_coarse_baseline"},{"prediction_method":"mixed_day_night"}])
def test_withdrawn_raster_rejected_before_open(client,monkeypatch,changes):
    manager=client.app.state.jobs
    result=manager.get(JOB)["result"]
    result.update(changes)
    manager._update(JOB,result=json.dumps(result))
    monkeypatch.setattr("lst_pilot.raster_pixel.read_pixel",lambda *a:pytest.fail("withdrawn raster opened"))
    response=lookup(client)
    assert response.status_code == 409 and "withdrawn" in response.json()["detail"]


def test_unknown_job_and_missing_published_canonical_file_are_404(client):
    assert client.get("/api/jobs/not-a-uuid/pixel",params={"lon":0,"lat":0}).status_code == 404
    assert client.get(f"/api/jobs/{'b'*32}/pixel",params={"lon":0,"lat":0}).status_code == 404
    manager=client.app.state.jobs
    result=manager.get(JOB)["result"]
    result["available_files"]={}
    result["files"]={"prediction_tif":"/etc/passwd"}
    manager._update(JOB,result=json.dumps(result))
    assert lookup(client).status_code == 404


def test_file_and_parent_directory_symlink_escape_are_rejected(client,tmp_path):
    directory=client.app.state.jobs.settings.jobs_dir/JOB
    path=directory/"prediction.tif"
    path.rename(directory/"original.tif")
    path.symlink_to(directory/"original.tif")
    assert lookup(client).status_code == 404
    path.unlink()
    (directory/"original.tif").rename(path)
    outside=tmp_path/"elsewhere"
    directory.rename(outside)
    directory.symlink_to(outside,target_is_directory=True)
    assert lookup(client).status_code == 404


def test_result_url_cannot_redirect_lookup_to_another_file(client):
    manager=client.app.state.jobs
    result=manager.get(JOB)["result"]
    result["available_files"]["prediction.tif"]="https://example.invalid/secret.tif"
    result["files"]={"prediction_tif":"../../other/prediction.tif"}
    manager._update(JOB,result=json.dumps(result))
    assert lookup(client).json()["values"]["prediction"] == 0


def test_raster_failure_does_not_expose_private_path_or_traceback(client):
    path=client.app.state.jobs.settings.jobs_dir/JOB/"prediction.tif"
    path.write_bytes(b"invalid TIFF /opt/private/token")
    response=lookup(client)
    assert response.status_code == 500
    assert response.json() == {"detail":"The saved raster could not be inspected."}
    assert str(path) not in response.text and "Traceback" not in response.text
