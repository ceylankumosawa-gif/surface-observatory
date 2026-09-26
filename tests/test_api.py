"""API boundary, cache and budget tests; fake renders and no network access."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from lst_pilot.api import (JobInput, JobManager, Settings, catalog_regions,
                           create_app, fetch_scene, validate_request, _clean_json)


@pytest.fixture
def settings(tmp_path):
    catalog = {
        "pilots": {"type": "FeatureCollection", "features": [{
            "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
                [-.17, -.17], [.17, -.17], [.17, .17], [-.17, .17], [-.17, -.17]]]},
            "properties": {"id": "test", "epsg": 3857, "extent_m": [-20000, -20000, 20000, 20000],
                "scenes": [
                    {"scene_id": "SCENE_A", "datetime_utc": "2024-06-01T12:34:56.123456Z"},
                    {"scene_id": "SCENE_B", "datetime_utc": "2024-06-20T12:00:00Z"},
                    {"scene_id": "SCENE_C", "datetime_utc": "2024-07-01T12:00:00Z"},
                ]}
        }]}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    return Settings(root=tmp_path, catalog_path=path)


def payload(**overrides):
    data = {"pilot_id": "test", "mode": "observed", "scene_id": "SCENE_A",
            "polygon": {"type": "Polygon", "coordinates": [[[0, 0], [.005, 0], [.005, .005], [0, .005], [0, 0]]]}}
    data.update(overrides)
    return JobInput.model_validate(data)


def fake_scene(scene, cache):
    return {"id": scene["scene_id"], "properties": {"datetime": scene["datetime_utc"]}}


def fake_render(**kwargs):
    path = kwargs["output_dir"]
    (path / "overlay.png").write_bytes(b"fake PNG test fixture")
    (path / "prediction.tif").write_bytes(b"fake TIFF test fixture")
    (path / "provenance.json").write_text(json.dumps({
        "model_path": "/opt/private/model.joblib",
        "source": "https://example.invalid/data.tif?sig=not-a-real-token",
        "note": "Stored at /opt/private/data.txt for processing",
    }))
    kwargs["progress"]({"stage": "rendering", "fraction": .8})
    return {"files": {"overlay_png": str(path / "overlay.png"),
                      "prediction_tif": str(path / "prediction.tif"),
                      "provenance_json": str(path / "provenance.json")},
            "model_path": "/opt/private/model.joblib",
            "summary": {"mean_c": 22.1, "valid_pixels": 9},
            "output_dir": str(path)}


def finished(manager, job_id, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        record = manager.get(job_id)
        if record["status"] not in ("queued", "running"):
            return record
        time.sleep(.01)
    raise AssertionError("Fake job did not finish")


def test_polygon_holes_invalid_shape_boundary_and_bbox_budget(settings):
    regions = catalog_regions(json.loads(settings.catalog_path.read_text()))
    valid, _, _ = validate_request(payload(), regions)
    assert valid["datetime_utc"] == "2024-06-01T12:34:56.123456Z"
    invalid = [
        # Holes are deliberately outside the small first-version contract.
        [[[0, 0], [.005, 0], [.005, .005], [0, .005], [0, 0]],
         [[.001, .001], [.002, .001], [.002, .002], [.001, .001]]],
        [[[0, 0], [.005, .005], [.005, 0], [0, .005], [0, 0]]],
        [[[.17, 0], [.19, 0], [.19, .005], [.17, .005], [.17, 0]]],
    ]
    for coordinates in invalid:
        with pytest.raises(ValueError):
            validate_request(payload(polygon={"type": "Polygon", "coordinates": coordinates}), regions)
    with pytest.raises(ValueError, match="4–512"):
        validate_request(payload(polygon={"type": "Polygon", "coordinates": [[[0, 0]] * 513]}), regions)


def test_observed_exact_time_and_experimental_preceding_selection(settings, monkeypatch):
    regions = catalog_regions(json.loads(settings.catalog_path.read_text()))
    monkeypatch.setattr("lst_pilot.api._is_daylight", lambda *args: True)
    with pytest.raises(ValueError, match="exact"):
        validate_request(payload(datetime_utc="2024-06-01T12:00:00Z"), regions)
    selected, _, _ = validate_request(payload(mode="experimental", scene_id=None,
                                               datetime_utc="2024-06-25T14:00:00Z"), regions)
    assert selected["scene_id"] == "SCENE_B"  # Never the newer July 1 optical scene.
    for dt in ["2024-05-31T12:00:00Z", "2024-12-31T12:00:00Z"]:
        with pytest.raises(ValueError, match="earlier surface scene"):
            validate_request(payload(mode="experimental", scene_id=None, datetime_utc=dt), regions)
    for dt in ["2025-01-01T12:00:00Z", "2024-06-25T12:00:00"]:
        with pytest.raises(ValueError):
            validate_request(payload(mode="experimental", scene_id=None, datetime_utc=dt), regions)
    monkeypatch.setattr("lst_pilot.api._is_daylight", lambda *args: False)
    with pytest.raises(ValueError, match="Nighttime"):
        validate_request(payload(mode="experimental", scene_id=None, datetime_utc="2024-06-25T00:00:00Z"), regions)


def test_reported_air_scenarios_allow_supported_daytime_dates(settings):
    regions = catalog_regions(json.loads(settings.catalog_path.read_text()))
    for stamp in ["1900-01-01T12:00:00Z", "2025-07-03T12:00:00Z", "2100-12-31T12:00:00Z"]:
        request, _, _ = validate_request(payload(mode="scenario", scene_id="SCENE_C",
            datetime_utc=stamp, air_override=-10), regions)
        assert request["datetime_utc"] == stamp and request["scene_id"] == "SCENE_C"
    for values in [{"air_override": None}, {"scene_id": "EXTERNAL"},
                   {"datetime_utc": "2101-01-01T00:00:00Z"}, {"datetime_utc": None}]:
        data = dict(mode="scenario", datetime_utc="2025-07-03T12:00:00Z", air_override=20)
        data.update(values)
        with pytest.raises(ValueError):
            validate_request(payload(**data), regions)


def test_night_request_is_rejected_before_creating_or_reusing_jobs(settings):
    def forbidden(*args, **kwargs):
        raise AssertionError("No source retrieval or rendering should start")
    with TestClient(create_app(settings, forbidden, forbidden)) as client:
        body = payload(mode="scenario", datetime_utc="2025-07-03T00:00:00Z", air_override=20).model_dump()
        response = client.post("/api/jobs", json=body)
        assert response.status_code == 422 and "withdrawn" in response.json()["detail"]
        with client.app.state.jobs._db() as db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_archived_night_result_is_marked_withdrawn_without_rewriting_files(settings):
    manager = JobManager(settings, fake_render, fake_scene)
    try:
        identifier = manager.submit(payload(), "one")["id"]
        record = finished(manager, identifier)
        old = record["result"]
        old["prediction_method"] = "nighttime_coarse_baseline"
        manager._update(identifier, result=json.dumps(old))
        found = manager.get(identifier)
        assert found["status"] == "succeeded" and found["result"]["withdrawn"]
        assert "does not learn" in found["result"]["withdrawal_reason"]
        with manager._db() as db:
            assert "withdrawn" not in json.loads(db.execute("SELECT result FROM jobs WHERE id=?", (identifier,)).fetchone()[0])
    finally:
        manager.close()


def test_complete_canonical_london_pilot_fits_api_and_renderer_limits():
    from lst_pilot.raster import build_grid
    catalog = json.loads((Path(__file__).resolve().parents[1]/"web/catalog.json").read_text())
    regions = catalog_regions(catalog)
    london = regions["greater_london"]
    request = JobInput(pilot_id="greater_london", polygon=london["geometry"],
        mode="observed", scene_id=london["scenes"][0]["scene_id"])
    normalized, region, _ = validate_request(request, regions)
    grid = build_grid(region, normalized["polygon"])
    assert grid.shape == (800, 800) and grid.inside.sum() == 640000


def test_same_request_reuses_running_job_and_queue_is_bounded(settings):
    gate = threading.Event()
    entered = threading.Event()

    def blocked(**kwargs):
        entered.set()
        assert gate.wait(5)
        return fake_render(**kwargs)

    manager = JobManager(settings, blocked, fake_scene)
    try:
        first = manager.submit(payload(), "one")
        assert entered.wait(2)
        again = manager.submit(payload(), "two")
        assert again["id"] == first["id"] and again["cached"]
        second = manager.submit(payload(air_override=20), "one")
        third = manager.submit(payload(air_override=21), "one")
        with pytest.raises(HTTPException) as caught:
            manager.submit(payload(air_override=22), "one")
        assert caught.value.status_code == 429 and "queue" in caught.value.detail
        gate.set()
        for record in (first, second, third):
            assert finished(manager, record["id"])["status"] == "succeeded"
    finally:
        gate.set()
        manager.close()


def test_rate_limits_and_completed_cache_survive_restart(settings):
    settings = replace(settings, hourly_per_ip=2, daily_global=3)
    manager = JobManager(settings, fake_render, fake_scene)
    try:
        first = manager.submit(payload(), "one")
        finished(manager, first["id"])
        second = manager.submit(payload(air_override=20), "one")
        finished(manager, second["id"])
    finally:
        manager.close()
    restarted = JobManager(settings, fake_render, fake_scene)
    try:
        assert restarted.submit(payload(), "one")["id"] == first["id"]
        with pytest.raises(HTTPException) as caught:
            restarted.submit(payload(air_override=21), "one")
        assert caught.value.status_code == 429 and "hour" in caught.value.detail
        third = restarted.submit(payload(air_override=21), "two")
        finished(restarted, third["id"])
        with pytest.raises(HTTPException) as caught:
            restarted.submit(payload(air_override=22), "three")
        assert caught.value.status_code == 429 and "day" in caught.value.detail
        assert restarted.submit(payload(), "four")["cached"]
    finally:
        restarted.close()


def test_scenario_cache_can_refresh_weather_on_a_later_utc_day(settings):
    manager = JobManager(settings, fake_render, fake_scene)
    stamp = [1788825600.]
    manager.clock = lambda: stamp[0]
    request = payload(mode="scenario", datetime_utc="2026-09-07T12:00:00Z", air_override=25)
    try:
        first = manager.submit(request, "one")
        finished(manager, first["id"])
        assert manager.submit(request, "one")["id"] == first["id"]
        observed = manager.submit(payload(), "one")
        finished(manager, observed["id"])
        stamp[0] += 86400
        second = manager.submit(request, "one")
        assert second["id"] != first["id"]
        finished(manager, second["id"])
        assert manager.submit(payload(), "one")["id"] == observed["id"]
    finally:
        manager.close()


def test_restart_marks_unfinished_jobs_failed(settings):
    manager = JobManager(settings, fake_render, fake_scene)
    record = manager.submit(payload(), "one")
    finished(manager, record["id"])
    # Simulate an on-disk job left running by an abruptly terminated process.
    manager._update(record["id"], status="running", result=None)
    manager.close()
    restarted = JobManager(settings, fake_render, fake_scene)
    try:
        found = restarted.get(record["id"])
        assert found["status"] == "failed" and found["stage"] == "interrupted"
        assert "restarted" in found["error"]
    finally:
        restarted.close()


def test_download_allowlist_semantic_keys_and_public_metadata(settings):
    manager = JobManager(settings, fake_render, fake_scene)
    try:
        job = manager.submit(payload(), "one")
        record = finished(manager, job["id"])
        assert record["status"] == "succeeded"
        assert record["result"]["files"]["overlay_png"] == f"/api/jobs/{job['id']}/files/overlay.png"
        assert "/opt/private" not in json.dumps(record)
        published = json.loads(manager.file(job["id"], "provenance.json").read_text())
        assert "sig=" not in json.dumps(published) and "/opt/private" not in json.dumps(published)
        assert "model_path" not in published
        with pytest.raises(HTTPException):
            manager.file(job["id"], "../../jobs.sqlite3")
        # Swapping an allowed filename for a symlink still must not expose it.
        image = manager.file(job["id"], "overlay.png")
        image.unlink()
        image.symlink_to(settings.catalog_path)
        with pytest.raises(HTTPException):
            manager.file(job["id"], "overlay.png")
    finally:
        manager.close()


def test_untrusted_scene_url_is_never_used_and_cached_identity_is_checked(tmp_path):
    cached = {"id": "WRONG", "collection": "landsat-c2-l2", "properties": {"datetime": "2024-06-01T12:00:00Z"}}
    (tmp_path / "SCENE_A.json").write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="does not match"):
        fetch_scene({"scene_id": "SCENE_A", "datetime_utc": "2024-06-01T12:00:00Z",
                     "stac_url": "http://127.0.0.1/private"}, tmp_path)


def test_http_validation_body_limit_status_and_files(settings):
    with TestClient(create_app(settings, fake_render, fake_scene)) as client:
        assert client.get("/api/health").json()["pilots"] == 1
        assert client.get("/api/catalog").json()["pilots"]["type"] == "FeatureCollection"
        assert client.post("/api/jobs", content=b"x" * 70000).status_code == 413
        invalid = payload().model_dump()
        invalid["air_override"] = float("nan")
        assert client.post("/api/jobs", content=json.dumps(invalid), headers={"Content-Type": "application/json"}).status_code == 422
        response = client.post("/api/jobs", json=payload().model_dump())
        assert response.status_code == 202
        record = finished(client.app.state.jobs, response.json()["id"])
        assert client.get(record["result"]["files"]["overlay_png"]).status_code == 200
        assert client.get(f"/api/jobs/{record['id']}/files/model.joblib").status_code == 404
        assert client.get("/api/jobs/not-a-uuid").status_code == 404


def test_render_errors_do_not_expose_private_details(settings):
    def broken(**kwargs):
        raise ValueError("Private /opt/secret/token.txt https://example.invalid/?sig=not-a-real-token")
    manager = JobManager(settings, broken, fake_scene)
    try:
        record = finished(manager, manager.submit(payload(), "one")["id"])
        assert record["status"] == "failed"
        assert "secret" not in record["error"] and "sig=" not in record["error"]
        assert "smaller area" in record["error"]
    finally:
        manager.close()


def test_documentation_query_survives_while_signed_credentials_are_removed():
    docs = "https://confluence.ecmwf.int/pages/viewpage.action?pageId=216495456"
    signed = "https://example.invalid/data.tif?sv=2024&st=now&se=later&sp=r&sig=not-a-real-token"
    clean = _clean_json({"documentation": docs, "asset": signed,
                        "mixed": "https://example.invalid/page?pageId=123&access_token=placeholder",
                        "note": "Documentation " + docs})
    assert clean["documentation"] == docs
    assert clean["asset"] == "https://example.invalid/data.tif"
    assert clean["mixed"] == "https://example.invalid/page?pageId=123"
    assert clean["note"] == "Documentation " + docs


def test_source_and_model_contents_invalidate_cache_after_restart(settings):
    source_dir = settings.root / "src/lst_pilot"
    source_dir.mkdir(parents=True)
    source = source_dir / "raster.py"
    source.write_text("MASK_VERSION = 1\n")
    settings.model_path.parent.mkdir(parents=True)
    settings.model_path.write_bytes(b"model-version-one")

    def run_once():
        manager = JobManager(settings, fake_render, fake_scene)
        try:
            submitted = manager.submit(payload(), "one")
            assert finished(manager, submitted["id"])["status"] == "succeeded"
            return submitted
        finally:
            manager.close()

    first = run_once()
    same = run_once()
    assert same["id"] == first["id"] and same["cached"]
    source.write_text("MASK_VERSION = 2\n")
    changed_source = run_once()
    assert changed_source["id"] != first["id"] and not changed_source["cached"]
    # Equal byte count: identity depends on model bytes, not just file size.
    settings.model_path.write_bytes(b"model-version-two")
    changed_model = run_once()
    assert changed_model["id"] != changed_source["id"] and not changed_model["cached"]
