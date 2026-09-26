import copy

import numpy as np
import pandas as pd
import pytest

from lst_global import weather_access as w


def availability():
    return {"checked_utc":"2026-09-18T14:00:00Z", "metadata_sha256":"a"*64,
            "final_last_valid_hour_utc":"2026-06-30T23:00:00Z",
            "provisional_last_valid_hour_utc":"2026-09-12T23:00:00Z"}


def plan(stamp="2026-09-18T14:00:00Z", **kwargs):
    return w.choose_weather_source(stamp, availability(), now="2026-09-18T14:04:00Z", **kwargs)


def test_publication_bounds_not_assumed_delay():
    assert plan("2026-06-30T23:00:00Z")["source"] == "era5_final"
    assert plan("2026-07-01T00:00:00Z")["source"] == "era5_provisional"
    assert plan("2026-09-12T23:00:00Z")["source"] == "era5_provisional"
    recent = plan("2026-09-13T00:00:00Z")
    assert recent["source"] == "experimental_gfs" and not recent["enabled"]
    assert recent["requested_time_utc"] == recent["valid_time_utc"]
    assert not recent["temporal_substitution"]
    assert plan(allow_operational=True)["enabled"]


@pytest.mark.parametrize("stamp", ["2026-09-18T15:00:00Z", "2026-09-18T14:01:00Z", "2026-09-18T14:00:00", "2020-01-01T00:00:00Z"])
def test_invalid_target_rejected(stamp):
    with pytest.raises(ValueError):
        plan(stamp)


def test_stale_publication_proof_rejected():
    with pytest.raises(ValueError, match="expired"):
        w.choose_weather_source("2026-09-12T00:00:00Z", availability(), now="2026-09-20T00:00:00Z")


def test_native_longwave_deaveraging_and_invalid_origin():
    np.testing.assert_array_equal(w.hourly_longwave([350,360], [340,355], 6,8,6), [360,365])
    np.testing.assert_array_equal(w.hourly_longwave([350], None,6,7),[350])
    with pytest.raises(ValueError):
        w.hourly_longwave([350],[340],6,8,3)
    assert np.isnan(w.hourly_longwave([-1,2000],None,6,7)).all()


def test_native_index_requires_exact_cycle_time_support():
    data = b"1:0:d=2026091806:WEASD:surface:8 hour fcst:\n2:100:d=2026091806:DLWRF:surface:6-8 hour ave fcst:\n"
    result = w._index_records(data, pd.Timestamp("2026-09-18T06:00:00Z"),8)
    assert result["DLWRF"]["start"] == 6
    with pytest.raises(ValueError):
        w._index_records(data,pd.Timestamp("2026-09-18T00:00:00Z"),8)
    with pytest.raises(ValueError):
        w._index_records(data,pd.Timestamp("2026-09-18T06:00:00Z"),9)


def response():
    times = pd.date_range("2026-09-14", periods=120, freq="h")
    hourly = {k:[1.0]*120 for k in w.GFS_VARIABLES}
    hourly["temperature_2m"] = [float(x)/10 for x in range(120)]
    hourly["surface_pressure"] = [1010.]*120
    hourly["rain"] = [2.]*120
    hourly["time"] = times.strftime("%Y-%m-%dT%H:%M").tolist()
    return {"utc_offset_seconds":0, "hourly":hourly, "hourly_units":w.GFS_UNITS.copy(), "latitude":51.75, "longitude":-1.25}


def test_operational_derived_history_is_exact_and_honest():
    data = response()
    result = w._gfs_table(data)
    row = result.iloc[100]
    assert row.air_temperature_lag24_c == 7.6
    assert row.rain_mm_72h == 144
    assert row.soil_moisture_m3_m3 == 1
    assert "soil_moisture_0_to_10cm" in w.GFS_VARIABLES
    assert "snow_depth" not in w.GFS_VARIABLES
    data["hourly"]["rain"][99] = None
    assert np.isnan(w._gfs_table(data).iloc[100].rain_mm_72h)


@pytest.mark.parametrize("fault",["units","gap","duplicate"])
def test_operational_contract_rejects_fault(fault):
    data = response()
    if fault == "units":
        data["hourly_units"]["temperature_2m"] = "K"
    elif fault == "gap":
        data["hourly"]["time"][0] = "2026-09-13T23:00"
    else:
        data["hourly"]["time"][1] = data["hourly"]["time"][0]
    with pytest.raises(ValueError):
        w._gfs_table(data)


def test_unavailable_never_requests_or_uses_older_hour(tmp_path):
    frame = pd.DataFrame({"datetime_utc":[pd.Timestamp("2026-09-18T14:00Z")], "latitude":[51.75],"longitude":[-1.25]})
    with pytest.raises(ValueError,match="disabled"):
        w.prepare_background(frame,tmp_path,plan())


def test_era5_exact_hour_guard(monkeypatch,tmp_path):
    frame = pd.DataFrame({"datetime_utc":[pd.Timestamp("2023-06-21T12:00Z")], "latitude":[51.75],"longitude":[-1.25]})
    bad = frame.assign(weather_datetime_utc=pd.Timestamp("2023-06-21T11:00Z"))
    monkeypatch.setattr(w.legacy_weather,"enrich_weather",lambda *a,**k:(bad,[]))
    with pytest.raises(ValueError,match="exact"):
        w.prepare_background(frame,tmp_path,plan("2023-06-21T12:00Z"))


def test_public_session_never_reads_netrc_or_proxy_credentials():
    assert not w.BoundedHTTP().session.trust_env


def test_atomic_public_cache_group_permissions(tmp_path):
    path=tmp_path/"weather_access"/"gfs"/"object.json"
    w._atomic(path,b"{}")
    assert path.stat().st_mode & 0o777 == 0o664
    assert path.parent.stat().st_mode & 0o2777 == 0o2775
    assert path.stat().st_gid == tmp_path.stat().st_gid


def test_operational_physical_domain_invalid_is_not_clipped():
    data=response()
    data["hourly"]["soil_moisture_0_to_10cm"][100]=-0.1
    with pytest.raises(ValueError,match="physical-domain"):
        w._gfs_table(data)


def test_native_multiple_cell_indices_and_bounds():
    from affine import Affine
    transform = Affine(.25,0,-1.625,0,-.25,52.125)
    rows,cols=w._native_rowcol(transform,np.array([-1.5,-1.25]),np.array([52,51.75]),3,3)
    np.testing.assert_array_equal(rows,[0,1])
    np.testing.assert_array_equal(cols,[0,1])
    with pytest.raises(ValueError):
        w._native_rowcol(transform,[0.],[51.75],3,3)


def test_metar_exact_time_raw_temperature_and_precise_negative():
    from lst_global.weather_stations import verified_metar
    raw = {"icaoId":"KAAA","obsTime":int(pd.Timestamp("2026-09-18T13:50Z").timestamp()),
           "rawOb":"METAR KAAA 181350Z 00000KT 10SM CLR M02/M05 A3000 RMK T10231050",
           "temp":-2.3,"lat":40.,"lon":-105.,"elev":1500,"qcField":0}
    result, reason = verified_metar(raw,"2026-09-18T14:00Z",{"KAAA"})
    assert reason=="verified" and result["air_temperature_c"]==-2.3
    assert result["precision"]=="0.1C_T_remark"
    raw["temp"]=-2
    assert verified_metar(raw,"2026-09-18T14:00Z",{"KAAA"})[1]=="raw_temperature_mismatch_or_out_of_range"


@pytest.mark.parametrize("fault,reason",[("future","not_timely_backward_report"),("stale","not_timely_backward_report"),
                                        ("wrongday","raw_observation_time_mismatch"),("wrongstation","station_identity_mismatch")])
def test_metar_never_fabricates_timely_station(fault,reason):
    from lst_global.weather_stations import verified_metar
    raw = {"icaoId":"EGTK","obsTime":int(pd.Timestamp("2026-09-18T13:50Z").timestamp()),
           "rawOb":"METAR EGTK 181350Z 22019KT 9999 BKN043 18/10 Q1013",
           "temp":18,"lat":51.838,"lon":-1.317,"elev":78}
    target = "2026-09-18T14:00Z"
    if fault=="future": target="2026-09-18T13:00Z"
    if fault=="stale": target="2026-09-18T16:00Z"
    if fault=="wrongday": raw["rawOb"]=raw["rawOb"].replace("181350Z","171350Z")
    if fault=="wrongstation": raw["icaoId"]="EGGW"
    assert verified_metar(raw,target,{"EGTK"})[1]==reason


def test_no_recent_station_leaves_air_missing_and_makes_no_report_request(monkeypatch,tmp_path):
    from lst_global import weather_stations as s
    inv=tmp_path/"stations"/"ghcnh-station-list.csv"
    inv.parent.mkdir();inv.write_text("inventory test fixture")
    candidates=pd.DataFrame(columns=["station_id","icao","distance_to_center_km"])
    monkeypatch.setattr(s,"station_inventory",lambda *a:candidates)
    monkeypatch.setattr(s,"nearby_stations",lambda *a,**k:candidates)
    monkeypatch.setattr(s,"region_bbox",lambda a:[-1.5,51.5,-1,52])
    class NoHTTP:
        def get(self,*a,**k):
            raise AssertionError("Missing geographic candidates must not initiate report fetch")
    frame=pd.DataFrame({"region_id":["x"],"datetime_utc":[pd.Timestamp("2026-09-18T14:00Z")],
                        "latitude":[51.75],"longitude":[-1.25],"background_air_temperature_c":[18.]})
    output,receipt=s.attach_recent_stations(frame,[{"id":"x"}],tmp_path,plan(allow_operational=True),http=NoHTTP())
    assert output.air_temperature_c.isna().all()
    assert output.station_id.eq("").all()
    assert receipt["status_counts"]=={"missing_timely_verified_METAR":1}
