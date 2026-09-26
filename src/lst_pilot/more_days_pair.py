"""Sequential, checkpointed research pairing and strict new-fit-only admission.

Numerical feature, optical and station adapters are reused without modification.
Only immutable completed 2021–2022 acquisition batches may enter this stage.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import legacy_isd, option_b_features as f, option_b_merge

VERSION = "more-days-pair-v1"
VERIFIED = {"exact_cached_report_verified", "exact_cached_legacy_isd_qc1_report_verified"}


def verify_new_identity(frame, areas, old, allowed_region=None):
    """Fail on provenance/identity violations; never silently turn them into loss."""
    data = f.validate_samples(frame, areas)
    dates = data.datetime_utc.dt.strftime("%Y-%m-%d")
    if not data.datetime_utc.dt.year.isin([2021, 2022]).all():
        raise ValueError("New fitting inputs must be 2021–2022 only.")
    if data.region_id.eq("cabauw").any() or (allowed_region and not data.region_id.eq(allowed_region).all()):
        raise ValueError("Unexpected pilot or forbidden Cabauw rows.")
    if not data.cohort_origin.eq("expanded").all() or not data.label_product.eq("landsat_c2_l2").all():
        raise ValueError("Require expanded Landsat C2 L2 observations.")
    if data.sample_id.isin(old.sample_id).any() or data.acquisition_id.isin(old.acquisition_id).any():
        raise ValueError("New IDs overlap the original Option-B cohort.")
    old_keys = set(zip(old.region_id, pd.to_datetime(old.datetime_utc, utc=True).dt.strftime("%Y-%m-%d")))
    if set(zip(data.region_id, dates)) & old_keys:
        raise ValueError("New pilot/date overlaps an original observation, including held-out/night rows.")
    if data.assign(day=dates).groupby(["region_id", "day"]).acquisition_id.nunique().gt(1).any():
        raise ValueError("Multiple acquisitions on one new pilot/date.")
    for rid, part in data.groupby("region_id"):
        a = areas[rid]
        x, y = Transformer.from_crs(4326, a["epsg"], always_xy=True).transform(part.longitude, part.latitude)
        error = np.hypot(np.asarray(x)-(a["extent_m"][0]+(part.grid_col.to_numpy()+.5)*100),
                         np.asarray(y)-(a["extent_m"][3]-(part.grid_row.to_numpy()+.5)*100))
        if not np.isfinite(error).all() or (error > 1).any():
            raise ValueError("New coordinates must match the fixed cell centre within 1 m.")
    return data


def fit_only_rows(admitted, excluded, new_ids):
    """Apply frozen fit restrictions after the existing source-admission stage."""
    data = pd.concat([admitted, excluded], ignore_index=True)
    data = data.loc[data.sample_id.isin(set(new_ids))].copy()
    if len(data) != len(new_ids) or data.sample_id.duplicated().any():
        raise ValueError("Merge did not preserve every new sample identity exactly once.")
    reason = data.research_admissibility_reason.copy()
    masks = {
        "spatial_holdout": data.spatial_holdout.eq(True),
        "in_holdout_buffer": data.in_holdout_buffer.eq(True),
        "incomplete_base40": ~f._complete(data, f.BASE_FEATURES),
        "unverified_actual_station_report": ~data.verified_station_report_status.isin(VERIFIED),
        "solar_elevation_below_ten_degrees": ~data.solar_elevation_deg.ge(10),
    }
    for label, mask in masks.items():
        reason.loc[reason.eq("") & mask] = label
    data["new_fit_exclusion_reason"] = reason
    keep = reason.eq("")
    selected = data.loc[keep].copy()
    if not selected.research_admissibility_reason.eq("").all():
        raise AssertionError("Fit rows must retain explicit empty source-admission reason.")
    audit = {"input_rows": len(data), "fit_rows": int(keep.sum()),
             "input_dates": int(pd.to_datetime(data.datetime_utc, utc=True).dt.floor("D").nunique()),
             "fit_dates": int(pd.to_datetime(selected.datetime_utc, utc=True).dt.floor("D").nunique()),
             "primary_exclusion_counts": reason[~keep].value_counts().to_dict(),
             "independent_failure_counts": {k: int(np.asarray(v).sum()) for k, v in masks.items()},
             "station_status_counts": data.verified_station_report_status.value_counts().to_dict()}
    date_records = []
    for (rid, day), part in data.groupby([data.region_id, pd.to_datetime(data.datetime_utc, utc=True).dt.strftime("%Y-%m-%d")]):
        date_records.append({"region_id": rid, "utc_date": day, "sampled_rows": len(part),
                             "fit_rows": int(part.new_fit_exclusion_reason.eq("").sum()),
                             "exclusions": part.loc[part.new_fit_exclusion_reason.ne(""), "new_fit_exclusion_reason"].value_counts().to_dict()})
    audit["dates"] = date_records
    return selected, data.loc[~keep].copy(), audit


def verify_isd_rows(frame, observations):
    """Match raw QC1 report time/value and original provenance, cache-only."""
    part = frame.loc[frame.station_id.eq(legacy_isd.STATION_ID)].copy()
    times = (pd.to_datetime(part.datetime_utc, utc=True)-pd.to_timedelta(part.station_age_minutes, unit="m")).dt.round("us")
    matched = observations.set_index("isd_observation_datetime_utc").reindex(pd.DatetimeIndex(times))
    ok = (part.station_age_minutes.between(0, 90).to_numpy()
          & part.station_distance_km.between(0, 100).to_numpy()
          & np.equal(matched.isd_temperature_c.to_numpy(), part.observed_station_air_temperature_c.to_numpy())
          & matched.isd_quality_code.eq("1").to_numpy()
          & np.equal(matched.isd_raw_sha256.to_numpy(), part.station_raw_sha256.to_numpy())
          & np.equal(matched.isd_raw_url.to_numpy(), part.station_raw_url.to_numpy())
          & part.air_temperature_source.eq(legacy_isd.AIR_SOURCE).to_numpy())
    result = frame.copy()
    result.loc[part.index, "verified_station_report_status"] = "legacy_isd_report_time_value_or_provenance_mismatch"
    good = part.index[ok]
    result.loc[good, "verified_station_report_status"] = "exact_cached_legacy_isd_qc1_report_verified"
    result.loc[good, "verified_station_observation_datetime_utc"] = times.loc[good].to_numpy()
    mapping = {"isd_raw_sha256": "verified_station_raw_sha256", "isd_raw_url": "verified_station_raw_url",
               "isd_source_code": "verified_station_temperature_source_code", "isd_quality_code": "verified_station_temperature_quality_code",
               "isd_report_type": "verified_station_report_type"}
    for src, dst in mapping.items():
        result.loc[good, dst] = matched.loc[np.asarray(ok), src].to_numpy()
    result.loc[good, "verified_station_parser_sha256"] = f._sha(legacy_isd.__file__)
    result.loc[good, "verified_station_dataset"] = "NOAA Global Hourly / legacy ISD"
    original = [c for c in frame if not c.startswith("verified_station_")]
    pd.testing.assert_frame_equal(frame[original], result[original], check_exact=True)
    return result


def station_audit(paired_path, output, cache, root):
    spec = importlib.util.spec_from_file_location("more_days_ghcnh_audit", root/"reports/night_replacement/audit_option_b_stations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = pd.read_parquet(paired_path)
    audit_input = paired_path
    isd_provenance = None
    if original.region_id.eq(legacy_isd.REGION_ID).any():
        # This reuses the previously reviewed Darwin-only QC1 policy, unchanged.
        changed, isd_provenance = legacy_isd.apply_darwin_fallback(original, cache, root/"runs/darwin_qc_audit/legacy_isd")
        changed["station_pair_available"] = changed.station_id.fillna("").ne("")
        changed["features_A_complete"] = f._complete(changed, f.BASE_FEATURES)
        changed["features_B_complete"] = changed.features_A_complete & f._complete(changed, f.MEMORY_FEATURES)
        changed["features_C_complete"] = changed.features_A_complete & f._complete(changed, f.COVER_FEATURES)
        changed["features_D_complete"] = changed.features_B_complete & changed.features_C_complete
        output.mkdir(parents=True, exist_ok=True)
        audit_input = output/"features_darwin_qc1.parquet"
        changed.to_parquet(audit_input, index=False)
        f._write_json(output/"darwin_qc1_pairing.json", isd_provenance)
    audited, report = module.audit(audit_input, output/"ghcnh", cache)
    if isd_provenance is not None:
        stamps = pd.to_datetime(audited.datetime_utc, utc=True)
        years = set(stamps.dt.year) | set((stamps-pd.Timedelta(minutes=90)).dt.year)
        observations, archive = legacy_isd.load_isd_archive(root/"runs/darwin_qc_audit/legacy_isd", years)
        audited = verify_isd_rows(audited, observations)
        report["legacy_isd_cache_audit"] = archive
    destination = output/"features_station_audited.parquet"
    audited.to_parquet(destination, index=False)
    report.update(output_path=str(destination), output_sha256=f._sha(destination),
                  statuses=audited.verified_station_report_status.value_counts().to_dict(),
                  orchestration_sha256=f._sha(__file__), reused_darwin_adapter=isd_provenance is not None)
    f._write_json(output/"station_source_audit.json", report)
    return destination


def process_batch(completion, experiment, root, areas_path, old, legacy, cache):
    rid = completion["region_id"]
    output = experiment/"pairing"/rid
    record_path = output/"completion.json"
    source = Path(completion["path"])
    if f._sha(source) != completion["sha256"]:
        raise ValueError("Immutable acquisition source hash changed.")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["source_sha256"] != completion["sha256"] or (record.get("fit_path") and f._sha(record["fit_path"]) != record["fit_sha256"]):
            raise ValueError("Existing paired completion no longer matches its inputs/outputs.")
        return record
    output.mkdir(parents=True, exist_ok=True)
    if completion["rows"] == 0:
        record = {"region_id": rid, "source_sha256": completion["sha256"], "input_rows": 0, "fit_rows": 0,
                  "status": "completed_no_acquisition_rows"}
        f._write_json(record_path, record)
        return record
    areas = {a["id"]: a for a in json.loads(areas_path.read_text())["areas"]}
    samples = verify_new_identity(pd.read_parquet(source), areas, old, rid)
    start = time.monotonic()
    subprocess.run([sys.executable, "-m", "lst_pilot.option_b_parallel", "--input", str(source),
                    "--output-dir", str(output/"features"), "--areas", str(areas_path), "--cache", str(cache), "--workers", "4"], check=True)
    audited_path = output/"station_audit/features_station_audited.parquet"
    if not audited_path.exists():
        audited_path = station_audit(output/"features/features.parquet", output/"station_audit", cache, root)
    admission = output/"admission"
    if not (admission/"paired_input.parquet").exists():
        option_b_merge.merge_cohort(legacy, [audited_path], areas_path, admission)
    admitted = pd.read_parquet(admission/"paired_input.parquet")
    excluded = pd.read_parquet(admission/"excluded_rows.parquet")
    fit, lost, record = fit_only_rows(admitted, excluded, samples.sample_id.to_list())
    verify_new_identity(fit, areas, old, rid) if len(fit) else None
    destination = output/"new_fit_only.parquet"
    fit.to_parquet(destination, index=False)
    lost.to_parquet(output/"new_fit_excluded.parquet", index=False)
    record.update(region_id=rid, status="completed", source_path=str(source), source_sha256=completion["sha256"],
                  source_completion=completion, paired_path=str(audited_path), paired_sha256=f._sha(audited_path),
                  fit_path=str(destination), fit_sha256=f._sha(destination),
                  excluded_path=str(output/"new_fit_excluded.parquet"), excluded_sha256=f._sha(output/"new_fit_excluded.parquet"),
                  elapsed_seconds=time.monotonic()-start, source_admission_sha256=f._sha(admission/"cohort_manifest.json"),
                  station_audit_sha256=f._sha(output/"station_audit/station_source_audit.json"),
                  orchestration_sha256=f._sha(__file__))
    f._write_json(record_path, record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"dates", "source_completion"}}), flush=True)
    return record


def run(experiment, root, legacy, cache, *, wait_seconds=30, max_wait_hours=8):
    root, experiment, legacy, cache = map(lambda p: Path(p).resolve(), [root, experiment, legacy, cache])
    areas_path = root/"pilot/areas_resolved.json"
    plan_path = experiment/"acquisition/plan.json"
    plan = json.loads(plan_path.read_text())
    prior_path = Path(plan["old_cohort"])
    if f._sha(prior_path) != plan["old_cohort_sha256"]:
        raise ValueError("Original cohort identity hash differs from frozen acquisition plan.")
    old = pd.read_parquet(prior_path, columns=["region_id", "datetime_utc", "sample_id", "acquisition_id"])
    ids = [g["region"]["id"] for g in plan["groups"]]
    started = time.monotonic()
    complete = {}
    while len(complete) < len(ids):
        progress = False
        for rid in ids:
            if rid in complete:
                continue
            path = experiment/"acquisition"/rid/"completion.json"
            if path.exists():
                record = process_batch(json.loads(path.read_text()), experiment, root, areas_path, old, legacy, cache)
                complete[rid] = record
                progress = True
        if len(complete) == len(ids):
            break
        if time.monotonic()-started > max_wait_hours*3600:
            raise TimeoutError("Acquisition completion wait expired; completed pairing checkpoints retained.")
        if not progress:
            time.sleep(wait_seconds)
    paths = [Path(complete[rid]["fit_path"]) for rid in ids if complete[rid].get("fit_path")]
    data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if len(data):
        areas = {a["id"]: a for a in json.loads(areas_path.read_text())["areas"]}
        verify_new_identity(data, areas, old)
    output = experiment/"paired_new_fit_only.parquet"
    data.to_parquet(output, index=False)
    f._write_json(experiment/"pairing_manifest.json", {"version": VERSION, "source_sha256": f._sha(__file__),
                  "plan_sha256": f._sha(plan_path), "original_cohort_identity_source_sha256": f._sha(prior_path),
                  "legacy_admission_reference_path": str(legacy), "legacy_admission_reference_sha256": f._sha(legacy),
                  "fit_path": str(output), "fit_sha256": f._sha(output), "fit_rows": len(data),
                  "per_pilot": [complete[rid] for rid in ids], "workers_global_max": 4,
                  "input_years": [2021, 2022], "old_rows_combined": False, "model_fitted": False,
                  "fit_policy": "Existing source admission; recomputed spatial flags; no reserved/buffer cells; complete frozen base40; exact audited actual station; solar elevation >=10 degrees; independent new pilot/date and sample/acquisition IDs."})
    print(json.dumps({"completed": True, "fit_path": str(output), "fit_sha256": f._sha(output), "rows": len(data)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--legacy", required=True)
    p.add_argument("--cache", default="cache")
    a = p.parse_args()
    run(a.experiment, a.root, a.legacy, a.cache)
