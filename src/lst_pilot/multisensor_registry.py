"""Metadata-only registry for previous dates and causal context target times."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import pandas as pd

from .option_b_features import _sha, _write_json


def build(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Use a new immutable registry directory.")
    dates = defaultdict(set)
    provenance = []

    def add(region, stamp, source):
        value = pd.Timestamp(stamp)
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        if value.year in (2021, 2022, 2023):
            dates[(region, value.strftime("%Y-%m-%d"))].add(source)

    inputs = ["runs/option_b_retrain_20260909/cohort_final_v1/paired_input.parquet",
              "runs/more_days_20260909_v1/paired_new_fit_only.parquet"]
    for relative in inputs:
        path = root/relative
        frame = pd.read_parquet(path, columns=["region_id", "datetime_utc"])
        for row in frame.drop_duplicates().itertuples(index=False):
            add(row.region_id, row.datetime_utc, relative)
        provenance.append({"path": str(path), "sha256": _sha(path), "columns_read": ["region_id", "datetime_utc"]})
    audits = ["runs/pilot_v1_satellite/satellite_manifest.json",
              "runs/option_b_retrain_20260909/acquisition_day/audit.json",
              "runs/option_b_retrain_20260909/acquisition_night/manifest.json"]
    for relative in audits:
        path = root/relative
        audit = json.loads(path.read_text())
        records = audit.get("records", audit.get("scenes", []))+audit.get("failures", [])
        for record in records:
            region = record.get("region_id", record.get("pilot_id"))
            stamp = record.get("datetime_utc", record.get("time_start", record.get("utc_date")))
            if stamp is None:
                match = re.search(r"LC0[89]_L2\w+_\d{6}_(\d{8})_", record.get("scene_id", ""))
                if match:
                    stamp = pd.to_datetime(match.group(1), format="%Y%m%d", utc=True)
            if region is None or stamp is None:
                raise ValueError("A prior acquisition record lacks a resolvable region/date.")
            add(region, stamp, relative)
        provenance.append({"path": str(path), "sha256": _sha(path), "record_count": len(records)})
    registry = {"version": "multisensor-prior-date-registry-v1", "source_sha256": _sha(__file__),
                "scope": "2021–2023 admitted observations and prior acquisition attempts; metadata-only rejected attempts are conservatively included.",
                "thermal_array_columns_read": False, "sources": provenance,
                "dates": [{"region_id": region, "utc_date": stamp, "prior_sources": sorted(sources)}
                          for (region, stamp), sources in sorted(dates.items())]}
    output.mkdir(parents=True)
    _write_json(output/"prior_inspected_dates.json", registry)
    targets = defaultdict(set)
    target_sources = []
    for scope, relative in [
        ("E_fit", "runs/more_days_20260909_v1/model_experiment/fitting_rows.parquet"),
        ("old_pre2024_evaluation", "runs/more_days_20260909_v1/model_experiment/predictions.parquet")]:
        path = root/relative
        frame = pd.read_parquet(path, columns=["region_id", "datetime_utc"])
        for row in frame.drop_duplicates().itertuples(index=False):
            stamp = pd.Timestamp(row.datetime_utc)
            if stamp.tzinfo is None or stamp.year not in (2021, 2022, 2023):
                raise ValueError("Reference targets must have aware UTC times within 2021–2023.")
            targets[(row.region_id, stamp.tz_convert("UTC").isoformat())].add(scope)
        target_sources.append({"scope": scope, "path": str(path), "sha256": _sha(path), "columns_read": ["region_id", "datetime_utc"]})
    area_path = root/"pilot/areas_resolved.json"
    areas = {a["id"]: a for a in json.loads(area_path.read_text())["areas"]}
    target_plan = {"version": "multisensor-reference-targets-v1", "sources": target_sources,
                   "areas_sha256": _sha(area_path), "source_sha256": _sha(__file__),
                   "thermal_array_columns_read": False, "new_2024_or_2025_targets": False,
                   "usage": "Metadata selection of native satellite context at or before each target; observations are not independent 100 m labels.",
                   "targets": [{"region_id": region, "datetime_utc": stamp, "scopes": sorted(scopes),
                               "epsg": areas[region]["epsg"], "extent_m": areas[region]["extent_m"]}
                               for (region, stamp), scopes in sorted(targets.items())]}
    _write_json(output/"reference_targets.json", target_plan)
    summary = {"prior_pilot_dates": len(dates), "prior_2023_pilot_dates": sum(y.startswith("2023-") for _, y in dates),
               "reference_target_times": len(targets),
               "prior_registry_sha256": _sha(output/"prior_inspected_dates.json"),
               "targets_sha256": _sha(output/"reference_targets.json")}
    _write_json(output/"summary.json", summary)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
