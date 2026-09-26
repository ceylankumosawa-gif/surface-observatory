"""Pair new fine-resolution sources without changing frozen numerical adapters.

Coarse satellite products cannot enter this label table. Source engineering
must establish QA and native support before the feature stage is invoked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import more_days_pair, option_b_features as features
from .option_b_cohort import spatial_flags

VERSION = "multisensor-fine-pair-v2"
PRODUCTS = frozenset(("ecostress_v2", "aster_ast08_v004"))
PROOF_FIELDS = ("cohort_origin", "source_screen_pass", "label_source_sha256", "native_fit_support_pass",
                "fresh_2023", "freshness_audit_sha256", "day_night")


def inspect_time_boundary(path):
    """Inspect identities/times before loading any thermal target columns."""
    meta = pd.read_parquet(path, columns=["region_id", "datetime_utc", "label_product"])
    stamps = pd.to_datetime(meta.datetime_utc, utc=True, errors="raise")
    if meta.empty or stamps.isna().any() or not stamps.dt.year.isin([2021, 2022, 2023]).all():
        raise ValueError("Only nonempty 2021–2023 fine observations may enter pairing.")
    if not meta.label_product.isin(PRODUCTS).all():
        raise ValueError("Coarse or unknown products cannot enter the fine-label table.")
    return meta


def validate_identity(frame, areas, old, registry, registry_sha256):
    data = features.validate_samples(frame, areas)
    required = {"cohort_origin", "source_screen_pass", "label_source_sha256", "native_fit_support_pass"}
    if not required.issubset(data):
        raise ValueError(f"Fine source proof is incomplete: {sorted(required-set(data))}")
    if not data.datetime_utc.dt.year.isin([2021, 2022, 2023]).all():
        raise ValueError("Reserved years cannot enter new fine pairing.")
    if not data.label_product.isin(PRODUCTS).all() or not data.cohort_origin.eq("expanded").all():
        raise ValueError("Only explicitly expanded ECOSTRESS/ASTER fine sources are supported.")
    if not data.label_source_sha256.astype("string").str.fullmatch(r"[0-9a-f]{64}").fillna(False).all():
        raise ValueError("Every label needs its verified source artifact hash.")
    if not re.fullmatch(r"[0-9a-f]{64}", registry_sha256):
        raise ValueError("A hash-bound previously inspected date registry is required.")
    if data.sample_id.isin(old.sample_id).any():
        raise ValueError("New sample identities overlap the previous research cohort.")
    keys = ["region_id", "acquisition_id", "grid_row", "grid_col"]
    if data.duplicated(keys).any():
        raise ValueError("A physical acquisition/cell occurs more than once.")
    old_keys = pd.MultiIndex.from_frame(old[keys])
    if pd.MultiIndex.from_frame(data[keys]).isin(old_keys).any():
        raise ValueError("A previously used physical observation has a new sample identity.")
    for rid, part in data.groupby("region_id"):
        area = areas[rid]
        x, y = Transformer.from_crs(4326, area["epsg"], always_xy=True).transform(part.longitude, part.latitude)
        distance = np.hypot(np.asarray(x)-(area["extent_m"][0]+(part.grid_col.to_numpy()+.5)*100),
                            np.asarray(y)-(area["extent_m"][3]-(part.grid_row.to_numpy()+.5)*100))
        if not np.isfinite(distance).all() or (distance > 1).any():
            raise ValueError("A fine observation is not at its fixed 100 m cell centre.")
    seen = {(r["region_id"], r["utc_date"]) for r in registry["dates"]}
    dates = data.datetime_utc.dt.strftime("%Y-%m-%d")
    data["fresh_2023"] = [stamp.year == 2023 and (rid, day) not in seen
                           for rid, day, stamp in zip(data.region_id, dates, data.datetime_utc)]
    data["freshness_audit_sha256"] = registry_sha256
    return data


def source_admission(frame, areas):
    """Preserve every input row and report explicit, ordered rejection reasons."""
    data = spatial_flags(frame, areas)
    stamp = pd.to_datetime(data.datetime_utc, utc=True)
    reasons = pd.Series("", index=data.index, dtype="object")

    def reject(mask, reason):
        reasons.loc[reasons.eq("") & pd.Series(mask, index=data.index).fillna(True)] = reason

    def passed(name):
        return data.get(name, pd.Series(False, index=data.index)).eq(True).fillna(False)

    reject(~np.isfinite(pd.to_numeric(data.lst_c, errors="coerce")), "nonfinite_label")
    reject(~data.label_product.isin(PRODUCTS), "not_an_approved_fine_product")
    reject(~passed("source_screen_pass"), "fine_source_screen_incomplete")
    reject(~data.worldcover_land_fraction.ge(.8), "insufficient_independent_land_fraction")
    reject(~data.worldcover_water_fraction.le(.05), "independent_water_fraction_exceeds_five_percent")
    reject(~features._complete(data, features.BASE_FEATURES), "incomplete_base40")
    reject(data.climate_class.isna() | data.climate_class.astype(str).str.strip().str.lower().isin(
        ["", "unknown", "__unknown__", "nan", "none", "0"]), "unknown_climate")
    reject(~passed("station_pair_available"), "no_actual_station_pair")
    reject(~data.station_age_minutes.between(0, 90), "station_time_mismatch")
    reject(~data.station_distance_km.between(0, 100), "station_distance_mismatch")
    reject(~data.verified_station_report_status.isin(more_days_pair.VERIFIED), "unverified_actual_station_report")
    for column in ("optical_earliest_source_utc", "optical_latest_source_utc"):
        times = pd.to_datetime(data[column], utc=True)
        reject(times.isna() | times.gt(stamp) | times.lt(stamp-pd.Timedelta(days=32)), "missing_future_or_stale_optical")
    reject(pd.to_datetime(data.optical_earliest_source_utc, utc=True).gt(
        pd.to_datetime(data.optical_latest_source_utc, utc=True)), "inverted_optical_time_interval")
    phase = pd.Series(np.select([data.solar_elevation_deg.ge(10), data.solar_elevation_deg.le(-6)],
                                 ["day", "night"], default="twilight"), index=data.index)
    reject(phase.eq("twilight"), "unsupported_twilight")
    if "day_night" in data:
        reject(~data.day_night.eq(phase), "declared_phase_disagrees_with_per_cell_solar_geometry")
    data["phase"] = phase
    data["research_admissibility_reason"] = reasons
    # Source-admissible spatial/date diagnostics are retained, even when not fit eligible.
    fit_reason = reasons.copy()
    checks = [(~stamp.dt.year.isin([2021, 2022]), "evaluation_year"),
              (data.region_id.eq("cabauw"), "withheld_region"),
              (data.spatial_holdout.eq(True), "spatial_holdout"),
              (data.in_holdout_buffer.eq(True), "holdout_buffer"),
              (~passed("native_fit_support_pass"), "native_support_intersects_geographic_exclusion")]
    for mask, reason in checks:
        fit_reason.loc[fit_reason.eq("") & mask] = reason
    data["new_fit_exclusion_reason"] = fit_reason
    eval_reason = reasons.copy()
    for mask, reason in [(~stamp.dt.year.eq(2023), "not_new_evaluation_year"),
                         (data.in_holdout_buffer.eq(True), "holdout_buffer")]:
        eval_reason.loc[eval_reason.eq("") & mask] = reason
    data["new_evaluation_exclusion_reason"] = eval_reason
    data["split"] = np.select([data.region_id.eq("cabauw"), data.spatial_holdout, data.in_holdout_buffer,
                                stamp.dt.year.isin([2021, 2022]), stamp.dt.month.le(6)],
                               ["heldout_region", "heldout_spatial", "buffer", "fit", "development"], default="calibration")
    fit = data.loc[fit_reason.eq("")].copy()
    evaluation = data.loc[eval_reason.eq("")].copy()
    excluded = data.loc[~data.sample_id.isin(set(fit.sample_id) | set(evaluation.sample_id))].copy()
    return fit, evaluation, excluded, data


def station_checkpoint(output, samples, features_path, root):
    """Accept only a complete station-audit checkpoint bound to its real input."""
    directory = Path(output)/"station_audit"
    destination = directory/"features_station_audited.parquet"
    report_path = directory/"station_source_audit.json"
    if not destination.exists() or not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    darwin = samples.region_id.eq(more_days_pair.legacy_isd.REGION_ID).any()
    expected_input = directory/"features_darwin_qc1.parquet" if darwin else Path(features_path)
    if (report.get("reused_darwin_adapter") is not bool(darwin)
            or not expected_input.exists()
            or report.get("input_sha256") != features._sha(expected_input)
            or report.get("output_sha256") != features._sha(destination)
            or report.get("audit_code_sha256") != features._sha(Path(root)/"reports/night_replacement/audit_option_b_stations.py")
            or report.get("orchestration_sha256") != features._sha(more_days_pair.__file__)):
        raise ValueError("Station audit checkpoint identity, source hash or completeness changed.")
    if darwin and (not (directory/"darwin_qc1_pairing.json").exists() or not report.get("legacy_isd_cache_audit")):
        raise ValueError("Darwin station checkpoint lacks its separately audited fallback provenance.")
    audited = pd.read_parquet(destination)
    ordered = features.preserve_identity(samples, audited)
    for field in PROOF_FIELDS:
        if field in samples:
            pd.testing.assert_series_equal(samples[field].reset_index(drop=True), ordered[field].reset_index(drop=True), check_exact=True)
    original = pd.read_parquet(expected_input)
    unchanged = [c for c in original if not c.startswith("verified_station_")]
    after = audited.set_index("sample_id").loc[original.sample_id].reset_index()
    pd.testing.assert_frame_equal(original[unchanged].reset_index(drop=True), after[unchanged].reset_index(drop=True), check_exact=True)
    return destination


def run(samples, output, registry_path, old_paths, root, cache):
    samples, output, registry_path, root, cache = map(Path, (samples, output, registry_path, root, cache))
    inspect_time_boundary(samples)
    areas_path = root/"pilot/areas_resolved.json"
    areas = {a["id"]: a for a in json.loads(areas_path.read_text())["areas"]}
    registry = json.loads(registry_path.read_text())
    old = pd.concat([pd.read_parquet(p, columns=["sample_id", "region_id", "acquisition_id", "grid_row", "grid_col"])
                     for p in old_paths], ignore_index=True)
    data = validate_identity(pd.read_parquet(samples), areas, old, registry, features._sha(registry_path))
    output.mkdir(parents=True, exist_ok=True)
    signature = {"version": VERSION, "samples_sha256": features._sha(samples),
                 "registry_sha256": features._sha(registry_path), "areas_sha256": features._sha(areas_path),
                 "source_sha256": features._sha(__file__),
                 "dependencies_sha256": {name: features._sha(Path(__file__).with_name(name)) for name in
                                          ["option_b_features.py", "option_b_parallel.py", "more_days_pair.py", "option_b_cohort.py", "legacy_isd.py"]},
                 "raw_station_audit_script_sha256": features._sha(root/"reports/night_replacement/audit_option_b_stations.py"),
                 "old_id_sources": {str(p): features._sha(p) for p in old_paths}}
    marker = output/"signature.json"
    if marker.exists() and json.loads(marker.read_text()) != signature:
        raise ValueError("Pairing checkpoint inputs or code changed; use a new output directory.")
    features._write_json(marker, signature)
    prepared = output/"source_samples.parquet"
    if prepared.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(prepared), data, check_exact=True)
    else:
        data.to_parquet(prepared, index=False)
    subprocess.run([sys.executable, "-m", "lst_pilot.option_b_parallel", "--input", str(prepared),
                    "--output-dir", str(output/"features"), "--areas", str(areas_path), "--cache", str(cache), "--workers", "4"], check=True)
    paired_path = output/"features/features.parquet"
    audited = station_checkpoint(output, data, paired_path, root)
    if audited is None:
        more_days_pair.station_audit(paired_path, output/"station_audit", cache, root)
        audited = station_checkpoint(output, data, paired_path, root)
        if audited is None:
            raise ValueError("Station audit did not produce a complete verified checkpoint.")
    fit, evaluation, excluded, all_rows = source_admission(pd.read_parquet(audited), areas)
    result = {"signature_sha256": features._sha(marker), "input_rows": len(data),
              "source_reason_counts": all_rows.research_admissibility_reason.value_counts().to_dict(),
              "fit_reason_counts": all_rows.new_fit_exclusion_reason.value_counts().to_dict(), "outputs": {}}
    for name, frame in [("new_fine_fit", fit), ("new_fine_evaluation", evaluation), ("excluded", excluded)]:
        path = output/(name+".parquet")
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
        else:
            frame.to_parquet(path, index=False)
        result["outputs"][name] = {"path": str(path.resolve()), "sha256": features._sha(path), "rows": len(frame),
                                  "pilot_dates": len(frame.assign(utc_date=pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D"))
                                                       [["region_id", "utc_date"]].drop_duplicates())}
    features._write_json(output/"completion.json", result)
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "output", "registry", "root", "cache"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--old-inputs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    run(args.samples, args.output, args.registry, args.old_inputs, args.root, args.cache)


if __name__ == "__main__":
    main()
