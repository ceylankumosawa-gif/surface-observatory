"""Freeze completed, source-audited fine batches without resampling observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import multisensor_pair as pairing

F = pairing.features
KINDS = ("new_fine_fit", "new_fine_evaluation", "excluded")


def distinct_rows(parts):
    data = pd.concat(parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    keys = ["region_id", "acquisition_id", "grid_row", "grid_col"]
    if data.sample_id.duplicated().any() or data.duplicated(keys).any():
        raise ValueError("Fine batches contain duplicate sample identities or physical observations.")
    return data


def combine(directories, output, root):
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError("A combined cohort is immutable; choose a new output.")
    if not directories or len(set(map(str, directories))) != len(directories):
        raise ValueError("Require distinct, completed source batches.")
    parts = {kind: [] for kind in KINDS}
    audits = []
    registries = set()
    areas_path = root/"pilot/areas_resolved.json"
    areas = {a["id"]: a for a in json.loads(areas_path.read_text())["areas"]}
    for directory in map(Path, directories):
        completion = json.loads((directory/"completion.json").read_text())
        signature = json.loads((directory/"signature.json").read_text())
        if completion["signature_sha256"] != F._sha(directory/"signature.json"):
            raise ValueError("A completed pairing signature changed.")
        registries.add(signature["registry_sha256"])
        pairing.inspect_time_boundary(directory/"source_samples.parquet")
        sources = pd.read_parquet(directory/"source_samples.parquet")
        feature_manifest = json.loads((directory/"features/manifest.json").read_text())
        if (signature["areas_sha256"] != F._sha(areas_path)
                or feature_manifest["input_sha256"] != F._sha(directory/"source_samples.parquet")
                or feature_manifest["output_sha256"] != F._sha(directory/"features/features.parquet")
                or Path(feature_manifest["input_path"]).resolve() != (directory/"source_samples.parquet").resolve()
                or Path(feature_manifest["output_path"]).resolve() != (directory/"features/features.parquet").resolve()
                or feature_manifest.get("complete_input_processed") is not True):
            raise ValueError("Area or feature input/output provenance changed.")
        station_path = pairing.station_checkpoint(directory, sources, directory/"features/features.parquet", root)
        if station_path is None:
            raise ValueError("Completed pairing lacks a verified station checkpoint.")
        # The first v1 engineering batch is preserved as rejected evidence only.
        total_admitted = sum(completion["outputs"][name]["rows"] for name in KINDS[:2])
        if total_admitted and (signature["version"] != pairing.VERSION or signature["source_sha256"] != F._sha(pairing.__file__)):
            raise ValueError("Admitted rows require the reviewed, unchanged pairing adapter.")
        if total_admitted:
            expected_dependencies = {name: F._sha(Path(pairing.__file__).with_name(name)) for name in
                                     ["option_b_features.py", "option_b_parallel.py", "more_days_pair.py", "option_b_cohort.py", "legacy_isd.py"]}
            if (signature["dependencies_sha256"] != expected_dependencies
                    or signature["raw_station_audit_script_sha256"] != F._sha(root/"reports/night_replacement/audit_option_b_stations.py")):
                raise ValueError("Admitted pairing dependencies changed.")
        actual_partitions = pairing.source_admission(pd.read_parquet(station_path), areas)[:3]
        recomputed = dict(zip(KINDS, actual_partitions))
        ids = []
        for kind in KINDS:
            record = completion["outputs"][kind]
            path = Path(record["path"])
            if path.resolve() != (directory/(kind+".parquet")).resolve() or F._sha(path) != record["sha256"]:
                raise ValueError("A paired cohort output changed or escaped its batch.")
            part = pd.read_parquet(path)
            if len(part) != record["rows"]:
                raise ValueError("A paired cohort row count changed.")
            if kind != "excluded" and not part.research_admissibility_reason.eq("").fillna(False).all():
                raise ValueError("An admitted cohort contains an unresolved exclusion.")
            # Parquet round-trips can convert empty/object string columns to
            # pandas' string extension dtype; every actual value stays exact.
            pd.testing.assert_frame_equal(part.reset_index(drop=True), recomputed[kind].reset_index(drop=True),
                                          check_exact=True, check_dtype=False)
            ids.extend(part.sample_id.tolist())
            parts[kind].append(part)
        if len(ids) != len(set(ids)) or set(ids) != set(sources.sample_id) or len(ids) != completion["input_rows"]:
            raise ValueError("Batch partitions do not exactly preserve every source identity once.")
        audits.append({"path": str(directory.resolve()), "completion_sha256": F._sha(directory/"completion.json"),
                       "signature_sha256": completion["signature_sha256"],
                       "feature_manifest_sha256": F._sha(directory/"features/manifest.json"),
                       "station_report_sha256": F._sha(directory/"station_audit/station_source_audit.json"),
                       "source_rows": len(sources), "partitions_and_values_recomputed_from_verified_station_table": True})
    if len(registries) != 1:
        raise ValueError("Batches disagree on previously inspected dates.")
    joined = {kind: distinct_rows(values) for kind, values in parts.items()}
    # Also reject collisions across fit, evaluation and exclusions, not only within.
    all_rows = distinct_rows(list(joined.values()))
    output.mkdir(parents=True)
    report = {"source_sha256": F._sha(__file__), "pairing_sha256": F._sha(pairing.__file__),
              "registry_sha256": registries.pop(), "input_batches": audits,
              "total_source_rows": len(all_rows), "outputs": {}}
    for kind, frame in joined.items():
        path = output/(kind+".parquet")
        frame.to_parquet(path, index=False)
        date_counts = frame.assign(utc_date=pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D"))
        groups = []
        for (region, phase), group in date_counts.groupby(["region_id", "phase"]):
            groups.append({"region_id": region, "phase": phase, "rows": len(group),
                           "utc_dates": int(group.utc_date.nunique()), "acquisitions": int(group.acquisition_id.nunique())})
        report["outputs"][kind] = {"path": str(path.resolve()), "sha256": F._sha(path),
                                  "rows": len(frame), "support": groups}
    report["source_reason_counts"] = all_rows.research_admissibility_reason.value_counts().to_dict()
    F._write_json(output/"manifest.json", report)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    combine(args.batches, args.output, args.root)
