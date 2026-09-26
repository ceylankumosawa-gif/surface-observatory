"""Freeze explicit sample eligibility before the first heldout assessment.

Unknown climate is NOT equivalent to water. An independent 20-point WorldCover
audit found both frozen-water contamination and legitimate coastal land among
these rows. They stay quarantined until climate and land support are resolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .context import CLIMATE_CODES


def select_eligible(frame):
    if not {"climate_class", "water_fraction"}.issubset(frame):
        raise ValueError("Climate and QA water fraction are required.")
    data = frame.copy()
    water = pd.to_numeric(data.water_fraction, errors="raise")
    if (water.notna() & ~water.between(0, 1)).any():
        raise ValueError("Invalid water fraction.")
    climate_ok = data.climate_class.isin(CLIMATE_CODES[1:])
    water_ok = water.eq(0)
    reasons = pd.Series("", index=data.index)
    reasons.loc[~climate_ok] = "unresolved_climate_or_coastal_support"
    reasons.loc[~water_ok] += ";nonzero_or_unknown_QA_water_fraction"
    keep = climate_ok & water_ok
    quarantine = data.loc[~keep].copy()
    quarantine["quarantine_reason"] = reasons.loc[~keep].str.strip(";")
    eligible = data.loc[keep].copy()
    if eligible.empty:
        raise ValueError("No samples meet the frozen eligibility criteria.")
    return eligible, quarantine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Choose a new output file; preserve source and prior runs.")
    data = pd.read_parquet(source)
    eligible, quarantine = select_eligible(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path = output.with_suffix(".quarantine.parquet")
    eligible.to_parquet(output, index=False)
    quarantine.to_parquet(quarantine_path, index=False)
    audit = {
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_rows": len(data), "eligible_rows": len(eligible), "quarantined_rows": len(quarantine),
        "quarantined_by_region": quarantine.region_id.value_counts().to_dict(),
        "quarantine_file": str(quarantine_path),
        "criterion": "QA water_fraction exactly zero AND a recognised Köppen class; frozen before model assessment",
        "reason": "Twenty independent WorldCover point checks found both permanent water and real coastal land in unknown-climate samples. These unresolved rows are quarantined, not relabelled ocean.",
        "scope": "Temporary pilot eligibility; does not establish an independent 100 m coastline/land-fraction mask. Terrestrial snow with known climate is retained.",
        "evidence": "runs/data_audit/unknown_climate_mask/summary.json",
        "worldcover_documentation": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf",
    }
    output.with_suffix(".eligibility.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
