"""Cache-only source trace of already paired NOAA reports; never changes predictors.

This audit performs no network calls and does not parse a missing source cache.
It writes a separate table with additional verified_* audit columns. Existing
columns, including all numeric features and selected air temperatures, must be
exactly unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lst_pilot import option_b_features as features, stations


def verify_rows(frame, parsed):
    observations = parsed.dropna(subset=["air_temperature_c"]).sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    by_time = observations.set_index("timestamp_utc")
    times = (pd.to_datetime(frame.datetime_utc, utc=True) - pd.to_timedelta(frame.station_age_minutes, unit="m")).dt.round("us")
    matched = by_time.reindex(pd.DatetimeIndex(times))
    values = pd.to_numeric(frame.observed_station_air_temperature_c, errors="coerce").to_numpy()
    valid = np.isfinite(values) & np.equal(matched.air_temperature_c.to_numpy(), values)
    # Existing station selection accepted only backwards reports. Recheck rather
    # than infer provenance from a plausible temperature at a different time.
    valid &= times.le(pd.to_datetime(frame.datetime_utc, utc=True)).to_numpy()
    valid &= frame.station_age_minutes.between(0, 90).to_numpy()
    out = pd.DataFrame(index=frame.index)
    out["verified_station_report_status"] = np.where(valid, "exact_cached_report_verified", "report_time_or_value_mismatch")
    out["verified_station_observation_datetime_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    out.loc[valid, "verified_station_observation_datetime_utc"] = times.loc[valid].to_numpy()
    mappings = {"air_temperature_c_source_code": "verified_station_temperature_source_code",
                "air_temperature_c_quality_code": "verified_station_temperature_quality_code",
                "air_temperature_c_report_type": "verified_station_report_type"}
    for source, destination in mappings.items():
        out[destination] = np.where(valid, matched[source].fillna("").astype(str).to_numpy(), "") if source in matched else ""
    return out


def audit(input_path, output_dir, cache):
    input_path, output, cache = map(Path, (input_path, output_dir, cache))
    original = pd.read_parquet(input_path)
    if original.sample_id.duplicated().any():
        raise ValueError("Unique sample_id required.")
    if any(str(name).startswith("verified_station_") for name in original):
        raise ValueError("Already audited input; choose the original paired table.")
    data = original.copy()
    data["verified_station_report_status"] = "no_station_pair"
    data["verified_station_raw_sha256"] = ""
    data["verified_station_raw_url"] = ""
    data["verified_station_parser_sha256"] = ""
    data["verified_station_dataset"] = ""
    data["verified_station_observation_datetime_utc"] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    records = []
    usable = data.station_id.fillna("").ne("") & data.station_age_minutes.notna()
    observation_times = pd.to_datetime(data.datetime_utc, utc=True) - pd.to_timedelta(data.station_age_minutes, unit="m")
    groups = data.loc[usable].groupby([data.loc[usable, "station_id"], observation_times.loc[usable].dt.year])
    for (station_id, year), part in groups:
        url = stations.station_year_url(station_id, int(year))
        raw = cache / "stations" / str(int(year)) / url.rsplit("/", 1)[-1]
        record = {"station_id": station_id, "year": int(year), "rows": len(part), "raw_path": str(raw), "raw_url": url}
        if not raw.exists():
            data.loc[part.index, "verified_station_report_status"] = "raw_cache_missing_or_non_GHCNh_source"
            record["status"] = "raw_cache_missing_or_non_GHCNh_source"
            records.append(record)
            continue
        raw_sha, parser_sha = features._sha(raw), features._sha(stations.__file__)
        identity = {"schema_version": 1, "raw_sha256": raw_sha, "parser_source_sha256": parser_sha, "keep_flags": True}
        parser_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        parsed_path = raw.parent / f"{station_id}_parsed_{parser_key}.parquet"
        provenance_path = parsed_path.with_suffix(".provenance.json")
        if not parsed_path.exists() or not provenance_path.exists() or json.loads(provenance_path.read_text()) != identity:
            data.loc[part.index, "verified_station_report_status"] = "matching_parser_cache_missing"
            record["status"] = "matching_parser_cache_missing"
            records.append(record)
            continue
        parsed = pd.read_parquet(parsed_path)
        additions = verify_rows(part, parsed)
        for name in additions:
            if name not in data:
                data[name] = ""
            data.loc[part.index, name] = additions[name].to_numpy()
        valid = additions.verified_station_report_status.eq("exact_cached_report_verified")
        indices = part.index[valid]
        data.loc[indices, "verified_station_raw_sha256"] = raw_sha
        data.loc[indices, "verified_station_raw_url"] = url
        data.loc[indices, "verified_station_parser_sha256"] = parser_sha
        data.loc[indices, "verified_station_dataset"] = "NOAA GHCNh"
        record.update(status="audited", verified_rows=int(valid.sum()), mismatched_rows=int((~valid).sum()),
                      raw_sha256=raw_sha, parser_sha256=parser_sha, parsed_cache_path=str(parsed_path),
                      parsed_cache_sha256=features._sha(parsed_path))
        records.append(record)
    pd.testing.assert_frame_equal(original, data[original.columns], check_exact=True)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "features_station_audited.parquet"
    data.to_parquet(destination, index=False)
    report = {"input_path": str(input_path), "input_sha256": features._sha(input_path), "output_path": str(destination),
              "output_sha256": features._sha(destination), "audit_code_sha256": features._sha(__file__),
              "original_columns_exactly_preserved": True, "network_requests": 0,
              "statuses": data.verified_station_report_status.value_counts().to_dict(), "sources": records,
              "timestamp_matching_rule": "Selected observation time recovered from target minus age and rounded only to 1 microsecond; parsed exact timestamp and observed Celsius value must match."}
    features._write_json(output / "station_source_audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2), flush=True)
    return data, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", default="cache")
    args = parser.parse_args()
    audit(args.input, args.output_dir, args.cache)
