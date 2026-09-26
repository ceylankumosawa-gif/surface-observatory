"""Meaningful final-cohort admission tests; remote, synthetic fixtures only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst_pilot import option_b_merge as merge


def row(sample_id="expanded-good", origin="expanded", **changes):
    result = dict(sample_id=sample_id, cohort_origin=origin, region_id="greater_london",
                  datetime_utc="2021-06-01T12:00:00Z", lst_c=25.0, climate_class="Cfb",
                  worldcover_land_fraction=.99, worldcover_water_fraction=.0,
                  station_pair_available=True, station_age_minutes=20.0, station_distance_km=5.0,
                  source_screen_pass=True, solar_elevation_deg=30.0,
                  optical_earliest_source_utc="2021-05-02T12:00:00Z",
                  optical_latest_source_utc="2021-05-31T12:00:00Z",
                  acquisition_id=sample_id, grid_row=150, grid_col=150,
                  label_product="landsat_c2_l2", features_A_complete=True, features_D_complete=True,
                  spatial_holdout=False, in_holdout_buffer=False, block_id="stale_source_flag")
    result.update(changes)
    return result


def input_files(tmp_path, legacy=None, expanded=None, extra=None):
    legacy = [row("legacy-good", "legacy")] if legacy is None else legacy
    expanded = [row()] if expanded is None else expanded
    lp = tmp_path / "legacy.parquet"
    ep = tmp_path / "expanded.parquet"
    pd.DataFrame(legacy).to_parquet(lp, index=False)
    pd.DataFrame(expanded).to_parquet(ep, index=False)
    sources = [ep]
    if extra is not None:
        more = tmp_path / "expanded_next.parquet"
        pd.DataFrame(extra).to_parquet(more, index=False)
        sources.append(more)
    areas = tmp_path / "areas.json"
    areas.write_text(json.dumps({"areas": [{"id": "greater_london", "epsg": 32630,
                                           "extent_m": [0, 0, 30000, 30000], "grid_shape": [300, 300]}]}))
    return lp, sources, areas


def run(tmp_path, **kwargs):
    lp, sources, areas = input_files(tmp_path, **kwargs)
    output = tmp_path / "merged"
    merge.merge_cohort(lp, sources, areas, output)
    return (pd.read_parquet(output / "paired_input.parquet"),
            pd.read_parquet(output / "excluded_rows.parquet"),
            json.loads((output / "cohort_manifest.json").read_text()))


@pytest.mark.parametrize("field,value,reason", [
    ("lst_c", np.nan, "nonfinite_label"),
    ("climate_class", None, "unknown_climate"),
    ("worldcover_land_fraction", .79, "missing_or_insufficient_independent_land_fraction"),
    ("worldcover_land_fraction", np.nan, "missing_or_insufficient_independent_land_fraction"),
    ("worldcover_water_fraction", .051, "independent_water_fraction_exceeds_five_percent"),
    ("worldcover_water_fraction", np.nan, "independent_water_fraction_exceeds_five_percent"),
    ("station_pair_available", False, "no_actual_station_pair"),
    ("station_pair_available", None, "no_actual_station_pair"),
    ("station_age_minutes", -1, "station_time_mismatch"),
    ("station_age_minutes", 90.1, "station_time_mismatch"),
    ("station_distance_km", -1, "station_distance_mismatch"),
    ("station_distance_km", 100.1, "station_distance_mismatch"),
    ("station_distance_km", np.nan, "station_distance_mismatch"),
])
def test_expanded_admission_gates_have_specific_reasons(tmp_path, field, value, reason):
    admitted, excluded, report = run(tmp_path, expanded=[row(**{field: value})])
    assert admitted.sample_id.tolist() == ["legacy-good"]
    assert excluded.research_admissibility_reason.tolist() == [reason]
    assert report["exclusion_counts"] == {reason: 1}


def test_valid_exact_gate_endpoints_are_retained(tmp_path):
    r = row(worldcover_land_fraction=.8, worldcover_water_fraction=.05,
            station_age_minutes=90, station_distance_km=100,
            optical_earliest_source_utc="2021-04-30T12:00:00Z",
            optical_latest_source_utc="2021-06-01T12:00:00Z")
    admitted, excluded, _ = run(tmp_path, expanded=[r])
    assert len(admitted) == 2
    assert excluded.empty


@pytest.mark.parametrize("column,timestamp", [
    ("optical_earliest_source_utc", None),
    ("optical_latest_source_utc", None),
    ("optical_latest_source_utc", "2021-06-01T12:00:01Z"),
    ("optical_earliest_source_utc", "2021-04-30T11:59:59Z"),
])
def test_optical_sources_must_exist_and_be_past_within_32_days(tmp_path, column, timestamp):
    admitted, excluded, _ = run(tmp_path, expanded=[row(**{column: timestamp})])
    assert admitted.sample_id.tolist() == ["legacy-good"]
    assert excluded.research_admissibility_reason.tolist() == ["missing_future_or_stale_optical"]


def test_inverted_optical_provenance_is_rejected(tmp_path):
    _, excluded, _ = run(tmp_path, expanded=[row(optical_earliest_source_utc="2021-05-31T12:00:00Z",
                                                optical_latest_source_utc="2021-05-02T12:00:00Z")])
    assert len(excluded) == 1
    assert "optical" in excluded.research_admissibility_reason.iloc[0]


@pytest.mark.parametrize("changes,reason", [
    ({"source_screen_pass": False}, "night_source_screen_incomplete"),
    ({"source_screen_pass": None}, "night_source_screen_incomplete"),
    ({"solar_elevation_deg": -5.99}, "night_solar_geometry_not_supported"),
])
def test_night_requires_source_screen_and_per_row_solar_geometry(tmp_path, changes, reason):
    r = row(label_product="ecostress_v2", solar_elevation_deg=-10)
    r.update(changes)
    admitted, excluded, _ = run(tmp_path, expanded=[r])
    assert admitted.sample_id.tolist() == ["legacy-good"]
    assert excluded.research_admissibility_reason.tolist() == [reason]


def test_legacy_retains_disclosed_original_admission_policy(tmp_path):
    old = row("legacy-good", "legacy", worldcover_land_fraction=np.nan,
              worldcover_water_fraction=np.nan, station_pair_available=False,
              station_age_minutes=np.nan, station_distance_km=np.nan,
              optical_earliest_source_utc=None, optical_latest_source_utc=None)
    admitted, excluded, _ = run(tmp_path, legacy=[old])
    assert set(admitted.sample_id) == {"legacy-good", "expanded-good"}
    assert excluded.empty


def test_legacy_then_earlier_batch_wins_duplicate_identity(tmp_path):
    old = row("same", "legacy", lst_c=11)
    duplicate = row("same", lst_c=99)
    first = row("expanded-same", lst_c=22)
    later = row("expanded-same", lst_c=88)
    admitted, excluded, report = run(tmp_path, legacy=[old], expanded=[duplicate, first], extra=[later])
    assert admitted.set_index("sample_id").lst_c.to_dict() == {"same": 11, "expanded-same": 22}
    assert excluded.research_admissibility_reason.str.startswith("duplicate_sample_prefer_legacy").all()
    assert report["exclusion_counts"]["duplicate_sample_prefer_legacy_then_earlier_batch"] == 2


def test_duplicate_physical_observation_with_new_id_is_not_silently_kept(tmp_path):
    old = row("old-id", "legacy", acquisition_id="one-scene")
    other = row("different-id", acquisition_id="one-scene")
    with pytest.raises(ValueError, match="Duplicate physical observations"):
        run(tmp_path, legacy=[old], expanded=[other])


def test_duplicate_inside_one_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicated input sample IDs"):
        run(tmp_path, expanded=[row(), row()])


def test_origin_cannot_be_overridden_by_input_order(tmp_path):
    with pytest.raises(ValueError, match="origin differs"):
        run(tmp_path, expanded=[row(origin="legacy")])


def test_whole_cell_spatial_flags_are_recomputed_but_not_hidden_as_admission(tmp_path):
    # Centre (10750,10750) is 1060.7 m from a reserved corner, but its
    # southwest footprint corner is only 989.95 m away and enters the buffer.
    edge = row("diagonal-edge", grid_col=107, grid_row=192,
               in_holdout_buffer=False, spatial_holdout=False, block_id="wrong")
    reserved = row("reserved", grid_col=50, grid_row=250, spatial_holdout=False)
    admitted, excluded, report = run(tmp_path, expanded=[edge, reserved])
    rows = admitted.set_index("sample_id")
    assert bool(rows.loc["diagonal-edge", "in_holdout_buffer"])
    assert not bool(rows.loc["diagonal-edge", "spatial_holdout"])
    assert bool(rows.loc["reserved", "spatial_holdout"])
    assert not bool(rows.loc["reserved", "in_holdout_buffer"])
    assert rows.loc["diagonal-edge", "block_id"] == "greater_london_r01_c01"
    assert excluded.empty  # Trainer excludes buffers and retains reserved evaluation rows separately.
    new = next(c for c in report["counts"] if c["origin"] == "expanded")
    assert new["reserved_rows"] == 1 and new["buffer_rows"] == 1


def test_completeness_is_a_separate_later_matched_cohort_gate(tmp_path):
    admitted, _, report = run(tmp_path, expanded=[row(features_A_complete=True, features_D_complete=False)])
    assert len(admitted) == 2
    new = next(c for c in report["counts"] if c["origin"] == "expanded")
    assert new["complete_A"] == 1 and new["complete_D"] == 0


@pytest.mark.parametrize("date", ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"])
def test_reserved_year_is_rejected_before_its_thermal_columns_are_read(tmp_path, monkeypatch, date):
    lp, sources, areas = input_files(tmp_path, expanded=[row(datetime_utc=date)])
    real_read = pd.read_parquet
    reserved_reads = []
    def guarded_read(path, *args, **kwargs):
        if Path(path) == sources[0]:
            reserved_reads.append(kwargs.get("columns"))
            assert kwargs.get("columns") == ["datetime_utc"], "Reserved thermal values were decoded before rejection"
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(merge.pd, "read_parquet", guarded_read)
    with pytest.raises(ValueError, match="[Rr]eserved|2021|2023"):
        merge.merge_cohort(lp, sources, areas, tmp_path / "merged")
    assert reserved_reads == [["datetime_utc"]]


def test_existing_frozen_output_cannot_be_overwritten(tmp_path):
    lp, sources, areas = input_files(tmp_path)
    output = tmp_path / "merged"
    merge.merge_cohort(lp, sources, areas, output)
    prior = (output / "paired_input.parquet").read_bytes()
    with pytest.raises(ValueError, match="Frozen cohort exists"):
        merge.merge_cohort(lp, sources, areas, output)
    assert (output / "paired_input.parquet").read_bytes() == prior
