from copy import deepcopy

import pytest

from lst_pilot import more_days_acquire as m


def item(date, identity="scene", cloud=10):
    return {"id": identity, "properties": {"datetime": date, "platform": "landsat-8", "eo:cloud_cover": cloud},
            "assets": {b: {"href": "not-opened"} for b in (*m.satellite.REQUIRED_ASSETS, "qa")}}


@pytest.mark.parametrize("date", ["2020-12-31T23:59:59Z", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2022-01-01"])
def test_only_timezone_aware_fitting_dates(date):
    with pytest.raises(ValueError):
        m.fitting_date(item(date))


def test_existing_weather_date_excluded_even_for_different_scene(monkeypatch):
    monkeypatch.setattr(m.satellite, "scene_coverage_fraction", lambda i, r: 1.)
    groups, rejected = m.queues([item("2021-06-15T10:00:00Z", "different_scene")], {}, {"2021-06-15"}, set())
    assert sum(len(g["items"]) for g in groups) == 0
    assert rejected == {"date_already_in_original_cohort": 1}


def test_months_interleave_quarters_and_years_before_backup_scenes(monkeypatch):
    monkeypatch.setattr(m.satellite, "scene_coverage_fraction", lambda i, r: 1.)
    groups, _ = m.queues([], {}, set(), set())
    assert [(g["year"], g["month"]) for g in groups[:8]] == [
        (2021, 1), (2022, 1), (2021, 7), (2022, 7), (2021, 4), (2022, 4), (2021, 10), (2022, 10)]
    assert len({(g["year"], g["month"]) for g in groups}) == 24


def test_metadata_ranking_is_bounded_and_does_not_use_temperature(monkeypatch):
    monkeypatch.setattr(m.satellite, "scene_coverage_fraction", lambda i, r: 1.)
    items = [item(f"2022-01-{j:02d}T10:00:00Z", str(j), j) for j in range(1, 7)]
    a, _ = m.queues(items, {}, set(), set())
    changed = deepcopy(items)
    for j, i in enumerate(changed):
        i["temperature"] = -999 if j % 2 else 999
        i["model_residual"] = j * 999
    b, _ = m.queues(changed, {}, set(), set())
    ids = lambda gs: [[i["id"] for i in g["items"]] for g in gs]
    assert ids(a) == ids(b)
    jan = next(g for g in a if g["year"] == 2022 and g["month"] == 1)
    assert jan["available_candidates"] == 6 and len(jan["items"]) == 4


def test_reject_attempted_and_unusable_metadata(monkeypatch):
    monkeypatch.setattr(m.satellite, "scene_coverage_fraction", lambda i, r: .1 if i["id"] == "outside" else 1.)
    items = [item("2022-01-01T10:00:00Z", x) for x in ("old", "cloud", "assets", "outside")]
    items[1]["properties"]["eo:cloud_cover"] = 90
    items[2]["assets"].pop("qa")
    groups, rejected = m.queues(items, {}, set(), {"old"})
    assert sum(len(g["items"]) for g in groups) == 0
    assert set(rejected) == {"scene_already_attempted", "scene_cloud_above_80_percent", "missing_source_assets", "footprint_below_15_percent"}
