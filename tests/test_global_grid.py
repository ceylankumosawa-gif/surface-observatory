import sqlite3

import numpy as np
import pytest
from pyproj import Geod

from lst_global.grid import zones, iter_tiles, geographic_boxes, Tile, TILE_CELLS
from lst_global.planner import create_plan, materialize_work, utc_hour, sha


def test_one_owner_at_zone_equator_dateline_and_polar_boundaries():
    lon = np.array([-180, 180, -174, -0.001, 0, 6, 179.99, 120, 0, 0, -60, 80])
    lat = np.array([0, 0, -0.001, 51, 0, 83.99, -80, -80.001, 84, 90, -90, -79.99])
    ownership = np.array([z.owns(lon, lat) for z in zones()])
    np.testing.assert_array_equal(ownership.sum(axis=0), np.ones(len(lon)))


def test_small_london_and_unseen_tokyo_use_deterministic_metric_tiles():
    london = list(iter_tiles((-.3, 51.4, -.1, 51.6)))
    tokyo = list(iter_tiles((139.6, 35.5, 139.9, 35.8)))
    assert london and tokyo
    assert {t.zone.id for t in london} == {"utm-30n"}
    assert {t.zone.id for t in tokyo} == {"utm-54n"}
    assert [t.id for t in london] == [t.id for t in iter_tiles((-.3, 51.4, -.1, 51.6))]
    geod = Geod(ellps="WGS84")
    lon, lat, owned = london[0].centres()
    assert lon.shape == (TILE_CELLS, TILE_CELLS)
    assert owned.any()
    _, _, distance = geod.inv(lon[200, 200], lat[200, 200], lon[200, 201], lat[200, 201])
    assert 99 < distance < 101


def test_antimeridian_is_two_local_strips_not_a_worldwide_rectangle():
    crossing = list(iter_tiles((179.8, -17, -179.8, -16.8)))
    assert {t.zone.id for t in crossing} == {"utm-01s", "utm-60s"}
    assert len(crossing) < 20
    assert len({t.id for t in crossing}) == len(crossing)


@pytest.mark.parametrize("bounds,zone", [((-20, 88, 20, 89), "ups-n"), ((-20, -89, 20, -88), "ups-s")])
def test_poles_are_represented(bounds, zone):
    tiles = list(iter_tiles(bounds))
    assert tiles and {t.zone.id for t in tiles} == {zone}
    assert any(t.centres()[2].any() for t in tiles)


@pytest.mark.parametrize("bounds", [(0, 0, 0, 1), (0, -91, 1, 0), (0, 0, float('nan'), 1), (180, 0, -180, 1)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        geographic_boxes(bounds)


def test_planning_is_compact_and_cannot_start_missing_inputs(tmp_path):
    root = tmp_path / "plan"
    report = create_plan(root, (-.3, 51.4, -.1, 51.6), "2023-06-21T01:00:00+01:00", 24, "a" * 64)
    assert report['start_utc'] == '2023-06-21T00:00:00Z'
    assert report['candidate_tile_hours'] == report['candidate_tiles'] * 24
    assert report['automated_generation_enabled'] is False
    with sqlite3.connect(root / 'plan.sqlite') as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='work'").fetchone()
        tile = db.execute('SELECT tile_id FROM tiles LIMIT 1').fetchone()[0]
    for _ in range(2):
        assert materialize_work(root / 'plan.sqlite', tile, report['start_utc']) == 'awaiting_inputs'
    with sqlite3.connect(root / 'work.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM work').fetchone()[0] == 1
    assert sha(root / 'plan.sqlite') == report['plan_sha256']
    with pytest.raises(ValueError):
        materialize_work(root / 'plan.sqlite', 'unknown', report['start_utc'])
    with pytest.raises(FileExistsError):
        create_plan(root, (-.3, 51.4, -.1, 51.6), report['start_utc'], 24, 'a' * 64)


@pytest.mark.parametrize('stamp', ['2023-01-01', '2023-01-01T00:30:00Z', 'not a date'])
def test_hour_requires_timezone_and_exact_hour(stamp):
    with pytest.raises(ValueError):
        utc_hour(stamp)


def test_first_work_item_rejects_tampered_plan(tmp_path):
    root = tmp_path / 'tampered'
    report = create_plan(root, (-.3, 51.4, -.1, 51.6), '2023-06-21T00:00:00Z', 1, 'a' * 64)
    with sqlite3.connect(root / 'plan.sqlite') as db:
        tile = db.execute('SELECT tile_id FROM tiles LIMIT 1').fetchone()[0]
        db.execute("UPDATE metadata SET value='tampered' WHERE key='grid_version'")
    with pytest.raises(ValueError, match='published content hash'):
        materialize_work(root / 'plan.sqlite', tile, report['start_utc'])
    assert not (root / 'work.sqlite').exists()
