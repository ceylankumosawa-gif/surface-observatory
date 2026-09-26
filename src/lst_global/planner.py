"""Immutable compact hourly plans; no automatic worldwide data acquisition.

One tile table and one hour table represent the Cartesian schedule. Work is
materialized lazily only when input preparation exists. The planning operation
cannot launch a download or model fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from .grid import GRID_VERSION, TILE_CELLS, RESOLUTION_M, iter_tiles, geographic_boxes

SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE tiles (tile_id TEXT PRIMARY KEY, zone TEXT NOT NULL, epsg INTEGER NOT NULL,
  col INTEGER NOT NULL, row INTEGER NOT NULL, minx REAL NOT NULL, miny REAL NOT NULL,
  maxx REAL NOT NULL, maxy REAL NOT NULL, land_status TEXT NOT NULL DEFAULT 'unclassified');
CREATE INDEX tiles_zone ON tiles(zone);
CREATE TABLE hours (time_utc TEXT PRIMARY KEY);
"""


def utc_hour(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, AttributeError, ValueError):
        raise ValueError("Use an ISO timestamp with a timezone offset.") from None
    if stamp.tzinfo is None:
        raise ValueError("A timezone offset is required.")
    stamp = stamp.astimezone(timezone.utc)
    if stamp.minute or stamp.second or stamp.microsecond:
        raise ValueError("Choose an exact UTC hour.")
    return stamp


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create_plan(output, bounds, start, hours, model_sha256, *, weather_mode="historical_reanalysis"):
    geographic_boxes(bounds)
    stamp = utc_hour(start)
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 744:
        raise ValueError("Plan between 1 and 744 hourly timestamps.")
    if not isinstance(model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise ValueError("Pin the frozen model's SHA-256 before planning.")
    if weather_mode != "historical_reanalysis":
        raise ValueError("The first implementation uses historical reanalysis; forecast inputs need a separate adapter.")
    if stamp.year < 1940 or stamp + timedelta(hours=hours) > datetime.now(timezone.utc) - timedelta(days=7):
        raise ValueError("This historical plan requires dates from 1940 to at least seven days ago. Source availability is checked separately.")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    path = root / "plan.sqlite"
    started = time.monotonic()
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        n = 0
        counts = {}
        for tile in iter_tiles(bounds):
            db.execute("INSERT INTO tiles(tile_id,zone,epsg,col,row,minx,miny,maxx,maxy) VALUES(?,?,?,?,?,?,?,?,?)",
                       (tile.id, tile.zone.id, tile.zone.epsg, tile.column, tile.row, *tile.bounds))
            n += 1
            counts[tile.zone.id] = counts.get(tile.zone.id, 0) + 1
        times = [(stamp + timedelta(hours=i)).isoformat().replace("+00:00", "Z") for i in range(hours)]
        db.executemany("INSERT INTO hours VALUES(?)", ((x,) for x in times))
        summary = {
            "version": "global-plan-v1", "grid_version": GRID_VERSION,
            "bounds_wgs84": list(bounds), "start_utc": times[0], "end_utc_inclusive": times[-1],
            "hours": hours, "candidate_tiles": n, "zone_tile_counts": counts,
            "candidate_pixel_slots_per_hour": n * TILE_CELLS ** 2,
            "candidate_tile_hours": n * hours, "resolution_m": RESOLUTION_M,
            "temperature_float32_bytes_per_hour_upper_bound": n * TILE_CELLS ** 2 * 4,
            "temperature_float32_bytes_all_hours_upper_bound": n * TILE_CELLS ** 2 * 4 * hours,
            "sizing_basis": "Candidate tile storage includes ocean, boundary padding and unowned cells. No compression assumption. Additional masks, inputs, overviews and replicas are excluded.",
            "model_sha256": model_sha256, "weather_mode": weather_mode,
            "source_code_sha256": {"planner.py": sha(__file__),
                                    "grid.py": sha(Path(__file__).with_name("grid.py"))},
            "state": "planned_only", "source_availability": "not_yet_checked",
            "land_mask": "pending_source_preparation", "support": "global_extrapolation",
            "automated_generation_enabled": False, "model_training_enabled": False,
            "plan_seconds": time.monotonic() - started,
        }
        db.executemany("INSERT INTO metadata VALUES(?,?)", ((k, json.dumps(v)) for k, v in summary.items()))
    summary["plan_sha256"] = sha(path)
    summary["plan_bytes"] = path.stat().st_size
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def materialize_work(plan, tile_id, time_utc):
    """Idempotently create one job; missing input preparation cannot be skipped."""
    canonical = utc_hour(time_utc).isoformat().replace("+00:00", "Z")
    plan = Path(plan).resolve()
    published = json.loads(plan.with_name("summary.json").read_text())
    digest = sha(plan)
    if published.get("plan_sha256") != digest:
        raise ValueError("Plan differs from its published content hash.")
    with sqlite3.connect(plan.as_uri() + "?mode=ro", uri=True) as source:
        if not source.execute("SELECT 1 FROM tiles WHERE tile_id=?", (tile_id,)).fetchone():
            raise ValueError("Tile does not belong to this plan.")
        if not source.execute("SELECT 1 FROM hours WHERE time_utc=?", (canonical,)).fetchone():
            raise ValueError("Hour does not belong to this plan.")
    # Mutable execution state has its own database: the frozen schedule and its
    # content hash remain unchanged when work is resumed or completed.
    with sqlite3.connect(plan.with_name("work.sqlite")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS binding (plan_sha256 TEXT PRIMARY KEY)")
        prior = db.execute("SELECT plan_sha256 FROM binding").fetchall()
        if prior and prior != [(digest,)]:
            raise ValueError("Work queue belongs to a different frozen plan.")
        db.execute("INSERT OR IGNORE INTO binding VALUES(?)", (digest,))
        db.execute("""CREATE TABLE IF NOT EXISTS work (tile_id TEXT NOT NULL,
          time_utc TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN
          ('awaiting_inputs','ready','running','complete','failed')),
          input_digest TEXT, output_path TEXT, detail TEXT, PRIMARY KEY(tile_id,time_utc))""")
        db.execute("INSERT OR IGNORE INTO work(tile_id,time_utc,state) VALUES(?,?,'awaiting_inputs')", (tile_id, canonical))
        return db.execute("SELECT state FROM work WHERE tile_id=? AND time_utc=?", (tile_id, canonical)).fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, default=(-180, -90, 180, 90))
    parser.add_argument("--start", required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(create_plan(args.output, args.bbox, args.start, args.hours, args.model_sha256), indent=2))


if __name__ == "__main__":
    main()
