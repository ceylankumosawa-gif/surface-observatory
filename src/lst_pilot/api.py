"""Bounded public API for the LST pilot; run behind a local HTTPS proxy.

One process / one render thread. SQLite preserves job state and rate accounting
across restarts. Importing this module does not start jobs or create directories.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Literal
from uuid import uuid4
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pyproj import Transformer
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)
STAC_BASE = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/"
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
SCENE_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
ACTIVE = ("queued", "running")
MAX_VERTICES = 512
MAX_PIXELS = 640_000
MAX_AREA_KM2 = 6400
MAX_BODY_BYTES = 65_536
SENSITIVE_QUERY_KEYS = frozenset({
    "sig", "se", "sp", "sv", "st", "sr", "spr", "sdd", "skoid", "sktid",
    "skt", "ske", "sks", "skv", "saoid", "suoid", "scid", "si", "sip",
    "token", "access_token", "refresh_token", "api_key", "apikey", "key",
    "secret", "password", "authorization", "signature", "key-pair-id", "policy",
})
DOWNLOAD_NAMES = frozenset({
    "prediction.tif", "observation.tif", "residual.tif", "uncertainty.tif",
    "prediction.png", "observation.png", "residual.png", "uncertainty.png",
    "predicted_lst.tif", "observed_lst.tif", "predicted_lst.png", "observed_lst.png",
    "overlay.png", "manifest.json", "summary.json", "statistics.json", "statistics.csv",
    "observed_overlay.png", "residual_overlay.png", "provenance.json", "quality.tif", "features.parquet",
})


@dataclass(frozen=True)
class Settings:
    root: Path
    catalog_path: Path | None = None
    jobs_dir: Path | None = None
    cache_dir: Path | None = None
    model_path: Path | None = None
    max_pending: int = 3
    hourly_per_ip: int = 6
    daily_global: int = 24

    def __post_init__(self):
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        for name, default in {
            "catalog_path": root / "web/catalog.json",
            "jobs_dir": root / "runs/web_jobs",
            "cache_dir": root / "cache",
            "model_path": root / "runs/pilot_v1_model/model.joblib",
        }.items():
            object.__setattr__(self, name, Path(getattr(self, name) or default).resolve())

    @classmethod
    def from_env(cls):
        root = Path(os.environ.get("LST_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        overrides = {k: os.environ.get("LST_" + k.upper()) for k in
                     ("catalog_path", "jobs_dir", "cache_dir", "model_path")}
        return cls(root, **overrides)


class PolygonInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pilot_id: str = Field(min_length=1, max_length=80)
    polygon: PolygonInput
    mode: Literal["observed", "experimental", "scenario"] = "observed"
    scene_id: str | None = Field(default=None, max_length=120)
    datetime_utc: str | None = Field(default=None, max_length=48)
    air_override: float | None = Field(default=None, ge=-90, le=65)


def _utc(value: str, scenario: bool = False) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Use an ISO timestamp with a UTC offset, for example 2024-07-01T12:00:00Z.") from None
    if dt.tzinfo is None:
        raise ValueError("The requested time must include Z or a UTC offset.")
    dt = dt.astimezone(timezone.utc)
    if scenario and not 1900 <= dt.year <= 2100:
        raise ValueError("Reported-air scenarios accept dates from 1900 through 2100.")
    if not scenario and not 2021 <= dt.year <= 2024:
        raise ValueError("This pilot supports historical times in 2021–2024 only.")
    return dt


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_daylight(dt: datetime, lon: float, lat: float) -> bool:
    import pandas as pd
    from pvlib.solarposition import get_solarposition
    sun = get_solarposition(pd.DatetimeIndex([dt]), latitude=lat, longitude=lon)
    return bool(sun.apparent_elevation.iloc[0] > 0)


def catalog_regions(catalog: dict) -> dict[str, dict]:
    pilots = catalog["pilots"]
    if isinstance(pilots, dict) and pilots.get("type") == "FeatureCollection":
        return {str(f["properties"]["id"]): {**f["properties"], "geometry": f["geometry"]}
                for f in pilots["features"]}
    return {str(p["id"]): p for p in pilots}


def validate_request(payload: JobInput, regions: dict) -> tuple[dict, dict, dict]:
    region = regions.get(payload.pilot_id)
    if region is None:
        raise ValueError("Choose one of the catalogued pilot areas.")
    geo = payload.polygon.model_dump()
    rings = geo["coordinates"]
    if len(rings) != 1:
        raise ValueError("Draw one polygon without holes.")
    ring = rings[0]
    if not 4 <= len(ring) <= MAX_VERTICES:
        raise ValueError("Use 4–512 polygon points, including the closing point.")
    if any(len(p) != 2 or not all(math.isfinite(x) for x in p) or
           not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90) for p in ring):
        raise ValueError("Polygon points must be finite [longitude, latitude] pairs.")
    if ring[0] != ring[-1]:
        raise ValueError("Close the polygon by repeating its first point.")
    poly = shape(geo)
    if poly.is_empty or not poly.is_valid or poly.area <= 0 or poly.bounds[2] - poly.bounds[0] > 180:
        raise ValueError("Draw a valid polygon with positive area and no self-intersections.")
    project = Transformer.from_crs(4326, int(region["epsg"]), always_xy=True)
    projected = transform(project.transform, poly)
    if not projected.is_valid or not all(math.isfinite(x) for x in projected.bounds):
        raise ValueError("The polygon cannot be projected into this pilot area.")
    extent = [float(x) for x in region["extent_m"]]
    # Canonical display boundaries have sub-metre chord/projection roundoff.
    # Clip the small accepted tolerance to the exact projected pilot extent.
    if not box(*extent).buffer(1.0).covers(projected):
        raise ValueError("The whole polygon must lie inside the selected pilot boundary.")
    projected = projected.intersection(box(*extent))
    xmin, ymin, xmax, ymax = projected.bounds
    if (xmax - xmin) * (ymax - ymin) > MAX_AREA_KM2 * 1_000_000:
        raise ValueError("The projected bounding box must be at most 6,400 km².")
    nx = math.ceil((xmax - extent[0]-.001) / 100) - math.floor((xmin - extent[0]+.001) / 100)
    ny = math.ceil((ymax - extent[1]-.001) / 100) - math.floor((ymin - extent[1]+.001) / 100)
    if nx < 1 or ny < 1 or nx * ny > MAX_PIXELS:
        raise ValueError("The aligned 100 m grid must contain at most 640,000 cells.")

    scenes = region.get("scenes", [])
    scene_map = {s.get("scene_id", s.get("id")): s for s in scenes}
    if payload.mode == "observed":
        scene = scene_map.get(payload.scene_id)
        if scene is None:
            raise ValueError("Choose an actual satellite scene listed for this pilot.")
        dt = _utc(scene["datetime_utc"])
        if payload.datetime_utc and _utc(payload.datetime_utc) != dt:
            raise ValueError("Observation mode requires the exact listed satellite acquisition time.")
    elif payload.mode == "scenario":
        if payload.air_override is None:
            raise ValueError("Enter the reported air temperature for the requested date and time.")
        if not payload.datetime_utc:
            raise ValueError("Enter the scenario date and time in UTC.")
        dt = _utc(payload.datetime_utc, scenario=True)
        from .scenario import select_surface_scene, require_daylight_area
        require_daylight_area(dt, poly)
        scene = select_surface_scene(scenes, dt, payload.scene_id)
    else:
        if not payload.datetime_utc:
            raise ValueError("Select a historical UTC time for the experimental estimate.")
        dt = _utc(payload.datetime_utc)
        prior = [s for s in scenes if timedelta(0) <= dt - _utc(s["datetime_utc"]) <= timedelta(days=90)]
        if not prior:
            raise ValueError("No catalogued earlier surface scene is available within 90 days of this time.")
        scene = max(prior, key=lambda s: _utc(s["datetime_utc"]))
        selected_id = scene.get("scene_id", scene.get("id"))
        if payload.scene_id is not None and payload.scene_id != selected_id:
            raise ValueError("Experimental mode uses the latest catalogued preceding scene within 90 days.")
        if not _is_daylight(dt, poly.centroid.x, poly.centroid.y):
            raise ValueError("Nighttime is unavailable in this pilot; choose a daylight time.")
    scene_id = scene.get("scene_id", scene.get("id"))
    if not isinstance(scene_id, str) or not SCENE_ID.fullmatch(scene_id):
        raise ValueError("The catalog contains an invalid scene identifier.")
    normalized = {
        "pilot_id": payload.pilot_id, "polygon": mapping(poly), "mode": payload.mode,
        "scene_id": scene_id, "datetime_utc": _iso(dt), "air_override": payload.air_override,
    }
    return normalized, region, scene


def fetch_scene(scene: dict, cache_dir: Path) -> dict:
    """Fetch only a trusted catalog ID from the fixed public STAC collection."""
    scene_id = scene.get("scene_id", scene.get("id"))
    if not isinstance(scene_id, str) or not SCENE_ID.fullmatch(scene_id):
        raise ValueError("Invalid catalogued satellite identifier.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (scene_id + ".json")
    if path.exists():
        item = json.loads(path.read_text())
    else:
        retry = Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], respect_retry_after_header=False)
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.headers["User-Agent"] = "LST-pilot-atlas/1.0 (bounded catalog scene access)"
            with session.get(STAC_BASE + scene_id, timeout=(10, 30), stream=True) as response:
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > 2_000_000:
                        raise ValueError("Satellite metadata exceeds the supported size.")
                item = json.loads(data)
        # API returns unsigned assets; strip temporary query credentials defensively.
        for asset in item.get("assets", {}).values():
            if "href" in asset:
                asset["href"] = asset["href"].split("?")[0]
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(item, allow_nan=False))
        temp.replace(path)
    if item.get("id") != scene_id or item.get("collection") != "landsat-c2-l2":
        raise ValueError("Satellite metadata does not match the requested catalogue entry.")
    if _utc(item["properties"]["datetime"]) != _utc(scene["datetime_utc"]):
        raise ValueError("Satellite metadata acquisition time differs from the catalogue.")
    return item


def _render(**kwargs):
    from .raster import render_raster
    return render_raster(**kwargs)


def _sanitize_url(value: str) -> str:
    """Retain functional documentation queries, remove signed credentials."""
    try:
        parsed = urlsplit(value)
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        safe = [(k, v) for k, v in pairs if k.lower() not in SENSITIVE_QUERY_KEYS
                and not k.lower().startswith(("x-amz-", "x-goog-"))]
        if safe == pairs and "@" not in parsed.netloc:
            return value
        return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path,
                           urlencode(safe), parsed.fragment))
    except ValueError:
        return value.split("?")[0]


def _pipeline_fingerprint(root: Path) -> str:
    """Freeze all pipeline source contents when the service process starts."""
    digest = hashlib.sha256()
    for path in sorted((root / "src/lst_pilot").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _clean_json(value: Any, job_dir: Path | None = None, file_urls: dict | None = None):
    """Remove private paths and URL query credentials from public metadata."""
    if isinstance(value, dict):
        return {str(k): _clean_json(v, job_dir, file_urls) for k, v in value.items()
                if str(k) not in {"traceback", "exception", "model_path", "cache_dir", "output_dir"}}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v, job_dir, file_urls) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return _sanitize_url(value)
        if value.startswith("/") and not value.startswith("/api/"):
            return (file_urls or {}).get(Path(value).name) if job_dir else None
        if value.startswith("file:"):
            return None
        value = re.sub(r"https?://[^\s'\"]+", lambda m: _sanitize_url(m.group(0)), value)
        return re.sub(r"(?<![\w:/])/(?:opt|home|root|var|srv|tmp|Users|mnt|etc)/[^\s'\"<>]+", "[private path]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _clean_json(value.item(), job_dir, file_urls)
    return value


class JobManager:
    def __init__(self, settings: Settings, renderer: Callable = _render,
                 scene_fetcher: Callable = fetch_scene, clock: Callable = time.time):
        self.settings, self.renderer, self.scene_fetcher, self.clock = settings, renderer, scene_fetcher, clock
        self.catalog = json.loads(settings.catalog_path.read_text())
        self.regions = catalog_regions(self.catalog)
        self.catalog_hash = hashlib.sha256(settings.catalog_path.read_bytes()).hexdigest()
        model = settings.model_path
        self.model_hash = hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else "missing-model"
        self.pipeline_hash = _pipeline_fingerprint(settings.root)
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = settings.jobs_dir / "jobs.sqlite3"
        self.lock = threading.RLock()
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lst-render")
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, ip TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL, request TEXT NOT NULL, result TEXT, error TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_hash ON jobs(request_hash)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created)")
            db.execute("UPDATE jobs SET status='failed', stage='interrupted', error=?, updated=? WHERE status IN ('queued','running')",
                       ("The server restarted before this job finished. Submit the request again to retry.", clock()))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)

    def public(self, row: sqlite3.Row, cached: bool = False) -> dict:
        result = json.loads(row["result"]) if row["result"] else None
        if result and result.get("prediction_method") in ("nighttime_coarse_baseline", "mixed_day_night"):
            from .scenario import NIGHT_WITHDRAWAL_REASON
            result.update(withdrawn=True, withdrawal_reason=NIGHT_WITHDRAWAL_REASON)
        return {
            "id": row["id"], "status": row["status"], "stage": row["stage"],
            "progress": row["progress"], "created_at": _iso(datetime.fromtimestamp(row["created"], timezone.utc)),
            "updated_at": _iso(datetime.fromtimestamp(row["updated"], timezone.utc)),
            "request": json.loads(row["request"]), "result": result,
            "error": row["error"], "cached": cached,
        }

    def get(self, job_id: str) -> dict:
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "Job not found.")
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found.")
        return self.public(row)

    def submit(self, payload: JobInput, ip: str) -> dict:
        try:
            normalized, region, scene = validate_request(payload, self.regions)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        serial = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        now = self.clock()
        # Revisit reference/provisional weather on a later day instead of
        # pinning a recent-date request to its first reference-weather result.
        weather_epoch = int(now // 86400) if normalized["mode"] == "scenario" else None
        key = hashlib.sha256(json.dumps(["web-api-v2", self.catalog_hash, self.model_hash,
                                        self.pipeline_hash, serial, weather_epoch]).encode()).hexdigest()
        with self.lock, self._db() as db:
            if self.closed:
                raise HTTPException(503, "The service is restarting. Please retry shortly.")
            cached = db.execute("SELECT * FROM jobs WHERE request_hash=? AND status IN ('queued','running','succeeded') ORDER BY created DESC LIMIT 1", (key,)).fetchone()
            if cached:
                return self.public(cached, cached=True)
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= self.settings.max_pending:
                raise HTTPException(429, "The render queue is full. Please wait for a current job to finish.", headers={"Retry-After": "60"})
            if db.execute("SELECT COUNT(*) FROM jobs WHERE ip=? AND created>?", (ip, now - 3600)).fetchone()[0] >= self.settings.hourly_per_ip:
                raise HTTPException(429, "The limit is six new jobs per hour for this connection. Cached requests remain available.", headers={"Retry-After": "3600"})
            if db.execute("SELECT COUNT(*) FROM jobs WHERE created>?", (now - 86400,)).fetchone()[0] >= self.settings.daily_global:
                raise HTTPException(429, "The shared pilot has reached its 24 new jobs per day budget. Cached requests remain available.", headers={"Retry-After": "3600"})
            job_id = uuid4().hex
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (job_id, key, ip, now, now, "queued", "queued", 0.0, serial, None, None))
            db.commit()
            self.executor.submit(self._run, job_id, normalized, region, scene)
        return self.get(job_id)

    def _update(self, job_id: str, **fields):
        with self.lock, self._db() as db:
            fields["updated"] = self.clock()
            sql = ",".join(f"{key}=?" for key in fields)
            db.execute(f"UPDATE jobs SET {sql} WHERE id=?", (*fields.values(), job_id))

    def _progress(self, job_id: str, event: dict):
        if not isinstance(event, dict):
            return
        try:
            progress = float(event.get("progress", event.get("fraction", 0)))
        except (TypeError, ValueError):
            progress = 0
        progress = max(0, min(.99, progress)) if math.isfinite(progress) else 0
        # Worker stages are short labels, not raw errors, paths or remote URLs.
        stage = str(event.get("stage", "processing"))
        if not re.fullmatch(r"[A-Za-z0-9 _.-]{1,80}", stage):
            stage = "processing"
        self._update(job_id, stage=stage, progress=progress)

    def _publish(self, job_id: str, result: dict) -> dict:
        directory = (self.settings.jobs_dir / job_id).resolve()
        urls = {}
        for name in DOWNLOAD_NAMES:
            path = directory / name
            if path.is_file() and not path.is_symlink() and path.resolve().parent == directory:
                urls[name] = f"/api/jobs/{job_id}/files/{name}"
        clean = _clean_json(result, directory, urls)
        if not isinstance(clean, dict):
            raise ValueError("The renderer did not return a result object.")
        # Preserve frontend semantic names while separately retaining the strict
        # filename allowlist used by the download endpoint.
        semantic = {}
        for role, value in result.get("files", {}).items():
            if isinstance(value, (str, Path)) and Path(value).name in urls:
                semantic[str(role)] = urls[Path(value).name]
        clean["files"] = semantic or urls
        clean["available_files"] = urls
        for key, names in {"png_url": ("prediction.png", "predicted_lst.png", "overlay.png"),
                           "geotiff_url": ("prediction.tif", "predicted_lst.tif")}.items():
            name = next((n for n in names if n in urls), None)
            if name:
                clean[key] = urls[name]
        for name in urls:
            if name.endswith(".json"):
                path = directory / name
                value = _clean_json(json.loads(path.read_text()), directory, urls)
                path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return clean

    def _run(self, job_id: str, normalized: dict, region: dict, scene: dict):
        try:
            from .satellite import configure_safe_logging
            configure_safe_logging()
            self._update(job_id, status="running", stage="satellite metadata", progress=.01)
            item = self.scene_fetcher(scene, self.settings.cache_dir / "stac")
            directory = self.settings.jobs_dir / job_id
            directory.mkdir(parents=True, exist_ok=True)
            result = self.renderer(
                region=region, polygon=normalized["polygon"], scene=item,
                requested_datetime=normalized["datetime_utc"], output_dir=directory,
                cache_dir=self.settings.cache_dir, model_path=self.settings.model_path,
                progress=lambda event: self._progress(job_id, event),
                air_override=normalized["air_override"], mode=normalized["mode"],
            )
            clean = self._publish(job_id, result)
            self._update(job_id, status="succeeded", stage="complete", progress=1.0,
                         result=json.dumps(clean, allow_nan=False), error=None)
        except Exception as exc:
            LOG.exception("Render job %s failed (%s)", job_id, type(exc).__name__)
            if isinstance(exc, (requests.RequestException, TimeoutError, ConnectionError)):
                public = "A required public data service could not be reached. Please retry later; cached jobs remain available."
            elif isinstance(exc, ValueError):
                public = "This area and time could not produce a valid raster. Try a smaller area or another listed satellite scene."
            elif isinstance(exc, FileNotFoundError):
                public = "A required model or data file is unavailable on the server. The service needs attention."
            else:
                public = "The raster could not be completed. Try another listed scene; the failure was recorded for diagnosis."
            self._update(job_id, status="failed", stage="failed", error=public)

    def file(self, job_id: str, filename: str) -> Path:
        record = self.get(job_id)
        if record["status"] != "succeeded" or filename not in DOWNLOAD_NAMES or filename not in (record["result"] or {}).get("available_files", {}):
            raise HTTPException(404, "Output file not found.")
        job_path = self.settings.jobs_dir / job_id
        directory = job_path.resolve()
        if job_path.is_symlink() or directory.parent != self.settings.jobs_dir:
            raise HTTPException(404, "Output file not found.")
        path = directory / filename
        if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
            raise HTTPException(404, "Output file not found.")
        return path

    def pixel(self, job_id: str, lon: float, lat: float) -> dict:
        if not (math.isfinite(lon) and math.isfinite(lat) and
                -180 <= lon <= 180 and -90 <= lat <= 90):
            raise HTTPException(422, "Use finite longitude and latitude within their valid ranges.")
        record = self.get(job_id)
        if record["status"] != "succeeded" or not record["result"]:
            raise HTTPException(409, "A completed raster is required for pixel inspection.")
        result = record["result"]
        if result.get("withdrawn") is True or result.get("prediction_method") in (
                "nighttime_coarse_baseline", "mixed_day_night"):
            raise HTTPException(409, "This archived nighttime result was withdrawn; pixel inspection is unavailable.")
        # Fixed canonical filename: never trust a caller or a result URL as a
        # filesystem path. Reuse the download allowlist and containment checks.
        path = self.file(job_id, "prediction.tif")
        try:
            from .raster_pixel import read_pixel
            return {"job_id": job_id, **read_pixel(path, lon, lat)}
        except Exception:
            LOG.exception("Pixel inspection failed for job %s", job_id)
            raise HTTPException(500, "The saved raster could not be inspected.") from None

    def examples(self) -> list:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status='succeeded' ORDER BY created DESC LIMIT 12").fetchall()
        return [self.public(row, cached=True) for row in rows]


def create_app(settings: Settings | None = None, renderer: Callable = _render,
               scene_fetcher: Callable = fetch_scene) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application):
        application.state.jobs = JobManager(settings, renderer, scene_fetcher)
        yield
        application.state.jobs.close()

    application = FastAPI(title="LST pilot API", version="1.0", lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)

    @application.middleware("http")
    async def bounded_body(request: Request, call_next):
        if request.method == "POST":
            try:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > MAX_BODY_BYTES:
                        return JSONResponse({"detail": "Request is too large; use at most 512 polygon points."}, status_code=413)
                request._body = bytes(body)
            except Exception:
                return JSONResponse({"detail": "Could not read the request body."}, status_code=400)
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Avoid reflecting arbitrary request content, NaNs or exception objects.
        errors = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": "Invalid request fields.", "errors": errors}, status_code=422)

    @application.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "service": "lst-pilot", "pilots": len(request.app.state.jobs.regions)}

    @application.get("/api/catalog")
    def catalog(request: Request):
        return _clean_json(request.app.state.jobs.catalog)

    @application.post("/api/jobs", status_code=202)
    def submit(payload: JobInput, request: Request):
        # Uvicorn trusts only Caddy. Never parse client-provided X-Forwarded-For here.
        ip = request.client.host if request.client else "unknown"
        return request.app.state.jobs.submit(payload, ip)

    @application.get("/api/jobs/{job_id}")
    def status(job_id: str, request: Request):
        return request.app.state.jobs.get(job_id)

    @application.get("/api/jobs/{job_id}/files/{filename}")
    def file(job_id: str, filename: str, request: Request):
        path = request.app.state.jobs.file(job_id, filename)
        media = {".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff", ".json": "application/json", ".csv": "text/csv", ".parquet": "application/vnd.apache.parquet"}.get(path.suffix, "application/octet-stream")
        return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"})

    @application.get("/api/jobs/{job_id}/pixel")
    def pixel(job_id: str, request: Request,
              lon: float = Query(..., ge=-180, le=180, allow_inf_nan=False),
              lat: float = Query(..., ge=-90, le=90, allow_inf_nan=False)):
        return request.app.state.jobs.pixel(job_id, lon, lat)

    @application.get("/api/examples")
    def examples(request: Request):
        return {"jobs": request.app.state.jobs.examples()}

    return application


app = create_app()
