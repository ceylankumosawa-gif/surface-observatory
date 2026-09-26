"""Isolated single-worker global API, with persisted budgets and subprocess limits."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from lst_pilot.api import JobManager, Settings, JOB_ID, _clean_json
from .model import MODEL_PATH
from .render import fingerprint
from .service_contract import GlobalInput, validate, capabilities, MAX_PENDING, JOB_TIMEOUT_SECONDS

LOG = logging.getLogger(__name__)
FILES = {'prediction.tif', 'quality.tif', 'coverage.tif', 'overlay.png', 'provenance.json'}
MIN_FREE_BYTES = 15 * 1024**3
TILE_CACHE_LIMIT = 40 * 1024**3
OUTPUT_RETENTION_SECONDS = 30 * 86400


def global_settings():
    root = Path(os.environ.get('LST_PROJECT_ROOT', '/opt/lst-pilot'))
    return Settings(root=root, jobs_dir=root/'runs/global_web_jobs', model_path=MODEL_PATH,
                    max_pending=MAX_PENDING, hourly_per_ip=6, daily_global=48)


class GlobalManager(JobManager):
    def __init__(self, settings, runner=None, clock=time.time):
        super().__init__(settings, clock=clock)
        self.pipeline_hash = fingerprint(settings.root)
        self.runner = runner or self._subprocess
        self.tile_cache = settings.jobs_dir/'canonical_tiles'
        self.tile_cache.mkdir(exist_ok=True)
        self.process = None
        self.operational = os.environ.get('LST_ENABLE_OPERATIONAL_WEATHER')=='1'

    def close(self):
        super().close()
        process = self.process
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    def pending(self):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    def submit(self, payload, ip):
        try:
            normalized = validate(payload, datetime.fromtimestamp(self.clock(), timezone.utc),self.operational)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        serial = json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False)
        now = self.clock()
        # Historical source products may be revised; revisit cached predictions
        # weekly. Existing job URLs keep their original bytes and provenance.
        key = hashlib.sha256(json.dumps(['global-web-v1', self.pipeline_hash, self.model_hash,
                                        serial, int(now//(7*86400))]).encode()).hexdigest()
        with self.lock, self._db() as db:
            if self.closed:
                raise HTTPException(503, 'The service is restarting. Please retry shortly.')
            row = db.execute("SELECT * FROM jobs WHERE request_hash=? AND status IN ('queued','running','succeeded') ORDER BY created DESC LIMIT 1", (key,)).fetchone()
            if row and (row['status'] != 'succeeded' or now-row['created'] < OUTPUT_RETENTION_SECONDS):
                return self.public(row, cached=True)
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= self.settings.max_pending:
                raise HTTPException(429, 'The global queue is full. Please wait for a job to finish.', headers={'Retry-After':'60'})
            if db.execute('SELECT COUNT(*) FROM jobs WHERE ip=? AND created>?', (ip,now-3600)).fetchone()[0] >= self.settings.hourly_per_ip:
                raise HTTPException(429, 'Six new requests per connection per hour are allowed; cached results remain available.', headers={'Retry-After':'3600'})
            if db.execute('SELECT COUNT(*) FROM jobs WHERE created>?', (now-86400,)).fetchone()[0] >= self.settings.daily_global:
                raise HTTPException(429, 'The shared daily computation budget is reached. Try again tomorrow.', headers={'Retry-After':'3600'})
            reserved_tiles = sum(len(json.loads(row[0])['source_tiles']) for row in
                                 db.execute('SELECT request FROM jobs WHERE created>?',(now-86400,)))
            if reserved_tiles+len(normalized['source_tiles'])>64:
                raise HTTPException(429, 'The shared daily source-download budget is reached. Try again tomorrow.', headers={'Retry-After':'3600'})
            if min(shutil.disk_usage(self.settings.jobs_dir).free, shutil.disk_usage(self.settings.cache_dir).free) < MIN_FREE_BYTES:
                raise HTTPException(503, 'New generation is paused to preserve server storage. Existing results remain available.')
            job_id = uuid4().hex
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (job_id,key,ip,now,now,'queued','queued',0.,serial,None,None))
            db.commit()
            self.executor.submit(self._run_global, job_id, normalized)
        return self.get(job_id)

    def _maintain(self):
        # The ledger is retained. Only this new service's own disposable output
        # directories and generation cache may be removed; never source data.
        now = self.clock()
        with self._db() as db:
            old = db.execute("SELECT id FROM jobs WHERE status IN ('succeeded','failed') AND created<?", (now-OUTPUT_RETENTION_SECONDS,)).fetchall()
        for row in old:
            directory = self.settings.jobs_dir/row['id']
            if JOB_ID.fullmatch(row['id']) and directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
            self._update(row['id'], status='expired', stage='expired', result=None,
                         error='The 30-day output retention period ended. Submit again to regenerate.')
        entries = []
        for directory in self.tile_cache.iterdir():
            if directory.is_dir() and not directory.is_symlink() and len(directory.name)==64 and all(c in '0123456789abcdef' for c in directory.name):
                size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
                entries.append((directory.stat().st_mtime, size, directory))
        total = sum(size for _,size,_ in entries)
        for modified,size,directory in sorted(entries):
            if total <= TILE_CACHE_LIMIT and now-modified <= 7*86400:
                continue
            shutil.rmtree(directory)
            total -= size

    def _run_global(self, job_id, normalized):
        try:
            self._maintain()
            self._update(job_id, status='running', stage='preparing sources', progress=.01)
            directory = self.settings.jobs_dir/job_id
            directory.mkdir(exist_ok=True)
            result = self.runner(job_id, normalized, directory)
            urls = {name:f'/api/global/jobs/{job_id}/files/{name}' for name in FILES
                    if (directory/name).is_file() and not (directory/name).is_symlink()}
            clean = _clean_json(result, directory, urls)
            clean['files'] = {role:urls[Path(name).name] for role,name in result.get('files',{}).items()
                              if isinstance(name,str) and Path(name).name in urls}
            clean['available_files'] = urls
            clean['png_url'] = urls.get('overlay.png')
            clean['geotiff_url'] = urls.get('prediction.tif')
            clean['retention_days'] = 30
            (directory/'provenance.json').write_text(json.dumps(clean, indent=2, allow_nan=False)+'\n')
            # Keep polling light: repeated native scene receipts belong in the
            # downloadable provenance, not every status response or DB row.
            clean['source_tiles'] = [{k:v for k,v in tile.items() if k!='source_provenance'}
                                     for tile in clean.get('source_tiles',[])]
            self._update(job_id, status='succeeded', stage='complete', progress=1.,
                         result=json.dumps(clean, allow_nan=False), error=None)
        except Exception as exc:
            LOG.exception('Global job %s failed: %s', job_id, type(exc).__name__)
            message = ('This request exceeded the 30-minute processing limit. Try a smaller area; completed source tiles remain cached.'
                       if isinstance(exc, TimeoutError) else
                       'Required source data could not produce this raster. The failure was recorded; try another date or a smaller area.')
            self._update(job_id, status='failed', stage='failed', error=message)

    def _subprocess(self, job_id, normalized, directory):
        request_path = directory/'request.json'
        request_path.write_text(json.dumps(normalized, allow_nan=False)+'\n')
        command = [sys.executable, '-m', 'lst_global.render', '--request', str(request_path),
                   '--output', str(directory), '--tile-cache', str(self.tile_cache),
                   '--source-cache', str(self.settings.cache_dir), '--pipeline-hash',
                   self.pipeline_hash+'-'+str(int(self.clock()//(7*86400)))]
        environment = {**os.environ, 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4',
                       'MKL_NUM_THREADS':'4', 'NUMEXPR_MAX_THREADS':'4', 'PYTHONUNBUFFERED':'1'}
        # systemd caps memory/CPU for the whole service; this subprocess adds a
        # true per-job wall limit and may be killed without losing API liveness.
        process = subprocess.Popen(command, cwd=self.settings.root, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=False)
        self.process = process
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        pending = b''
        try:
            with (directory/'worker.log').open('wb') as log:
                while process.poll() is None or selector.get_map():
                    if time.monotonic()-started > JOB_TIMEOUT_SECONDS:
                        raise TimeoutError('Worker deadline reached')
                    for key,_ in selector.select(timeout=.5):
                        block = os.read(key.fileobj.fileno(), 8192)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        if log.tell() < 2*1024**2:
                            log.write(block)
                        pending += block
                        while b'\n' in pending:
                            line,pending = pending.split(b'\n',1)
                            try:
                                event = json.loads(line)
                                if 'progress' in event:
                                    self._progress(job_id,event)
                            except (ValueError,TypeError):
                                pass
                        pending = pending[-65536:]
                if process.wait() != 0:
                    raise RuntimeError('Source worker failed')
            return json.loads((directory/'result.json').read_text())
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()
            self.process = None

    def file(self, job_id, filename):
        record = self.get(job_id)
        if record['status'] != 'succeeded' or filename not in FILES or filename not in (record['result'] or {}).get('available_files',{}):
            raise HTTPException(404, 'Output file not found.')
        directory = self.settings.jobs_dir/job_id
        path = directory/filename
        if directory.is_symlink() or path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise HTTPException(404, 'Output file not found.')
        return path


def create_app(settings=None, runner=None):
    settings = settings or global_settings()
    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = GlobalManager(settings, runner)
        yield
        app.state.jobs.close()
    app = FastAPI(title='Global LST experimental API', lifespan=lifespan,
                  docs_url=None,redoc_url=None,openapi_url=None)

    @app.middleware('http')
    async def body_limit(request, call_next):
        if request.method == 'POST':
            body = bytearray()
            async for block in request.stream():
                body.extend(block)
                if len(body)>65536:
                    return JSONResponse({'detail':'Request is too large.'},status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return JSONResponse({'detail':'Invalid request fields.'},status_code=422)

    @app.get('/api/global/capabilities')
    def caps(request: Request, lon:float|None=Query(None,ge=-180,le=180,allow_inf_nan=False),
             lat:float|None=Query(None,ge=-90,le=90,allow_inf_nan=False)):
        result = capabilities(operational=request.app.state.jobs.operational)
        result['queue'] = {'pending':request.app.state.jobs.pending()}
        result['time_zone'] = None
        if lon is not None and lat is not None:
            from timezonefinder import timezone_at
            result['time_zone'] = timezone_at(lng=lon,lat=lat)
            result['time_zone_source'] = 'timezonefinder 8.2.4 polygon lookup at selection centre; civil rules use IANA time zones'
        return result

    @app.post('/api/global/jobs',status_code=202)
    def submit(payload:GlobalInput,request:Request):
        return request.app.state.jobs.submit(payload,request.client.host if request.client else 'unknown')

    @app.get('/api/global/jobs/{job_id}')
    def status(job_id:str,request:Request):
        return request.app.state.jobs.get(job_id)

    @app.get('/api/global/jobs/{job_id}/files/{filename}')
    def file(job_id:str,filename:str,request:Request):
        path = request.app.state.jobs.file(job_id,filename)
        return FileResponse(path,media_type={'.png':'image/png','.tif':'image/tiff','.json':'application/json'}[path.suffix],
                            headers={'Cache-Control':'public, max-age=86400','X-Content-Type-Options':'nosniff'})

    @app.get('/api/global/jobs/{job_id}/pixel')
    def pixel(job_id:str,request:Request,lon:float=Query(...,ge=-180,le=180,allow_inf_nan=False),
              lat:float=Query(...,ge=-90,le=90,allow_inf_nan=False)):
        return request.app.state.jobs.pixel(job_id,lon,lat)
    return app


app = create_app()
