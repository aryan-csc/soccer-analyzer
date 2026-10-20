"""Touchline Kick Lab — local FastAPI server and durable single-worker analysis queue."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.pipeline import MAX_BYTES, run_analysis, write_json
from src.settings import Settings

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('KICKLAB_DATA_DIR', ROOT / 'data' / 'sessions')).resolve()
EXAMPLE = ROOT / 'data' / 'examples' / 'instep_side.mp4'
logger = logging.getLogger('kicklab')


class SessionManifest(BaseModel):
    """Validate local state before it can affect startup, paths, or API lists."""
    model_config = ConfigDict(allow_inf_nan=False)
    id: str = Field(pattern=r'^[a-f0-9]{32}$')
    name: str = Field(min_length=1)
    status: Literal['queued', 'processing', 'complete', 'failed']
    progress: float = Field(ge=0, le=1)
    message: str
    created_at: datetime
    updated_at: datetime
    settings: Settings
    source: Literal['original.mp4', 'original.mov']
    error: str | None = None


class JobStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kicklab')
        for manifest in directory.glob('*/job.json'):
            try:
                job = SessionManifest.model_validate_json(manifest.read_text()).model_dump(mode='json')
                if job['id'] != manifest.parent.name:
                    continue
                if job['status'] in ('queued', 'processing'):
                    job.update(status='failed', message='Analysis interrupted by server restart.',
                               error='The server stopped before completion. Retry this session or upload again.')
                    write_json(manifest, job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                logger.warning('Skipped invalid session manifest %s', manifest)

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, 'Analysis not found.')
            return dict(self.jobs[job_id])

    def update(self, job_id: str, **changes):
        with self.lock:
            self.jobs[job_id].update(changes)
            self.jobs[job_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            write_json(self.directory / job_id / 'job.json', self.jobs[job_id])

    def create(self, job_id: str, name: str, source: Path, settings: Settings) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.jobs[job_id] = {'id': job_id, 'name': name, 'status': 'queued', 'progress': 0,
                                 'message': 'Waiting for analysis', 'created_at': now, 'updated_at': now,
                                 'settings': settings.model_dump(), 'source': source.name, 'error': None}
            self.update(job_id)
        self.executor.submit(self.work, job_id, False)
        return self.get(job_id)

    def work(self, job_id: str, reuse_pose: bool):
        job = self.get(job_id)
        directory = self.directory / job_id
        last_update = 0.0
        def progress(value, message):
            nonlocal last_update
            now = time.monotonic()
            if now - last_update >= 0.3 or value >= 1:
                self.update(job_id, progress=round(float(value), 3), message=message)
                last_update = now
        self.update(job_id, status='processing', error=None)
        try:
            report = run_analysis(directory / job['source'], directory, job['settings'], progress, reuse_pose)
            self.update(job_id, status='complete', progress=1, message='Analysis complete', settings=report['settings'])
        except Exception as exc:
            logger.exception('Analysis %s failed', job_id)
            self.update(job_id, status='failed', message='Analysis could not be completed', error=str(exc))


def create_app(data_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.store = JobStore(Path(data_dir or DATA_DIR))
        try:
            yield
        finally:
            application.state.store.executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(title='Touchline Kick Lab', version='1.0.0', lifespan=lifespan)

    @application.middleware('http')
    async def upload_limit(request: Request, call_next):
        if request.method == 'POST' and request.url.path == '/api/analyses':
            try:
                if int(request.headers.get('content-length', 0)) > MAX_BYTES + 1024 * 1024:
                    return JSONResponse({'detail': 'Video is too large. Maximum upload is 200 MB.'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail': 'Invalid content length.'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    def store() -> JobStore:
        return application.state.store

    @application.get('/api/health')
    def health():
        return {'status': 'ok', 'version': '1.0.0'}

    @application.get('/api/analyses')
    def analyses():
        with store().lock:
            return {'analyses': sorted((dict(job) for job in store().jobs.values()),
                                       key=lambda job: job['created_at'], reverse=True)}

    @application.get('/api/examples')
    def examples():
        return {'examples': [{'name': 'Instep kick · side view', 'path': EXAMPLE.name,
                              'source': 'https://www.youtube.com/watch?v=WmsaKPhKSCU'}] if EXAMPLE.exists() else []}

    @application.post('/api/analyses/example', status_code=202)
    def example():
        if not EXAMPLE.exists():
            raise HTTPException(404, 'No example footage is installed. Upload your own clip.')
        job_id = uuid4().hex
        directory = store().directory / job_id
        directory.mkdir()
        source = directory / 'original.mp4'
        shutil.copy2(EXAMPLE, source)
        return store().create(job_id, 'Instep kick · side view (sample)', source, Settings())

    @application.post('/api/analyses', status_code=202)
    async def upload(file: UploadFile = File(...), foot: str = Form('right'),
                     shot_type: str = Form('shot'), camera_view: str = Form('side'),
                     camera_motion: str = Form('fixed'), direction: str = Form('auto'),
                     contact_frame: str | None = Form(None), scale_m_per_px: str | None = Form(None)):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in ('.mp4', '.mov'):
            await file.close()
            raise HTTPException(415, 'Upload an MP4 or MOV video.')
        try:
            settings = Settings(foot=foot, shot_type=shot_type, camera_view=camera_view,
                                camera_motion=camera_motion, direction=direction,
                                contact_frame=int(contact_frame) if contact_frame else None,
                                scale_m_per_px=float(scale_m_per_px) if scale_m_per_px else None)
        except (ValidationError, ValueError) as exc:
            await file.close()
            raise HTTPException(422, 'Invalid analysis settings: ' + str(exc)) from exc
        job_id = uuid4().hex
        directory = store().directory / job_id
        directory.mkdir()
        source = directory / ('original' + suffix)
        size = 0
        try:
            with source.open('wb') as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, 'Video is too large. Maximum upload is 200 MB.')
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, 'The uploaded file is empty.')
        except Exception:
            source.unlink(missing_ok=True)
            directory.rmdir()
            raise
        finally:
            await file.close()
        return store().create(job_id, Path(file.filename or 'Kick video').name, source, settings)

    @application.get('/api/analyses/{job_id}')
    def analysis(job_id: str):
        return store().get(job_id)

    @application.patch('/api/analyses/{job_id}', status_code=202)
    def revise(job_id: str, values: dict):
        with store().lock:
            job = store().get(job_id)
            if job['status'] in ('queued', 'processing'):
                raise HTTPException(409, 'Wait for the current analysis to finish.')
            try:
                settings = Settings(**{**job['settings'], **values})
            except ValidationError as exc:
                raise HTTPException(422, str(exc)) from exc
            raw_path = store().directory / job_id / 'pose.json'
            if settings.contact_frame is not None and raw_path.exists():
                try:
                    count = json.loads(raw_path.read_text())['metadata']['frame_count']
                except (OSError, ValueError, KeyError, TypeError):
                    # The pipeline can rebuild a damaged cache. Do not block
                    # its retry here; decoded frames will validate contact.
                    count = None
                if type(count) is int and count > 0 and settings.contact_frame >= count:
                    raise HTTPException(422, 'Contact frame is outside the video.')
            store().update(job_id, status='queued', progress=0, message='Updating measurements',
                           settings=settings.model_dump(), error=None)
        store().executor.submit(store().work, job_id, raw_path.exists())
        return store().get(job_id)

    @application.get('/api/analyses/{job_id}/report')
    def report(job_id: str):
        with store().lock:
            job = store().get(job_id)
            if job['status'] != 'complete':
                raise HTTPException(409, 'The report is available when analysis completes.')
            directory = store().directory / job_id
            try:
                if any(not (directory / filename).is_file() for filename in
                       ('video.mp4', 'overlay.mp4', 'metrics.csv', 'report.json')):
                    raise ValueError('Missing analysis exports')
                data = json.loads((directory / 'report.json').read_text())
                if (not isinstance(data, dict) or not isinstance(data.get('frames'), list) or not data['frames']
                        or not isinstance(data.get('metadata'), dict)
                        or data['metadata'].get('frame_count') != len(data['frames'])
                        or any(not isinstance(data.get(key), kind) for key, kind in
                               {'settings': dict, 'quality': dict, 'contact': dict, 'phases': list,
                                'summary': list, 'observations': list, 'metric_definitions': list}.items())):
                    raise ValueError('Invalid report structure')
            except (OSError, ValueError, TypeError):
                message = 'Saved analysis files are missing or damaged. Retry the analysis to rebuild them.'
                store().update(job_id, status='failed', message='Saved analysis needs repair', error=message)
                raise HTTPException(409, message) from None
            revision = job['updated_at']
            data.update(id=job_id, name=job['name'], urls={key: f'/media/{job_id}/{name}?v={quote(revision, safe="")}' for key, name in
                        {'video': 'video.mp4', 'overlay': 'overlay.mp4', 'csv': 'metrics.csv', 'json': 'report.json'}.items()})
            return data

    @application.get('/media/{job_id}/{filename}')
    def media(job_id: str, filename: str, v: str | None = None):
        job = store().get(job_id)
        allowed = {'video.mp4': 'video/mp4', 'overlay.mp4': 'video/mp4',
                   'metrics.csv': 'text/csv', 'report.json': 'application/json'}
        if filename not in allowed:
            raise HTTPException(404, 'File not found.')
        if filename != 'video.mp4' and job['status'] != 'complete':
            raise HTTPException(409, 'Exports are available when analysis completes. Retry failed analyses to rebuild them.')
        if v is not None and v != job.get('updated_at'):
            raise HTTPException(409, 'This analysis has changed. Reload the session before downloading or playing its files.')
        path = store().directory / job_id / filename
        if not path.is_file():
            raise HTTPException(404, 'This file is not ready yet.')
        return FileResponse(path, media_type=allowed[filename], filename=filename,
                            content_disposition_type='inline' if filename.endswith('.mp4') else 'attachment',
                            headers={'Cache-Control': 'no-cache'})

    application.mount('/', StaticFiles(directory=ROOT / 'web', html=True), name='dashboard')
    return application


app = create_app()
