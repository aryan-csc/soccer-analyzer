"""Regression cases for damaged local state and source preservation."""
import json
from urllib.parse import urlencode

import pytest

pytest.importorskip("fastapi", reason="Optional dashboard dependencies are not installed")
pytest.importorskip("httpx", reason="Optional dashboard test dependencies are not installed")

from fastapi.testclient import TestClient

from app import Settings, create_app


def manifest(job_id='a' * 32, status='failed'):
    return {'id': job_id, 'name': 'Kick.mp4', 'status': status,
            'progress': 0, 'message': 'Analysis stopped',
            'created_at': '2026-10-20T12:00:00+00:00', 'updated_at': '2026-10-20T12:00:00+00:00',
            'settings': Settings().model_dump(), 'source': 'original.mp4', 'error': None}


@pytest.mark.parametrize('damage', [[], None, {}, {'status': 'unknown'},
    {'created_at': None}, {'progress': float('nan')}, {'source': '../another.mp4'},
    {'settings': {'foot': 'both'}}, {'id': '../outside'}])
def test_bad_manifest_does_not_break_startup_or_session_list(tmp_path, damage):
    directory = tmp_path / ('a' * 32)
    directory.mkdir()
    value = {**manifest(), **damage} if isinstance(damage, dict) and damage else damage
    (directory / 'job.json').write_text(json.dumps(value))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/health').status_code == 200
        assert client.get('/api/analyses').json() == {'analyses': []}


@pytest.mark.parametrize('contents', [None, '{interrupted write', '[]', '{}'])
def test_missing_or_damaged_report_becomes_retryable_failure(tmp_path, contents):
    job = manifest(status='complete')
    directory = tmp_path / job['id']
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps(job))
    # Other exports are present, isolating report corruption from missing media.
    for name in ['video.mp4', 'overlay.mp4', 'metrics.csv']:
        (directory / name).write_bytes(b'present')
    if contents is not None:
        (directory / 'report.json').write_text(contents)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/api/analyses/{job['id']}/report")
        assert response.status_code == 409
        assert 'retry' in response.json()['detail'].lower()
        assert client.get(f"/api/analyses/{job['id']}").json()['status'] == 'failed'
        # Broken pose JSON should not prevent the retry request itself.
        (directory / 'pose.json').write_text('{broken cache')
        assert client.patch(f"/api/analyses/{job['id']}", json={'contact_frame': 0}).status_code == 202


def test_interrupted_job_is_retryable_after_restart(tmp_path):
    job = manifest(status='processing')
    directory = tmp_path / job['id']
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps(job))
    with TestClient(create_app(tmp_path)) as client:
        saved = client.get(f"/api/analyses/{job['id']}").json()
        assert saved['status'] == 'failed'
        assert 'restart' in saved['message']
        assert client.patch(f"/api/analyses/{job['id']}", json={}).status_code == 202


def test_stale_exports_are_not_downloadable_during_recalculation(tmp_path):
    job = manifest(status='failed')
    directory = tmp_path / job['id']
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps(job))
    for name in ['overlay.mp4', 'metrics.csv', 'report.json']:
        (directory / name).write_bytes(b'old result')
    with TestClient(create_app(tmp_path)) as client:
        for status in ['queued', 'processing', 'failed']:
            client.app.state.store.update(job['id'], status=status)
            for name in ['overlay.mp4', 'metrics.csv', 'report.json']:
                assert client.get(f"/media/{job['id']}/{name}").status_code == 409


def test_old_revision_links_cannot_download_new_measurements(tmp_path):
    job = manifest(status='complete')
    directory = tmp_path / job['id']
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps(job))
    (directory / 'metrics.csv').write_text('frame,knee_flexion\n0,90\n')
    with TestClient(create_app(tmp_path)) as client:
        current = client.get(f"/api/analyses/{job['id']}").json()
        url = f"/media/{job['id']}/metrics.csv?" + urlencode({'v': current['updated_at']})
        assert client.get(url).status_code == 200
        client.app.state.store.update(job['id'], message='Recalculated')
        assert client.get(url).status_code == 409
