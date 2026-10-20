import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="Optional dashboard dependencies are not installed")
pytest.importorskip("httpx", reason="Optional dashboard test dependencies are not installed")

from fastapi.testclient import TestClient

from app import Settings, create_app
from src.pipeline import normalize_video, probe_video


def test_upload_validation_and_failure_reporting(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/health').json()['status'] == 'ok'
        assert client.get('/api/analyses').json() == {'analyses': []}
        assert client.get('/api/analyses/missing').status_code == 404
        assert client.post('/api/analyses', files={'file': ('notes.txt', b'hello')}).status_code == 415
        assert client.post('/api/analyses', files={'file': ('empty.mp4', b'')}).status_code == 400
        assert client.post('/api/analyses', files={'file': ('bad.mp4', b'data')}, data={'foot': 'both'}).status_code == 422
        response = client.post('/api/analyses', files={'file': ('corrupt.mov', b'not a video')})
        assert response.status_code == 202
        job_id = response.json()['id']
        for _ in range(100):
            job = client.get(f'/api/analyses/{job_id}').json()
            if job['status'] == 'failed':
                break
            time.sleep(0.05)
        assert job['status'] == 'failed'
        assert job['error']
        assert client.get(f'/api/analyses/{job_id}/report').status_code == 409
        assert client.get(f'/media/{job_id}/original.mov').status_code == 404
        assert client.patch(f'/api/analyses/{job_id}', json={'contact_frame': -1}).status_code == 422
        assert client.patch(f'/api/analyses/{job_id}', json={'scale_m_per_px': 0}).status_code == 422
        assert client.patch(f'/api/analyses/{job_id}', json={'surprise': True}).status_code == 422
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get('/api/analyses').json()['analyses'][0]['id'] == job_id


def test_settings_reject_invalid_scale_and_frame():
    for settings in [{'scale_m_per_px': float('nan')}, {'scale_m_per_px': float('inf')},
                     {'contact_frame': 1.5}, {'contact_frame': True}, {'scale_m_per_px': True}]:
        with pytest.raises(ValueError):
            Settings(**settings)


def test_mp4_mov_normalization_and_browser_range(tmp_path):
    import cv2
    import numpy as np
    source = tmp_path / 'phone.mov'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 240))
    assert writer.isOpened()
    for index in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index * 4, 60), (index * 4 + 30, 90), (0, 255, 100), -1)
        writer.write(frame)
    writer.release()
    target = tmp_path / 'normalized.mp4'
    metadata = normalize_video(source, target)
    assert metadata['frame_count'] == 10
    assert metadata['fps'] == 25
    assert metadata['width'] == 320 and metadata['height'] == 240
    assert probe_video(target)['duration'] == pytest.approx(0.4)
    assert target.read_bytes().find(b'moov') < target.read_bytes().find(b'mdat')
    # Verify the actual media response supports byte-range browser seeking.
    with TestClient(create_app(tmp_path / 'store')) as client:
        job_id = 'a' * 32
        directory = client.app.state.store.directory / job_id
        directory.mkdir()
        target.rename(directory / 'video.mp4')
        client.app.state.store.jobs[job_id] = {'id': job_id, 'status': 'complete'}
        response = client.get(f'/media/{job_id}/video.mp4', headers={'Range': 'bytes=0-63'})
        assert response.status_code == 206
        assert len(response.content) == 64
        assert response.headers['content-type'] == 'video/mp4'
        assert 'bytes 0-63/' in response.headers['content-range']
