"""Opt-in upload-to-export test using actual soccer footage and MediaPipe."""
import csv
import io
import json
import os
from pathlib import Path
import time

import cv2
import imageio_ffmpeg
import pytest

pytest.importorskip("fastapi", reason="Optional dashboard dependencies are not installed")
pytest.importorskip("httpx", reason="Optional dashboard test dependencies are not installed")

from fastapi.testclient import TestClient

from app import create_app

SAMPLE = Path(__file__).resolve().parents[1] / 'input/instep_side.mp4'
pytestmark = pytest.mark.skipif(
    os.environ.get('KICKLAB_RUN_VIDEO_E2E') != '1' or not SAMPLE.is_file(),
    reason='Set KICKLAB_RUN_VIDEO_E2E=1 and provide input/instep_side.mp4 for real footage analysis.',
)


def wait_for_job(client, job_id, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f'/api/analyses/{job_id}').json()
        if job['status'] in ('complete', 'failed'):
            assert job['status'] == 'complete', job.get('error')
            return job
        time.sleep(0.1)
    pytest.fail(f'Analysis {job_id} timed out')


def test_real_upload_tracking_exports_manual_contact_and_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        with SAMPLE.open('rb') as source:
            response = client.post('/api/analyses', files={'file': ('kick.mp4', source, 'video/mp4')},
                                   data={'foot': 'right', 'shot_type': 'penalty'})
        assert response.status_code == 202
        job_id = response.json()['id']
        wait_for_job(client, job_id)
        report = client.get(f'/api/analyses/{job_id}/report').json()
        count = report['metadata']['frame_count']
        assert count == imageio_ffmpeg.count_frames_and_secs(str(SAMPLE))[0]
        assert report['quality']['pose_coverage'] > 0
        assert any(frame['metrics']['knee_flexion'] is not None for frame in report['frames'])
        assert any(frame['landmarks'].get('left_ankle') for frame in report['frames'])
        assert any(frame['landmarks'].get('right_ankle') for frame in report['frames'])
        csv_response = client.get(report['urls']['csv'])
        rows = list(csv.DictReader(io.StringIO(csv_response.text)))
        assert len(rows) == count
        assert 'knee_flexion' in rows[0] and 'right_ankle_visibility' in rows[0]
        assert client.get(report['urls']['json']).json()['metadata']['frame_count'] == count
        # Decode the actual rendered file through its last frame.
        capture = cv2.VideoCapture(str(tmp_path / job_id / 'overlay.mp4'))
        decoded = 0
        try:
            while capture.read()[0]:
                decoded += 1
        finally:
            capture.release()
        assert decoded == count
        pose_path = tmp_path / job_id / 'pose.json'
        tracked_at = pose_path.stat().st_mtime_ns
        # Pick a visible contact frame from this particular normalized sample.
        manual_frame = 345
        revised = client.patch(f'/api/analyses/{job_id}', json={
            'contact_frame': manual_frame, 'foot': 'left', 'shot_type': 'free_kick',
            'camera_view': 'oblique', 'camera_motion': 'moving', 'scale_m_per_px': 0.01,
        })
        assert revised.status_code == 202
        wait_for_job(client, job_id)
        updated = client.get(f'/api/analyses/{job_id}/report').json()
        assert updated['contact']['source'] == 'manual'
        assert updated['contact']['frame'] == manual_frame
        assert updated['settings']['foot'] == 'left'
        assert updated['settings']['shot_type'] == 'free_kick'
        assert updated['frames'][manual_frame]['phase'] == 'contact'
        assert any('Camera movement' in warning for warning in updated['quality']['warnings'])
        assert next(d for d in updated['metric_definitions'] if d['key'] == 'foot_speed')['unit'] == 'm/s'
        assert pose_path.stat().st_mtime_ns == tracked_at
        assert client.patch(f'/api/analyses/{job_id}', json={'contact_frame': count}).status_code == 422
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get(f'/api/analyses/{job_id}').json()['status'] == 'complete'
        assert restarted.get(f'/api/analyses/{job_id}/report').json()['contact']['frame'] == manual_frame
        # Older normalization versions could lose frames or use different
        # image dimensions. Their manual contact/scale must not silently
        # carry over to a newly standardized video with different geometry.
        old = json.loads(pose_path.read_text())
        old['cache_version'] = 0
        old['metadata'].update(frame_count=count - 2, width=240, height=180)
        pose_path.write_text(json.dumps(old))
        assert restarted.patch(f'/api/analyses/{job_id}', json={}).status_code == 202
        migrated = wait_for_job(restarted, job_id)
        assert migrated['settings']['contact_frame'] is None
        assert migrated['settings']['scale_m_per_px'] is None
        repaired = restarted.get(f'/api/analyses/{job_id}/report').json()
        assert repaired['metadata']['frame_count'] == count
        assert repaired['contact']['source'] != 'manual'
        assert any('contact selection was cleared' in warning for warning in repaired['quality']['warnings'])
        assert any('calibration was cleared' in warning for warning in repaired['quality']['warnings'])
