"""Opt-in Chromium regression: real footage, local server, no mocked reports."""
import csv
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'input/instep_side.mp4'
pytestmark = pytest.mark.skipif(
    os.environ.get('KICKLAB_RUN_BROWSER_E2E') != '1' or not SAMPLE.is_file(),
    reason='Set KICKLAB_RUN_BROWSER_E2E=1 and provide the optional soccer sample and Playwright Chromium.',
)


def test_upload_review_playback_graph_contact_exports_and_mobile(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    with socket.socket() as address:
        address.bind(('127.0.0.1', 0))
        port = address.getsockname()[1]
    origin = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'KICKLAB_DATA_DIR': str(tmp_path / 'sessions'),
           'MPLCONFIGDIR': str(tmp_path / 'matplotlib')}
    artifacts = Path(os.environ.get('KICKLAB_BROWSER_ARTIFACTS', tmp_path / 'screenshots'))
    artifacts.mkdir(parents=True, exist_ok=True)
    with (tmp_path / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app',
                                   '--host', '127.0.0.1', '--port', str(port)],
                                  cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urlopen(origin + '/api/health', timeout=1) as response:
                        assert response.status == 200
                    break
                except (URLError, OSError):
                    if server.poll() is not None or time.monotonic() > deadline:
                        pytest.fail((tmp_path / 'server.log').read_text())
                    time.sleep(0.1)
            with sync_playwright() as runtime:
                browser = runtime.chromium.launch()
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(origin)
                expect(page.locator('#upload-panel')).to_be_visible()
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
                page.screenshot(path=str(artifacts / 'upload-desktop.png'), full_page=True)
                page.locator('#video-file').set_input_files(SAMPLE)
                page.locator('#shot-type').select_option('penalty')
                page.locator('#analyze-button').click()
                expect(page.locator('#results-panel')).to_be_visible(timeout=180000)
                page.wait_for_function("document.querySelector('video').readyState >= 2")
                assert page.locator('video').evaluate('(v) => v.videoWidth') == 480
                assert page.locator('#chart-series path').count() > 0
                assert page.locator('#phase-timeline button').count() > 0
                assert not page.locator('#quality-notice').evaluate('(details) => details.open')
                page.locator('#quality-notice summary').click()
                expect(page.locator('#quality-notice p').first).to_be_visible()
                page.locator('#quality-notice summary').click()

                def seek(frame):
                    page.locator('#frame-slider').evaluate('''(slider, value) => {
                        slider.value = value;
                        slider.dispatchEvent(new Event('input', {bubbles: true}));
                    }''', frame)
                    page.wait_for_function('''(frame) => Math.abs(
                        document.querySelector('video').currentTime - frame / state.report.metadata.fps
                    ) < 0.002 && state.currentFrame === frame''', arg=frame)

                seek(345)
                page.locator('#next-frame').click()
                expect(page.locator('#frame-badge')).to_have_text('FRAME 346')
                page.locator('#previous-frame').click()
                expect(page.locator('#frame-badge')).to_have_text('FRAME 345')
                page.locator('[data-metric="foot_speed"]').click()
                expect(page.locator('#chart-current-unit')).to_have_text('px/s')
                page.locator('#chart-hit-area').click(position={'x': 400, 'y': 50})
                assert 300 < page.evaluate('state.currentFrame') < 450
                page.locator('#overlay-toggle').uncheck()
                assert page.evaluate('state.overlay') is False
                page.locator('#overlay-toggle').check()
                initial = page.locator('video').evaluate('(v) => v.currentTime')
                page.locator('#play-pause').click()
                page.wait_for_function('(initial) => document.querySelector("video").currentTime > initial + 0.1', arg=initial)
                page.locator('#play-pause').click()
                assert page.locator('video').evaluate('(v) => v.paused')

                seek(345)
                job_id = page.url.split('=')[-1]
                pose_path = tmp_path / 'sessions' / job_id / 'pose.json'
                tracked_at = pose_path.stat().st_mtime_ns
                page.locator('#set-contact').click()
                expect(page.locator('#results-panel')).to_be_visible(timeout=120000)
                expect(page.locator('#contact-status')).to_have_text('Manually selected', timeout=120000)
                assert page.evaluate('state.report.contact.frame') == 345
                assert pose_path.stat().st_mtime_ns == tracked_at

                page.locator('#edit-settings').click()
                page.locator('#settings-form [name="foot"]').select_option('left')
                page.locator('#settings-form [name="shot_type"]').select_option('free_kick')
                page.locator('#settings-form [name="camera_view"]').select_option('oblique')
                page.locator('#settings-form [name="camera_motion"]').select_option('moving')
                page.locator('#settings-form [name="scale_m_per_px"]').fill('0.01')
                page.locator('#save-settings').click()
                expect(page.locator('#results-panel')).to_be_visible(timeout=120000)
                expect(page.locator('#result-context')).to_contain_text('Left foot', timeout=120000)
                expect(page.locator('#chart-current-unit')).to_have_text('m/s')
                assert page.evaluate('state.report.settings.shot_type') == 'free_kick'
                assert pose_path.stat().st_mtime_ns == tracked_at
                with page.expect_download() as download:
                    page.locator('#download-csv').click()
                csv_path = download.value.path()
                with open(csv_path) as content:
                    assert len(list(csv.DictReader(content))) == page.evaluate('state.report.frames.length')
                for name in ['json', 'video']:
                    with page.expect_download() as download:
                        page.locator('#download-' + name).click()
                    assert Path(download.value.path()).stat().st_size > 0
                page.screenshot(path=str(artifacts / 'results-desktop.png'), full_page=True)
                for width, height in [(768, 1024), (390, 844)]:
                    page.set_viewport_size({'width': width, 'height': height})
                    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
                    page.screenshot(path=str(artifacts / f'results-{width}.png'), full_page=True)
                # A failed request must reconnect without a page refresh or
                # duplicate analysis. The next request goes to the real server.
                page.route(f'**/api/analyses/{job_id}', lambda route: route.abort(), times=1)
                page.reload()
                expect(page.locator('#reload-session')).to_be_visible()
                expect(page.locator('#connection-status')).to_have_class(re.compile('offline'))
                expect(page.locator('#results-panel')).to_be_visible(timeout=20000)
                expect(page.locator('#global-error')).to_be_hidden()
                expect(page.locator('#contact-status')).to_have_text('Manually selected', timeout=120000)

                # Exercise an actual browser without requestVideoFrameCallback.
                fallback = browser.new_page(viewport={'width': 1000, 'height': 800})
                fallback.on('pageerror', lambda error: errors.append(str(error)))
                fallback.add_init_script('''
                    HTMLVideoElement.prototype.requestVideoFrameCallback = undefined;
                    HTMLVideoElement.prototype.cancelVideoFrameCallback = undefined;
                ''')
                fallback.goto(origin + '/#analysis=' + job_id)
                expect(fallback.locator('#results-panel')).to_be_visible()
                fallback.wait_for_function("document.querySelector('video').readyState >= 2")
                fallback.evaluate('''() => { for(let frame=300;frame<=320;frame++) seekFrame(frame); }''')
                fallback.wait_for_function('state.currentFrame === 320 && state.pendingSeekFrame === null')
                fallback.evaluate('updateFromTime((320 + 0.75) / fps())')
                assert fallback.evaluate('state.currentFrame') == 320
                fallback.locator('#play-pause').click()
                fallback.wait_for_function('state.currentFrame > 322')
                fallback.locator('#play-pause').click()
                fallback.wait_for_function('state.animationFrame === null')
                fallback.locator('#new-analysis').click()
                assert fallback.evaluate('''state.report === null && state.graph === null
                    && state.playbackCallback === null && state.animationFrame === null
                    && !video.getAttribute('src')''')
                fallback.close()

                # Corrupt saved results become a visible repair action, and
                # an intact pose cache avoids redoing model inference.
                (pose_path.parent / 'report.json').write_text('{broken report')
                page.reload()
                expect(page.locator('#retry-analysis')).to_be_visible(timeout=15000)
                expect(page.locator('#processing-failure')).to_contain_text('missing or damaged')
                page.locator('#retry-analysis').click()
                expect(page.locator('#results-panel')).to_be_visible(timeout=120000)
                assert pose_path.stat().st_mtime_ns == tracked_at

                # A damaged pose cache is rebuilt using the original footage.
                pose_path.write_text('{broken tracking')
                response = page.request.patch(origin + '/api/analyses/' + job_id, data={})
                assert response.status == 202
                page.reload()
                expect(page.locator('#results-panel')).to_be_visible(timeout=180000)
                expect(page.locator('#contact-status')).to_have_text('Manually selected')
                assert pose_path.stat().st_mtime_ns > tracked_at
                page.locator('#new-analysis').click()
                page.locator('#video-file').set_input_files({'name': 'empty.mp4', 'mimeType': 'video/mp4', 'buffer': b''})
                expect(page.locator('#global-error')).to_contain_text('empty')
                assert page.evaluate('state.selectedFile === null')
                page.locator('#video-file').set_input_files({'name': 'corrupt.mp4', 'mimeType': 'video/mp4', 'buffer': b'not a video'})
                page.locator('#analyze-button').click()
                expect(page.locator('#processing-failure')).to_be_visible(timeout=15000)
                expect(page.locator('#retry-analysis')).to_be_visible()
                with page.expect_response(lambda response: response.request.method == 'PATCH') as retried:
                    page.locator('#retry-analysis').click()
                assert retried.value.status == 202
                expect(page.locator('#processing-failure')).to_be_visible(timeout=15000)
                assert not errors, errors
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
