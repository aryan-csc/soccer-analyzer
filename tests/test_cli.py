"""Exercise the standalone CLI without requiring any web-server packages."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'input/instep_side.mp4'


def run_cli(*args, cwd=ROOT, timeout=180):
    return subprocess.run([str(ROOT / 'run.sh'), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)


def test_cli_import_does_not_load_web_server():
    code = '''
import sys
class RejectWeb:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'app', 'fastapi', 'uvicorn'}:
            raise AssertionError('CLI imported a web server: ' + fullname)
sys.meta_path.insert(0, RejectWeb())
from scripts.analyze_kick import main
from scripts import process_landmarks
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_without_arguments_prints_cli_help():
    result = run_cli()
    assert result.returncode == 0, result.stderr
    assert '--contact-frame' in result.stdout and '--foot' in result.stdout
    assert 'uvicorn' not in result.stdout and 'http://localhost' not in result.stdout


def test_cli_validates_settings_without_tracebacks(tmp_path):
    result = run_cli('missing.mp4', '--contact-frame', '-1', cwd=tmp_path)
    assert result.returncode == 1
    assert 'contact_frame' in result.stderr and 'Analysis failed:' in result.stderr
    assert 'Traceback' not in result.stderr


def test_launcher_resolves_relative_paths_from_callers_directory(tmp_path):
    working = tmp_path / 'folder with spaces'
    working.mkdir()
    for package in ['src', 'scripts']:
        (working / package).mkdir()
        (working / package / '__init__.py').write_text("raise RuntimeError('Unrelated caller package imported')")
    clip = working / 'empty scene.mp4'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'mp4v'), 12, (320, 240))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    result = run_cli(clip.name, '--output', 'results folder', cwd=working)
    assert result.returncode == 1
    assert 'No visible human pose' in result.stderr
    assert (working / 'results folder/.analysis/video.mp4').is_file()
    assert clip.is_file()
    assert 'Traceback' not in result.stderr


@pytest.mark.skipif(os.environ.get('KICKLAB_RUN_VIDEO_E2E') != '1' or not SAMPLE.is_file(),
                    reason='Opt in with KICKLAB_RUN_VIDEO_E2E=1 and provide the soccer sample.')
def test_real_cli_analysis_reports_plots_and_manual_contact(tmp_path):
    output = tmp_path / 'analysis'
    result = run_cli(SAMPLE, '--output', output, '--foot', 'right')
    assert result.returncode == 0, result.stderr
    assert 'SOCCER KICK ANALYSIS' in result.stdout
    assert 'MEASUREMENTS' in result.stdout and 'MEASUREMENT LIMITS' in result.stdout
    for filename in ['.analysis/video.mp4', 'overlay.mp4', '.analysis/metrics.csv', '.analysis/report.json', 'report.txt', 'metrics.png', '.analysis/pose.json']:
        assert (output / filename).stat().st_size > 0
    report = json.loads((output / '.analysis/report.json').read_text())
    assert report['metadata']['frame_count'] == 766
    with (output / '.analysis/metrics.csv').open() as file:
        assert len(list(csv.DictReader(file))) == 766
    image = cv2.imread(str(output / 'metrics.png'))
    assert image is not None and image.shape[0] > 1000 and image.std() > 10
    pose_mtime = (output / '.analysis/pose.json').stat().st_mtime_ns
    revised = run_cli(SAMPLE, '--output', output, '--foot', 'right', '--contact-frame', '346', '--reuse-pose', '--no-plots')
    assert revised.returncode == 0, revised.stderr
    assert 'Using saved pose tracking' in revised.stderr
    assert 'frame 346 (manual)' in revised.stdout
    assert (output / '.analysis/pose.json').stat().st_mtime_ns == pose_mtime
    assert not (output / 'metrics.png').exists()
    assert json.loads((output / '.analysis/report.json').read_text())['contact']['source'] == 'manual'


def test_default_output_uses_clear_output_folder(tmp_path):
    result = run_cli('missing.mp4', cwd=tmp_path)
    assert result.returncode == 1
    assert (tmp_path / 'output/missing/.analysis').is_dir()
    assert not (tmp_path / 'data').exists()


def test_cli_never_overwrites_input_with_export(tmp_path):
    output = tmp_path / 'results'
    output.mkdir()
    source = output / 'overlay.mp4'
    source.write_bytes(b'preserve original')
    result = run_cli(source, '--output', output)
    assert result.returncode == 1
    assert 'overwrite the source' in result.stderr
    assert source.read_bytes() == b'preserve original'
