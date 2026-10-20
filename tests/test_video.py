"""Exercise the real media decoder/encoder; no network or display is required."""

from pathlib import Path
import json
import shutil
import subprocess
from fractions import Fraction

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from src.extractor import VideoAnalysisError, extract_pose
from src.renderer import render_overlay
from src.pipeline import POSE_CACHE_VERSION, file_digest, load_pose_cache, normalize_video, run_analysis


def _write_clip(path: Path, count=8, fps=12, size=(320, 240), sar="1/1"):
    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=fps, codec="libx264", macro_block_size=1,
        output_params=["-movflags", "+faststart", "-vf", f"setsar={sar}",
                       "-r", str(Fraction(fps).limit_denominator(1001))],
    )
    writer.send(None)
    try:
        for index in range(count):
            frame = np.full((size[1], size[0], 3), (20 + index * 8, 45, 35), dtype=np.uint8)
            writer.send(frame)
    finally:
        writer.close()


def test_renderer_encodes_all_frames_and_keeps_missing_landmarks(tmp_path):
    source, output = tmp_path / "input.mp4", tmp_path / "annotated.mp4"
    _write_clip(source)
    frames = []
    for index in range(8):
        # Known image-plane geometry provides real, exact 90-degree joint angles
        # in a small renderer fixture. It never enters the application's data.
        landmarks = {} if index == 3 else {
            "right_shoulder": {"x": 100, "y": 100, "visibility": 1.0},
            "right_hip": {"x": 100, "y": 150, "visibility": 1.0},
            "right_knee": {"x": 150, "y": 150, "visibility": 1.0},
            "right_ankle": {"x": 150, "y": 200, "visibility": 1.0},
        }
        frames.append({
            "frame": index, "time": index / 12, "landmarks": landmarks,
            "phase": "backswing" if index < 4 else "follow_through",
            "metrics": {"knee_flexion": 90.0 if landmarks else None,
                        "hip_angle": 90.0 if landmarks else None},
        })
    report = {"settings": {"foot": "right"}, "frames": frames,
              "contact": {"frame": 4, "source": "manual"}}
    events = []
    assert render_overlay(source, report, output, lambda value, message: events.append((value, message))) == output
    assert events[-1][0] == 1
    content = output.read_bytes()
    assert 0 < content.index(b"moov") < content.index(b"mdat")
    capture = cv2.VideoCapture(str(output))
    decoded = []
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12, abs=0.01)
        assert int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little") in (b"avc1", b"h264")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
    finally:
        capture.release()
    assert len(decoded) == 8
    assert decoded[0].shape == (240, 320, 3)
    # Distinct final source frames remain visible below the overlay, proving
    # output is not truncated at contact or frozen on the last detected pose.
    assert decoded[-1][225:, :, 2].mean() > decoded[0][225:, :, 2].mean() + 40
    assert decoded[0][:80].std() > 15


def test_renderer_rejects_frame_mismatch_without_partial_output(tmp_path):
    source, output = tmp_path / "input.mp4", tmp_path / "annotated.mp4"
    _write_clip(source)
    with pytest.raises(ValueError, match="out of sync"):
        render_overlay(source, {"frames": [{"frame": 0}]}, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".annotated-*.mp4"))


def test_extractor_rejects_non_video(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_text("not a media file")
    with pytest.raises(VideoAnalysisError, match="Cannot open"):
        extract_pose(source)


def test_extractor_reports_missing_human_pose(tmp_path):
    source = tmp_path / "empty-scene.mp4"
    _write_clip(source, count=3)
    with pytest.raises(VideoAnalysisError, match="No visible human pose"):
        extract_pose(source)


def test_pose_cache_from_another_source_is_not_reused(tmp_path):
    import json
    import shutil
    from src.pipeline import run_analysis

    source = tmp_path / 'new-empty-scene.mp4'
    _write_clip(source, count=3)
    output = tmp_path / 'reused-output'
    output.mkdir()
    shutil.copyfile(source, output / 'video.mp4')
    (output / 'pose.json').write_text(json.dumps({'source_sha256': 'previous-source'}))
    # A different source must run real tracking and detect the absent person,
    # rather than using the stale cache (which intentionally has no frames).
    with pytest.raises(VideoAnalysisError, match='No visible human pose'):
        run_analysis(source, output, {'foot': 'right'}, reuse_pose=True)


@pytest.mark.parametrize('artifact', ['video.mp4', 'overlay.mp4'])
def test_analysis_never_overwrites_its_source(tmp_path, artifact):
    source = tmp_path / artifact
    _write_clip(source, count=3)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='source'):
        run_analysis(source, tmp_path, {'foot': 'right'})
    assert source.read_bytes() == original


def test_normalization_never_overwrites_its_source(tmp_path):
    source = tmp_path / 'source.mp4'
    _write_clip(source, count=3)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='source'):
        normalize_video(source, source)
    assert source.read_bytes() == original


@pytest.mark.parametrize('filename', ['video.encoding.mp4', 'video.audio.mp4'])
def test_temporary_video_names_cannot_collide_with_input_files(tmp_path, filename):
    source = tmp_path / filename
    _write_clip(source, count=3)
    original = source.read_bytes()
    normalize_video(source, tmp_path / 'video.mp4')
    assert source.is_file() and source.read_bytes() == original
    assert not list(tmp_path.glob('.*.encoding.mp4'))
    assert not list(tmp_path.glob('.*.audio.mp4'))


@pytest.mark.parametrize('contents', ['{truncated', 'null', '[]', '{}'])
def test_damaged_pose_cache_is_rebuilt(tmp_path, contents):
    source = tmp_path / 'source.mp4'
    _write_clip(source, count=3)
    output = tmp_path / 'output'
    output.mkdir()
    shutil.copyfile(source, output / 'video.mp4')
    (output / 'pose.json').write_text(contents)
    # This input really has no person. Rebuilding should report this rather
    # than failing while reading a broken cache or serving stale measurements.
    with pytest.raises(VideoAnalysisError, match='No visible human pose'):
        run_analysis(source, output, {'foot': 'right'}, reuse_pose=True)


def test_pose_cache_requires_matching_video_bytes_and_frame_timing(tmp_path):
    video = tmp_path / 'video.mp4'
    _write_clip(video, count=3)
    digest = file_digest(video)
    cache = {'cache_version': POSE_CACHE_VERSION, 'source_sha256': digest, 'video_sha256': digest,
             'metadata': {'width': 320, 'height': 240, 'fps': 12, 'frame_count': 3, 'duration': 0.25},
             'frames': [{'frame': index, 'time': index / 12, 'landmarks': {}} for index in range(3)]}
    pose = tmp_path / 'pose.json'
    pose.write_text(json.dumps(cache))
    assert load_pose_cache(tmp_path, digest) == cache
    assert load_pose_cache(tmp_path, 'different source') is None
    cache['frames'][1]['time'] = 0.1
    pose.write_text(json.dumps(cache))
    assert load_pose_cache(tmp_path, digest) is None
    cache['frames'][1]['time'] = 1 / 12
    pose.write_text(json.dumps(cache))
    video.write_bytes(b'damaged normalized media')
    assert load_pose_cache(tmp_path, digest) is None


@pytest.mark.parametrize('audio_duration', [0.1, 2.0])
def test_audio_length_never_truncates_or_extends_video_playback(tmp_path, audio_duration):
    silent = tmp_path / 'silent.mp4'
    source = tmp_path / 'audio.mov'
    _write_clip(silent, count=12, fps=24)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(silent),
                    '-f', 'lavfi', '-i', f'sine=frequency=440:duration={audio_duration}',
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', str(source)], check=True)
    normalized = tmp_path / 'normalized.mp4'
    metadata = normalize_video(source, normalized)
    assert metadata['frame_count'] == 12
    report = {'frames': [{'frame': i, 'time': i / 24} for i in range(12)]}
    overlay = tmp_path / 'overlay.mp4'
    render_overlay(normalized, report, overlay)
    for output in [normalized, overlay]:
        reader = imageio_ffmpeg.read_frames(str(output))
        try:
            assert next(reader)['duration'] == pytest.approx(0.5, abs=0.08)
        finally:
            reader.close()
        capture = cv2.VideoCapture(str(output))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        finally:
            capture.release()


@pytest.mark.parametrize('rotation,expected_size', [(90, (240, 320)), (180, (320, 240)), (270, (240, 320))])
def test_phone_rotation_is_baked_into_pixels_once(tmp_path, rotation, expected_size):
    original, rotated, output = (tmp_path / name for name in ['original.mp4', 'phone.mov', 'normalized.mp4'])
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*'mp4v'), 24, (320, 240))
    assert writer.isOpened()
    for _ in range(4):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:120, :160] = (30, 150, 240)
        frame[120:, 160:] = (200, 80, 40)
        writer.write(frame)
    writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-display_rotation', str(rotation),
                    '-i', str(original), '-c', 'copy', str(rotated)], check=True)
    capture = cv2.VideoCapture(str(rotated))
    try:
        assert int(capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360 == (360 - rotation) % 360
        ok, expected = capture.read()
        assert ok
    finally:
        capture.release()
    metadata = normalize_video(rotated, output)
    assert (metadata['width'], metadata['height']) == expected_size
    capture = cv2.VideoCapture(str(output))
    try:
        ok, actual = capture.read()
        assert ok and actual.shape == expected.shape
        assert np.abs(actual.astype(float) - expected.astype(float)).mean() < 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    finally:
        capture.release()


@pytest.mark.parametrize('fps', [30000 / 1001, 60000 / 1001, 24000 / 1001])
def test_fractional_frame_rates_do_not_lose_decoded_frames(tmp_path, fps):
    source, normalized, overlay = (tmp_path / name for name in ['source.mp4', 'normalized.mp4', 'overlay.mp4'])
    _write_clip(source, count=24, fps=fps)
    metadata = normalize_video(source, normalized)
    assert metadata['fps'] == pytest.approx(fps, rel=1e-5)
    assert metadata['frame_count'] == 24
    report = {'frames': [{'frame': index, 'time': index / metadata['fps']} for index in range(24)]}
    render_overlay(normalized, report, overlay)
    for output in [normalized, overlay]:
        assert imageio_ffmpeg.count_frames_and_secs(str(output))[0] == 24
        capture = cv2.VideoCapture(str(output))
        try:
            count = 0
            while capture.read()[0]:
                count += 1
            assert count == 24
        finally:
            capture.release()
@pytest.mark.parametrize("size,sar,expected", [
    ((320, 240), "2/1", (640, 240)),
    ((320, 240), "1/2", (160, 240)),
    ((1920, 1080), "1/1", (1280, 720)),
    ((1080, 1920), "1/1", (720, 1280)),
    ((720, 480), "8/9", (640, 480)),
])
def test_normalization_preserves_display_geometry_and_timing(tmp_path, size, sar, expected):
    from src.pipeline import normalize_video

    source, output = tmp_path / "original.mov", tmp_path / "normalized.mp4"
    _write_clip(source, count=4, fps=24, size=size, sar=sar)
    metadata = normalize_video(source, output)
    assert (metadata["width"], metadata["height"]) == expected
    assert metadata["fps"] == pytest.approx(24, abs=0.001)
    assert metadata["frame_count"] == 4
    assert metadata["duration"] == pytest.approx(4 / 24, abs=0.001)
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.get(cv2.CAP_PROP_SAR_NUM) == capture.get(cv2.CAP_PROP_SAR_DEN) == 1
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape[:2] == (expected[1], expected[0])
            decoded += 1
        assert decoded == 4
    finally:
        capture.release()
