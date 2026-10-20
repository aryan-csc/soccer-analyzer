"""Shared local video pipeline used by the web application and CLI."""
from __future__ import annotations

import csv
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from uuid import uuid4
from typing import Callable

import cv2
import imageio_ffmpeg

MAX_DURATION = 120.0
MAX_DIMENSION = 1280
MAX_BYTES = 200 * 1024 * 1024
POSE_CACHE_VERSION = 2
Progress = Callable[[float, str], None]


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('This file cannot be opened as a video. Upload a valid MP4 or MOV.')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or min(width, height) < 32 or not math.isfinite(fps) or fps <= 0 or count < 3:
            raise ValueError('The video has no usable frames or timing information. Export it as MP4 and retry.')
        duration = count / fps
        if duration > MAX_DURATION + 0.1:
            raise ValueError('Please trim the clip to one kick, at most 120 seconds.')
        if fps > 240:
            raise ValueError('Please export the clip at 240 fps or less before uploading.')
        return {'width': width, 'height': height, 'fps': fps, 'frame_count': count, 'duration': duration}
    finally:
        cap.release()


def normalize_video(source: Path, target: Path) -> dict:
    """Decode MOV/MP4 orientation and variable timing into a seekable H264 CFR clip."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('The normalized output must have a different path from the source video.')
    meta = probe_video(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = target.with_name(f'.{target.stem}-{token}.encoding.mp4')
    aligned = target.with_name(f'.{target.stem}-{token}.audio.mp4')
    # A decimal expression such as fps=29.970029970 can produce a huge
    # encoder time base and overflow MP4 composition timestamps. Rational
    # rates and a bounded track time base preserve every decoded frame.
    fps = Fraction(meta['fps']).limit_denominator(1001)
    track_timescale = fps.numerator * max(1, 90000 // fps.numerator)
    # Preserve display geometry, including anamorphic (non-square-pixel) input.
    # Applying setsar=1 to coded iw/ih alone would distort measured joint angles.
    # Round both output dimensions to even values required by H.264 yuv420p.
    fit = f"min(1,min({MAX_DIMENSION}/(iw*sar),{MAX_DIMENSION}/ih))"
    filters = (f"scale=w='max(2,2*trunc(iw*sar*{fit}/2))':"
               f"h='max(2,2*trunc(ih*{fit}/2))',setsar=1,fps={fps}")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(source), '-map', '0:v:0', '-map', '0:a?', '-vf', filters,
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-video_track_timescale', str(track_timescale),
               '-c:a', 'aac', '-b:a', '128k', '-af',
               f"apad=whole_dur={meta['duration']:.9f},atrim=duration={meta['duration']:.9f}",
               '-movflags', '+faststart', '-t', str(MAX_DURATION),
               str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise ValueError('Video conversion failed. Re-export the clip as H.264 MP4. ' + result.stderr[-600:])
        normalized = probe_video(temporary)
        if normalized['frame_count'] < 3:
            raise ValueError('The converted video is too short to analyze.')
        if abs(normalized['duration'] - meta['duration']) > 0.001:
            # CFR conversion can drop/duplicate source frames. Align audio to
            # the actual decoded video duration, copying the encoded video.
            duration = normalized['duration']
            mux = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', str(temporary), '-map', '0:v:0', '-map', '0:a?', '-c:v', 'copy',
                   '-c:a', 'aac', '-af', f'apad=whole_dur={duration:.9f},atrim=duration={duration:.9f}',
                   '-movflags', '+faststart', str(aligned)]
            result = subprocess.run(mux, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise ValueError('Audio alignment failed. Re-export the clip as H.264 MP4. ' + result.stderr[-600:])
            aligned.replace(temporary)
        temporary.replace(target)
        normalized['source_width'] = meta['width']
        normalized['source_height'] = meta['height']
        normalized['source_fps'] = meta['fps']
        return normalized
    finally:
        temporary.unlink(missing_ok=True)
        aligned.unlink(missing_ok=True)


def file_digest(path: Path) -> str:
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load_pose_cache(directory: Path, source_digest: str) -> dict | None:
    """Reuse only intact tracking matched to both source and normalized video.

    Interrupted JSON writes, deleted media, changed timing, and older cache
    formats trigger a clean rebuild. They must not trap Retry in a failure loop.
    """
    from src.physics_utils import _validate

    try:
        raw = json.loads((directory / 'pose.json').read_text())
        if (not isinstance(raw, dict) or raw.get('cache_version') != POSE_CACHE_VERSION
                or raw.get('source_sha256') != source_digest
                or raw.get('video_sha256') != file_digest(directory / 'video.mp4')):
            return None
        metadata = probe_video(directory / 'video.mp4')
        frames, _, _ = _validate(raw['frames'], raw['metadata'], None)
        if frames != raw['frames'] or len(frames) != metadata['frame_count']:
            return None
        if any(frame['frame'] != index or not math.isclose(frame['time'], index / metadata['fps'], abs_tol=1e-6)
               for index, frame in enumerate(frames)):
            return None
        if any(raw['metadata'][key] != metadata[key] for key in ('width', 'height', 'frame_count')):
            return None
        if not math.isclose(raw['metadata']['fps'], metadata['fps'], rel_tol=1e-6):
            return None
        return raw
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def export_csv(report: dict, path: Path) -> None:
    frames = report['frames']
    metrics = [item['key'] for item in report['metric_definitions']]
    names = sorted({name for frame in frames for name in frame.get('landmarks', {})})
    fields = ['frame', 'time', 'phase', *metrics]
    fields += [f'{name}_{axis}' for name in names for axis in ('x', 'y', 'visibility')]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            row = {key: frame[key] for key in ('frame', 'time', 'phase')}
            row.update({key: frame['metrics'].get(key) for key in metrics})
            for name in names:
                landmark = frame.get('landmarks', {}).get(name, {})
                row.update({f'{name}_{axis}': landmark.get(axis) for axis in ('x', 'y', 'visibility')})
            writer.writerow(row)
    temporary.replace(path)


def run_analysis(source: Path, directory: Path, settings: dict, progress: Progress | None = None,
                 reuse_pose: bool = False) -> dict:
    # Lazy imports keep API startup and validation independent of model initialization.
    from src.extractor import extract_pose
    from src.physics_utils import analyze_kick
    from src.renderer import render_overlay

    directory = Path(directory)
    source = Path(source)
    if source.resolve() in {(directory / name).resolve() for name in ('video.mp4', 'overlay.mp4')}:
        raise ValueError('Choose an output directory that does not overwrite the source video.')
    directory.mkdir(parents=True, exist_ok=True)
    update = progress or (lambda value, message: None)
    raw_path = directory / 'pose.json'
    video_path = directory / 'video.mp4'
    # A CLI output directory may be reused for a different video. Cached pose
    # data must never silently produce a report for the previous source clip.
    if not source.is_file():
        raise ValueError('The source video is missing. Upload the clip again.')
    source_digest = file_digest(source)
    previous_metadata = None
    if reuse_pose:
        try:
            previous = json.loads(raw_path.read_text())
            if isinstance(previous, dict) and previous.get('source_sha256') == source_digest:
                previous_metadata = previous.get('metadata')
        except (OSError, ValueError):
            pass
    raw = load_pose_cache(directory, source_digest) if reuse_pose else None
    if raw is not None:
        update(0.12, 'Using saved pose tracking')
    if raw is None:
        update(0.03, 'Preparing browser-compatible video')
        metadata = normalize_video(Path(source), video_path)
        update(0.12, 'Tracking both legs and upper body')
        raw = extract_pose(video_path, progress=lambda value, message: update(0.12 + 0.55 * value, message))
        raw['source_sha256'] = source_digest
        raw['cache_version'] = POSE_CACHE_VERSION
        raw['video_sha256'] = file_digest(video_path)
        raw['metadata'].update({k: v for k, v in metadata.items() if k.startswith('source_')})
        write_json(raw_path, raw)
    update(0.69, 'Measuring 2D motion and estimating kick phases')
    settings = dict(settings)
    rebuild_notes = []
    if isinstance(previous_metadata, dict):
        previous_fps = previous_metadata.get('fps')
        if (previous_metadata.get('frame_count') != raw['metadata']['frame_count']
                or not isinstance(previous_fps, (int, float))
                or not math.isclose(previous_fps, raw['metadata']['fps'], rel_tol=1e-6)):
            if settings.get('contact_frame') is not None:
                settings['contact_frame'] = None
                rebuild_notes.append('The standardized video timing changed when tracking was rebuilt. '
                                     'The previous contact selection was cleared; select contact again in this video.')
        if any(previous_metadata.get(key) != raw['metadata'][key] for key in ('width', 'height')):
            if settings.get('scale_m_per_px') is not None:
                settings['scale_m_per_px'] = None
                rebuild_notes.append('The analyzed image dimensions changed when tracking was rebuilt. '
                                     'The previous spatial calibration was cleared; measure the scale in this video again.')
    report = analyze_kick(raw['frames'], raw['metadata'], settings)
    report['quality']['warnings'][:0] = rebuild_notes
    update(0.75, 'Rendering annotated video')
    render_overlay(video_path, report, directory / 'overlay.mp4',
                   progress=lambda value, message: update(0.75 + 0.23 * value, message))
    export_csv(report, directory / 'metrics.csv')
    write_json(directory / 'report.json', report)
    update(1.0, 'Analysis complete')
    return report
