"""Render measured soccer kick overlays as browser-compatible H.264 video."""

from __future__ import annotations

import math
from fractions import Fraction
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg

# OpenCV colors are BGR. The kicking and plant sides keep distinct colors.
KICK_COLOR = (103, 226, 123)
PLANT_COLOR = (240, 189, 92)
WHITE = (245, 247, 244)
GRAY = (192, 204, 199)
BONES = (
    ("shoulder", "elbow"), ("elbow", "wrist"),
    ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ("ankle", "heel"), ("heel", "foot_index"), ("ankle", "foot_index"),
)
PHASE_LABELS = {
    "approach": "Approach", "plant": "Plant-foot placement",
    "plant_foot": "Plant-foot placement", "backswing": "Backswing",
    "forward_swing": "Forward swing", "contact": "Ball-contact estimate",
    "ball_contact": "Ball-contact estimate", "follow_through": "Follow-through",
    "unclassified": "Phase unavailable", "unknown": "Phase unavailable",
}


def _point(landmarks: dict, name: str) -> tuple[int, int] | None:
    point = landmarks.get(name)
    if not point or point.get("visibility", 1) < 0.5:
        return None
    x, y = point.get("x"), point.get("y")
    if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
        return None
    return round(x), round(y)


def _number(value: object, unit: str = "deg") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unavailable"
    return f"{value:.1f} {unit}"


def _text(frame, label, point, scale, color=WHITE, thickness=1):
    cv2.putText(frame, label, (point[0] + 1, point[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 20, 16), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, row: dict, report: dict):
    """Draw one report row on a BGR image without inferring absent landmarks."""
    height, width = frame.shape[:2]
    scale = max(0.32, min(0.65, width / 1300))
    stroke = max(1, round(min(width, height) / 250))
    landmarks = row.get("landmarks", {})
    foot = report.get("settings", {}).get("foot", "right")
    for side in ("left", "right"):
        color = KICK_COLOR if side == foot else PLANT_COLOR
        for first, second in BONES:
            p1 = _point(landmarks, f"{side}_{first}")
            p2 = _point(landmarks, f"{side}_{second}")
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (15, 30, 20), stroke + 2, cv2.LINE_AA)
                cv2.line(frame, p1, p2, color, stroke, cv2.LINE_AA)
        for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index"):
            point = _point(landmarks, f"{side}_{joint}")
            if point:
                cv2.circle(frame, point, stroke + 2, color, -1, cv2.LINE_AA)
                cv2.circle(frame, point, stroke + 2, WHITE, 1, cv2.LINE_AA)
    for joint in ("shoulder", "hip"):
        first = _point(landmarks, f"left_{joint}")
        second = _point(landmarks, f"right_{joint}")
        if first and second:
            cv2.line(frame, first, second, GRAY, stroke, cv2.LINE_AA)

    metrics = row.get("metrics", {})
    phase = row.get("phase") or "unclassified"
    phase_label = next((item["label"] for item in report.get("phases", [])
                        if item.get("id") == phase), PHASE_LABELS.get(phase, phase.replace("_", " ").title()))
    contact = report.get("contact", {})
    if row.get("frame") == contact.get("frame"):
        phase_label = "Contact: manual selection" if contact.get("source") == "manual" else "Contact: estimate - review"
    speed_unit = "m/s" if report.get("settings", {}).get("scale_m_per_px") else "px/s"
    lines = [
        (f"KICK ANALYSIS  |  {row.get('time', 0):.2f} s  |  Frame {row.get('frame', 0)}", WHITE),
        (phase_label, KICK_COLOR),
        (f"Kick knee flexion  {_number(metrics.get('knee_flexion'))}  |  Hip  {_number(metrics.get('hip_angle'))}", KICK_COLOR),
        (f"Plant knee  {_number(metrics.get('plant_knee_flexion'))}  |  Torso lean  {_number(metrics.get('torso_lean'))}", PLANT_COLOR),
        (f"Foot speed  {_number(metrics.get('foot_speed'), speed_unit)}  |  2D image-plane measurements", GRAY),
    ]
    if not landmarks:
        lines.append(("Pose unavailable in this frame", WHITE))
    # Keep long labels inside narrow videos. The readable dashboard remains the
    # main detailed view, while the downloadable video retains compact context.
    max_text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                         for line, _ in lines)
    if max_text_width > width - 24:
        scale *= max(0.35, (width - 24) / max_text_width)
    line_height = max(14, round(29 * scale / 0.6))
    panel_height = min(height, line_height * len(lines) + 14)
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, panel_height), (22, 32, 27), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
    for index, (label, color) in enumerate(lines):
        _text(frame, label, (10, 12 + line_height * index + line_height // 2), scale, color)

    # Per-joint labels put the same measured angles beside their source joints.
    for name, key, color in (
        (f"{foot}_knee", "knee_flexion", KICK_COLOR),
        (f"{foot}_hip", "hip_angle", KICK_COLOR),
        (f"{'left' if foot == 'right' else 'right'}_knee", "plant_knee_flexion", PLANT_COLOR),
    ):
        point = _point(landmarks, name)
        value = metrics.get(key)
        if point and isinstance(value, (int, float)) and math.isfinite(value):
            x = min(max(5, point[0] + 9), max(5, width - 75))
            y = min(max(panel_height + 14, point[1] - 9), height - 8)
            _text(frame, f"{value:.0f} deg", (x, y), scale, color)
    return frame


def render_overlay(video_path: str | Path, report: dict, output_path: str | Path,
                   progress: Callable[[float, str], None] | None = None) -> Path:
    """Render every source frame and atomically save an H.264/yuv420p MP4.

    ffmpeg stderr is written to a temporary file so verbose encoder errors
    cannot fill a pipe and deadlock the frame writer. Source audio is optional.
    """
    source, destination = Path(video_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Annotated output must have a different path from the source video.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}-{uuid.uuid4().hex}.mp4")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open the video to render the analysis overlay.")
    process = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot render a video with invalid dimensions or frame rate.")
        rows = report.get("frames", [])
        if not rows:
            raise ValueError("The analysis has no frame records to render.")
        rate = Fraction(fps).limit_denominator(1001)
        track_timescale = rate.numerator * max(1, 90000 // rate.numerator)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(rate), "-i", "pipe:0", "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-video_track_timescale", str(track_timescale),
            "-c:a", "aac", "-af",
            f"apad=whole_dur={len(rows) / fps:.9f},atrim=duration={len(rows) / fps:.9f}",
            "-movflags", "+faststart", str(temporary),
        ]
        with tempfile.TemporaryFile(mode="w+b") as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=error_log)
            count = 0
            try:
                while True:
                    decoded, frame = capture.read()
                    if not decoded:
                        break
                    if count >= len(rows) or rows[count].get("frame") != count:
                        raise ValueError("Video and analysis frames are out of sync. Run analysis again.")
                    rendered = draw_overlay(frame, rows[count], report)
                    process.stdin.write(rendered.tobytes())
                    count += 1
                    if progress and count % 10 == 0:
                        progress(min(0.99, count / len(rows)), f"Rendering annotated video: frame {count}")
                process.stdin.close()
                returncode = process.wait(timeout=120)
            except BrokenPipeError as exc:
                process.wait(timeout=10)
                error_log.seek(0)
                error = error_log.read().decode("utf-8", errors="replace")[-2500:]
                raise RuntimeError(f"Video encoding failed: {error or 'ffmpeg stopped accepting frames'}") from exc
            if returncode != 0:
                error_log.seek(0)
                error = error_log.read().decode("utf-8", errors="replace")[-2500:]
                raise RuntimeError(f"Video encoding failed: {error}")
            if count != len(rows):
                raise ValueError(f"Decoded {count} video frames but analysis contains {len(rows)}. Run analysis again.")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("The encoder did not produce an annotated video.")
        temporary.replace(destination)
        if progress:
            progress(1.0, "Annotated video ready")
        return destination
    finally:
        capture.release()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        temporary.unlink(missing_ok=True)
