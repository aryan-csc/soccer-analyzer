"""Headless, image-plane pose extraction for soccer kick analysis.

MediaPipe's 3-D/world landmarks are deliberately not used. A low-visibility
landmark is omitted, rather than carried forward or filled with a made-up point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2

Progress = Callable[[float, str], None]
MIN_VISIBILITY = 0.5
LANDMARK_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


class VideoAnalysisError(ValueError):
    """An input cannot produce a usable video analysis."""


def extract_pose(video_path: str | Path, progress: Progress | None = None) -> dict:
    """Extract visible landmarks, in source-image pixels, for every video frame.

    Use a constant-frame-rate input so frame index / fps is its playback time.
    Both anatomical sides are retained regardless of the selected kicking foot.
    Visibility >= 0.5 is a tracking-quality filter, not a technique threshold.
    """
    path = Path(video_path)
    if not path.is_file():
        raise VideoAnalysisError(f"Video file does not exist: {path.name}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoAnalysisError("Cannot open this video. Upload a readable MP4 or MOV file.")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise VideoAnalysisError("The video has invalid dimensions or frame rate.")

        # Import lazily so the web server and media checks do not initialize the
        # pose runtime until analysis actually starts.
        import mediapipe as mp

        model = Path(mp.__file__).parent / "modules/pose_landmark/pose_landmark_full.tflite"
        if not model.is_file():
            raise VideoAnalysisError(
                "The bundled MediaPipe pose model is missing. Reinstall mediapipe==0.10.14."
            )
        landmark_ids = {
            name: getattr(mp.solutions.pose.PoseLandmark, name.upper()).value
            for name in LANDMARK_NAMES
        }
        frames = []
        visible_frames = 0
        if progress:
            progress(0.0, "Loading pose tracking model")
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                index = len(frames)
                if frame.shape[:2] != (height, width):
                    raise VideoAnalysisError("The video's frame dimensions change during playback.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = pose.process(rgb)
                landmarks = {}
                if result.pose_landmarks:
                    for name, landmark_id in landmark_ids.items():
                        point = result.pose_landmarks.landmark[landmark_id]
                        values = (float(point.x), float(point.y), float(point.visibility))
                        if not all(math.isfinite(value) for value in values):
                            continue
                        if point.visibility < MIN_VISIBILITY:
                            continue
                        # An off-image joint is not visually observed and must
                        # not become a valid measurement solely on confidence.
                        if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                            continue
                        landmarks[name] = {
                            "x": float(point.x) * width,
                            "y": float(point.y) * height,
                            "visibility": float(point.visibility),
                        }
                if landmarks:
                    visible_frames += 1
                frames.append({"frame": index, "time": index / fps, "landmarks": landmarks})
                if progress and (index % 10 == 0):
                    fraction = min(0.99, (index + 1) / max(expected_count, index + 2))
                    progress(fraction, f"Tracking body landmarks: frame {index + 1}")

        if not frames:
            raise VideoAnalysisError("The video contains no decodable frames.")
        if not visible_frames:
            raise VideoAnalysisError(
                "No visible human pose was detected. Use a clear clip with one player, "
                "the whole body in frame, and adequate lighting."
            )
        metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": len(frames),
            "duration": len(frames) / fps,
        }
        if progress:
            progress(1.0, f"Pose tracking complete: {visible_frames} of {len(frames)} frames")
        return {"metadata": metadata, "frames": frames}
    finally:
        capture.release()
