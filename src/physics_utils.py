"""Measured 2D soccer-kick kinematics and explicitly tentative phase estimates.

Phase heuristics are navigation aids, not a validated ball/contact detector.
All timing refers to playback time in the supplied video.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np
import pandas as pd
from src.math_utils import calculate_angle, smooth_data, valid_runs

# Signal/quality parameters; never ideal-technique limits.
MIN_VISIBILITY = 0.5
SPEED_SMOOTHING_SECONDS = 0.12
PRE_CONTACT_SEARCH_SECONDS = 1.5
POST_CONTACT_SEARCH_SECONDS = 1.0
NUMERICAL_MOTION_EPSILON = 1e-6
DEFAULT_SETTINGS = {
    "foot": "right", "shot_type": "shot", "camera_view": "side",
    "camera_motion": "fixed", "direction": "auto", "contact_frame": None,
    "scale_m_per_px": None,
}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean(value):
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _point(landmarks, name):
    point = landmarks.get(name)
    if point is None or point["visibility"] < MIN_VISIBILITY:
        return None
    return np.array([point["x"], point["y"]], dtype=float)


def _angle(a, b, c):
    return None if any(p is None for p in (a, b, c)) else calculate_angle(*a, *b, *c)


def _flexion(hip, knee, ankle):
    angle = _angle(hip, knee, ankle)
    return None if angle is None else 180.0 - angle


def _trajectory(points):
    return np.array([p if p is not None else [np.nan, np.nan] for p in points], dtype=float)


def _smooth_trajectory(points, fps):
    window = max(3, round(fps * SPEED_SMOOTHING_SECONDS))
    if window % 2 == 0:
        window += 1
    return np.column_stack([
        smooth_data(pd.Series(points[:, axis]), window, 2).to_numpy() for axis in (0, 1)
    ])


def _speed(points, times, fps):
    """Central differences inside visible runs; endpoints remain unavailable."""
    smoothed = _smooth_trajectory(points, fps)
    speed = np.full(len(points), np.nan)
    for start, end in valid_runs(np.isfinite(smoothed).all(axis=1)):
        if end - start >= 3:
            difference = smoothed[start + 2:end] - smoothed[start:end - 2]
            dt = times[start + 2:end] - times[start:end - 2]
            speed[start + 1:end - 1] = np.linalg.norm(difference, axis=1) / dt
    return speed, smoothed


def _finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def _metric_definitions(calibrated):
    length_unit, speed_unit = ("m", "m/s") if calibrated else ("px", "px/s")
    return [
        {"key": "knee_flexion", "label": "Kicking knee flexion", "unit": "°", "description": "180° minus the hip–knee–ankle image-plane interior angle; 0° is straight."},
        {"key": "hip_angle", "label": "Kicking hip angle", "unit": "°", "description": "Shoulder–hip–knee image-plane interior angle; not anatomical 3D hip flexion."},
        {"key": "plant_knee_flexion", "label": "Plant knee flexion", "unit": "°", "description": "Opposite-leg 180° minus the hip–knee–ankle image-plane angle."},
        {"key": "torso_lean", "label": "Torso inclination", "unit": "°", "description": "Absolute selected-side hip-to-shoulder inclination from image vertical; no forward/backward claim."},
        {"key": "foot_speed", "label": "Estimated foot speed", "unit": speed_unit, "description": "2D foot-index landmark speed from a smoothed image trajectory and video time; not ball speed. Gaps and run endpoints are unavailable."},
        {"key": "plant_displacement", "label": "Plant ankle displacement", "unit": length_unit, "description": "Image-plane distance from the first visible plant ankle in the stated support window; camera movement also contributes."},
    ]


def _identity_suspicions(frames):
    """Flag abrupt bilateral assignments that fit the crossed identities better.

    This cannot prove a swap (real crossing can look similar). Never relabel the
    skeleton; conservatively invalidate the local discontinuity and flag review.
    """
    suspected = []
    for index in range(1, len(frames)):
        previous, current = frames[index - 1]["landmarks"], frames[index]["landmarks"]
        same = crossed = 0.0
        usable = 0
        lengths = []
        for joint in ("knee", "ankle", "foot_index"):
            points = [_point(lm, f"{side}_{joint}") for lm in (previous, current) for side in ("left", "right")]
            if any(point is None for point in points):
                continue
            left_before, right_before, left_after, right_after = points
            same += np.linalg.norm(left_after - left_before) + np.linalg.norm(right_after - right_before)
            crossed += np.linalg.norm(left_after - right_before) + np.linalg.norm(right_after - left_before)
            usable += 1
        for side in ("left", "right"):
            hip, knee, ankle = [_point(current, f"{side}_{joint}") for joint in ("hip", "knee", "ankle")]
            if all(point is not None for point in (hip, knee, ankle)):
                lengths.append(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))
        if usable >= 2 and lengths and same > 0.2 * float(np.mean(lengths)) and crossed < 0.6 * same:
            suspected.append(index)
    return suspected


def _validate(frames, metadata, settings):
    if not isinstance(frames, list) or not frames:
        raise ValueError("The video has no decoded frames to analyze.")
    if not isinstance(metadata, Mapping) or (settings is not None and not isinstance(settings, Mapping)):
        raise ValueError("Video metadata and analysis settings must be objects.")
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    choices = {"foot": {"left", "right"}, "shot_type": {"penalty", "free_kick", "shot"},
               "camera_view": {"side", "oblique", "front"}, "camera_motion": {"fixed", "moving"},
               "direction": {"auto", "left", "right"}}
    for key, values in choices.items():
        if not isinstance(settings[key], str) or settings[key] not in values:
            raise ValueError(f"Invalid {key}: choose one of {', '.join(sorted(values))}.")
    fps = _number(metadata.get("fps"))
    if fps is None or fps <= 0:
        raise ValueError("A positive, finite video frame rate is required.")
    for key in ("width", "height"):
        value = _number(metadata.get(key))
        if value is None or value <= 0:
            raise ValueError(f"Video {key} must be positive and finite.")
    scale = settings["scale_m_per_px"]
    if scale is not None:
        scale = _number(scale)
        if scale is None or scale <= 0:
            raise ValueError("Spatial calibration must be a positive number of meters per pixel.")
        settings["scale_m_per_px"] = scale
    clean_frames = []
    last_frame, last_time = -1, -math.inf
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError("Each decoded frame must be an object containing landmarks and timing.")
        frame_id = frame.get("frame", index)
        if isinstance(frame_id, bool) or _number(frame_id) is None or int(frame_id) != frame_id or frame_id <= last_frame:
            raise ValueError("Frame numbers must be nonnegative, increasing integers.")
        time = _number(frame.get("time", frame_id / fps))
        if time is None or time < 0 or time <= last_time:
            raise ValueError("Frame times must be finite, nonnegative, and strictly increasing.")
        landmarks = {}
        raw_landmarks = frame.get("landmarks") or {}
        if not isinstance(raw_landmarks, Mapping):
            raise ValueError("Frame landmarks must map joint names to image points.")
        for name, point in raw_landmarks.items():
            if not isinstance(point, Mapping):
                continue
            x, y, visibility = (_number(point.get(key)) for key in ("x", "y", "visibility"))
            if (x is not None and y is not None and visibility is not None
                    and 0 <= x <= float(metadata['width']) and 0 <= y <= float(metadata['height'])):
                landmarks[name] = {"x": x, "y": y, "visibility": max(0.0, min(1.0, visibility))}
        clean_frames.append({"frame": int(frame_id), "time": time, "landmarks": landmarks})
        last_frame, last_time = frame_id, time
    contact = settings["contact_frame"]
    if contact is not None:
        if isinstance(contact, bool) or not isinstance(contact, (int, np.integer)) or contact not in {f["frame"] for f in clean_frames}:
            raise ValueError("Contact frame must be an integer identifying a decoded video frame.")
        settings["contact_frame"] = int(contact)
    return clean_frames, settings, fps


def _phase_estimates(relative_foot, plant_speed, times, contact_index, settings):
    """Return observable motion landmarks, without manufacturing six phases."""
    count = len(times)
    if contact_index is None:
        return ["unclassified"] * count, None, None, None, None
    c = contact_index
    start = int(np.searchsorted(times, max(times[0], times[c] - PRE_CONTACT_SEARCH_SECONDS)))
    stop = min(count - 1, int(np.searchsorted(times, times[c] + POST_CONTACT_SEARCH_SECONDS, side="right")) - 1)
    sign = {"left": -1, "right": 1}.get(settings["direction"])
    if sign is None and 0 < c < count - 1 and np.isfinite(relative_foot[c - 1:c + 2]).all():
        delta = relative_foot[c + 1, 0] - relative_foot[c - 1, 0]
        sign = 1 if delta > NUMERICAL_MOTION_EPSILON else (-1 if delta < -NUMERICAL_MOTION_EPSILON else None)
    backswing_start = reversal = plant = follow_end = None
    ids = ["unclassified"] * count
    if sign is not None:
        projected = sign * relative_foot[:, 0]
        candidates = [i for i in range(start + 1, c - 1)
                      if np.isfinite(projected[i - 1:i + 2]).all()
                      and projected[i] < projected[i - 1] and projected[i] <= projected[i + 1]]
        if candidates:
            reversal = min(candidates, key=lambda i: projected[i])
            backswing_start = reversal
            while backswing_start > start and np.isfinite(projected[backswing_start - 1]) and projected[backswing_start - 1] >= projected[backswing_start]:
                backswing_start -= 1
            # A reversal before a tracking gap cannot establish the swing up
            # to contact. Do not paint an unobserved interval on the timeline.
            if reversal - backswing_start >= 2 and np.isfinite(projected[reversal:c + 1]).all():
                ids[backswing_start:reversal + 1] = ["backswing"] * (reversal - backswing_start + 1)
                ids[reversal + 1:c] = ["forward_swing"] * (c - reversal - 1)
            else:
                backswing_start = reversal = None
        if c < stop and np.isfinite(projected[c:c + 2]).all() and projected[c + 1] > projected[c]:
            follow_end = c + 1
            while follow_end < stop and np.isfinite(projected[follow_end + 1]) and projected[follow_end + 1] >= projected[follow_end]:
                follow_end += 1
            ids[c + 1:follow_end + 1] = ["follow_through"] * (follow_end - c)
    # An already-stationary ankle does not reveal when it was planted.
    plant_start = int(np.searchsorted(times, max(times[0], times[c] - 0.8)))
    slowing = [(plant_speed[i - 1] - plant_speed[i], i)
               for i in range(max(1, plant_start), c)
               if np.isfinite(plant_speed[i - 1:i + 1]).all()]
    if slowing and settings["camera_motion"] == "fixed":
        amount, candidate = max(slowing)
        local = _finite(plant_speed[plant_start:c + 1])
        if amount > NUMERICAL_MOTION_EPSILON and local and plant_speed[candidate] <= float(np.median(local)):
            plant = candidate
            ids[plant] = "plant"
    ids[c] = "contact"
    return ids, plant, backswing_start, reversal, follow_end


def analyze_kick(frames, metadata, settings=None):
    """Analyze per-frame pixel landmarks and return a JSON-serializable report."""
    frames, settings, fps = _validate(frames, metadata, settings)
    count = len(frames)
    times = np.array([frame["time"] for frame in frames])
    suspect_indices = _identity_suspicions(frames)
    suspect_neighborhood = set()
    for event in suspect_indices:
        start = int(np.searchsorted(times, times[event] - 0.15))
        end = int(np.searchsorted(times, times[event] + 0.15, side="right"))
        suspect_neighborhood.update(range(start, end))
    kick, plant = settings["foot"], "left" if settings["foot"] == "right" else "right"
    scale = settings["scale_m_per_px"] or 1.0
    calibrated = settings["scale_m_per_px"] is not None
    length_unit, speed_unit = ("m", "m/s") if calibrated else ("px", "px/s")
    foot_points, hip_points, plant_points = [], [], []
    for index, frame in enumerate(frames):
        lm = frame["landmarks"]
        shoulder, hip, knee, ankle = [_point(lm, f"{kick}_{name}") for name in ("shoulder", "hip", "knee", "ankle")]
        p_hip, p_knee, p_ankle = [_point(lm, f"{plant}_{name}") for name in ("hip", "knee", "ankle")]
        torso = None
        if shoulder is not None and hip is not None and np.linalg.norm(shoulder - hip) > NUMERICAL_MOTION_EPSILON:
            delta = shoulder - hip
            torso = math.degrees(math.atan2(abs(delta[0]), -delta[1]))
        frame["metrics"] = {
            "knee_flexion": _flexion(hip, knee, ankle), "hip_angle": _angle(shoulder, hip, knee),
            "plant_knee_flexion": _flexion(p_hip, p_knee, p_ankle), "torso_lean": torso,
            "foot_speed": None, "plant_displacement": None,
        }
        if index in suspect_neighborhood:
            for key in ("knee_flexion", "hip_angle", "plant_knee_flexion"):
                frame["metrics"][key] = None
        foot_points.append(_point(lm, f"{kick}_foot_index") if index not in suspect_neighborhood else None)
        hip_points.append(hip)
        plant_points.append(p_ankle if index not in suspect_neighborhood else None)
    foot, hips, planted = (_trajectory(points) for points in (foot_points, hip_points, plant_points))
    speeds, _ = _speed(foot, times, fps)
    relative_speeds, relative = _speed(foot - hips, times, fps)
    plant_speeds, _ = _speed(planted, times, fps)
    for frame, speed in zip(frames, speeds):
        frame["metrics"]["foot_speed"] = float(speed * scale) if math.isfinite(speed) else None
    warnings = [
        "All angles and trajectories are 2D image-plane estimates; perspective, occlusion and pose-tracking error affect measurements.",
        "Speed uses video playback time. Slow-motion or retimed footage does not reveal real-time speed, even with spatial calibration.",
        "The ball is not tracked. A foot-motion peak is only a contact candidate; confirm contact using frame navigation.",
        "Overlapping legs can cause the pose model to swap anatomical left/right labels. Verify the selected kicking leg on the skeleton, especially around contact.",
    ]
    if suspect_indices:
        warnings.append("Possible left/right leg-identity discontinuities at frames " + ", ".join(str(frames[i]["frame"]) for i in suspect_indices[:15]) + ". Leg measurements within ±0.15 video seconds of these events are withheld. Anatomical labels after an event may remain swapped; review the skeleton before interpreting later measurements.")
    if settings["camera_view"] != "side":
        warnings.append("An oblique or front view foreshortens limb segments. Treat the displayed angles as projected geometry, not sagittal-plane joint angles.")
    if settings["camera_motion"] == "moving":
        warnings.append("Camera movement contributes to foot speed and plant-ankle displacement. These cannot be interpreted as ground-referenced motion or balance.")
    if calibrated:
        warnings.append("Meters use the supplied single-plane scale. Calibration must lie in the player's movement plane; perspective and depth changes remain uncorrected.")
    else:
        warnings.append("No spatial calibration supplied: speed is in pixels per video second and displacement in pixels. Values are not comparable across differently framed clips.")
    contact_index = None
    contact = {"frame": None, "source": "unavailable", "needs_review": True,
               "reason": "No distinct, continuously tracked foot-motion peak was found. Select contact manually if a kick is visible."}
    if settings["contact_frame"] is not None:
        contact_index = next(i for i, frame in enumerate(frames) if frame["frame"] == settings["contact_frame"])
        contact = {"frame": settings["contact_frame"], "source": "manual", "needs_review": False,
                   "reason": "Contact frame selected by the user; surrounding phase boundaries remain motion-based estimates."}
    else:
        visible = np.flatnonzero(np.isfinite(relative_speeds))
        if len(visible) >= 5:
            baseline = float(np.median(relative_speeds[visible]))
            noise = float(np.median(np.abs(relative_speeds[visible] - baseline)))
            peaks = [i for i in visible if 1 < i < count - 2 and np.isfinite(relative_speeds[i - 1:i + 2]).all()
                     and relative_speeds[i] >= relative_speeds[i - 1] and relative_speeds[i] >= relative_speeds[i + 1]]
            if peaks:
                candidate = max(peaks, key=lambda i: relative_speeds[i])
                if relative_speeds[candidate] > baseline + max(2.5 * noise, NUMERICAL_MOTION_EPSILON):
                    contact_index = int(candidate)
                    contact = {"frame": frames[candidate]["frame"], "source": "estimated", "needs_review": True,
                               "reason": "Candidate at a prominent foot-speed peak relative to the hip. Foot–ball impact is not detected; verify or replace this frame."}
    if contact_index is None:
        warnings.append("Kick phases and contact-dependent measurements are unavailable until a contact frame is selected.")
    phase_ids, plant_index, backswing_start, reversal, follow_end = _phase_estimates(relative, plant_speeds, times, contact_index, settings)
    if contact_index is not None:
        c = contact_index
        if frames[c]["metrics"]["knee_flexion"] is None:
            warnings.append("The kicking leg is not reliably visible at the selected contact frame; contact angles are unavailable.")
        if plant_index is None:
            warnings.append("Plant-foot placement could not be localized from ankle slowing. Support measurements use the stated contact-centered window.")
        approach_end = min([v for v in (plant_index, backswing_start) if v is not None], default=0) - 1
        if settings["camera_motion"] == "fixed" and approach_end >= 2:
            approach_start = max(0, int(np.searchsorted(times, times[approach_end] - 1.5)))
            section = hips[approach_start:approach_end + 1]
            if np.isfinite(section).all():
                steps = np.linalg.norm(np.diff(section, axis=0), axis=1)
                displacement = np.linalg.norm(section[-1] - section[0])
                if len(steps) and displacement > max(3 * float(np.median(steps)), NUMERICAL_MOTION_EPSILON):
                    phase_ids[approach_start:approach_end + 1] = ["approach"] * (approach_end - approach_start + 1)
    labels = {"approach": "Approach", "plant": "Plant-foot placement", "backswing": "Backswing", "forward_swing": "Forward swing", "contact": "Ball-contact candidate" if contact["source"] != "manual" else "Selected ball contact", "follow_through": "Observed follow-through"}
    phases = []
    for index, phase in enumerate(phase_ids):
        frames[index]["phase"] = phase
        if phase == "unclassified":
            continue
        if index > 0 and phase_ids[index - 1] == phase:
            phases[-1]["end_frame"] = frames[index]["frame"]
        else:
            phases.append({"id": phase, "label": labels[phase], "start_frame": frames[index]["frame"], "end_frame": frames[index]["frame"],
                           "source": "manual" if phase == "contact" and contact["source"] == "manual" else "estimated"})
    summary = []
    def add(key, label, value, unit, detail):
        summary.append({"key": key, "label": label, "value": value, "unit": unit, "detail": detail})
    visible_speeds = _finite([frame["metrics"]["foot_speed"] for frame in frames])
    add("peak_foot_speed", "Peak observed foot speed", max(visible_speeds) if visible_speeds else None, speed_unit,
        "Maximum foot-index speed over the full clip, in video time. It may occur outside the selected kick and is not ball speed.")
    coverage = sum(frame["metrics"]["knee_flexion"] is not None and frame["metrics"]["hip_angle"] is not None for frame in frames) / count
    if coverage < 0.8:
        warnings.append(f"Usable kicking-leg angle coverage is {coverage:.0%}. Missing or low-visibility landmarks remain gaps; reposition the camera or use a clearer clip.")
    support_window = None
    contact_identity_uncertain = contact_index is not None and any(abs(times[contact_index] - times[i]) <= 0.4 for i in suspect_indices)
    if contact_index is not None:
        c = contact_index
        for key, label in [("knee_flexion", "Knee flexion at contact"), ("hip_angle", "Hip angle at contact"), ("torso_lean", "Torso inclination at contact")]:
            add(f"{key}_at_contact", label, frames[c]["metrics"][key], "°", f"Measured at {'selected' if contact['source'] == 'manual' else 'estimated'} contact frame {frames[c]['frame']}.")
        pre_start = backswing_start if backswing_start is not None else int(np.searchsorted(times, max(times[0], times[c] - 0.5)))
        pre_knees = _finite([frame["metrics"]["knee_flexion"] for frame in frames[pre_start:c]])
        knee_at_contact = frames[c]["metrics"]["knee_flexion"]
        add("knee_extension_to_contact", "Knee extension toward contact", max([*pre_knees, knee_at_contact]) - knee_at_contact if pre_knees and knee_at_contact is not None else None, "°",
            f"Largest observed knee flexion minus contact flexion between frames {frames[pre_start]['frame']} and {frames[c]['frame']}; descriptive, not an ideal target.")
        support_start = plant_index if plant_index is not None else int(np.searchsorted(times, max(times[0], times[c] - 0.2)))
        support_end = min(count - 1, int(np.searchsorted(times, times[c] + 0.2, side="right")) - 1)
        support_window = {"start_frame": frames[support_start]["frame"], "end_frame": frames[support_end]["frame"], "source": "estimated_plant" if plant_index is not None else "contact_centered"}
        available = [i for i in range(support_start, support_end + 1) if np.isfinite(planted[i]).all()]
        if available:
            origin = planted[available[0]]
            for i in available:
                frames[i]["metrics"]["plant_displacement"] = float(np.linalg.norm(planted[i] - origin) * scale)
        displacements = _finite([frame["metrics"]["plant_displacement"] for frame in frames[support_start:support_end + 1]])
        support_knees = _finite([frame["metrics"]["plant_knee_flexion"] for frame in frames[support_start:support_end + 1]])
        window_text = f"Support window: frames {frames[support_start]['frame']}–{frames[support_end]['frame']}. Image motion includes tracking noise and any camera movement; not a balance diagnosis."
        add("plant_displacement", "Plant ankle excursion", max(displacements) if len(displacements) >= 2 else None, length_unit, window_text)
        add("plant_knee_variability", "Plant knee variability", float(np.std(support_knees)) if len(support_knees) >= 2 else None, "°", "Standard deviation of visible plant-knee flexion. " + window_text)
        excursion = None
        if follow_end is not None and np.isfinite(relative[[c, follow_end]]).all():
            excursion = float(np.linalg.norm(relative[follow_end] - relative[c]) * scale)
        add("follow_through_excursion", "Observed follow-through excursion", excursion, length_unit,
            "Foot displacement relative to the hip from contact to the first observed forward-motion maximum within one video second. Missing when no continuous forward trajectory is visible.")
        add("follow_through_duration", "Observed follow-through duration", times[follow_end] - times[c] if follow_end is not None else None, "s", "Duration of the observed forward portion in video time; not proof that the full follow-through was captured.")
        if contact_identity_uncertain:
            warnings.append("Leg identity is uncertain near contact. Contact knee/hip angles, knee extension, plant summaries, and follow-through summaries are withheld; torso inclination remains available.")
            for entry in summary:
                if entry["key"] not in {"peak_foot_speed", "torso_lean_at_contact"}:
                    entry["value"] = None
                    entry["detail"] += " Withheld because a possible anatomical leg-identity swap occurs within 0.4 video seconds of contact."
    observations = [
        {"title": "Review contact and phases", "body": contact["reason"] + " Phase boundaries are motion heuristics; unobservable phases are omitted. A clip with several kicks should be analyzed as separate attempts.", "kind": "review"},
        {"title": "Measurement scope", "body": "These measurements describe the visible projection. They do not establish shot power, accuracy, injury risk, an ideal technique, or a performance score.", "kind": "limitation"},
    ]
    summaries = {entry["key"]: entry for entry in summary}
    if contact_index is not None:
        extension = summaries["knee_extension_to_contact"]["value"]
        if extension is not None:
            observations.append({"title": "Kicking-leg extension", "body": f"The knee extended by {extension:.1f}° from its largest visible flexion in the pre-contact window to contact. Review the synchronized curve to see when extension occurred; contact selection affects this value.", "kind": "measurement"})
        excursion = summaries["plant_displacement"]["value"]
        if excursion is not None:
            variability = summaries["plant_knee_variability"]["value"]
            variation = f" Plant-knee flexion standard deviation was {variability:.1f}°." if variability is not None else ""
            observations.append({"title": "Plant-leg motion", "body": f"The plant ankle moved up to {excursion:.2f} {length_unit} from its first visible position in the support window.{variation} Inspect foot contact and camera stability before interpreting this as player movement.", "kind": "measurement"})
        torso = summaries["torso_lean_at_contact"]["value"]
        if torso is not None:
            observations.append({"title": "Torso at contact", "body": f"The torso was {torso:.1f}° from image vertical at contact. This view does not establish forward/backward lean or predict shot height.", "kind": "measurement"})
    report = {
        "metadata": {**metadata, "frame_count": count, "fps": fps, "duration": times[-1] + 1.0 / fps},
        "settings": settings, "quality": {"pose_coverage": coverage, "warnings": warnings,
            "identity_suspect_frames": [frames[i]["frame"] for i in suspect_indices],
            "metric_coverage": {key: sum(frame["metrics"][key] is not None for frame in frames) / count for key in frames[0]["metrics"]},
            "coverage_definition": "Fraction of frames with usable kicking-side knee and hip angles."},
        "contact": contact, "phases": phases, "metric_definitions": _metric_definitions(calibrated),
        "summary": summary, "observations": observations, "support_window": support_window, "frames": frames,
        "algorithm_notes": [
            f"Landmark visibility cutoff: {MIN_VISIBILITY}. This controls tracking quality, not technique quality.",
            "Angles use pixel coordinates directly. Low-visibility/invalid points and coincident segments yield null values; no missing spans are filled.",
            "Possible leg-identity jumps require crossed knee/ankle/toe assignment cost below 60% of same-label cost, with total jump above 20% of visible leg length. This quality heuristic never automatically relabels legs. Local ±0.15 s leg samples and contact-dependent leg summaries within 0.4 s are withheld.",
            f"Speed uses centered differences after quadratic Savitzky–Golay filtering (~{SPEED_SMOOTHING_SECONDS} s) within continuous visible runs only; run endpoints have no speed estimate.",
            "A contact candidate requires an interior relative-foot-speed maximum above the median plus 2.5 median absolute deviations; this is an unvalidated signal heuristic, not observed ball contact.",
            "Backswing and forward swing use continuous visible horizontal foot reversals relative to the hip in the 1.5 s before contact. Plant placement uses local opposite-ankle slowing in the preceding 0.8 s and requires a declared fixed camera. Approach requires observed hip translation with a declared fixed camera.",
            "Follow-through uses the continuous forward foot trajectory relative to the hip, ending at its first maximum or one second after contact. Phases may overlap physiologically; the timeline gives plant and contact events display priority.",
            "Support motion uses the estimated plant event through 0.2 s after contact, or ±0.2 s around contact when placement is unavailable. No ideal performance thresholds or normative scores are used.",
        ],
        "references": [{"title": "Kellis & Katis (2007), Biomechanical Characteristics and Determinants of Instep Soccer Kick", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3786235/", "scope": "Background on kick biomechanics; does not validate this application's 2D phase heuristics."}],
    }
    return _clean(report)
