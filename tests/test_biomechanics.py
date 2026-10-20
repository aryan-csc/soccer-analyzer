"""Exact geometry and motion fixtures test the measurements, not pose accuracy.

These fixtures never supply data to the dashboard. Real footage is exercised
separately by the opt-in end-to-end test.
"""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.math_utils import calculate_angle, smooth_data
from src.physics_utils import _phase_estimates, analyze_kick


def motion_clip(count=30, fps=30, foot='right'):
    frames = []
    for index in range(count):
        landmarks = {}
        for side, offset in [('right', 200), ('left', 400)]:
            points = {'shoulder': (offset, 80), 'hip': (offset, 120),
                      'knee': (offset + 40, 120), 'ankle': (offset + 40, 160),
                      'foot_index': (offset + 50 + (100 * index / fps if side == foot else 0), 160)}
            landmarks.update({f'{side}_{name}': {'x': x, 'y': y, 'visibility': 1.0}
                              for name, (x, y) in points.items()})
        frames.append({'frame': index, 'time': index / fps, 'landmarks': landmarks})
    return frames, {'width': 640, 'height': 360, 'fps': fps, 'frame_count': count, 'duration': count / fps}


def summaries(report):
    return {entry['key']: entry['value'] for entry in report['summary']}


@pytest.mark.parametrize('points,expected', [
    ((0, 0, 0, 1, 1, 1), 90), ((0, 0, 0, 1, 0, 2), 180),
    ((1, 0, 0, 0, 1, 1), 45), ((0, 0, 0, 0, 1, 1), None),
    ((None, 0, 0, 0, 1, 1), None), ((math.nan, 0, 0, 0, 1, 1), None),
])
def test_pixel_geometry_and_degenerate_segments(points, expected):
    result = calculate_angle(*points)
    assert result is None if expected is None else result == pytest.approx(expected)


@pytest.mark.parametrize('foot', ['left', 'right'])
def test_measured_angles_speed_calibration_and_foot_selection(foot):
    frames, metadata = motion_clip(foot=foot)
    original = copy.deepcopy(frames)
    report = analyze_kick(frames, metadata, {'foot': foot, 'contact_frame': 15})
    assert frames == original  # Cached tracking must survive repeated revisions.
    row = report['frames'][15]['metrics']
    assert row['knee_flexion'] == pytest.approx(90)
    assert row['hip_angle'] == pytest.approx(90)
    assert row['plant_knee_flexion'] == pytest.approx(90)
    assert row['torso_lean'] == pytest.approx(0)
    assert row['foot_speed'] == pytest.approx(100)
    assert report['contact']['source'] == 'manual'
    assert report['contact']['frame'] == 15
    assert report['frames'][15]['phase'] == 'contact'
    assert summaries(report)['plant_displacement'] == pytest.approx(0)
    assert summaries(report)['plant_knee_variability'] == pytest.approx(0)
    assert report['frames'][0]['metrics']['foot_speed'] is None
    assert report['frames'][-1]['metrics']['foot_speed'] is None
    calibrated = analyze_kick(frames, metadata, {'foot': foot, 'contact_frame': 15, 'scale_m_per_px': 0.01})
    assert calibrated['frames'][15]['metrics']['foot_speed'] == pytest.approx(1)
    assert next(d for d in calibrated['metric_definitions'] if d['key'] == 'foot_speed')['unit'] == 'm/s'
    assert calibrated['frames'][15]['metrics']['knee_flexion'] == pytest.approx(90)
    json.dumps(calibrated, allow_nan=False)


def test_missing_and_low_visibility_landmarks_never_become_measurements():
    frames, metadata = motion_clip()
    frames[10]['landmarks'] = {}
    frames[15]['landmarks']['right_knee']['visibility'] = 0.1
    frames[20]['landmarks']['right_ankle']['x'] = math.nan
    report = analyze_kick(frames, metadata, {'contact_frame': 15})
    for index in [10, 15, 20]:
        assert report['frames'][index]['metrics']['knee_flexion'] is None
    for index in [9, 10, 11]:
        assert report['frames'][index]['metrics']['foot_speed'] is None
    assert summaries(report)['knee_flexion_at_contact'] is None
    assert report['quality']['pose_coverage'] < 1
    json.dumps(report, allow_nan=False)


def test_off_image_joints_are_not_treated_as_visible_measurements():
    frames, metadata = motion_clip()
    frames[10]['landmarks']['right_ankle']['x'] = -10
    frames[11]['landmarks']['right_knee']['y'] = 400
    report = analyze_kick(frames, metadata)
    assert report['frames'][10]['metrics']['knee_flexion'] is None
    assert report['frames'][11]['metrics']['knee_flexion'] is None
    assert 'right_ankle' not in report['frames'][10]['landmarks']


@pytest.mark.parametrize('frames,metadata', [([None], {}), ([{'landmarks': ['bad']}], {'fps': 30, 'width': 640, 'height': 360})])
def test_malformed_saved_frames_fail_with_a_readable_error(frames, metadata):
    with pytest.raises(ValueError):
        analyze_kick(frames, metadata)


def test_smoothing_does_not_join_gaps():
    values = smooth_data(pd.Series([0, 1, 2, math.nan, 100, 101, 102]))
    assert values.iloc[:3].tolist() == pytest.approx([0, 1, 2])
    assert math.isnan(values.iloc[3])
    assert values.iloc[4:].tolist() == pytest.approx([100, 101, 102])


def test_constant_or_absent_motion_does_not_invent_contact_or_phases():
    frames, metadata = motion_clip()
    report = analyze_kick(frames, metadata)
    assert report['contact']['source'] == 'unavailable'
    assert report['phases'] == []
    assert report['support_window'] is None
    for frame in frames:
        frame['landmarks'] = {}
    report = analyze_kick(frames, metadata)
    assert report['quality']['pose_coverage'] == 0
    assert all(value is None for row in report['frames'] for value in row['metrics'].values())


def test_distinct_foot_motion_peak_is_only_an_estimated_contact():
    frames, metadata = motion_clip(count=61)
    for index, frame in enumerate(frames):
        frame['landmarks']['right_foot_index']['x'] = 250 + 100 * math.erf((index - 30) / 8)
    report = analyze_kick(frames, metadata)
    assert report['contact']['source'] == 'estimated'
    assert abs(report['contact']['frame'] - 30) <= 1
    assert report['contact']['needs_review'] is True
    assert 'not detected' in report['contact']['reason']


def test_phase_estimates_require_continuous_swing_and_fixed_plant_camera():
    times = np.arange(14) / 10
    x = np.array([4, 3, 2, 1, 0, 1, 2, 4, 6, 7, 8, 9, 8, 7], dtype=float)
    relative = np.column_stack((x, np.zeros(14)))
    plant_speed = np.array([10, 10, 10, 10, 9, 8, 2, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    settings = {'direction': 'right', 'camera_motion': 'fixed'}
    ids, plant, start, reversal, end = _phase_estimates(relative, plant_speed, times, 9, settings)
    assert start == 0 and reversal == 4 and plant == 6 and end == 11
    assert ids[2] == 'backswing' and ids[7] == 'forward_swing'
    assert ids[9] == 'contact' and ids[10] == 'follow_through'
    # The same movement mirrored on screen should identify the same phases.
    mirrored = _phase_estimates(-relative, plant_speed, times, 9, {**settings, 'direction': 'left'})
    assert mirrored == (ids, plant, start, reversal, end)
    relative[7] = np.nan
    broken = _phase_estimates(relative, plant_speed, times, 9, settings)
    assert 'forward_swing' not in broken[0]
    assert broken[0][7] == 'unclassified'
    assert _phase_estimates(relative, plant_speed, times, 9, {**settings, 'camera_motion': 'moving'})[1] is None


def test_identity_discontinuity_withholds_contact_leg_summaries():
    frames, metadata = motion_clip()
    for frame in frames[15:]:
        lm = frame['landmarks']
        for name in ['knee', 'ankle', 'foot_index']:
            lm['left_' + name], lm['right_' + name] = lm['right_' + name], lm['left_' + name]
    report = analyze_kick(frames, metadata, {'contact_frame': 15})
    assert 15 in report['quality']['identity_suspect_frames']
    assert summaries(report)['knee_flexion_at_contact'] is None
    assert summaries(report)['follow_through_duration'] is None
    assert report['frames'][15]['metrics']['foot_speed'] is None


def test_torso_inclination_is_measured_from_upward_vertical():
    frames, metadata = motion_clip()
    frames[10]['landmarks']['right_shoulder']['y'] = 160
    report = analyze_kick(frames, metadata)
    assert report['frames'][10]['metrics']['torso_lean'] == pytest.approx(180)


@pytest.mark.parametrize('contact', [0, 10])
def test_knee_extension_is_unavailable_without_any_pre_contact_measurement(contact):
    frames, metadata = motion_clip()
    for frame in frames[:contact]:
        frame['landmarks'].pop('right_knee')
    report = analyze_kick(frames, metadata, {'contact_frame': contact})
    assert summaries(report)['knee_flexion_at_contact'] is not None
    assert summaries(report)['knee_extension_to_contact'] is None
    assert not any(observation['title'] == 'Kicking-leg extension' for observation in report['observations'])


@pytest.mark.parametrize('settings', [{'contact_frame': -1}, {'contact_frame': 30},
    {'contact_frame': True}, {'scale_m_per_px': -1}, {'scale_m_per_px': math.inf}, {'foot': 'both'}])
def test_invalid_analysis_settings_are_rejected(settings):
    frames, metadata = motion_clip()
    with pytest.raises(ValueError):
        analyze_kick(frames, metadata, settings)
