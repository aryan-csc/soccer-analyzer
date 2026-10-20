"""Image-plane geometry helpers shared by the soccer analysis pipeline."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def valid_runs(mask):
    """Yield half-open contiguous valid ranges, never joining missing samples."""
    start = None
    for index, valid in enumerate(mask):
        if valid and start is None:
            start = index
        if start is not None and (not valid or index == len(mask) - 1):
            yield start, index if not valid else index + 1
            start = None


def smooth_data(series: pd.Series, window_length: int = 5, polyorder: int = 2) -> pd.Series:
    """Smooth finite runs separately, preserving missing values and gaps."""
    values = np.asarray(series, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    for start, end in valid_runs(np.isfinite(values)):
        window = min(int(window_length), end - start)
        if window % 2 == 0:
            window -= 1
        order = min(int(polyorder), window - 1)
        if window >= 3 and order >= 1:
            values[start:end] = savgol_filter(values[start:end], window, order, mode="interp")
    return pd.Series(values, index=series.index, name=series.name)


def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    """2D interior angle ABC in degrees, or None for missing/degenerate points.

    Inputs must share the same unit (pixels, not independently normalized axes).
    """
    try:
        values = tuple(float(v) for v in (a_x, a_y, b_x, b_y, c_x, c_y))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    ax, ay, bx, by, cx, cy = values
    ux, uy, vx, vy = ax - bx, ay - by, cx - bx, cy - by
    denominator = math.hypot(ux, uy) * math.hypot(vx, vy)
    if denominator <= 1e-12:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / denominator))
    return math.degrees(math.acos(cosine))
