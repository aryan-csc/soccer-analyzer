import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_data(
    series: pd.Series, window_length: int = 13, polyorder: int = 3
) -> pd.Series:
    filled_series = series.interpolate(method="linear").bfill().ffill()

    if len(filled_series) < window_length:
        window_length = (
            len(filled_series) - 1
            if len(filled_series) % 2 == 0
            else len(filled_series)
        )

    if window_length <= polyorder:
        polyorder = window_length - 1

    if window_length < 3 or polyorder < 1:
        return filled_series

    smoothed_values = savgol_filter(filled_series, window_length, polyorder)
    return pd.Series(smoothed_values, index=filled_series.index, name=series.name)


def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    ba = np.array([a_x - b_x, a_y - b_y])
    bc = np.array([c_x - b_x, c_y - b_y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
