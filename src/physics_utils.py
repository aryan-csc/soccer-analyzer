import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import json
from src.math_utils import smooth_data, calculate_angle


class KinematicsProcessor:
    def __init__(self, input_csv: str, output_csv: str):
        self.input_csv = input_csv
        self.output_csv = output_csv

    def process(self):
        df = pd.read_csv(self.input_csv)

        for col in df.columns:
            if col.endswith("_y"):
                df[f"{col}_inv"] = 1.0 - df[col]

        for col in df.columns:
            if col not in ["frame", "time"]:
                df[col] = smooth_data(df[col])

        dt = max(df["time"].iloc[1] - df["time"].iloc[0], 0.001)
        df["vel_y"] = smooth_data(df["bar_y_inv"].diff() / dt)

        df["hip_angle"] = df.apply(
            lambda row: calculate_angle(
                row["shoulder_x"],
                row["shoulder_y_inv"],
                row["hip_x"],
                row["hip_y_inv"],
                row["knee_x"],
                row["knee_y_inv"],
            ),
            axis=1,
        )

        df["hip_angle"] = smooth_data(df["hip_angle"], window_length=9)
        df.to_csv(self.output_csv, index=False)


class RepDetector:
    def __init__(self, input_csv: str, output_json: str):
        self.input_csv = input_csv
        self.output_json = output_json

    def process(self):
        df = pd.read_csv(self.input_csv)

        global_peak_idx = df["bar_y_inv"].idxmax()

        max_vel_idx = df.loc[:global_peak_idx, "vel_y"].idxmax()

        start_idx = 0
        for i in range(max_vel_idx, -1, -1):
            if df.loc[i, "bar_y_inv"] < 0.35 and df.loc[i, "vel_y"] <= 0.05:
                start_idx = i
                break

        peak_idx = max_vel_idx
        for i in range(max_vel_idx, len(df)):
            if df.loc[i, "vel_y"] <= 0:
                peak_idx = i
                break

        end_idx = min(len(df) - 1, df.loc[peak_idx:, "bar_y_inv"].idxmax() + 20)

        metadata = {
            "reps": [
                {
                    "rep_number": 1,
                    "start_frame": int(df.loc[start_idx, "frame"]),
                    "peak_frame": int(df.loc[peak_idx, "frame"]),
                    "end_frame": int(df.loc[end_idx, "frame"]),
                }
            ]
        }

        with open(self.output_json, "w") as f:
            json.dump(metadata, f, indent=4)


class IdealPathCalculator:
    def __init__(self, input_csv: str, input_json: str, output_csv: str):
        self.input_csv = input_csv
        self.input_json = input_json
        self.output_csv = output_csv

    def process(self):
        df = pd.read_csv(self.input_csv)
        with open(self.input_json, "r") as f:
            rep = json.load(f)["reps"][0]

        start_idx = df.index[df["frame"] == rep["start_frame"]].tolist()[0]
        peak_idx = df.index[df["frame"] == rep["peak_frame"]].tolist()[0]

        start_x = df.loc[start_idx, "bar_x"]
        start_y = df.loc[start_idx, "bar_y_inv"]
        peak_y = df.loc[peak_idx, "bar_y_inv"]

        h = max(peak_y - start_y, 0.1)
        facing_dir = (
            1 if df.loc[start_idx, "knee_x"] > df.loc[start_idx, "hip_x"] else -1
        )

        keypoints_y = [
            start_y,
            start_y + (0.25 * h),
            start_y + (0.60 * h),
            start_y + (0.85 * h),
            start_y + (1.00 * h),
        ]

        keypoints_x = [
            start_x,
            start_x + (0.015 * h * facing_dir),
            start_x - (0.040 * h * facing_dir),
            start_x + (0.020 * h * facing_dir),
            start_x - (0.060 * h * facing_dir),
        ]

        sorted_indices = np.argsort(keypoints_y)
        spline = CubicSpline(
            np.array(keypoints_y)[sorted_indices],
            np.array(keypoints_x)[sorted_indices],
            bc_type="natural",
        )

        ideal_y = np.linspace(start_y, peak_y, 100)
        pd.DataFrame({"ideal_bar_x": spline(ideal_y), "ideal_bar_y": ideal_y}).to_csv(
            self.output_csv, index=False
        )
