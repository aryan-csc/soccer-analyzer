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
        
        # --- UPDATED BARBELL PROXY ---
        # Map the new CSRT plate coordinates to the bar variables
        df['bar_x'] = df['plate_x']
        # We still invert the Y-axis so 0 is the floor for standard physics!
        df['bar_y'] = 1.0 - df['plate_y'] 
        
        for col in df.columns:
            if col.endswith('_y') and col not in ['bar_y', 'plate_y']:
                df[col + '_inv'] = 1.0 - df[col]

        df['smooth_bar_x'] = smooth_data(df['bar_x'])
        df['smooth_bar_y'] = smooth_data(df['bar_y'])

        # Constant dt derived from metadata timeframe
        dt = df['time_sec'].iloc[1] - df['time_sec'].iloc[0]
        dt = max(dt, 0.001) # Safety clamp against zero division
        
        df['vel_y'] = smooth_data(df['smooth_bar_y'].diff() / dt)

        df['hip_angle'] = df.apply(lambda row: calculate_angle(
            row['left_shoulder_x'], row['left_shoulder_y_inv'],
            row['left_hip_x'], row['left_hip_y_inv'],
            row['left_knee_x'], row['left_knee_y_inv']
        ), axis=1)
        
        df['hip_angle'] = smooth_data(df['hip_angle'], window_length=9)
        df.to_csv(self.output_csv, index=False)

class RepDetector:
    def __init__(self, input_csv: str, output_json: str):
        self.input_csv = input_csv
        self.output_json = output_json

    def process(self):
        df = pd.read_csv(self.input_csv)
        peak_idx = df['smooth_bar_y'].idxmax()
        
        start_idx = 0
        for i in range(peak_idx, -1, -1):
            if df.loc[i, 'smooth_bar_y'] < 0.35 and df.loc[i, 'vel_y'] <= 0.05:
                start_idx = i
                break
                
        end_idx = len(df) - 1
        for i in range(peak_idx, len(df)):
            if df.loc[i, 'smooth_bar_y'] < 0.6 and df.loc[i, 'vel_y'] < -0.2:
                end_idx = i
                break
                
        start_frame = int(df.loc[start_idx, 'frame'])
        peak_frame = int(df.loc[peak_idx, 'frame'])
        end_frame = int(df.loc[end_idx, 'frame'])
        
        metadata = {
            "reps": [{
                "rep_number": 1,
                "start_frame": start_frame,
                "peak_frame": peak_frame,
                "end_frame": end_frame
            }]
        }
        
        with open(self.output_json, 'w') as f:
            json.dump(metadata, f, indent=4)

class IdealPathCalculator:
    def __init__(self, input_csv: str, input_json: str, output_csv: str):
        self.input_csv = input_csv
        self.input_json = input_json
        self.output_csv = output_csv

    def process(self):
        df = pd.read_csv(self.input_csv)
        with open(self.input_json, 'r') as f:
            rep = json.load(f)['reps'][0]
            
        start_idx = df.index[df['frame'] == rep['start_frame']].tolist()[0]
        peak_idx = df.index[df['frame'] == rep['peak_frame']].tolist()[0]
        
        start_x = df.loc[start_idx, 'smooth_bar_x']
        start_y = df.loc[start_idx, 'smooth_bar_y']
        peak_y = df.loc[peak_idx, 'smooth_bar_y']
        
        h = max(peak_y - start_y, 0.5) 
            
        keypoints_y = [
            start_y,
            start_y + (0.25 * h),
            start_y + (0.45 * h),
            start_y + (0.80 * h),
            start_y + (1.00 * h)
        ]
        keypoints_x = [
            start_x,
            start_x - (0.04 * h),
            start_x + (0.01 * h),
            start_x - (0.05 * h),
            start_x - (0.06 * h)
        ]
        
        sorted_indices = np.argsort(keypoints_y)
        spline = CubicSpline(np.array(keypoints_y)[sorted_indices], np.array(keypoints_x)[sorted_indices], bc_type='natural')
        
        ideal_y = np.linspace(start_y, peak_y, 100)
        pd.DataFrame({'ideal_bar_x': spline(ideal_y), 'ideal_bar_y': ideal_y}).to_csv(self.output_csv, index=False)