# src/physics.py
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
        """Calculates velocity, acceleration, and normalizes coordinates."""
        df = pd.read_csv(self.input_csv)
        
        # 1. BARBELL PROXY & Y-INVERSION
        # MediaPipe Y is 0 at the top of the screen. We invert it so 0 is the floor (Standard Physics)
        df['bar_x'] = (df['left_wrist_x'] + df['right_wrist_x']) / 2.0
        df['bar_y'] = 1.0 - ((df['left_wrist_y'] + df['right_wrist_y']) / 2.0)
        
        # Invert body Y coordinates for accurate angle math
        for col in df.columns:
            if col.endswith('_y') and col not in ['bar_y']:
                df[col + '_inv'] = 1.0 - df[col]

        # 2. SMOOTHING
        df['smooth_bar_x'] = smooth_data(df['bar_x'])
        df['smooth_bar_y'] = smooth_data(df['bar_y'])

        # 3. KINEMATICS (Velocity)
        # dt is the time difference between frames
        dt = df['time_sec'].diff().fillna(df['time_sec'].iloc[1] - df['time_sec'].iloc[0])
        dt = dt.replace(0, 0.033) # Fallback to ~30fps if dt is 0
        
        # Vertical Velocity (v = dy/dt)
        df['vel_y'] = df['smooth_bar_y'].diff() / dt
        df['vel_y'] = smooth_data(df['vel_y']) # We MUST smooth the derivative too!

        # 4. BIOMECHANICS (Hip Extension)
        # Calculate the angle between Shoulder, Hip, and Knee
        df['hip_angle'] = df.apply(lambda row: calculate_angle(
            row['left_shoulder_x'], row['left_shoulder_y_inv'],
            row['left_hip_x'], row['left_hip_y_inv'],
            row['left_knee_x'], row['left_knee_y_inv']
        ), axis=1)
        
        # Apply slight smoothing to the angle to remove jitter
        df['hip_angle'] = smooth_data(df['hip_angle'], window_length=9)

        # Save to Processed Directory
        df.to_csv(self.output_csv, index=False)
        print(f"Kinematics processed successfully.")

class RepDetector:
    def __init__(self, input_csv: str, output_json: str):
        self.input_csv = input_csv
        self.output_json = output_json

    def process(self):
        """Isolates the lifting phase by analyzing bar height and velocity."""
        df = pd.read_csv(self.input_csv)
        
        # 1. Find the peak of the lift (Maximum bar height)
        # We use idxmax() to get the DataFrame index of the highest point
        peak_idx = df['smooth_bar_y'].idxmax()
        
        # 2. Find the start (Scan backwards from the peak)
        # Look for the moment the bar was low to the ground AND velocity was roughly 0
        start_idx = 0
        for i in range(peak_idx, -1, -1):
            # 0.35 is a rough normalized height for plates resting on the floor
            if df.loc[i, 'smooth_bar_y'] < 0.35 and df.loc[i, 'vel_y'] <= 0.05:
                start_idx = i
                break
                
        # 3. Find the end (Scan forwards from the peak)
        # Look for the moment the bar drops significantly
        end_idx = len(df) - 1
        for i in range(peak_idx, len(df)):
            if df.loc[i, 'smooth_bar_y'] < 0.6 and df.loc[i, 'vel_y'] < -0.2:
                end_idx = i
                break
                
        # Convert pandas indices back to actual frame numbers
        start_frame = int(df.loc[start_idx, 'frame'])
        peak_frame = int(df.loc[peak_idx, 'frame'])
        end_frame = int(df.loc[end_idx, 'frame'])
        
        # 4. Save metadata
        metadata = {
            "reps": [
                {
                    "rep_number": 1,
                    "start_frame": start_frame,
                    "peak_frame": peak_frame,
                    "end_frame": end_frame
                }
            ]
        }
        
        with open(self.output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Rep detected: Frames {start_frame} to {end_frame}")
        print(f"Metadata saved to {self.output_json}")

class IdealPathCalculator:
    def __init__(self, input_csv: str, input_json: str, output_csv: str):
        self.input_csv = input_csv
        self.input_json = input_json
        self.output_csv = output_csv

    def process(self):
        """Generates a mathematical S-Curve based on the lifter's dimensions."""
        df = pd.read_csv(self.input_csv)
        
        with open(self.input_json, 'r') as f:
            metadata = json.load(f)
            
        rep = metadata['reps'][0]
        start_idx = df.index[df['frame'] == rep['start_frame']].tolist()[0]
        peak_idx = df.index[df['frame'] == rep['peak_frame']].tolist()[0]
        
        # Get starting coordinates
        start_x = df.loc[start_idx, 'smooth_bar_x']
        start_y = df.loc[start_idx, 'smooth_bar_y']
        peak_y = df.loc[peak_idx, 'smooth_bar_y']
        
        # Calculate total vertical distance of the lift
        # (Assuming positive Y is UP, based on our earlier inversion)
        h = peak_y - start_y
        
        if h <= 0:
            print("Error: Lift height is negative or zero. Check coordinates.")
            h = 0.5 # Fallback to prevent crashes
            
        # Define the S-Curve Keypoints (Relative to Start X)
        # These coefficients represent the ideal biomechanical deviations
        keypoints_y = [
            start_y,                 # Floor
            start_y + (0.25 * h),    # Above Knee (Pulled back)
            start_y + (0.45 * h),    # Power Position (Knees shift forward)
            start_y + (0.80 * h),    # High Pull (Max extension, traveling back)
            start_y + (1.00 * h)     # Catch (Overhead)
        ]
        
        keypoints_x = [
            start_x,                 # Floor
            start_x - (0.04 * h),    # Pulled back 4% of lift height
            start_x + (0.01 * h),    # Shifted forward slightly past start line
            start_x - (0.05 * h),    # Extended back
            start_x - (0.06 * h)     # Locked out overhead
        ]
        
        # We must ensure the Y coordinates are strictly increasing for CubicSpline
        # If the data is messy, this forces it to be monotonic
        sorted_indices = np.argsort(keypoints_y)
        kp_y_sorted = np.array(keypoints_y)[sorted_indices]
        kp_x_sorted = np.array(keypoints_x)[sorted_indices]
        
        # Fit a smooth cubic spline through these keypoints
        # bc_type='natural' ensures the ends of the curve aren't forced into weird angles
        spline = CubicSpline(kp_y_sorted, kp_x_sorted, bc_type='natural')
        
        # Generate 100 smooth points along this ideal path
        ideal_y = np.linspace(start_y, peak_y, 100)
        ideal_x = spline(ideal_y)
        
        # Save the ideal path
        ideal_df = pd.DataFrame({
            'ideal_bar_x': ideal_x,
            'ideal_bar_y': ideal_y
        })
        
        ideal_df.to_csv(self.output_csv, index=False)
        print(f"Generated ideal S-Curve. Saved to {self.output_csv}")