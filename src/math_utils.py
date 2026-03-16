# src/math_utils.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import json
import matplotlib.pyplot as plt

def smooth_data(series: pd.Series, window_length: int = 13, polyorder: int = 3) -> pd.Series:
    """
    Applies Savitzky-Golay filter to remove tracking jitter.
    """
    # Forward/backward fill any NaNs (missing tracking frames)
    filled_series = series.bfill().ffill()
    
    # Safety check: window length must be odd and less than the data length
    if len(filled_series) < window_length:
        window_length = len(filled_series) - 1 if len(filled_series) % 2 == 0 else len(filled_series)
        if window_length < 3: 
            return filled_series # Too short to smooth
            
    return savgol_filter(filled_series, window_length, polyorder)

def calculate_angle(a_x, a_y, b_x, b_y, c_x, c_y):
    """
    Calculates the interior angle ABC (with B as the vertex) in degrees.
    Used for calculating joint extension (e.g., Shoulder -> Hip -> Knee).
    """
    ba = np.array([a_x - b_x, a_y - b_y])
    bc = np.array([c_x - b_x, c_y - b_y])
    
    # Cosine rule: dot product divided by magnitudes
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    
    # Clip to [-1.0, 1.0] to prevent floating point precision errors in arccos
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

class ReportGenerator:
    def __init__(self, input_csv: str, input_json: str, output_report: str):
        self.input_csv = input_csv
        self.input_json = input_json
        self.output_report = output_report

    def generate(self):
        """Generates a static dashboard of the lift's kinematics."""
        print("Generating telemetry report...")
        
        # Load data
        df = pd.read_csv(self.input_csv)
        with open(self.input_json, 'r') as f:
            metadata = json.load(f)
            
        rep = metadata['reps'][0]
        
        # Filter the dataframe to just the actual lift (with a tiny buffer)
        buffer = 5
        start_frame = max(0, rep['start_frame'] - buffer)
        end_frame = min(len(df) - 1, rep['end_frame'] + buffer)
        
        rep_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()
        
        # Normalize X so the graph starts at 0 for cleaner reading
        start_x = rep_df.iloc[0]['smooth_bar_x']
        rep_df['norm_x'] = rep_df['smooth_bar_x'] - start_x

        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Olympic Snatch Telemetry Report', fontsize=20, fontweight='bold')

        # --- Plot 1: Bar Path ---
        axes[0].plot(rep_df['norm_x'], rep_df['smooth_bar_y'], color='blue', linewidth=3)
        axes[0].scatter(rep_df['norm_x'].iloc[0], rep_df['smooth_bar_y'].iloc[0], color='green', s=100, label='Start', zorder=5)
        axes[0].scatter(rep_df['norm_x'].iloc[-1], rep_df['smooth_bar_y'].iloc[-1], color='red', s=100, label='Catch', zorder=5)
        axes[0].axvline(0, color='gray', linestyle='--', alpha=0.7)
        axes[0].set_title('Bar Path (Side View)', fontsize=14)
        axes[0].set_xlabel('Horizontal Displacement (Relative)')
        axes[0].set_ylabel('Vertical Height (Relative)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # --- Plot 2: Vertical Velocity ---
        axes[1].plot(rep_df['time_sec'], rep_df['vel_y'], color='red', linewidth=2)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_title('Barbell Vertical Velocity', fontsize=14)
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Velocity')
        axes[1].grid(True, alpha=0.3)
        
        # --- Plot 3: Hip Angle ---
        axes[2].plot(rep_df['time_sec'], rep_df['hip_angle'], color='green', linewidth=2)
        axes[2].axhline(180, color='gray', linestyle='--', alpha=0.7, label='Full Extension')
        axes[2].set_title('Hip Extension Angle', fontsize=14)
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Angle (Degrees)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # Save and close
        plt.tight_layout()
        # Adjust layout so the suptitle doesn't overlap
        plt.subplots_adjust(top=0.88) 
        plt.savefig(self.output_report, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Report saved to {self.output_report}")