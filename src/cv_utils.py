import cv2
import mediapipe as mp
import pandas as pd
import json
import numpy as np
import sys

class PoseExtractor:
    def __init__(self, video_path: str, output_csv_path: str):
        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}", file=sys.stderr)
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                data.append({
                    'frame': frame_idx,
                    'time_sec': frame_idx / fps,
                    'left_wrist_x': lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                    'left_wrist_y': lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                    'right_wrist_x': lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    'right_wrist_y': lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                    'left_shoulder_x': lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    'left_shoulder_y': lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    'left_hip_x': lm[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                    'left_hip_y': lm[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                    'left_knee_x': lm[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                    'left_knee_y': lm[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                    'left_ankle_x': lm[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                    'left_ankle_y': lm[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                })
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...", end='\r')

        cap.release()
        pd.DataFrame(data).to_csv(self.output_csv_path, index=False)
        print(f"\nSaved {len(data)} frames to {self.output_csv_path}")


class VideoRenderer:
    def __init__(self, input_video, input_csv, input_ideal, input_json, output_video):
        self.input_video = input_video
        self.input_csv = input_csv
        self.input_ideal = input_ideal
        self.input_json = input_json
        self.output_video = output_video

    def render(self):
        df = pd.read_csv(self.input_csv).set_index('frame')
        ideal_df = pd.read_csv(self.input_ideal)
        
        with open(self.input_json, 'r') as f:
            metadata = json.load(f)

        if not metadata.get('reps'):
            print("Error: No reps detected.", file=sys.stderr)
            return
            
        rep = metadata['reps'][0]
        start_frame, end_frame = rep['start_frame'], rep['end_frame']
        
        cap = cv2.VideoCapture(self.input_video)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Vectorized Pixel Coordinates (Pre-computed for OpenCV performance)
        ideal_df['px_x'] = (ideal_df['ideal_bar_x'] * width).astype(int)
        ideal_df['px_y'] = ((1.0 - ideal_df['ideal_bar_y']) * height).astype(int)
        ideal_pts_array = ideal_df[['px_x', 'px_y']].values.reshape((-1, 1, 2))

        # 1. Bar coordinates (Un-invert for OpenCV)
        if 'smooth_bar_x' in df.columns:
            df['smooth_bar_px_x'] = (df['smooth_bar_x'] * width).fillna(0).astype(int)
            df['smooth_bar_px_y'] = ((1.0 - df['smooth_bar_y']) * height).fillna(0).astype(int)

        # 2. Body coordinates (Keep top-down, no inverting needed)
        body_cols = ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']
        for col in body_cols:
            if f'{col}_x' in df.columns:
                df[f'{col}_px_x'] = (df[f'{col}_x'] * width).fillna(0).astype(int)
                df[f'{col}_px_y'] = (df[f'{col}_y'] * height).fillna(0).astype(int)

        actual_path_history = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if start_frame <= frame_idx <= end_frame:
                cv2.polylines(frame, [ideal_pts_array], False, (0, 255, 0), 3, cv2.LINE_AA)
                
                if frame_idx in df.index and not df.loc[frame_idx].empty:
                    row = df.loc[frame_idx]
                    
                    joints = {
                        'shoulder': (int(row['left_shoulder_px_x']), int(row['left_shoulder_px_y'])),
                        'hip': (int(row['left_hip_px_x']), int(row['left_hip_px_y'])),
                        'knee': (int(row['left_knee_px_x']), int(row['left_knee_px_y'])),
                        'ankle': (int(row['left_ankle_px_x']), int(row['left_ankle_px_y']))
                    }
                    bar_pt = (int(row['smooth_bar_px_x']), int(row['smooth_bar_px_y']))
                    
                    cv2.line(frame, joints['shoulder'], joints['hip'], (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, joints['hip'], joints['knee'], (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, joints['knee'], joints['ankle'], (255, 255, 255), 4, cv2.LINE_AA)
                    
                    for pt in joints.values():
                        cv2.circle(frame, pt, 6, (0, 165, 255), -1, cv2.LINE_AA)

                    actual_path_history.append(bar_pt)
                    if len(actual_path_history) > 30: actual_path_history.pop(0)
                        
                    for i in range(1, len(actual_path_history)):
                        thick = int(max(1, (i / len(actual_path_history)) * 5))
                        cv2.line(frame, actual_path_history[i-1], actual_path_history[i], (0, 0, 255), thick, cv2.LINE_AA)
                    
                    cv2.circle(frame, bar_pt, 10, (0, 0, 255), -1, cv2.LINE_AA)
                    
                    cv2.putText(frame, f"Vel: {row['vel_y']:.2f} m/s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f"Vel: {row['vel_y']:.2f} m/s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Hip Ext: {row['hip_angle']:.0f} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f"Hip Ext: {row['hip_angle']:.0f} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Render complete! Saved {self.output_video}")