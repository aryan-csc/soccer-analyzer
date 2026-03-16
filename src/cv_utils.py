# src/cv_utils.py
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
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        # model_complexity=2 is the heaviest but most accurate model (ideal for offline processing)
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self):
        """Runs MediaPipe over the video and saves raw landmarks to CSV."""
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
            if not ret:
                break

            # MediaPipe requires RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract the specific nodes needed for Snatch kinematics
                row = {
                    'frame': frame_idx,
                    'time_sec': frame_idx / fps,
                    
                    # Barbell proxies (Nodes 15 and 16)
                    'left_wrist_x': landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                    'left_wrist_y': landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                    'right_wrist_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    'right_wrist_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                    
                    # Body angles (Shoulder, Hip, Knee, Ankle)
                    'left_shoulder_x': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    'left_shoulder_y': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    'left_hip_x': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                    'left_hip_y': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                    'left_knee_x': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                    'left_knee_y': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                    'left_ankle_x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                    'left_ankle_y': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                }
                data.append(row)
            
            frame_idx += 1
            
            # Simple progress tracker
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...", end='\r')

        cap.release()
        
        # Save to the interim directory
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv_path, index=False)
        print(f"\nSaved {len(df)} frames of coordinate data to {self.output_csv_path}")


class VideoRenderer:
    def __init__(self, input_video, input_csv, input_ideal, input_json, output_video):
        self.input_video = input_video
        self.input_csv = input_csv
        self.input_ideal = input_ideal
        self.input_json = input_json
        self.output_video = output_video

    def _denormalize(self, x, y, width, height):
        """Converts 0.0-1.0 coordinates back to pixel coordinates."""
        # Note: We must un-invert the Y coordinate here because OpenCV 
        # draws from the top-left (0,0) down.
        pixel_x = int(x * width)
        pixel_y = int((1.0 - y) * height)
        return (pixel_x, pixel_y)

    def render(self):
        """Draws skeletons and bar paths onto frames and saves video."""
        print(f"Opening video {self.input_video} for rendering...")
        
        # Load datasets
        df = pd.read_csv(self.input_csv)
        df.set_index('frame', inplace=True)
        ideal_df = pd.read_csv(self.input_ideal)
        with open(self.input_json, 'r') as f:
            metadata = json.load(f)

        # Check for reps before proceeding
        if not metadata['reps']:
            print("Error: No reps detected in metadata. Aborting render.", file=sys.stderr)
            return
            
        rep = metadata['reps'][0]
        start_frame = rep['start_frame']
        end_frame = rep['end_frame']
        
        # Video Setup
        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Use mp4v codec for standard MP4 output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        # Prepare Ideal Path pixel coordinates
        ideal_pts = []
        for _, row in ideal_df.iterrows():
            pt = self._denormalize(row['ideal_bar_x'], row['ideal_bar_y'], width, height)
            ideal_pts.append(pt)
            
        # Draw the ideal path as a static dashed line across the whole screen? 
        # Usually it looks better to just draw it as a solid semi-transparent line.
        ideal_pts_array = np.array(ideal_pts, np.int32).reshape((-1, 1, 2))

        # Prepare history for the actual trailing bar path
        actual_path_history = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only render frames within our detected rep
            if start_frame <= frame_idx <= end_frame:
                
                # 1. DRAW IDEAL PATH (Static Green Line)
                cv2.polylines(frame, [ideal_pts_array], isClosed=False, color=(0, 255, 0), thickness=3)
                
                # Get current frame data (efficiently)
                try:
                    row = df.loc[frame_idx]
                except KeyError:
                    # This frame has no landmark data, skip drawing on it
                    frame_idx +=1
                    continue
                
                if not row.empty:
                    
                    # 2. DRAW SKELETON (Simplified)
                    joints = {
                        'shoulder': self._denormalize(row['left_shoulder_x'], row['left_shoulder_y'], width, height),
                        'hip': self._denormalize(row['left_hip_x'], row['left_hip_y'], width, height),
                        'knee': self._denormalize(row['left_knee_x'], row['left_knee_y'], width, height),
                        'ankle': self._denormalize(row['left_ankle_x'], row['left_ankle_y'], width, height)
                    }
                    
                    # Draw bones (White lines)
                    cv2.line(frame, joints['shoulder'], joints['hip'], (255, 255, 255), 4)
                    cv2.line(frame, joints['hip'], joints['knee'], (255, 255, 255), 4)
                    cv2.line(frame, joints['knee'], joints['ankle'], (255, 255, 255), 4)
                    
                    # Draw joints (Orange circles)
                    for joint_name, pt in joints.items():
                        cv2.circle(frame, pt, 6, (0, 165, 255), -1)

                    # 3. DRAW ACTUAL BAR PATH (Trailing Red Line)
                    bar_pt = self._denormalize(row['smooth_bar_x'], row['smooth_bar_y'], width, height)
                    actual_path_history.append(bar_pt)
                    
                    # Keep only the last 30 frames for the trail so it doesn't clutter the screen
                    if len(actual_path_history) > 30:
                        actual_path_history.pop(0)
                        
                    # Draw the trail
                    if len(actual_path_history) > 1:
                        for i in range(1, len(actual_path_history)):
                            # Fade effect: older points are drawn thinner
                            thickness = int(max(1, (i / len(actual_path_history)) * 5))
                            cv2.line(frame, actual_path_history[i-1], actual_path_history[i], (0, 0, 255), thickness)
                    
                    # Draw current bar position (Big Red Dot)
                    cv2.circle(frame, bar_pt, 10, (0, 0, 255), -1)
                    
                    # 4. OVERLAY METRICS (Top Left Corner)
                    vel = row['vel_y']
                    angle = row['hip_angle']
                    
                    # Define text properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1
                    color = (255, 255, 255) # White
                    thick = 2
                    
                    # Add black outline for readability against bright backgrounds
                    cv2.putText(frame, f"Vel: {vel:.2f} m/s", (30, 50), font, scale, (0,0,0), thick+2)
                    cv2.putText(frame, f"Vel: {vel:.2f} m/s", (30, 50), font, scale, color, thick)
                    
                    cv2.putText(frame, f"Hip Ext: {angle:.0f} deg", (30, 90), font, scale, (0,0,0), thick+2)
                    cv2.putText(frame, f"Hip Ext: {angle:.0f} deg", (30, 90), font, scale, color, thick)

                final_frame = frame

                # Write the annotated frame to the new video file
                out.write(final_frame)
                
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Render complete! Saved {self.output_video}")