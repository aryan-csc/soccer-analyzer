import cv2
import mediapipe as mp
import pandas as pd
import json
import sys

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

        # vectorize pixel coordinates
        ideal_df['px_x'] = (ideal_df['ideal_bar_x'] * width).astype(int)
        ideal_df['px_y'] = ((1.0 - ideal_df['ideal_bar_y']) * height).astype(int)
        ideal_pts_array = ideal_df[['px_x', 'px_y']].values.reshape((-1, 1, 2))

        # vectorize landmarks
        for col in df.columns:
            if col.endswith('_x'):
                df[f'{col.removesuffix("_x")}_px_x'] = (df[col] * width).fillna(0).astype(int)
            if col.endswith('_y'):
                df[f'{col.removesuffix("_y")}_px_y'] = (df[col] * height).fillna(0).astype(int)

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
                        'shoulder': (int(row['shoulder_px_x']), int(row['shoulder_px_y'])),
                        'elbow': (int(row['elbow_px_x']), int(row['elbow_px_y'])),
                        'hip': (int(row['hip_px_x']), int(row['hip_px_y'])),
                        'knee': (int(row['knee_px_x']), int(row['knee_px_y'])),
                        'ankle': (int(row['ankle_px_x']), int(row['ankle_px_y']))
                    }
                    bar_pt = (int(row['bar_px_x']), int(row['bar_px_y']))
                    
                    cv2.line(frame, joints['shoulder'], joints['hip'], (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, joints['shoulder'], joints['elbow'], (255, 255, 255), 4, cv2.LINE_AA)
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