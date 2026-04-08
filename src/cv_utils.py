import cv2
import mediapipe as mp
import pandas as pd
import json
import sys

class PoseExtractor:

    PoseLandmark = mp.solutions.pose.PoseLandmark

    JOINTS = {
        "shoulder": (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
        "elbow":    (PoseLandmark.LEFT_ELBOW,    PoseLandmark.RIGHT_ELBOW),
        "hip":      (PoseLandmark.LEFT_HIP,      PoseLandmark.RIGHT_HIP),
        "knee":     (PoseLandmark.LEFT_KNEE,     PoseLandmark.RIGHT_KNEE),
        "ankle":    (PoseLandmark.LEFT_ANKLE,    PoseLandmark.RIGHT_ANKLE),
    }

    def __init__(self, video_path, output_csv_path):
        self.video_path = video_path
        self.output_csv_path = output_csv_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def _open_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}", file=sys.stderr)
            sys.exit(1)
        meta = {
            "fps":    cap.get(cv2.CAP_PROP_FPS),
            "total_frames":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return cap, meta

    def _detect_side_index(self, lm):
        """Return 0 for left, 1 for right, based on visibility in the first frame."""
        left_vis  = sum(lm[ids[0]].visibility for ids in self.JOINTS.values())
        right_vis = sum(lm[ids[1]].visibility for ids in self.JOINTS.values())
        return 0 if left_vis >= right_vis else 1

    def _init_tracker(self, cap):
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.", file=sys.stderr)
            sys.exit(1)

        # detect side
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            self.side_index = self._detect_side_index(results.pose_landmarks.landmark)
        else:
            self.side_index = 0
            print("Warning: Could not detect pose in first frame, defaulting to left side.")

        # select bbox
        print("\n--- BARBELL CALIBRATION ---")
        print("Draw a box around the barbell plate, then press ENTER")
        bbox = cv2.selectROI("Select Barbell Plate", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Barbell Plate")

        tracker = cv2.TrackerCSRT_create()
        tracker.init(first_frame, bbox)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return tracker

    @staticmethod
    def _bbox_center_normalized(bbox, width, height):
        cx = (bbox[0] + bbox[2] / 2.0) / width
        cy = (bbox[1] + bbox[3] / 2.0) / height
        return cx, cy

    def _extract_landmarks(self, lm):
        return {
            name: (lm[ids[self.side_index]].x, lm[ids[self.side_index]].y)
            for name, ids in self.JOINTS.items()
        }

    def process(self):
        cap, meta = self._open_capture()
        tracker = self._init_tracker(cap)
        data = []
        plate_xy = (0.5, 0.5)

        for frame_idx in range(meta["total_frames"]):
            ret, frame = cap.read()
            if not ret:
                break

            # barbell tracking
            success, bbox = tracker.update(frame)
            if success:
                plate_xy = self._bbox_center_normalized(bbox, meta["width"], meta["height"])

            # pose tracking
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if not results.pose_landmarks:
                continue

            row = {
                "frame":    frame_idx,
                "time": frame_idx / meta["fps"],
                "bar_x":  plate_xy[0],
                "bar_y":  plate_xy[1],
            }
            for name, (x, y) in self._extract_landmarks(results.pose_landmarks.landmark).items():
                row[f"{name}_x"] = x
                row[f"{name}_y"] = y
            data.append(row)

            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{meta["total_frames"]} frames...", end="\r")

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