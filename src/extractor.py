import cv2
import mediapipe as mp
import pandas as pd
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