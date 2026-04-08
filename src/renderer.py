import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import json
import sys
import math


class VideoRenderer:
    def __init__(self, input_video, input_csv, input_ideal, input_json, output_video):
        self.input_video = input_video
        self.input_csv = input_csv
        self.input_ideal = input_ideal
        self.input_json = input_json
        self.output_video = output_video

    def _draw_clean_text(self, img, text, pos, font_scale, color, thickness=1):
        x, y = pos
        cv2.putText(
            img,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def render(self):
        df = pd.read_csv(self.input_csv).set_index("frame")
        ideal_df = pd.read_csv(self.input_ideal)

        with open(self.input_json, "r") as f:
            metadata = json.load(f)

        if not metadata.get("reps"):
            print("Error: No reps detected.", file=sys.stderr)
            return

        rep = metadata["reps"][0]
        start_frame, end_frame = rep["start_frame"], rep["end_frame"]

        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            self.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        SYS_BLUE = (244, 133, 66)
        SYS_WHITE = (245, 245, 245)
        SYS_GRAY = (180, 180, 180)
        SYS_RED = (58, 48, 255)

        ideal_df["px_x"] = (ideal_df["ideal_bar_x"] * width).astype(int)
        ideal_df["px_y"] = ((1.0 - ideal_df["ideal_bar_y"]) * height).astype(int)
        ideal_pts_array = ideal_df[["px_x", "px_y"]].values.reshape((-1, 1, 2))

        for col in df.columns:
            if col.endswith("_x"):
                df[f'{col.removesuffix("_x")}_px_x'] = (
                    (df[col] * width).fillna(0).astype(int)
                )
            if col.endswith("_y"):
                df[f'{col.removesuffix("_y")}_px_y'] = (
                    (df[col] * height).fillna(0).astype(int)
                )

        rep_df = df.loc[start_frame:end_frame]
        max_vel = rep_df["vel_y"].max()
        min_vel = rep_df["vel_y"].min()
        vel_range = max_vel - min_vel if (max_vel - min_vel) > 0 else 1.0

        gx = max(10, int(width * 0.025))
        gy = max(10, int(height * 0.025))
        gw = int(width / 3)
        gh = int(gw * 0.35)

        graph_dict = {}
        for f in range(start_frame, end_frame + 1):
            if f in df.index and not pd.isna(df.loc[f, "vel_y"]):
                v = df.loc[f, "vel_y"]
                px = gx + int(((f - start_frame) / (end_frame - start_frame)) * gw)
                py = gy + gh - int(((v - min_vel) / vel_range) * gh)
                graph_dict[f] = (px, py)

        base_x = ideal_pts_array[0][0][0]
        region_margin = max(15, int(width * 0.025))
        top_y = np.min(ideal_pts_array[:, 0, 1]) - 50
        bot_y = np.max(ideal_pts_array[:, 0, 1]) + 50

        actual_path_history = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if start_frame <= frame_idx <= end_frame:

                region_overlay = frame.copy()
                cv2.rectangle(
                    region_overlay,
                    (base_x - region_margin, top_y),
                    (base_x + region_margin, bot_y),
                    SYS_BLUE,
                    -1,
                )
                cv2.addWeighted(region_overlay, 0.15, frame, 0.85, 0, frame)
                cv2.rectangle(
                    frame,
                    (base_x - region_margin, top_y),
                    (base_x + region_margin, bot_y),
                    SYS_WHITE,
                    1,
                    cv2.LINE_AA,
                )

                cv2.polylines(frame, [ideal_pts_array], False, SYS_GRAY, 2, cv2.LINE_AA)

                if frame_idx in df.index and not df.loc[frame_idx].empty:
                    row = df.loc[frame_idx]

                    joints = {
                        "shoulder": (
                            int(row["shoulder_px_x"]),
                            int(row["shoulder_px_y"]),
                        ),
                        "elbow": (int(row["elbow_px_x"]), int(row["elbow_px_y"])),
                        "hip": (int(row["hip_px_x"]), int(row["hip_px_y"])),
                        "knee": (int(row["knee_px_x"]), int(row["knee_px_y"])),
                        "ankle": (int(row["ankle_px_x"]), int(row["ankle_px_y"])),
                    }
                    bar_pt = (int(row["bar_px_x"]), int(row["bar_px_y"]))

                    hip, shoulder, knee = (
                        joints["hip"],
                        joints["shoulder"],
                        joints["knee"],
                    )

                    angle1 = math.degrees(
                        math.atan2(shoulder[1] - hip[1], shoulder[0] - hip[0])
                    )
                    angle2 = math.degrees(
                        math.atan2(knee[1] - hip[1], knee[0] - hip[0])
                    )

                    interior_angle = abs(angle1 - angle2)
                    if interior_angle > 180.0:
                        interior_angle = 360.0 - interior_angle

                    start_angle = min(angle1, angle2)
                    end_angle = max(angle1, angle2)
                    if end_angle - start_angle > 180:
                        start_angle, end_angle = end_angle, start_angle + 360

                    overlay = frame.copy()
                    cv2.ellipse(
                        overlay,
                        hip,
                        (45, 45),
                        0,
                        start_angle,
                        end_angle,
                        SYS_BLUE,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                    self._draw_clean_text(
                        frame,
                        f"{interior_angle:.0f}",
                        (hip[0] + 15, hip[1] - 15),
                        0.6,
                        SYS_WHITE,
                        2,
                    )

                    bone_pairs = [
                        (joints["shoulder"], joints["hip"]),
                        (joints["shoulder"], joints["elbow"]),
                        (joints["hip"], joints["knee"]),
                        (joints["knee"], joints["ankle"]),
                    ]

                    for p1, p2 in bone_pairs:
                        cv2.line(frame, p1, p2, SYS_WHITE, 3, cv2.LINE_AA)

                    for pt in joints.values():
                        cv2.circle(frame, pt, 8, SYS_BLUE, -1, cv2.LINE_AA)
                        cv2.circle(frame, pt, 8, SYS_WHITE, 1, cv2.LINE_AA)

                    actual_path_history.append(bar_pt)
                    if len(actual_path_history) > 75:
                        actual_path_history.pop(0)

                    for i in range(1, len(actual_path_history)):
                        thick = int(max(1, (i / len(actual_path_history)) * 6))
                        segment_x = actual_path_history[i][0]
                        path_color = (
                            SYS_RED
                            if segment_x < (base_x - region_margin)
                            or segment_x > (base_x + region_margin)
                            else SYS_BLUE
                        )
                        cv2.line(
                            frame,
                            actual_path_history[i - 1],
                            actual_path_history[i],
                            path_color,
                            thick,
                            cv2.LINE_AA,
                        )

                    is_out_of_bounds = bar_pt[0] < (base_x - region_margin) or bar_pt[
                        0
                    ] > (base_x + region_margin)
                    cv2.circle(frame, bar_pt, 10, SYS_WHITE, -1, cv2.LINE_AA)
                    cv2.circle(
                        frame,
                        bar_pt,
                        10,
                        SYS_RED if is_out_of_bounds else SYS_BLUE,
                        2,
                        cv2.LINE_AA,
                    )

                    ui_overlay = frame.copy()
                    cv2.rectangle(
                        ui_overlay,
                        (gx - 15, gy - 15),
                        (gx + gw + 15, gy + gh + 40),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.addWeighted(ui_overlay, 0.65, frame, 0.35, 0, frame)

                    cv2.line(
                        frame, (gx, gy), (gx + gw, gy), (80, 80, 80), 1, cv2.LINE_AA
                    )
                    cv2.line(
                        frame,
                        (gx, gy + gh),
                        (gx + gw, gy + gh),
                        (80, 80, 80),
                        1,
                        cv2.LINE_AA,
                    )

                    if min_vel <= 0 <= max_vel:
                        zero_py = gy + gh - int(((0 - min_vel) / vel_range) * gh)
                        cv2.line(
                            frame,
                            (gx, zero_py),
                            (gx + gw, zero_py),
                            (100, 100, 100),
                            1,
                            cv2.LINE_AA,
                        )
                        self._draw_clean_text(
                            frame,
                            "0 m/s",
                            (gx + gw - 40, zero_py - 4),
                            0.35,
                            SYS_GRAY,
                            1,
                        )

                    full_pts = [
                        graph_dict[f]
                        for f in range(start_frame, end_frame + 1)
                        if f in graph_dict
                    ]
                    if len(full_pts) > 1:
                        cv2.polylines(
                            frame,
                            [np.array(full_pts, np.int32)],
                            False,
                            (100, 100, 100),
                            2,
                            cv2.LINE_AA,
                        )

                    prog_pts = [
                        graph_dict[f]
                        for f in range(start_frame, frame_idx + 1)
                        if f in graph_dict
                    ]
                    if len(prog_pts) > 1:
                        cv2.polylines(
                            frame,
                            [np.array(prog_pts, np.int32)],
                            False,
                            SYS_BLUE,
                            3,
                            cv2.LINE_AA,
                        )

                    if frame_idx in graph_dict:
                        curr_pt = graph_dict[frame_idx]
                        cv2.circle(frame, curr_pt, 5, SYS_WHITE, -1, cv2.LINE_AA)
                        cv2.circle(frame, curr_pt, 5, SYS_BLUE, 1, cv2.LINE_AA)

                        self._draw_clean_text(
                            frame, "Velocity", (gx, gy + gh + 22), 0.45, SYS_GRAY, 1
                        )
                        self._draw_clean_text(
                            frame,
                            f"{row['vel_y']:.2f} m/s",
                            (gx + 65, gy + gh + 22),
                            0.5,
                            SYS_WHITE,
                            2,
                        )

                out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Render complete! Saved {self.output_video}")
