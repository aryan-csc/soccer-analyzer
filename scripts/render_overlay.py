import argparse
import os
import tempfile
from src.cv_utils import VideoRenderer
from src.math_utils import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Renders tracking, ideal path, and report onto video.")
    parser.add_argument("--input_video", required=True, help="Raw input video")
    parser.add_argument("--input_csv", required=True, help="Clean kinematics CSV")
    parser.add_argument("--input_ideal", required=True, help="Ideal path CSV")
    parser.add_argument("--input_json", required=True, help="Rep metadata JSON")
    parser.add_argument("--output", required=True, help="Final rendered MP4")
    args = parser.parse_args()

    print(f"Rendering final video to {args.output}...")
    renderer = VideoRenderer(
        args.input_video, 
        args.input_csv, 
        args.input_ideal, 
        args.input_json, 
        args.output,
    )
    renderer.render()

if __name__ == "__main__":
    main()