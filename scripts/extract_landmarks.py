"""Optional standalone pose extraction; output contains 2D pixel coordinates."""
import argparse
from pathlib import Path
from src.extractor import extract_pose
from src.pipeline import write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='Standardized constant-frame-rate MP4')
    parser.add_argument('output', type=Path, help='Pose JSON file')
    args = parser.parse_args()
    write_json(args.output, extract_pose(args.video))
