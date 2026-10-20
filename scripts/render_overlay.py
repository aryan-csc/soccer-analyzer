"""Optional standalone annotated soccer video export."""
import argparse
import json
from pathlib import Path
from src.renderer import render_overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render_overlay(args.video, json.loads(args.report.read_text()), args.output)
