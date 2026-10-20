"""Optional standalone soccer measurements from saved pose JSON."""
import argparse
import json
from pathlib import Path
from src.settings import Settings
from src.physics_utils import analyze_kick
from src.pipeline import write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--foot', choices=['left', 'right'], default='right')
    parser.add_argument('--contact-frame', type=int)
    args = parser.parse_args()
    raw = json.loads(args.pose.read_text())
    settings = Settings(foot=args.foot, contact_frame=args.contact_frame)
    write_json(args.output, analyze_kick(raw['frames'], raw['metadata'], settings.model_dump()))
