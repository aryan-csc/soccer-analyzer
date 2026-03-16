import argparse
from src.cv_utils import PoseExtractor

def main():
    parser = argparse.ArgumentParser(description="Extracts MediaPipe landmarks from video.")
    parser.add_argument("--input", required=True, help="Path to input raw video")
    parser.add_argument("--output", required=True, help="Path to output raw CSV")
    args = parser.parse_args()

    print(f"Extracting poses from {args.input}...")
    extractor = PoseExtractor(args.input, args.output)
    extractor.process()

if __name__ == "__main__":
    main()