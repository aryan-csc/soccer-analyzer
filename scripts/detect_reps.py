import argparse
from src.physics import RepDetector

def main():
    parser = argparse.ArgumentParser(description="Detects actual lift phases.")
    parser.add_argument("--input", required=True, help="Path to clean kinematics CSV")
    parser.add_argument("--output", required=True, help="Path to output metadata JSON")
    args = parser.parse_args()

    print(f"Detecting reps in {args.input}...")
    detector = RepDetector(args.input, args.output)
    detector.process()

if __name__ == "__main__":
    main()