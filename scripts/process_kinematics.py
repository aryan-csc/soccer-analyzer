import argparse
from src.physics import KinematicsProcessor

def main():
    parser = argparse.ArgumentParser(description="Cleans data and calculates kinematics.")
    parser.add_argument("--input", required=True, help="Path to raw landmarks CSV")
    parser.add_argument("--output", required=True, help="Path to clean kinematics CSV")
    args = parser.parse_args()

    print(f"Processing kinematics for {args.input}...")
    processor = KinematicsProcessor(args.input, args.output)
    processor.process()

if __name__ == "__main__":
    main()