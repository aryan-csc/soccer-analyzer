import argparse
from src.math_utils import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Generates matplotlib summary charts.")
    parser.add_argument("--input_csv", required=True, help="Clean kinematics CSV")
    parser.add_argument("--input_json", required=True, help="Rep metadata JSON")
    parser.add_argument("--output", required=True, help="Output PNG report")
    args = parser.parse_args()

    print(f"Generating visual report to {args.output}...")
    generator = ReportGenerator(args.input_csv, args.input_json, args.output)
    generator.generate()

if __name__ == "__main__":
    main()