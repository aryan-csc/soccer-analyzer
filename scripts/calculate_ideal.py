import argparse
from src.physics import IdealPathCalculator

def main():
    parser = argparse.ArgumentParser(description="Calculates the ideal bar path.")
    parser.add_argument("--input_csv", required=True, help="Clean kinematics CSV")
    parser.add_argument("--input_json", required=True, help="Metadata JSON with rep frames")
    parser.add_argument("--output", required=True, help="Output ideal path CSV")
    args = parser.parse_args()

    print("Calculating ideal bar path...")
    calculator = IdealPathCalculator(args.input_csv, args.input_json, args.output)
    calculator.process()

if __name__ == "__main__":
    main()