import sys
from src.physics_utils import IdealPathCalculator

input_csv, input_json, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

print("Calculating ideal bar path...")
IdealPathCalculator(input_csv, input_json, output_path).process()