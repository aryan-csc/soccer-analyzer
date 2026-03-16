import sys
from src.physics_utils import RepDetector

input_path, output_path = sys.argv[1], sys.argv[2]

print(f"Detecting reps in {input_path}...")
RepDetector(input_path, output_path).process()