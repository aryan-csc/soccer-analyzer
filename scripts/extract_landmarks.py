import sys
from src.extractor import PoseExtractor

input_path, output_path = sys.argv[1], sys.argv[2]

print(f"Extracting poses from {input_path}...")
PoseExtractor(input_path, output_path).process()