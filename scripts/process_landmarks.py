import sys
from src.physics import KinematicsProcessor

input_path, output_path = sys.argv[1], sys.argv[2]

print(f"Processing kinematics for {input_path}...")
KinematicsProcessor(input_path, output_path).process()
