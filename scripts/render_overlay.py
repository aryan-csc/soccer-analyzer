import sys
from src.cv_utils import VideoRenderer

input_video, input_csv, input_ideal, input_json, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

print(f"Rendering final video to {output_path}...")
VideoRenderer(input_video, input_csv, input_ideal, input_json, output_path).render()