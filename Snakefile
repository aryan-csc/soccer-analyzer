import os
from glob import glob

shell.prefix("PYTHONPATH=. ")
VIDEOS = glob("data/raw/*.mp4")
VIDEO_NAMES = [os.path.splitext(os.path.basename(v))[0] for v in VIDEOS]

rule all:
    input:
        expand("results/{video_name}_analyzed.mp4", video_name=VIDEO_NAMES)

rule extract_pose:
    input:
        "data/raw/{video_name}.mp4"
    output:
        "data/interim/{video_name}_raw_landmarks.csv"
    shell:
        "venv/bin/python scripts/extract_pose.py --input {input} --output {output}"

rule process_kinematics:
    input:
        "data/interim/{video_name}_raw_landmarks.csv"
    output:
        "data/processed/{video_name}_clean_kinematics.csv"
    shell:
        "venv/bin/python scripts/process_kinematics.py --input {input} --output {output}"

rule detect_reps:
    input:
        "data/processed/{video_name}_clean_kinematics.csv"
    output:
        "data/processed/{video_name}_metadata.json"
    shell:
        "venv/bin/python scripts/detect_reps.py --input {input} --output {output}"

rule calculate_ideal:
    input:
        csv="data/processed/{video_name}_clean_kinematics.csv",
        json="data/processed/{video_name}_metadata.json"
    output:
        "data/processed/{video_name}_ideal_path.csv"
    shell:
        "venv/bin/python scripts/calculate_ideal.py --input_csv {input.csv} --input_json {input.json} --output {output}"

rule render_overlay:
    input:
        video="data/raw/{video_name}.mp4",
        csv="data/processed/{video_name}_clean_kinematics.csv",
        ideal="data/processed/{video_name}_ideal_path.csv",
        json="data/processed/{video_name}_metadata.json"
    output:
        "results/{video_name}_analyzed.mp4"
    shell:
        "venv/bin/python scripts/render_overlay.py --input_video {input.video} --input_csv {input.csv} --input_ideal {input.ideal} --input_json {input.json} --output {output}"