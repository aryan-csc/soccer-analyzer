import os
from glob import glob

shell.prefix("PYTHONPATH=. venv/bin/python ")
VIDEOS = glob("data/input/*.mp4")
VIDEO_NAMES = [os.path.splitext(os.path.basename(v))[0] for v in VIDEOS]

rule all:
    input:
        expand("data/output/{video_name}_analyzed.mp4", video_name=VIDEO_NAMES)

rule extract_landmarks:
    input:
        "data/input/{video_name}.mp4"
    output:
        "data/process/{video_name}_landmarks.csv"
    shell:
        "scripts/extract_landmarks.py {input} {output}"

rule process_landmarks:
    input:
        "data/process/{video_name}_landmarks.csv"
    output:
        "data/process/{video_name}_processed_landmarks.csv"
    shell:
        "scripts/process_landmarks.py {input} {output}"

rule detect_rep:
    input:
        "data/process/{video_name}_processed_landmarks.csv"
    output:
        "data/process/{video_name}_metadata.json"
    shell:
        "scripts/detect_rep.py {input} {output}"

rule calculate_ideal:
    input:
        csv="data/process/{video_name}_processed_landmarks.csv",
        json="data/process/{video_name}_metadata.json"
    output:
        "data/process/{video_name}_ideal_path.csv"
    shell:
        "scripts/calculate_ideal.py {input.csv} {input.json} {output}"

rule render_overlay:
    input:
        video="data/input/{video_name}.mp4",
        csv="data/process/{video_name}_processed_landmarks.csv",
        ideal="data/process/{video_name}_ideal_path.csv",
        json="data/process/{video_name}_metadata.json"
    output:
        "data/output/{video_name}_analyzed.mp4"
    shell:
        "scripts/render_overlay.py {input.video} {input.csv} {input.ideal} {input.json} {output}"