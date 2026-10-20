"""Optional batch workflow for soccer kick clips."""
from pathlib import Path
import sys

VIDEOS = sorted(p.name for p in Path('input').glob('*') if p.suffix.lower() in ('.mp4', '.mov'))
PYTHON = sys.executable

rule all:
    input:
        expand('output/{video}/.analysis/report.json', video=VIDEOS),
        expand('output/{video}/overlay.mp4', video=VIDEOS)

rule analyze_kick:
    input:
        video='input/{video}',
        code=['src/extractor.py', 'src/math_utils.py', 'src/physics_utils.py',
              'src/renderer.py', 'src/pipeline.py', 'src/settings.py',
              'src/cli_report.py', 'scripts/analyze_kick.py']
    output:
        report='output/{video}/.analysis/report.json',
        overlay='output/{video}/overlay.mp4'
    params:
        directory='output/{video}',
        foot=lambda wildcards: config.get('foot', 'right')
    shell:
        '{PYTHON:q} -m scripts.analyze_kick {input.video:q} --output {params.directory:q} --foot {params.foot:q}'
