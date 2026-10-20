"""Standalone command-line soccer kick analyzer."""
import argparse
from pathlib import Path
import sys

from src.settings import Settings
from src.cli_report import format_report, save_plots
from src.pipeline import run_analysis


def main():
    parser = argparse.ArgumentParser(description='Analyze a single soccer kick in a side-view video.')
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, help='Output folder (default: output/<video name>)')
    parser.add_argument('--foot', choices=['left', 'right'], default='right')
    parser.add_argument('--shot-type', choices=['penalty', 'free_kick', 'shot'], default='shot')
    parser.add_argument('--camera-view', choices=['side', 'oblique', 'front'], default='side')
    parser.add_argument('--camera-motion', choices=['fixed', 'moving'], default='fixed')
    parser.add_argument('--direction', choices=['auto', 'left', 'right'], default='auto')
    parser.add_argument('--contact-frame', type=int, help='Zero-based frame in the standardized video')
    parser.add_argument('--scale-m-per-px', type=float, help='Known meters per analyzed video pixel')
    parser.add_argument('--reuse-pose', action='store_true', help='Reuse saved tracking for the same input clip')
    parser.add_argument('--no-plots', action='store_true', help='Skip the PNG graph export')
    args = parser.parse_args()
    args.output = args.output or Path('output') / args.video.stem
    last_bucket = -1
    def progress(value, message):
        nonlocal last_bucket
        line = f'{value:5.0%}  {message}'
        if sys.stderr.isatty():
            print('\r' + line.ljust(110), file=sys.stderr, end='\n' if value >= 1 else '', flush=True)
        elif int(value * 10) != last_bucket or value in (0.03, 0.12, 0.69, 0.75, 1):
            print(line, file=sys.stderr, flush=True)
        last_bucket = int(value * 10)
    try:
        settings = Settings(**{key: getattr(args, key) for key in Settings.model_fields})
        if args.video.resolve() in {(args.output / name).resolve() for name in ('overlay.mp4', 'report.txt', 'metrics.png')}:
            raise ValueError('Choose an output directory that does not overwrite the source video.')
        work = args.output / '.analysis'
        report = run_analysis(args.video, work, settings.model_dump(),
                              progress=progress, reuse_pose=args.reuse_pose)
        (work / 'overlay.mp4').replace(args.output / 'overlay.mp4')
        text = format_report(report, args.video)
        (args.output / 'report.txt').write_text(text, encoding='utf-8')
        if not args.no_plots:
            save_plots(report, args.output / 'metrics.png')
        else:
            (args.output / 'metrics.png').unlink(missing_ok=True)
    except KeyboardInterrupt:
        print('\nAnalysis cancelled.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Analysis failed: {error}', file=sys.stderr)
        return 1
    print(text)
    print(f"Outputs: {args.output.resolve()}")
    print('  overlay.mp4   annotated video')
    print('  report.txt    readable analysis and observations')
    if not args.no_plots:
        print('  metrics.png   measurement graphs and phase timeline')
    print('  .analysis/    tracking cache, standardized video, CSV and JSON (advanced)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
