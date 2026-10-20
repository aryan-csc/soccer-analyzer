"""Terminal reports and image exports; no web server or graphical display."""
from pathlib import Path
import textwrap
from uuid import uuid4


def format_report(report: dict, source: Path) -> str:
    meta, settings, contact = (report[key] for key in ('metadata', 'settings', 'contact'))
    lines = [f'SOCCER KICK ANALYSIS — {source.name}',
             f"{settings['foot'].title()} foot | {settings['shot_type'].replace('_', ' ')} | "
             f"{meta['frame_count']} frames at {meta['fps']:.3f} fps",
             f"Usable kicking-leg angle coverage: {report['quality']['pose_coverage']:.1%}", '']
    if contact['frame'] is None:
        lines.append('Contact: not identified — supply --contact-frame after reviewing video.mp4.')
    else:
        review = '; review needed' if contact['needs_review'] else ''
        lines.append(f"Contact: frame {contact['frame']} ({contact['source']}{review})")
    lines += [textwrap.fill(contact['reason'], 100), '', 'MEASUREMENTS']
    for metric in report['summary']:
        value = 'unavailable' if metric['value'] is None else f"{metric['value']:.2f} {metric['unit']}"
        lines.append(f"  {metric['label']}: {value}")
        lines.append(textwrap.fill(metric['detail'], 100, initial_indent='    ', subsequent_indent='    '))
    lines += ['', 'OBSERVED PHASES (boundaries are estimates except manually selected contact)']
    for phase in report['phases']:
        lines.append(f"  {phase['label']}: frames {phase['start_frame']}–{phase['end_frame']}")
    if not report['phases']:
        lines.append('  No reliable phases identified.')
    lines += ['', 'OBSERVATIONS']
    for observation in report['observations']:
        lines.append(textwrap.fill(f"{observation['title']}: {observation['body']}", 100,
                                   initial_indent='  ', subsequent_indent='  '))
    lines += ['', 'MEASUREMENT LIMITS']
    for warning in report['quality']['warnings']:
        lines.append(textwrap.fill(warning, 100, initial_indent='  - ', subsequent_indent='    '))
    return '\n'.join(lines) + '\n'


def save_plots(report: dict, path: Path) -> None:
    """Plot actual time series with missing samples left as gaps."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    definitions, frames = report['metric_definitions'], report['frames']
    figure = Figure(figsize=(11, 2.1 * (len(definitions) + 1)), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(definitions) + 1, 1, sharex=True, squeeze=False).ravel()
    times = [frame['time'] for frame in frames]
    for axis, definition in zip(axes, definitions):
        values = [frame['metrics'].get(definition['key']) for frame in frames]
        axis.plot(times, [np.nan if value is None else value for value in values], color='#245e48', linewidth=1.3)
        axis.set_ylabel(definition['unit'])
        axis.set_title(definition['label'], loc='left', fontsize=10)
        axis.grid(alpha=0.2)
        if not any(value is not None for value in values):
            axis.text(0.5, 0.5, 'No reliable measurements', transform=axis.transAxes, ha='center')
    contact = report['contact']['frame']
    if contact is not None:
        time = next(frame['time'] for frame in frames if frame['frame'] == contact)
        for axis in axes:
            axis.axvline(time, color='#b78329', linestyle='--', linewidth=1)
    timeline = axes[-1]
    colors = {'approach': '#c7d7b5', 'plant': '#b2c997', 'backswing': '#96b47e',
              'forward_swing': '#5f8b61', 'contact': '#dab977', 'follow_through': '#91b4a0'}
    fps = report['metadata']['fps']
    for phase in report['phases']:
        timeline.broken_barh([(phase['start_frame'] / fps, (phase['end_frame'] - phase['start_frame'] + 1) / fps)],
                            (0, 1), facecolors=colors.get(phase['id'], '#aaaaaa'), label=phase['label'])
    timeline.set(title='Observed kick phases', xlabel='Video playback time (seconds)', yticks=[])
    timeline.set_xlim(0, report['metadata']['duration'])
    handles, labels = timeline.get_legend_handles_labels()
    if labels:
        unique = dict(zip(labels, handles))
        timeline.legend(unique.values(), unique.keys(), fontsize=8, loc='upper center',
                        bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    figure.suptitle('Soccer kick — 2D image-plane measurements', fontsize=14)
    temporary = path.with_name(f'.{path.stem}-{uuid4().hex}.png')
    try:
        figure.savefig(temporary, dpi=140, bbox_inches='tight')
        temporary.replace(path)
    finally:
        figure.clear()
        temporary.unlink(missing_ok=True)
