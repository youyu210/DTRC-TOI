"""Complete unaffected experiments while preserving stalled strict references."""
import json
from pathlib import Path
import subprocess
import time
import run_dual_tolerance_v2 as experiment

OUT = experiment.ROOT / 'experiments/dual-tolerance-v2'
real_run = subprocess.run


def supervised_run(command, **kwargs):
    if not str(command[0]).lower().endswith('gipc.exe'):
        return real_run(command, **kwargs)
    path = Path(command[command.index('--benchmark-output-dir') + 1])
    if 'references' not in path.parts or command[command.index('--benchmark-frames') + 1] == '1':
        return real_run(command, **kwargs)
    timeout = kwargs.pop('timeout')
    started = last_progress = time.monotonic()
    last_size = 0
    with subprocess.Popen(command, **kwargs) as process:
        while process.poll() is None:
            frames = path / 'frames.csv'
            size = frames.stat().st_size if frames.exists() else 0
            if size != last_size:
                last_progress = time.monotonic()
                last_size = size
            now = time.monotonic()
            if now - last_progress > 300 or now - started > timeout:
                import csv
                rows = list(csv.DictReader(frames.open())) if frames.exists() else []
                note = {'reason': 'Strict reference interrupted after 300 seconds without a completed frame or the original 1800-second total budget; reference convergence unverified.',
                        'completed_frames': len(rows), 'stalled_frame': len(rows) + 1,
                        'stalled_seconds': now - last_progress, 'elapsed_seconds': now - started}
                experiment.write(path / 'INTERRUPTION.json', note)
                process.terminate()
                break
            time.sleep(1)
        return subprocess.CompletedProcess(command, process.wait())


def main():
    subprocess.run = supervised_run
    for case in experiment.CASES:
        for name, scale in [('strict2', .5), ('strict4', .25)]:
            path = OUT / 'references' / case / name
            if not (path / 'INTERRUPTION.json').exists():
                try:
                    experiment.execute(path, case, 'graph', scale=scale)
                except AssertionError:
                    if not (path / 'INTERRUPTION.json').exists():
                        raise
            experiment.execute(OUT / 'probe-references' / case / name, case, 'graph', nf=1, scale=scale, offset=-.248, trace=True)
        for config in experiment.CONFIGS:
            experiment.execute(OUT / 'probe' / case / config, case, config, nf=1, offset=-.248, trace=True)
    experiment.execute(OUT / 'profile' / 'stack10', 'stack10', 'graph', nf=35, profile=True)
    experiment.execute(OUT / 'profile' / 'table', 'table', 'comp03', nf=50, profile=True)
    for config in ['graph', 'toi']:
        experiment.execute(OUT / 'production' / 'table' / config, 'table', config, production=True)
    experiment.summarize(OUT)


if __name__ == '__main__':
    main()
