"""Post-run completeness and diagnostic audit; does not launch GPU work."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/dual-tolerance-v2'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    summary = read(OUT / 'summary.json')
    runs = summary['runs']
    failures = summary['interrupted_references']
    assert len(runs) + len(failures) == 68
    assert sum(r['metadata']['frames'] for r in runs) + 50 * len(failures) == 2307
    partial_exports = {}
    for name, failure in failures.items():
        p = OUT / name
        meta = read(p / 'metadata.json')
        rows = list(csv.DictReader((p / 'frames.csv').open()))
        assert len(rows) == failure['completed_frames']
        frame_bytes = meta['vertex_count'] * 24
        partial_exports[name] = {
            filename: {'bytes': (p / filename).stat().st_size,
                       'complete_state_frames': (p / filename).stat().st_size // frame_bytes,
                       'partial_frame_bytes': (p / filename).stat().st_size % frame_bytes}
            for filename in ['states.f64', 'velocities.f64']}
    manifest = read(ROOT / 'versions/dual-tolerance-toi-v2/MANIFEST.json')
    assert sha(ROOT / 'build/gipc.exe') == manifest['binary_files']['gipc.exe']
    assert sha(ROOT / 'versions/cloth-four-100-v1/gipc.exe') == manifest['old_cloth_exe_sha256']
    assert sha(ROOT / 'EXPERIMENT_PLAN_V2.md') == manifest['plan_sha256']
    for name, digest in manifest['binary_files'].items():
        assert sha(ROOT / 'versions/dual-tolerance-toi-v2' / name) == digest
    totals = dict.fromkeys(['newton_cap_hits', 'fallbacks', 'nonfinite_vertices',
                           'intersection', 'history_rejections', 'guard_rejections'], 0)
    gate_history_veto_exits = []
    tail_without_residual_veto = []
    trace_samples = 0
    rhs_checks = 0
    for run in runs:
        p = OUT / run['relative_path']
        execution = read(p / 'execution.json')
        assert execution['returncode'] == 0
        assert execution['exe_sha256'] == manifest['binary_files']['gipc.exe']
        assert execution['source_commit'].startswith(manifest['source_commit'])
        rows = list(csv.DictReader((p / 'frames.csv').open()))
        for row in rows:
            for key in totals:
                totals[key] += int(row[key])
            if row['exit_reason'] == 'gated' and float(row['exit_budget']) > .001 + 1e-12:
                gate_history_veto_exits.append([run['relative_path'], int(row['frame'])])
            if run['metadata']['termination'] in ['gated', 'compensated']:
                if int(row['tail_iterations']) != int(row['residual_rejections']):
                    tail_without_residual_veto.append([run['relative_path'], int(row['frame'])])
        trace = p / 'termination_trace.jsonl'
        if trace.exists():
            for line in trace.read_text().splitlines():
                samples = json.loads(line)['samples']
                trace_samples += sum(s['active'] for s in samples)
                last = samples[-1]
                if rows[int(json.loads(line)['frame']) - 1]['exit_reason'] in ['gated', 'compensated']:
                    assert last['normalized'] <= run['metadata']['residual_threshold'] + 1e-12
            checks = [a for a in read(p / 'audits.json') if a.get('kind') == 'nonlinear_native_rhs']
            assert all(a['agrees'] for a in checks)
            rhs_checks += len(checks)
    audit = {
        'completed_runs': len(runs),
        'completed_frames': sum(r['metadata']['frames'] for r in runs),
        'interrupted_reference_runs': len(failures),
        'partial_frames_retained': sum(f['completed_frames'] for f in failures.values()),
        'partial_reference_exports': partial_exports,
        'source_commit': manifest['source_commit'],
        'exe_sha256': manifest['binary_files']['gipc.exe'],
        'original_binary_hash_preserved': True,
        'plan_hash_preserved': True,
        'totals': totals,
        'gate_exits_that_history_would_veto': gate_history_veto_exits,
        'tail_frames_not_explained_by_current_residual': tail_without_residual_veto,
        'formal_trace_active_samples': trace_samples,
        'formal_native_rhs_checks': rhs_checks,
        'max_fixed_drift_m': max(r['fixed_drift_m'] for r in runs),
    }
    (OUT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
