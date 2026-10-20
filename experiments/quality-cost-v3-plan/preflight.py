"""Create the V3 design matrix and audit retained inputs; no GPU simulation."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    configs = []
    for scale in [1, 2, 4, 8]:
        configs.append(dict(id=f'graph_n{scale}', termination='original',
                            newton_scale=scale, pcg_scale=1, epsilon_r=.001))
    configs += [dict(id='toi', termination='toi', newton_scale=1, pcg_scale=1, epsilon_r=.03),
                dict(id='comp001', termination='compensated', newton_scale=1, pcg_scale=1, epsilon_r=.001)]
    for mode, prefix in [('compensated', 'comp'), ('gated', 'gate')]:
        for label, eps in [('01', .01), ('03', .03), ('10', .1)]:
            configs.append(dict(id=prefix+label, termination=mode, newton_scale=1, pcg_scale=1, epsilon_r=eps))
    references = [dict(id=f'strict{label}', termination='original', newton_scale=scale, pcg_scale=scale)
                  for label, scale in [(2, .5), (4, .25)]]
    sensitivity = [dict(id=f'n{n}_p{p}', newton_scale=ns, pcg_scale=ps)
                   for n, ns in [(1, 1), (4, .25)] for p, ps in [(1, 1), (4, .25)]]
    matrix = {
        'schema_version': 1,
        'status': 'design_only_not_executed',
        'source_answer_message_id': '5c7a52b6-d6aa-4063-a19a-f04cdca7fd49',
        'base_results_commit': 'bbb9c2ce79408ca5251f782b54d337ab87485208',
        'base_solver_commit': 'cc1afb9f5a677ed046beab9121f004f566fc74bb',
        'baseline_newton_coefficient': .001, 'baseline_pcg_rtz_coefficient': .001,
        'epsilon_toi': .001, 'minimum_updates': 6, 'gpu_index': 0,
        'P0': {'requires_full_checkpoint_implementation': True, 'same_checkpoint_repeats': 5,
               'smoke_scenes': ['table', 'bunny'], 'smoke_frames': 5},
        'A1': {'scene': 'table', 'target_frame': 23, 'prefix_frames': 22,
               'prefix_newton_scale': .25, 'prefix_pcg_scale': .25,
               'configs': sensitivity, 'repeats': 3, 'planned_single_steps': 12,
               'seconds_per_step': 300, 'newton_samples_per_step': 2000,
               'stage_seconds': 1200, 'max_independent_prefixes': 3},
        'A2_A3': {'scenes': ['table', 'bunny'], 'selection_prefix_frames': 50,
                  'state_categories': ['contact_onset', 'dense_contact', 'contact_rearrangement', 'quiet_motion'],
                  'maximum_states': 8, 'selected_states': None,
                  'selection_must_be_frozen_before_candidates': True,
                  'configs': configs, 'references': references, 'repeats': 3,
                  'maximum_single_steps': 336, 'seconds_per_step': 120,
                  'newton_samples_per_step': 2000, 'stage_seconds': 3600},
        'quality_bands': {'position_rms_m': [1e-6, 1e-5, 1e-4],
                          'velocity_rms_rule': 'position_rms_m / actual_dt',
                          'max_error_multiplier': 10, 'reference_resolution_fraction': .1,
                          'production_tolerances_approved': False},
        'A4': {'shadow_epsilon_r': .03, 'maximum_additional_states': 2,
               'configs': ['graph_n1', 'toi', 'comp03', 'gate03'],
               'reference_ids': ['strict2', 'strict4'], 'repeats': 3, 'maximum_single_steps': 36,
               'artificial_ccd_alpha_changes': False},
        'B1': {'scene': 'stack10', 'frames': 35, 'profile_repeats': 1,
               'instrumentation_off_repeats': 3, 'frozen_baseline_repeats': 3, 'total_frames': 245},
        'B2': {'conditional_on_profile': True, 'maximum_optimizations': 1,
               'scene_frames': {'stack10': 35, 'table': 50, 'bunny': 50},
               'versions': ['pre_optimization', 'post_optimization'], 'repeats': 3,
               'total_frames': 810, 'target_median_time_reduction': .05,
               'other_scene_maximum_median_regression': .03},
    }
    assert len(configs) == 12 and len({c['id'] for c in configs}) == 12
    assert len(sensitivity)*3 == matrix['A1']['planned_single_steps']
    assert 8*(len(configs)+len(references))*3 == matrix['A2_A3']['maximum_single_steps']
    assert 2*(4+2)*3 == matrix['A4']['maximum_single_steps']
    assert 35*(1+3+3) == matrix['B1']['total_frames']
    assert sum(matrix['B2']['scene_frames'].values())*2*3 == matrix['B2']['total_frames']
    assert all(c['pcg_scale'] == 1 for c in configs)
    write(HERE/'matrix.json', matrix)

    assert git('rev-parse', 'dual-tolerance-toi-v2-key-scenes') == matrix['base_results_commit']
    assert git('rev-parse', 'dual-tolerance-toi-v2') == matrix['base_solver_commit']
    assert not git('diff', matrix['base_results_commit'], '--', 'StiffGIPC', 'CMakeLists.txt')
    version = read(ROOT/'versions/dual-tolerance-toi-v2/MANIFEST.json')
    for filename, expected in version['binary_files'].items():
        assert sha(ROOT/'versions/dual-tolerance-toi-v2'/filename) == expected, filename
    old_out = ROOT/'experiments/dual-tolerance-v2'
    evidence = read(old_out/'EVIDENCE_MANIFEST.json')
    for relative, expected in evidence.items():
        assert sha(old_out/relative) == expected, relative
    old_audit = read(old_out/'AUDIT.json')
    assert old_audit['completed_runs'] == 67 and old_audit['completed_frames'] == 2257
    assert old_audit['interrupted_reference_runs'] == 1

    required = {
        'StiffGIPC/GIPC.cu': ['GIPC::computeGradientAndHessian', 'GIPC::lineSearch',
                            'numOfLineSearch <= report_line_search_threshold', 'GIPC::postLineSearch'],
        'StiffGIPC/linear_system/linear_system/global_linear_system.cu':
            ['GlobalLinearSystem::build_linear_system', 'convert_new();', 'distribute_solution();'],
        'StiffGIPC/linear_system/utils/converter.cu':
            ['Converter::_radix_sort_indices_and_blocks', 'Converter::_make_unique_block_warp_reduction'],
        'StiffGIPC/abd_system/abd_sim_data.h': ['body_id_to_q_prev', 'body_id_to_q_v', 'body_id_to_q_tilde'],
    }
    source_checks = {}
    for relative, symbols in required.items():
        source = (ROOT/relative).read_text(encoding='utf-8')
        assert all(symbol in source for symbol in symbols), relative
        source_checks[relative] = {'sha256': sha(ROOT/relative), 'verified_symbols': symbols}

    files = ['EXPERIMENT_PLAN_V3.md', 'docs/TOI_RESEARCH_V3_SOURCE.md',
             'experiments/quality-cost-v3-plan/matrix.json', 'experiments/quality-cost-v3-plan/preflight.py']
    report = {
        'status': 'PASS_design_and_retention_preflight_only',
        'new_gpu_experiments_executed': False,
        'new_checkpoint_or_diagnostics_implemented': False,
        'baseline_results_commit': matrix['base_results_commit'],
        'baseline_solver_commit': matrix['base_solver_commit'],
        'retained_binary_files_verified': len(version['binary_files']),
        'retained_experiment_file_hashes_verified': len(evidence),
        'old_results_unmodified': True,
        'simulation_source_unmodified': True,
        'matrix_arithmetic_and_controls_verified': True,
        'v2_completed_runs': 67, 'v2_completed_frames': 2257,
        'v2_interrupted_reference_runs': 1,
        'new_plan_file_hashes': {p: sha(ROOT/p) for p in files},
        'source_checks': source_checks,
        'preexisting_timeCost_txt': {'present': (ROOT/'timeCost.txt').exists(),
                                     'sha256': sha(ROOT/'timeCost.txt') if (ROOT/'timeCost.txt').exists() else None},
    }
    write(HERE/'PREFLIGHT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_checks', 'new_plan_file_hashes', 'preexisting_timeCost_txt']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
