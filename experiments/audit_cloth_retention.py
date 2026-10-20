"""Retain independent source, native-RHS and short graph-direction evidence."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def function(path, signature):
    source = path.read_text(encoding='utf-8')
    start = source.index(signature)
    brace = source.index('{',start)
    depth = 1
    end = brace+1
    while depth:
        depth += (source[end]=='{') - (source[end]=='}')
        end += 1
    return source[start:end]


def main():
    source = 'StiffGIPC/linear_system/solver/pcg_solver.cu'
    assert function(ROOT/source,'SizeT PCGSolver::pcg(') == function(
        ROOT.parent/'upstream-c49982fa'/source,'SizeT PCGSolver::pcg(')
    paths = ['StiffGIPC/GIPC.cu','StiffGIPC/GIPC.cuh','StiffGIPC/gipc/gipc.cu',
             'StiffGIPC/linear_system','StiffGIPC/gipc/utils/toi_budget.h',
             'StiffGIPC/gipc/nonlinear_residual.cu']
    changes = subprocess.check_output(['git','diff','residual-toi-graph-v1','--',*paths],cwd=ROOT,text=True)
    assert not changes, 'Numerical source differs from retained TOI implementation'
    audits = []
    for path in (ROOT/'experiments/cloth-four-smoke-v2').glob('*/*/r01/audits.json'):
        for item in json.loads(path.read_text()):
            audits.append({'run':path.parent.relative_to(ROOT).as_posix(),**item})
    rhs = [a for a in audits if a.get('kind') == 'nonlinear_native_rhs']
    assert rhs and all(a['agrees'] for a in rhs)
    directions = [a for a in audits if 'relative_l2_direction_diff' in a]
    assert len({a['run'].split('/')[-3] for a in directions}) == 5
    assert all(a['actual_iterations'] == a['eager_iterations'] for a in directions)
    # CUDA reductions are not bitwise deterministic. This is a diagnostic
    # comparison of the same RHS, not a changed PCG stopping tolerance.
    direction_check_tolerance = 1e-6
    assert max(a['relative_l2_direction_diff'] for a in directions) < direction_check_tolerance
    assert max(abs(a['actual_true_relative_residual']-a['eager_true_relative_residual'])
               for a in directions) < direction_check_tolerance
    data = {'original_pcg_function_exact_match':True,'numerical_source_unchanged_from':'residual-toi-graph-v1',
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'new_exe_sha256':sha(ROOT/'build/gipc.exe'),
            'previous_exe_sha256':sha(ROOT/'versions/residual-toi-graph-v1/gipc.exe'),
            'native_rhs_audit_samples':len(rhs),'all_native_rhs_audits_agree':True,
            'graph_direction_audit_samples':len(directions),
            'direction_check_tolerance':direction_check_tolerance,
            'max_graph_direction_relative_l2_diff':max(a['relative_l2_direction_diff'] for a in directions),
            'all_audits':audits}
    assert data['previous_exe_sha256'] == '144aa9c868be179d0b02274e8435cff5a332846fe5870ecb9c1db6283753fbce'
    (ROOT/'experiments/cloth-four-100-v1/SOURCE_AND_SMOKE_AUDIT.json').write_text(
        json.dumps(data,indent=2),encoding='utf-8')
    snapshot = ROOT/'versions/cloth-four-100-v1'
    manifest = {'implementation_commit':data['source_commit'],'implementation_tag':'cloth-four-100-v1',
                'retained_parent_tag':'residual-toi-graph-v1-validated',
                'binary_files':{p.name:sha(p) for p in snapshot.iterdir() if p.suffix in ['.exe','.dll']},
                'asset_files':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'Assets').rglob('*') if p.is_file()},
                'scene_spec_sha256':sha(ROOT/'experiments/CLOTH_V1_SCENES.json')}
    (snapshot/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='all_audits'},indent=2))


if __name__ == '__main__':
    main()
