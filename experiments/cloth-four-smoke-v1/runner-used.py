"""Frozen cloth-v1 inputs, four solver controls, sequential GPU 0 runs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
import numpy as np
from run_three_versions import load, errors, geometry, verify_trace

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build/gipc.exe'
CASES = ['sphere7', 'bunny', 'table', 'stack10', 'hang']
MODES = ['base', 'graph', 'toi', 'compensated']
SPEC = json.loads((ROOT / 'experiments/CLOTH_V1_SCENES.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def validate(path, case, mode, frames):
    data = load(path)
    status = json.loads((path / 'status.json').read_text())
    assert status['completed'] and status['frames_completed'] == frames
    assert len(data['rows']) == frames
    cfg = json.loads((path / 'cloth_scene.json').read_text())
    for key, value in SPEC[case]['expected_scene'].items():
        assert cfg[key] == value, (case, key, cfg[key], value)
    assert sha(path / 'scene_initial.f64') == SPEC[case]['initial_sha256']
    assert sha(path / 'scene_triangles.u32') == SPEC[case]['triangles_sha256']
    assert data['metadata']['dt'] == SPEC[case]['dt']
    assert data['metadata']['pcg_tolerance'] == .001
    assert data['metadata']['newton_tolerance'] == .001
    assert data['metadata']['cloth_scene'] == case
    for row in data['rows']:
        assert int(row['nonfinite_vertices']) == int(row['newton_cap_hits']) == 0
        assert int(row['bounds_valid']) == 1
        assert math.isfinite(float(row['exit_residual']))
        if row['exit_reason'] in ['toi', 'compensated']:
            assert int(row['budget_valid']) == 1
            assert float(row['exit_beta']) <= .001 + 1e-12
            if row['exit_reason'] == 'compensated':
                assert float(row['exit_budget']) <= .001 + 1e-12
    if mode == 'base':
        assert data['direction_launches'] == data['tail_launches'] == 0
    else:
        assert data['direction_launches'] > 0 and data['tail_launches'] > 0
        assert data['fallbacks'] == 0
    n = data['metadata']['vertex_count']
    assert (path / 'states.f64').stat().st_size == frames*n*24
    assert (path / 'velocities.f64').stat().st_size == frames*n*24
    states = np.memmap(path / 'states.f64', dtype='<f8', mode='r', shape=(frames,n,3))
    assert np.isfinite(states).all()
    initial = np.fromfile(path / 'initial.f64', dtype='<f8').reshape(n,3)
    fixed = np.array(cfg['pinned'], dtype=int)
    drift = float(np.linalg.norm(states[:,fixed] - initial[fixed], axis=2).max()) if fixed.size else 0.
    # This loader and the legacy state arrays retain the same vertex numbering.
    assert drift < 1e-10, (case, mode, 'fixed drift', drift)
    data['fixed_max_drift_m'] = drift
    data['intersection_frames'] = sum(int(r['intersection']) for r in data['rows'])
    data['geometry'] = geometry(data)
    if (path / 'termination_trace.jsonl').exists():
        verify_trace(path)
    return data


def run(path, case, mode, frames, smoke):
    assert not path.exists(), f'Refusing overwrite: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(EXE), '--scene', '1', '--benchmark-frames', str(frames),
               '--benchmark-output-dir', str(path), '--export-states',
               '--termination', 'original' if mode in ['base','graph'] else mode,
               '--toi-min-updates', '6', '--toi-threshold', '.001']
    if mode != 'base':
        command += ['--pcg-dual-graph']
    if smoke:
        command += ['--trace-termination']
        if mode == 'graph':
            command += ['--audit-frame','1']
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GIPC_','CIPC_'))}
    env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CIPC_SCENE=case, CIPC_META_OUT=str(path / 'cloth_scene.json'),
               CIPC_INITIAL_OUT=str(path / 'scene_initial.f64'),
               CIPC_TRIANGLES_OUT=str(path / 'scene_triangles.u32'))
    env['PATH'] = str(EXE.parent) + os.pathsep + env['PATH']
    print(f'START {case}/{mode}: {frames} frames', flush=True)
    start = time.monotonic()
    with path.with_suffix('.stdout.log').open('wb') as stdout, path.with_suffix('.stderr.log').open('wb') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=5400)
    write(path / 'environment.json', {'command':command,'returncode':result.returncode,
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'exe_sha256':sha(EXE),'parameters_sha256':sha(ROOT / 'Assets/scene/parameterSetting.txt'),
          'environment':{k:v for k,v in env.items() if k.startswith(('CUDA_','CIPC_'))},
          'wall_seconds':time.monotonic()-start})
    assert result.returncode == 0, f'Run failed: {path}'
    data = validate(path,case,mode,frames)
    print(f"DONE {case}/{mode}: Newton={data['newton']}; step={data['step_seconds']:.3f}s; "
          f"Newton={data['newton_seconds']:.3f}s; auxiliary={data['auxiliary_exits']}",flush=True)
    return data


def summarize(out, repeats, frames):
    runs, comparisons = [], {}
    for case in CASES:
        groups = {}
        for mode in MODES:
            groups[mode] = []
            for repeat in range(1,repeats+1):
                path = out / case / mode / f'r{repeat:02d}'
                data = validate(path,case,mode,frames)
                base = load(out / case / 'base' / f'r{repeat:02d}')
                data['vs_base'] = errors(data,base)
                data['case'], data['mode'], data['repeat'] = case,mode,repeat
                runs.append({k:v for k,v in data.items() if k!='rows'})
                groups[mode].append(data)
        assert len({sha(Path(d['path'])/'initial.f64') for group in groups.values() for d in group}) == 1
        comparisons[case] = {mode:{
            'step_seconds_median':statistics.median(d['step_seconds'] for d in groups[mode]),
            'newton_seconds_median':statistics.median(d['newton_seconds'] for d in groups[mode]),
            'newton_median':statistics.median(d['newton'] for d in groups[mode]),
            'pcg_iterations_median':statistics.median(d['pcg_iterations'] for d in groups[mode]),
            'auxiliary_exits_median':statistics.median(d['auxiliary_exits'] for d in groups[mode]),
            'intersection_frames':sum(d['intersection_frames'] for d in groups[mode]),
            'max_position_rms_vs_base_m':max(d['vs_base']['max_position_rms_m'] for d in groups[mode]),
            'speedup_vs_base_median':statistics.median(a['step_seconds']/b['step_seconds'] for a,b in zip(groups['base'],groups[mode])),
            'speedup_vs_graph_median':statistics.median(a['step_seconds']/b['step_seconds'] for a,b in zip(groups['graph'],groups[mode]))
        } for mode in MODES}
    write(out / 'summary.json', {'frames':frames,'repeats':repeats,'runs':runs,'comparisons':comparisons})
    report = ['# 五布料场景四版本比较','',f'每项 {frames} 帧，{repeats} 次。GPU 0 串行运行，warmup=0；'+
              'epsilon=1e-3，Kmin=6。时间为包含 Graph 准备和真实非线性残差测量的整步同步墙钟；排除导出、渲染和帧末相交诊断。',
              '', '四项共用同一二进制：base 使用 c49982f 原 PCG 和原 Newton 停止规则；原 Graph 启用留存 PCG 双段图；'+
              '纯累计 TOI 和残差补偿 TOI 启用同一双段图并切换附加退出。此处原 Graph 不是旧表的 AGIPC f36c784/e8d638b。',
              '', '本次只新增冻结 cloth-v1-20260911 场景入口，求解数值源码保持 residual-toi-graph-v1 的实现。'+
              '场景初态、三角拓扑 SHA-256 和全部 expected_scene 参数逐项对照原 SCENES.json。',
              '', '| 场景 | base 秒 / Newton | 原 Graph 秒 / Newton | 纯 TOI 秒 / Newton | 补偿 TOI 秒 / Newton | 补偿相对 Graph |',
              '|---|---:|---:|---:|---:|---:|']
    for case in CASES:
        c = comparisons[case]
        values = [f"{c[m]['step_seconds_median']:.3f} / {c[m]['newton_median']:g}" for m in MODES]
        ratio = c['compensated']['speedup_vs_graph_median']
        report.append('| '+case+' | '+' | '.join(values)+f' | {ratio:.3f}× |')
    report += ['', '| 场景 / 方法 | PCG 总迭代 | 附加退出帧 | 相交标志帧 | 相对 base 最大帧位置 RMS (mm) |',
               '|---|---:|---:|---:|---:|']
    for case in CASES:
        for mode in MODES:
            d = comparisons[case][mode]
            report.append(f"| {case} / {mode} | {d['pcg_iterations_median']:g} | {d['auxiliary_exits_median']:g} | "
                          f"{d['intersection_frames']} | {1000*d['max_position_rms_vs_base_m']:.3f} |")
    report += ['', '所有固定顶点漂移、有限状态、Newton 上限、预算界、图运行/回退记录见 summary.json。'+
               '位置差是相对 base 轨迹的差异，base 不是严格收敛的真值；不能将差异直接解释为绝对误差。',
               '', '单次结果只反映本次测量，不支持稳定性能或等精度结论。前四场景此前为 200 帧，'+
               '本次 100 帧时长及碰撞阶段不同，不能直接与旧表总耗时比较或将旧表除以二。',
               '', '原 GPU 相交查询为有限诊断，未全面验证布料自交、穿透或退化；不能仅凭标志为零宣称全部几何安全。',
               '', 'source_commit 和二进制 SHA-256 见每项 environment.json；全部原始文件 SHA-256 见 EVIDENCE_MANIFEST.json。']
    (out / 'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(out / 'EVIDENCE_MANIFEST.json', {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
          if p.is_file() and p.name != 'EVIDENCE_MANIFEST.json'})
    print('SUMMARY',out / 'REPORT.md',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    frames = 2 if args.smoke else 100
    if args.summarize_only:
        summarize(out,args.repeats,frames)
        return
    if not args.resume:
        assert not out.exists(), f'Refusing overwrite: {out}'
        out.mkdir(parents=True)
        write(out / 'protocol.json', {'baseline':'cloth-v1-20260911','frames':frames,'repeats':args.repeats,
             'cases':CASES,'modes':MODES,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv'],text=True),
             'exe_sha256':sha(EXE),'spec_sha256':sha(ROOT / 'experiments/CLOTH_V1_SCENES.json'),
             'epsilon':.001,'minimum_accepted_updates':6,'warmup':0,'trace':args.smoke})
        shutil.copy2(__file__,out / 'runner-used.py')
    else:
        protocol = json.loads((out / 'protocol.json').read_text())
        assert protocol['exe_sha256'] == sha(EXE)
        assert protocol['frames'] == frames and protocol['repeats'] == args.repeats
    for repeat in range(1,args.repeats+1):
        for index,case in enumerate(CASES):
            offset = (index+repeat-1)%4
            order = MODES[offset:] + MODES[:offset]
            for mode in order:
                path = out / case / mode / f'r{repeat:02d}'
                if args.resume and path.exists():
                    validate(path,case,mode,frames)
                    continue
                run(path,case,mode,frames,args.smoke)
    summarize(out,args.repeats,frames)


if __name__ == '__main__':
    main()
