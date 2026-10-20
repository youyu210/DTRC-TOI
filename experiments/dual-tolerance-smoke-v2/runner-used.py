"""Preregistered key-scene dual-tolerance experiments; outputs never overwritten."""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, shutil, statistics, subprocess, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'build/gipc.exe'
CASES=['table','bunny']
CONFIGS={'graph':('original',.001),'toi':('toi',.001),'comp001':('compensated',.001),
         'comp01':('compensated',.01),'comp03':('compensated',.03),'comp10':('compensated',.1),
         'gate01':('gated',.01),'gate03':('gated',.03),'gate10':('gated',.1)}
CORE=['graph','toi','comp001','comp03','gate03']
SPEC=json.loads((ROOT/'experiments/CLOTH_V1_SCENES.json').read_text(encoding='utf-8'))

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def trace_check(path):
    meta=read(path/'metadata.json'); samples=0
    for line in (path/'termination_trace.jsonl').read_text().splitlines():
        last=None
        for s in json.loads(line)['samples']:
            if not s['active']: continue
            samples+=1
            assert s['valid']
            z,r,b,w=s['budget'],s['normalized'],s['beta'],s['weight']
            tol=1e-12*max(1,z,r,b)
            assert z+tol>=b and z+tol>=w*r and z<=s['unit_budget']+tol
            assert abs(z-(b+w*(s['unit_budget']-b)))<=tol
            if last:
                assert s['reference']==last['reference'] and w==last['weight']
                q=1-s['alpha'];excess=max(0,r-q*last['normalized'])
                assert abs(s['excess']-excess)<=tol
                assert abs(z-(q*last['budget']+w*excess))<=tol
                assert abs(b-q*last['beta'])<=tol
            else:
                assert s['accepted_updates']==meta['toi_min_updates']-1
                assert s['reference']==max(s['residual'],meta['residual_floor'])
                assert z==b==s['unit_budget']==1
            last=s
    audits=read(path/'audits.json')
    assert all(a['agrees'] for a in audits if a.get('kind')=='nonlinear_native_rhs')
    return samples

def validate(path):
    m=read(path/'metadata.json');status=read(path/'status.json')
    rows=list(csv.DictReader((path/'frames.csv').open()))
    assert status['completed'] and status['frames_completed']==m['frames']
    assert [int(x['frame']) for x in rows]==list(range(1,m['frames']+1))
    cfg=read(path/'cloth_scene.json');case=m['cloth_scene']
    if m['cloth_offset_y']==0:
        for key,value in SPEC[case]['expected_scene'].items(): assert cfg[key]==value,(key,cfg[key],value)
        assert sha(path/'scene_initial.f64')==SPEC[case]['initial_sha256']
    assert sha(path/'scene_triangles.u32')==SPEC[case]['triangles_sha256']
    assert m['pcg_tolerance']==.001*m['pcg_tolerance_scale']
    assert m['newton_tolerance']==.001*m['newton_tolerance_scale']
    for x in rows:
        assert int(x['newton_cap_hits'])==int(x['nonfinite_vertices'])==int(x['fallbacks'])==0
        assert int(x['bounds_valid'])==1 and int(x['newton_solves'])==1
        assert all(math.isfinite(float(x[k])) for k in ['step_ms','newton_ms','exit_residual','tail_ms'])
        ready,exit_k=int(x['first_toi_ready_k']),int(x['exit_k'])
        assert int(x['tail_iterations'])==(exit_k-ready if ready>=0 else 0)
        if x['exit_reason'] in ['toi','gated','compensated']:
            assert exit_k>=6 and int(x['budget_valid'])==1 and ready>=0
            assert float(x['exit_beta'])<=.001+1e-12
            if x['exit_reason']=='compensated': assert float(x['exit_budget'])<=.001+1e-12
        if m['production_measurement'] and m['termination'] in ['original','toi']:
            assert int(x['residual_evaluations'])==0
    assert sum(int(x['direction_launches']) for x in rows)>0
    n=m['vertex_count'];nf=m['frames']
    assert (path/'states.f64').stat().st_size==nf*n*24
    assert (path/'velocities.f64').stat().st_size==nf*n*24
    positions=np.memmap(path/'states.f64',dtype='<f8',mode='r',shape=(nf,n,3))
    velocity=np.memmap(path/'velocities.f64',dtype='<f8',mode='r',shape=(nf,n,3))
    assert np.isfinite(positions).all() and np.isfinite(velocity).all()
    ini=np.fromfile(path/'initial.f64',dtype='<f8').reshape(n,3)
    fixed=np.array(cfg['pinned'],dtype=int)
    drift=float(np.linalg.norm(positions[:,fixed]-ini[fixed],axis=2).max()) if fixed.size else 0.
    assert drift<1e-10
    total=lambda k:sum(float(x[k]) for x in rows)
    result={'path':str(path),'metadata':m,'newton':int(total('newton_iterations')),
            'pcg_iterations':int(total('pcg_iterations')),'seconds':total('step_ms')/1000,
            'first_step_ms':float(rows[0]['step_ms']),
            'continuation_ms_per_frame':statistics.mean(float(x['step_ms']) for x in rows[10:]) if nf>10 else None,
            'tail_iterations':int(total('tail_iterations')),'tail_seconds':total('tail_ms')/1000,
            'toi_ready_frames':sum(int(x['first_toi_ready_k'])>=0 for x in rows),
            'residual_rejections':int(total('residual_rejections')),'history_rejections':int(total('history_rejections')),
            'guard_rejections':int(total('guard_rejections')),'auxiliary_exits':sum(x['exit_reason']!='original_direction' for x in rows),
            'intersection_frames':int(total('intersection')),'fixed_drift_m':drift,
            'phases_ms':{k:total(k) for k in ['gradient_hessian_ms','residual_check_ms','linear_system_ms','ccd_ms','line_search_ms','graph_prepare_ms']}}
    if (path/'termination_trace.jsonl').exists(): result['trace_samples']=trace_check(path)
    return result

def execute(path,case,config,nf=50,repeat=1,scale=1,offset=0,profile=False,production=False,trace=False):
    if path.exists():
        assert (path/'execution.json').exists(),f'Partial run retained; refusing overwrite: {path}'
        e=read(path/'execution.json');assert e['returncode']==0 and e['exe_sha256']==sha(EXE)
        return validate(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    mode,eps=CONFIGS[config]
    cmd=[str(EXE),'--scene','1','--pcg-dual-graph','--benchmark-frames',str(nf),
         '--benchmark-output-dir',str(path),'--export-states','--termination',mode,
         '--toi-min-updates','6','--toi-threshold','.001','--residual-threshold',str(eps),
         '--newton-tolerance-scale',str(scale),'--pcg-tolerance-scale',str(scale),'--cloth-offset-y',str(offset)]
    if profile: cmd+=['--profile-stages']
    if production: cmd+=['--production-measurement']
    if trace: cmd+=['--trace-termination']
    env={k:v for k,v in os.environ.items() if not k.startswith(('GIPC_','CIPC_'))}
    env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CIPC_SCENE=case,
               CIPC_META_OUT=str(path/'cloth_scene.json'),CIPC_INITIAL_OUT=str(path/'scene_initial.f64'),
               CIPC_TRIANGLES_OUT=str(path/'scene_triangles.u32'))
    env['PATH']=str(EXE.parent)+os.pathsep+env['PATH']
    print('START',path.relative_to(ROOT),flush=True);begin=time.monotonic()
    with path.with_suffix('.stdout.log').open('wb') as so,path.with_suffix('.stderr.log').open('wb') as se:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,timeout=1800)
    e={'command':cmd,'returncode':result.returncode,'wall_seconds':time.monotonic()-begin,
       'exe_sha256':sha(EXE),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'gpu_after':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,temperature.gpu,clocks.sm','--format=csv,noheader'],text=True)}
    write(path.with_suffix('.execution.json'),e)
    assert result.returncode==0,f'Run failed; logs retained: {path}'
    write(path/'execution.json',e)
    data=validate(path)
    print(f"DONE {case}/{config}: {nf} frames, {data['seconds']:.3f}s, Newton={data['newton']}, tail={data['tail_iterations']}, history veto={data['history_rejections']}",flush=True)
    return data

def difference(path,reference):
    m=read(path/'metadata.json');r=read(reference/'metadata.json')
    assert m['frames']==r['frames'] and m['dt']==r['dt']
    for name in ['initial.f64','initial_velocities.f64','masses.f64','boundary.i32','scene_triangles.u32']:
        assert sha(path/name)==sha(reference/name),(path,name,'initial mismatch')
    cfg=read(path/'cloth_scene.json');boundary=np.fromfile(path/'boundary.i32',dtype='<i4')
    mask=(np.array(cfg['roles'])>0)&(boundary==0); n=m['vertex_count'];nf=m['frames']
    result={'free_cloth_vertices':int(mask.sum())}
    for name,label in [('states.f64','position'),('velocities.f64','velocity')]:
        a=np.memmap(path/name,dtype='<f8',mode='r',shape=(nf,n,3))
        b=np.memmap(reference/name,dtype='<f8',mode='r',shape=(nf,n,3))
        rms=[];maximum=[]
        for x,y in zip(a,b):
            delta=np.linalg.norm(x[mask]-y[mask],axis=1)
            rms.append(float(np.sqrt(np.mean(delta*delta))));maximum.append(float(delta.max()))
        result.update({f'{label}_max_frame_rms':max(rms),f'{label}_final_rms':rms[-1],f'{label}_max':max(maximum)})
    return result

def summarize(out):
    records=[]
    for p in out.rglob('execution.json'):
        d=validate(p.parent);d['relative_path']=p.parent.relative_to(out).as_posix()
        parts=d['relative_path'].split('/')
        if parts[0] in ['main','probe']:
            reference=out/('references' if parts[0]=='main' else 'probe-references')/parts[1]/'strict4'
            if reference.exists(): d['vs_strict4']=difference(p.parent,reference)
        records.append(d)
    agreements={case:difference(out/'references'/case/'strict2',out/'references'/case/'strict4') for case in CASES}
    probe_agreements={case:difference(out/'probe-references'/case/'strict2',out/'probe-references'/case/'strict4') for case in CASES}
    grouped={}
    for c in CASES:
        grouped[c]={}
        for config in CONFIGS:
            ds=[d for d in records if d['relative_path'].startswith(f'main/{c}/{config}/')]
            times=[d['seconds'] for d in ds]
            grouped[c][config]={'repeats':len(ds),'seconds_median':statistics.median(times),
                'seconds_min':min(times),'seconds_max':max(times),'newton_median':statistics.median(d['newton'] for d in ds),
                'tail_median':statistics.median(d['tail_iterations'] for d in ds),
                'tail_seconds_median':statistics.median(d['tail_seconds'] for d in ds),
                'history_veto_median':statistics.median(d['history_rejections'] for d in ds),
                'residual_veto_median':statistics.median(d['residual_rejections'] for d in ds),
                'max_position_rms_m':max(d['vs_strict4']['position_max_frame_rms'] for d in ds),
                'max_velocity_rms_m_s':max(d['vs_strict4']['velocity_max_frame_rms'] for d in ds),
                'first_step_ms_median':statistics.median(d['first_step_ms'] for d in ds),
                'continuation_ms_median':statistics.median(d['continuation_ms_per_frame'] for d in ds)}
    noise={}
    for case in CASES:
        base=out/'main'/case/'graph'/'r01'
        noise[case]=[difference(out/'main'/case/'graph'/f'r{i:02d}',base) for i in [2,3]]
    write(out/'summary.json',{'groups':grouped,'reference_agreement':agreements,
          'probe_reference_agreement':probe_agreements,'original_graph_repeat_difference':noise,'runs':records})
    lines=['# 双容差 TOI 关键场景实验','',
           'table/bunny 为原固定场景前 50 帧；ε_TOI=1e-3，Kmin=6，PCG 原 rᵀz 阈值 1e-3。',
           '预先固定 .03 为重复验证参数；其余探索参数各一次。时间含共同残差观测，剖析与生产路径另列。',
           '', '| 场景 / 配置 | 次数 | 总秒中位数 [范围] | Newton | N_tail | T_tail 秒 | 自由布料位置 RMS 最大值 mm | 自由布料速度 RMS 最大值 m/s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for c,configs in grouped.items():
        for k,d in configs.items():
            lines.append(f"| {c} / {k} | {d['repeats']} | {d['seconds_median']:.3f} [{d['seconds_min']:.3f}, {d['seconds_max']:.3f}] | {d['newton_median']:g} | {d['tail_median']:g} | {d['tail_seconds_median']:.3f} | {1000*d['max_position_rms_m']:.3f} | {d['max_velocity_rms_m_s']:.4f} |")
            csvrows.append({'scene':c,'config':k,**d})
    lines+=['','comp001=原补偿；comp01/03/10=双容差补偿，ε_R=.01/.03/.1；gate 同容差当前门控。',
            '误差相对原 Graph 的 4 倍严格参考，统计自由布料顶点；不是精确物理解。原出口不受新残差界约束。',
            '', '## 严格参考间差异', '', '| 场景 | 2× 对 4× 最大帧位置 RMS mm | 最大帧速度 RMS m/s |','|---|---:|---:|']
    for c,d in agreements.items(): lines.append(f"| {c} | {1000*d['position_max_frame_rms']:.4f} | {d['velocity_max_frame_rms']:.5f} |")
    lines+=['','## 首步与续运行窗口','','cold 在此指首个物理步，未包含整个进程/CUDA 初始化。第 11–50 帧是续运行窗口，不意味着物理状态稳定。',
            '', '| 场景 / 配置 | 首步 ms 中位数 | 第 11–50 帧平均 ms 中位数 |','|---|---:|---:|']
    for c in CASES:
        for k in CORE:
            d=grouped[c][k];lines.append(f"| {c} / {k} | {d['first_step_ms_median']:.3f} | {d['continuation_ms_median']:.3f} |")
    lines+=['','## 同完整初态单步接触探针','','布料整体下移 .248 m，在独立场景中零速度初始化；不是轨迹检查点回放。全部初始化输入哈希逐项一致。',
            '', '| 场景 / 配置 | Newton | 相对严格参考位置 RMS mm | 速度 RMS m/s |','|---|---:|---:|---:|']
    for d in records:
        if d['relative_path'].startswith('probe/'):
            e=d['vs_strict4'];lines.append(f"| {d['relative_path']} | {d['newton']} | {1000*e['position_max_frame_rms']:.5f} | {e['velocity_max_frame_rms']:.5f} |")
    lines+=['','## 阶段剖析','','剖析启用额外同步，其时间不并入未剖析性能排名。Graph prepare/update 是 linear_system 的内部子项。']
    for d in records:
        if d['metadata']['profile_stages']:
            phases=d['phases_ms'];other=d['seconds']*1000-sum(v for k,v in phases.items() if k!='graph_prepare_ms')
            lines += ['',d['relative_path']+':', '', '| 阶段 | 秒 | 占整步比例 |','|---|---:|---:|']
            for k,v in {**phases,'other':other}.items(): lines.append(f"| {k} | {v/1000:.4f} | {v/(1000*d['seconds']):.2%} |")
    lines+=['','## 生产观测成本','','仅 Graph/纯 TOI 关闭不需要的真实残差观测；TOI 此时只校验最终 alpha 有限且合法，失去逐轮梯度有限性观测，帧末状态检查保留。该运行独立报告。']
    for d in records:
        if d['metadata']['production_measurement']: lines.append(f"- {d['relative_path']}: {d['seconds']:.3f} s，Newton={d['newton']}。")
    lines+=['','完整参考差异、重复噪声、拒绝原因及逐帧数据见 summary.json 与 frames.csv。',
            '本轮只覆盖指定诊断窗口，不据此接受全场景默认阈值，不承诺位移/速度误差界。',
            '原 GPU 相交标志、有限性和固定点检查不能替代全面布料自交/穿透检查。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    write(out/'EVIDENCE_MANIFEST.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='EVIDENCE_MANIFEST.json'})
    print('SUMMARY',out/'REPORT.md',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();out=Path(args.out).resolve()
    if args.summarize_only: summarize(out);return
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():
        write(out/'protocol.json',{'exe_sha256':sha(EXE),'plan_sha256':sha(ROOT/'EXPERIMENT_PLAN_V2.md'),
              'scenes':CASES,'frames':50,'configs':CONFIGS,'three_repeat_configs':CORE,
              'epsilon_toi':.001,'minimum_updates':6,'strict_reference_scales':[.5,.25],
              'contact_probe_y_offset':-.248,'smoke':args.smoke})
        shutil.copy2(__file__,out/'runner-used.py')
    else: assert read(out/'protocol.json')['exe_sha256']==sha(EXE),'Binary changed; use new output directory'
    if args.smoke:
        for case in CASES:
            for k in ['comp001','comp03','gate03']:
                execute(out/case/k,case,k,nf=2,offset=-.248,trace=True,profile=True)
        execute(out/'table'/'production','table','toi',nf=2,production=True)
        execute(out/'table'/'strict','table','graph',nf=1,scale=.25,offset=-.248)
        print('SMOKE COMPLETE',flush=True);return
    for repeat in range(1,4):
        configs=list(CONFIGS) if repeat==1 else CORE
        for ci,case in enumerate(CASES):
            shift=(repeat+ci-1)%len(configs);order=configs[shift:]+configs[:shift]
            for k in order: execute(out/'main'/case/k/f'r{repeat:02d}',case,k)
    for case in CASES:
        for name,scale in [('strict2',.5),('strict4',.25)]:
            execute(out/'references'/case/name,case,'graph',scale=scale)
            execute(out/'probe-references'/case/name,case,'graph',nf=1,scale=scale,offset=-.248,trace=True)
        for k in CONFIGS: execute(out/'probe'/case/k,case,k,nf=1,offset=-.248,trace=True)
    execute(out/'profile'/'stack10','stack10','graph',nf=35,profile=True)
    execute(out/'profile'/'table','table','comp03',nf=50,profile=True)
    for k in ['graph','toi']: execute(out/'production'/'table'/k,'table',k,production=True)
    summarize(out)

if __name__=='__main__': main()
