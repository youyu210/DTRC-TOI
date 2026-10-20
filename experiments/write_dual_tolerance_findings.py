"""Interpret the frozen diagnostic matrix, including its negative results."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/dual-tolerance-v2'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    s = read(OUT / 'summary.json')
    a = read(OUT / 'AUDIT.json')
    groups = s['groups']
    labels = {'graph': '原 Graph', 'toi': '纯 TOI', 'comp001': '旧补偿 εR=.001',
              'comp03': '双容差补偿 εR=.03', 'gate03': '当前残差门控 εR=.03'}
    lines = [
        '# 双容差 TOI：关键场景实验结论', '',
        '本轮验证了性能假设：将残差容差独立放宽到 .03，能明显减少旧补偿的 Newton 尾部迭代。'
        '但这不是同精度加速证明，尚不能把 .03 接受为生产默认。历史预算在这批样本中没有提供额外拒绝；'
        'stack10 的主要开销在梯度/Hessian 与 PCG 外的线性系统准备。', '',
        '## 范围、控制与版本', '',
        '基于 `34f8e0f / cloth-four-100-v1-validated`。求解器实现冻结为 `cc1afb9 / dual-tolerance-toi-v2`，'
        '在 RTX 3090 Ti（GPU 0）、CUDA 12.6、Release 下串行运行。旧版源码、二进制和实验均保留。',
        'table/bunny 使用固定场景前 50 帧；stack10 仅剖析前 35 帧。εTOI=1e-3、Kmin=6、PCG 原 rᵀz 系数 1e-3。'
        '主实验统一测量残差；固定 λ=εTOI/εR，不改 PCG 迭代公式、预条件器、CCD、线搜索和双段图结构。',
        '残差容差 .01/.03/.1 各做探索，.03 在运行前已选定作三次重复参数，未按场景调参。'
        '原 Graph、纯 TOI、旧补偿、补偿 .03、门控 .03 各三次；其余参数各一次。完整参数扫描见 REPORT.md。',
        f"计划 68 组：实际完整完成 {a['completed_runs']} 组、{a['completed_frames']} 帧；"
        f"另有 {a['interrupted_reference_runs']} 组严格参考中止，保留 {a['partial_frames_retained']} 帧部分结果。"
        '38 组主实验（1,900 帧）、22 个同初态单步、阶段剖析和生产测量均完成。', '',
        '## 三次重复的主要结果', '',
        '时间为 50 帧整步耗时之和的中位数，排除渲染/导出；Newton、Ntail 也取中位数。'
        '位置指标先逐帧计算自由布料顶点 RMS，再对所有帧和重复取最大值，单位 mm。'
        '两场景各 22,801 个自由布料顶点，固定障碍不计入 RMS。', '',
        '| 场景 / 配置 | 秒 [最小, 最大] | Newton | Ntail | 最大帧位置 RMS mm |',
        '|---|---:|---:|---:|---:|',
    ]
    for case in ['table', 'bunny']:
        for config, label in labels.items():
            d = groups[case][config]
            lines.append(f"| {case} / {label} | {d['seconds_median']:.3f} [{d['seconds_min']:.3f}, {d['seconds_max']:.3f}] | "
                         f"{d['newton_median']:g} | {d['tail_median']:g} | {1000*d['max_position_rms_m']:.3f} |")
    lines += ['', 'table 的差异相对 2× 严格参考，bunny 相对 4× 严格参考；两者均不是已证明收敛的精确解。', '']
    for case in ['table', 'bunny']:
        old, new = groups[case]['comp001'], groups[case]['comp03']
        lines.append(f"- {case}：补偿 .03 比旧补偿耗时减少 {100*(1-new['seconds_median']/old['seconds_median']):.2f}%，"
                     f"Newton 减少 {100*(1-new['newton_median']/old['newton_median']):.2f}%；"
                     f"Ntail 从 {old['tail_median']:g} 降至 {new['tail_median']:g}，"
                     f"Ttail 从 {old['tail_seconds_median']:.3f} s 降至 {new['tail_seconds_median']:.3f} s。")
    noise = {c: max(x['position_max_frame_rms'] for x in s['original_graph_repeat_difference'][c])*1000 for c in groups}
    ref = s['reference_agreement']['bunny']
    lines += ['', '## 质量结论的边界', '',
              'table 的 4× 严格参考在第 23 帧超过 335 秒未完成，反复打印线搜索次数偏高；实际提前中止，'
              '没有声称触及 Newton 上限或证明数学不收敛。其前 22 帧及日志保留。'
              '2× 参考完成，但缺乏另一档完整严格参考的一致性验证。运行中修订见 ../../EXPERIMENT_AMENDMENT_V2.md。',
              '中止项 CSV 有 22 个已完成物理步；状态/速度流的第 22 帧尾部各有 1,680 字节未落盘，'
              '因此只保留 21 个完整原始状态帧和部分第 22 帧。文件原样保留，不补齐、不用于完整轨迹比较。',
              f"bunny 的 2× 与 4× 参考之间最大帧位置 RMS 为 {ref['position_max_frame_rms']*1000:.3f} mm，"
              f"速度 RMS 为 {ref['velocity_max_frame_rms']:.4f} m/s。原 Graph 重复间最大帧位置 RMS："
              f"table {noise['table']:.3f} mm、bunny {noise['bunny']:.3f} mm。"
              '不能把上述不同方法对某个参考的差值直接相减并归因于停止准则，也不能据此声称长期轨迹等精度。',
              '严格参考将 Newton 阈值和 PCG rᵀz 停止系数分别乘 .5 或 .25，实际值已核对元数据。'
              '这不是将线性残差的欧氏范数阈值简单缩小 2 或 4 倍。', '',
              '## 同完整初始化状态的单步接触探针', '',
              '统一下移布料 .248 m、初始向下速度 1 m/s，经原初始化流程构建状态。'
              '比较前核对初态位置、初速度、质量、边界和拓扑哈希。该测试不是从轨迹中仅恢复位置。'
              '位置差异相对同一探针的 4× 严格参考，下面用 μm 表示。', '',
              '| 场景 / 配置 | Newton | 出口 | 位置 RMS μm |', '|---|---:|---|---:|']
    for case in ['table', 'bunny']:
        for config, label in labels.items():
            run = next(r for r in s['runs'] if r['relative_path'] == f'probe/{case}/{config}')
            row = list(csv.DictReader((OUT/run['relative_path']/'frames.csv').open()))[0]
            lines.append(f"| {case} / {label} | {run['newton']} | {row['exit_reason']} | "
                         f"{1e6*run['vs_reference']['position_max_frame_rms']:.3f} |")
    lines.append('')
    for case, d in s['probe_reference_agreement'].items():
        lines.append(f"- {case} 单步两档参考位置 RMS 差 {1e6*d['position_max_frame_rms']:.3f} μm。")
    lines += ['table 这一步中补偿/门控均由原方向规则先退出，因此未区分它们的提前退出行为；'
              'bunny 中补偿 .03 和门控 .03 都是 8 次 Newton，旧补偿 9 次、纯 TOI 6 次。'
              '单步结果说明了局部折中，不能替代后续轨迹质量验收。', '',
              '## 历史预算是否有额外价值', '',
              f"所有完整运行的历史预算单独拒绝次数为 {a['totals']['history_rejections']}；"
              f"门控退出时若保留历史预算会被拒绝的样本数为 {len(a['gate_exits_that_history_would_veto'])}。"
              f"补偿/门控尾部不能由当前残差拒绝解释的帧数为 {len(a['tail_frames_not_explained_by_current_residual'])}。"
              '因此本轮性能改善主要来自独立放宽当前残差目标，没有观测到历史预算带来的附加质量保护。'
              '门控在诊断时维护影子预算，但退出规则不依赖它。',
              '补偿与门控的独立运行仍可能产生不同轨迹/迭代次数；不能将这种差异直接归因于历史预算。'
              '保留两种实现，不凭这两个诊断窗口删除历史状态。', '',
              '## 阶段剖析与下一处性能目标', '',
              '剖析使用额外同步，不与主实验直接合并排名。PCG 是线性系统阶段的子项；下表分解后不重复相加。', '',
              '| 阶段 | stack10 35 帧秒数 | 比例 |', '|---|---:|---:|']
    run = next(r for r in s['runs'] if r['relative_path'] == 'profile/stack10')
    p = run['phases_ms']
    rows = list(csv.DictReader((OUT/'profile/stack10/frames.csv').open()))
    pcg = sum(float(r['pcg_ms']) for r in rows)
    other = run['seconds']*1000-sum(v for k,v in p.items() if k != 'graph_prepare_ms')
    stages = {'梯度/Hessian': p['gradient_hessian_ms'], '线性系统中 PCG 以外部分': p['linear_system_ms']-pcg,
              'PCG 求解（含其内部准备）': pcg, 'CCD': p['ccd_ms'], '线搜索': p['line_search_ms'],
              '残差范数/预算检查': p['residual_check_ms'], '其余': other}
    for label, value in stages.items():
        lines.append(f"| {label} | {value/1000:.4f} | {value/(run['seconds']*1000):.2%} |")
    table_profile = next(r for r in s['runs'] if r['relative_path'] == 'profile/table')
    lines += ['', f"Graph capture/update 准备仅 {p['graph_prepare_ms']/1000:.4f} s，是 PCG 内部子项。"
              f"table 补偿 .03 的残差检查占整步 {table_profile['phases_ms']['residual_check_ms']/(table_profile['seconds']*1000):.2%}。",
              '优先拆解梯度/Hessian 和线性系统中 PCG 以外的准备工作，再确定矩阵组装、转换、预条件器各自占比。'
              '本轮数据不支持先把 CCD 当作主要瓶颈，也不支持仅靠减少残差范数归约获得大幅加速。'
              '当前梯度与 Hessian 混合组装；按需验收若要省去 Hessian，需要单独设计梯度复用。', '',
              '## 生产观测与验证', '',
              '生产测量只关闭原 Graph/纯 TOI 不需要的额外残差观察；仍执行原梯度/Hessian 组装。'
              '纯 TOI 此路径只验证 alpha 合法性，失去逐轮梯度有限性观察，帧末有限性检查保留。', '']
    for run in s['runs']:
        if run['metadata']['production_measurement']:
            lines.append(f"- {run['relative_path']}：{run['seconds']:.3f} s，Newton={run['newton']}（单次）。")
    lines += ['该单次生产路径不能与不同轨迹的主实验直接相减得到准确的观察成本。首步和第 11–50 帧续运行窗口见 REPORT.md；'
              '首步计时不包含进程/CUDA 初始化，续运行窗口也不意味着物理稳态。',
              f"完整运行：Newton 达上限 {a['totals']['newton_cap_hits']}、Graph 回退 {a['totals']['fallbacks']}、"
              f"非有限顶点 {a['totals']['nonfinite_vertices']}、原 GPU 相交标志 {a['totals']['intersection']}；"
              f"固定顶点最大漂移 {a['max_fixed_drift_m']:.3g} m。预算界和固定参考恒等式检查通过；"
              f"正式单步审计 {a['formal_native_rhs_checks']} 次真实组装 RHS 比对全部一致，另有短测 334 次。"
              'CPU 标量测试覆盖 λ=1 旧递推兼容、双容差界、历史拒绝和非法输入。',
              '相交标志与有限性检查不覆盖全面布料自交/穿透；这批结果是诊断验证，不是完整几何安全或物理精度证明。', '',
              '## 后续顺序', '',
              '1. 先查清 table 第 23 帧严格求解的平台/线搜索行为，并处理 bunny 长轨迹参考敏感性；在此之前不把 .03 设为默认。',
              '2. 保留 .03 作为速度/质量候选，与当前门控共同验证；需要区分历史预算时增加实际出现小步长和历史拒绝的受控样本。',
              '3. 对 stack10 拆分非 PCG 线性准备与 Hessian 成本，再决定是否进入按需验收或自适应 PCG 路线。本轮没有实现这些后续算法，也没有降低 Kmin。', '',
              '逐帧数据、配置、原始状态摘要和失败记录见本目录；复现及二进制留存说明见 ../../versions/dual-tolerance-toi-v2/README.md。']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(OUT/'FINDINGS.md')


if __name__ == '__main__':
    main()
