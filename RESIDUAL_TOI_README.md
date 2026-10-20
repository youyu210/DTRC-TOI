# 残差衰减补偿累计 TOI：Graph 上的独立版本

基于已留存 `pcg-dual-graph-v1-validated`（145c87e）；原目录 `../pcg-dual-graph-c49982fa` 完整保留。用户原方案存于 `docs/RESIDUAL_TOI_SPEC.md`。本版本按照用户的新请求，只修改外层 Newton 附加终止条件，PCG 双段图、碰撞检测、能量线搜索和物理参数保留。

## 三个主版本

统一启用 `--pcg-dual-graph`，用 `--termination` 切换：

| 选项 | 退出策略 |
|---|---|
| original | 现有 Graph 原停止规则 |
| toi | 原停止规则 + beta <= epsilon |
| compensated | 原停止规则 + z <= epsilon |
| gated | 消融：原停止规则 + beta <= epsilon 且 r <= epsilon |

现有 Graph 源码没有累计 TOI 路径，本次新增 pure-TOI 对照；它不是直接移植 barrier-free 求解器。所有附加路径使用相同 Kmin、阈值、有效性检查、边界动画目标已完成检查。原 `gradVanish` 实际是上一轮移动方向范数条件，完整保留并优先判断；未强迫原停止规则至少执行 Kmin 次。

## 对接约定

- 残差是原求解坐标的真实非线性梯度 infinity norm：ABD `system_gradient` 和自由 FEM 的 `fb+shape_grads`。固定 ABD 梯度已由原组装器置零，FEM `BoundaryType!=0` 被排除。不是 PCG 线性残差。
- 保留统一求解器的原坐标缩放。ABD 平移/仿射分量与 FEM 笛卡尔分量直接使用原始 RHS 约定；没有额外的坐标归一化。因此指标依赖原坐标表示，不解释为物理位置误差。
- Kmin 默认 6：前 5 次接受更新不累计；第 6 次更新**之前**的梯度设定固定 `R_ref=max(R_start,1e-30)`。第 6 次接受更新才首次进入预算。参考在该次 Newton solve 中不变。
- 保存能量回溯和相交回溯后的最终 alpha，经过 `postLineSearch()`，下一轮正常梯度/Hessian 组装时取得当前有效目标的 R_new，延后验收。pure-TOI 和 gated 也在该位置验收，获得相同的组装开销；与接受步后立即退出的 TOI 实现相比，最后一次多一次梯度/Hessian 组装。
- `d=max(0,r_new-(1-alpha)*r_old)`，`z=(1-alpha)*z+d`，补偿系数严格为 1，不把预算裁剪到 [0,1]。
- alpha、残差或预算无效时禁止附加退出，继续原停止路径。Kappa 在 `postLineSearch()` 改变时仍使用更新后的目标残差，记录 target_changes；不据固定目标 Taylor 展开解释这些样本。
- 相交回溯、CCD/CFL 限制、边界更新等安全步骤没有改动；附加退出额外要求 `animation_fullRate>0.99`，对应原外层目标完成条件。

## 构建与运行

`BUILD_WINDOWS.cmd` 构建本机 Release。`RUN_WINDOWS.cmd` 默认启动场景 1 的 Graph + compensated 可视窗口；可在 PowerShell 切换：

```powershell
.\RUN_WINDOWS.ps1 -Scene 3 -Termination original
.\RUN_WINDOWS.ps1 -Scene 5 -Termination toi
.\RUN_WINDOWS.ps1 -Scene 1 -Termination compensated
.\tests\RUN_BUDGET_TESTS.cmd
python experiments/run_three_versions.py --out experiments/three-versions-100-v1
```

默认主实验为源码场景 1、3、5，各方法 100 帧、三次，以循环轮换先后顺序串行运行，GPU 0，epsilon=1e-3、Kmin=6。其他资产与参数保持一致，比较累计 Newton/PCG 迭代、牛顿/PCG/整步时间、退出残差和附加退出次数。

每个场景另运行一个原停止规则参考，Newton 和 PCG 容差各缩小 2 倍。另留存最初的 10 倍严格探索：场景 1、3 完成；场景 5 第 50 帧需 3792 次 Newton、226.52 秒，第 51 帧再次长时间求解后中断，不纳入完整误差对照。消融包括 gated 的 epsilon=1e-3，以及 toi/gated/compensated 的 epsilon=1e-2 和 1e-1 各一次、100 帧。参考/消融与主实验分开，不混入主实验三次速度统计。

所有批量版本都评估相同定义的非线性残差，用于退出质量比较；该归约开销纳入时间。导出与原安全诊断在物理计时之外。开启 `--trace-termination` 的短测还逐轮核验参考、公式和上界，并用原组装器的全局 RHS 独立核对残差；诊断模式不用于速度实验。

## 数据与版本留存

- 输出目录必须不存在，避免覆盖。
- `frames.csv`、`pcg_iterations.jsonl`、metadata/status/environment 留存统计与参数；`--trace-termination` 额外导出逐轮 alpha/R/r/beta/d/z/Kappa。
- `states.f64` 与 `velocities.f64`：无头 little-endian double，形状 `(帧数,顶点数,3)`；初态、终态、末速度另存。后者是原共享速度缓冲，其 ABD 顶点值由原程序置零，ABD 速度另存于 q_v。因此报告的统一世界速度误差使用所有顶点相邻接受位置差 / dt，避免忽略 ABD 的速度误差。
- `tetrahedra.u32`：无头 little-endian uint32，形状 `(四面体数,4)`；`boundary.i32`：原顶点边界类型。用同一初态拓扑检查所有帧的体积符号和静态固定 FEM 顶点漂移；相交使用原 GPU 查询记录。
- `residual-toi-graph-v1` tag 留存实现，`residual-toi-graph-v1-validated` tag 留存结果与清单。`versions/residual-toi-graph-v1` 留存二进制和所需 DLL，资产仍使用本目录烘焙路径。

不能预先承诺本方案更快。预算界只适用于 compensated 分支的退出；原停止分支、位置/速度误差和非线性轨迹不由该界保证。三版本及消融的完整结果见 `experiments/three-versions-100-v1/REPORT.md` 与 `VALIDATION.md`。
