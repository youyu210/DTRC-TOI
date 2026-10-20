# PCG 双段 CUDA Graph

基于 `c49982fac7767225c7775683734c89eb74081142`，独立目录保留源码、构建与实验结果。原始版本位于同级 `upstream-c49982fa`，未修改其受版本管理的文件。

## 数学与执行顺序

原始 `PCGSolver::pcg()` 保留。新路径只改变 CUDA 提交方式：

1. 方向图：原始 SpMV → 原始归约计算 `pᵀAp` → `alpha=rho/(pᵀAp)` → 原式更新 `x,r`。
2. 主机按原顺序检查**旧 rho**，保留原容差和迭代上限。未改变原停止条件，也未添加零右端项快捷路径。
3. 原预条件器在图外照常执行。
4. 尾段图：原始归约计算 `rho_new=rᵀz` → `beta=rho_new/rho` → 原式更新 `p`；拷回 rho 供下一次主机检查。

保持原归约树、scratch 别名、SpMV 原子累加及所有物理参数。标量除法迁移到 GPU 双精度 round-to-nearest；浮点原子累加顺序仍可能导致不同运行间的微小差异。公式相同不代表逐位相同。

两段图分别缓存。指针、向量长度、矩阵非零项数量等变化时重新捕获，先尝试 `cudaGraphExecUpdate`，无法更新时重建。捕获只记录，不执行；准备失败时该段用原式普通提交。图启动失败直接报错，避免重复执行已部分完成的更新。

## 构建与运行

本机 `BUILD_WINDOWS.cmd` 配置 Release / CUDA 12.6 / MSVC / vcpkg 并构建。沿用本机可运行的 cuBLAS DLL 配对，放置于可执行文件目录；没有修改系统 CUDA 安装。

`RUN_WINDOWS.cmd` 默认启动双段图、场景 1 的可视窗口；PowerShell 可切换：

```powershell
.\RUN_WINDOWS.ps1 -Scene 3
.\RUN_WINDOWS.ps1 -Base -Scene 1
```

程序默认使用原始 PCG，直接运行时用 `--pcg-dual-graph` 开启。批量入口执行原有初始化和 `IPC_Solver()`，省去 OpenGL 渲染。

```powershell
.\build\gipc.exe --scene 1 --benchmark-frames 100 --benchmark-output-dir experiments/example --pcg-dual-graph --export-states
python experiments/run_comparison.py --out experiments/comparison-100-v1
```

输出目录必须不存在，防止覆盖已有版本/结果。场景按源码 `scene_no=1,3,5`；正式实验每种方法每场景 100 帧、各重复三次，交替先后顺序、串行运行、`CUDA_VISIBLE_DEVICES=0`，同一二进制和参数文件。

`frames.csv` 包含逐帧牛顿迭代、PCG 次数/迭代、同步墙钟时间和图回放/更新/回退计数。PCG 时间包含图准备；牛顿时间包含整个 `solve_subIP`；整步时间包含全部物理更新。状态导出、渲染及旧版报告写盘在计时外。`states.f64` 是 little-endian double，形状为 `(帧数, vertex_count, 3)`，无文件头；初态与终态另存。

`--audit-pcg-frame N` 在第 N 帧第一次 PCG 求解时，对完全相同的 A,b 独立运行原始 PCG，保留实际解，并记录方向相对差、迭代数、真实相对残差。审计计入牛顿/整步时间，只用于独立数值检查，不能作为速度实验。

第 100 帧的独立审计可用 `python experiments/run_late_audits.py --out experiments/late-audit-new` 重现，必须在速度实验结束后串行运行。已完成的验证及限制见 `experiments/comparison-100-v1/VALIDATION.md`。

## 版本留存

- `upstream-c49982fa` Git tag：原始提交。
- `pcg-dual-graph-v1` Git tag：实现与实验脚本。
- `pcg-dual-graph-v1-validated` Git tag：完整实验统计、独立审计、哈希清单及验证范围。
- 正式结果在 `experiments/comparison-100-v1/REPORT.md` 与 `summary.json`；逐帧数据及原始二进制状态在各场景方法子目录。
- `versions/pcg-dual-graph-v1` 留存可执行文件、所需 DLL 和 SHA-256 清单。资产绝对路径烘焙在二进制中，运行时应保留本目录及 Assets。

不修改碰撞、能量、预条件器、牛顿/线搜索或停止规则；未实施需要更改数学原理的优化。
