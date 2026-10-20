# 双容差 TOI v2 留存

- 实现：`cc1afb9`，标签 `dual-tolerance-toi-v2`。
- 关键场景诊断归档标签：`dual-tolerance-toi-v2-key-scenes`（67 组完整运行，1 组严格参考中止；精度收敛尚未确认）。
- 上一版：`34f8e0f` / `cloth-four-100-v1-validated`，其源码、五场景数据和二进制保持留存。
- 本目录保留实际实验用 `gipc.exe` 和运行 DLL；SHA-256 见 `MANIFEST.json`。
  大型二进制和原始状态文件留在本机，未纳入 Git；清理或搬迁前需连同这些文件一起备份。
- 原始实验：`experiments/dual-tolerance-v2`；逐文件摘要见其中的 `EVIDENCE_MANIFEST.json`。
- 本轮冻结计划：`EXPERIMENT_PLAN_V2.md`；最后一条参考回答留在 `docs/TOI_RESEARCH_LAST_ANSWER.md`。
- 严格参考的运行预算修订及未完成处理：`EXPERIMENT_AMENDMENT_V2.md`，与原计划分别保留。

## 复现实验

在 `D:\code_2\stiffGIPC\residual-toi-graph-c49982fa` 运行。可执行文件编译时使用了本仓库的绝对资源路径。
同一 GPU 上串行测试；完整矩阵约 2,307 帧，包含 38 组主实验、两档严格参考、同初态单步、阶段剖析和生产测量。

```powershell
python experiments/run_dual_tolerance_v2.py --out experiments/dual-tolerance-v2-repeat
```

脚本使用 `build/gipc.exe`；先核对其 SHA 与本目录快照一致。脚本拒绝覆盖未完成的既有输出。
原实验的后处理审计可用 `python experiments/audit_dual_tolerance_v2.py`，不会启动 GPU 求解。

单项双容差补偿示例（输出目录需使用新名称）：

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CIPC_SCENE = 'table'
.\versions\dual-tolerance-toi-v2\gipc.exe --scene 1 --pcg-dual-graph --benchmark-frames 50 --benchmark-output-dir experiments/manual-table-comp03 --export-states --termination compensated --toi-min-updates 6 --toi-threshold .001 --residual-threshold .03
```

`termination` 的 `original` / `toi` / `compensated` / `gated` 分别为原 Graph、纯 TOI、补偿、当前残差门控。
`compensated` 且 `residual-threshold=.001` 保留旧补偿递推；`.03` 使用固定权重 `λ=.001/.03`。
`gated` 的退出仅依赖 β 与当前残差；诊断模式保留影子历史预算，便于比较。

主排名使用共同残差观察；`--profile-stages` 和 `--production-measurement` 是独立诊断路径。
严格参考参数是 `--newton-tolerance-scale .5/.25` 与 `--pcg-tolerance-scale .5/.25`；PCG 缩放的是原 rᵀz 判据的系数。
没有改变 PCG 迭代公式、预条件器、CCD、线搜索或双段图结构。

## 适用范围

这是 table/bunny 前 50 帧及 stack10 前 35 帧的诊断验证。新容差明确放宽残差目标，不能声称与旧补偿具有同一精度要求。
尚未把 `.03` 设为生产默认，也未实现按需残差验收、自适应 PCG 或更换残差度量。
