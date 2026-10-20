# 五布料场景、四版本、各 100 帧

沿用 `cloth-v1-20260911` 固定基准的 sphere7、bunny、table、stack10、hang。
初态、三角拓扑 SHA-256 及材料、时间步、固定点等参数逐项核对
`experiments/CLOTH_V1_SCENES.json`。此前前四场景为 200 帧，本实验统一为
100 帧；球/兔子/桌子模拟 1 秒，十层模拟 2 秒，悬挂模拟 4 秒。

四模式共用同一 Release 二进制，以保持构建和诊断成本一致：

| 模式 | PCG | Newton 停止 |
|---|---|---|
| base | c49982f 原 PCG | 原规则 |
| graph | 留存 PCG 双段 CUDA Graph | 原规则 |
| toi | 同一双段 Graph | 原规则优先，附加纯累计 TOI |
| compensated | 同一双段 Graph | 原规则优先，附加残差衰减补偿累计 TOI |

附加退出 epsilon=1e-3、Kmin=6、residual floor=1e-30。PCG 及原 Newton
阈值系数均按冻结布料基准为 1e-3，保留原 PCG 的 rᵀz 判据和检查时序。
此处 graph 并非旧表中的 AGIPC `f36c784` 或 `e8d638b`。
仅新增场景入口，没有修改已留存 PCG/Graph/TOI 数值实现。

GPU 0 串行，每项一次，warmup=0。整步时间包含真实非线性残差测量、
图准备/更新与物理求解，排除导出、渲染和帧末诊断；Newton 和 PCG 时间
也单独留存。base 同样包含残差诊断成本，因此不是完全没有诊断的上游
可视化程序裸跑时间。单次测量不支持稳定性能结论。不同退出条件可能
产生不同轨迹，速度比较不代表等精度比较。

运行方法：

```powershell
.\BUILD_WINDOWS.cmd
python -X utf8 experiments/run_cloth_four_versions.py --out experiments/new-four-cloth-run
# 可配置三次重复；输出必须是新目录。
python -X utf8 experiments/run_cloth_four_versions.py --out experiments/new-four-cloth-repeat --repeats 3
```

五场景短测见 `experiments/cloth-four-smoke-v2`。最初短测 v1 的脚本图审计
选项名错误，失败日志保留；v2 已修正并全部完成，不计入正式速度。
短测共 370 个真实 RHS 残差核验，全部一致；五个同 RHS Graph/原 PCG
方向审计的迭代数一致，最大方向相对 L2 差 9.0354e-8，诊断容差 1e-6。
没有要求 GPU 浮点归约逐位一致，没有变更实际 PCG 停止阈值。

正式结果、原始帧指标和状态：`experiments/cloth-four-100-v1`。
报告为 `REPORT.md`，汇总为 `summary.json`；状态文件格式与
`RESIDUAL_TOI_README.md` 相同。位置 RMS 以全部顶点等权统计，包含固定障碍
顶点；不是仅布料顶点的指标。相对 base 位置差只表示轨迹差异，base
不是严格收敛真值。原 GPU 相交标志为有限诊断，不提供全面布料几何安全保证。

实现留存：提交 `ff8b5b9dcebfca6f50612967e2eee0baab295f0d`、标签
`cloth-four-100-v1`。独立二进制和 DLL 快照为 `versions/cloth-four-100-v1`；
其中 `MANIFEST.json` 记录二进制/资产 SHA-256。旧实现与旧结果标签、二进制
快照均保留。完整结果归档使用 `cloth-four-100-v1-validated` 标签。
