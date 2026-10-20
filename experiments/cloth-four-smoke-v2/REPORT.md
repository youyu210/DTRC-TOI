# 五布料场景四版本比较

每项 2 帧，1 次。GPU 0 串行运行，warmup=0；epsilon=1e-3，Kmin=6。时间为包含 Graph 准备和真实非线性残差测量的整步同步墙钟；排除导出、渲染和帧末相交诊断。

四项共用同一二进制：base 使用 c49982f 原 PCG 和原 Newton 停止规则；原 Graph 启用留存 PCG 双段图；纯累计 TOI 和残差补偿 TOI 启用同一双段图并切换附加退出。此处原 Graph 不是旧表的 AGIPC f36c784/e8d638b。

本次只新增冻结 cloth-v1-20260911 场景入口，求解数值源码保持 residual-toi-graph-v1 的实现。场景初态、三角拓扑 SHA-256 和全部 expected_scene 参数逐项对照原 SCENES.json。

| 场景 | base 秒 / Newton | 原 Graph 秒 / Newton | 纯 TOI 秒 / Newton | 补偿 TOI 秒 / Newton | 补偿相对 Graph |
|---|---:|---:|---:|---:|---:|
| sphere7 | 0.142 / 7 | 0.132 / 7 | 0.130 / 7 | 0.129 / 7 | 1.022× |
| bunny | 0.204 / 7 | 0.194 / 7 | 0.193 / 7 | 0.205 / 7 | 0.943× |
| table | 0.269 / 7 | 0.264 / 7 | 0.274 / 7 | 0.272 / 7 | 0.969× |
| stack10 | 0.839 / 10 | 0.835 / 10 | 0.822 / 10 | 0.851 / 10 | 0.982× |
| hang | 0.731 / 77 | 0.592 / 77 | 0.127 / 15 | 0.569 / 77 | 1.041× |

| 场景 / 方法 | PCG 总迭代 | 附加退出帧 | 相交标志帧 | 相对 base 最大帧位置 RMS (mm) |
|---|---:|---:|---:|---:|
| sphere7 / base | 85 | 0 | 0 | 0.000 |
| sphere7 / graph | 85 | 0 | 0 | 0.000 |
| sphere7 / toi | 85 | 0 | 0 | 0.000 |
| sphere7 / compensated | 85 | 0 | 0 | 0.000 |
| bunny / base | 112 | 0 | 0 | 0.000 |
| bunny / graph | 112 | 0 | 0 | 0.000 |
| bunny / toi | 112 | 0 | 0 | 0.000 |
| bunny / compensated | 112 | 0 | 0 | 0.000 |
| table / base | 114 | 0 | 0 | 0.000 |
| table / graph | 113 | 0 | 0 | 0.000 |
| table / toi | 114 | 0 | 0 | 0.000 |
| table / compensated | 114 | 0 | 0 | 0.000 |
| stack10 / base | 156 | 0 | 0 | 0.000 |
| stack10 / graph | 156 | 0 | 0 | 0.000 |
| stack10 / toi | 156 | 0 | 0 | 0.000 |
| stack10 / compensated | 156 | 0 | 0 | 0.000 |
| hang / base | 2326 | 0 | 0 | 0.000 |
| hang / graph | 2324 | 0 | 0 | 0.000 |
| hang / toi | 209 | 2 | 0 | 0.969 |
| hang / compensated | 2325 | 0 | 0 | 0.000 |

所有固定顶点漂移、有限状态、Newton 上限、预算界、图运行/回退记录见 summary.json。位置差是相对 base 轨迹的差异，base 不是严格收敛的真值；不能将差异直接解释为绝对误差。

单次结果只反映本次测量，不支持稳定性能或等精度结论。前四场景此前为 200 帧，本次 100 帧时长及碰撞阶段不同，不能直接与旧表总耗时比较或将旧表除以二。

原 GPU 相交查询为有限诊断，未全面验证布料自交、穿透或退化；不能仅凭标志为零宣称全部几何安全。

source_commit 和二进制 SHA-256 见每项 environment.json；全部原始文件 SHA-256 见 EVIDENCE_MANIFEST.json。
