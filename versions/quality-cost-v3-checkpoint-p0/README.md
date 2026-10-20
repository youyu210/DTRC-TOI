# quality-cost-v3-checkpoint-p0

这是 V3 同状态实验使用的首个已验证诊断二进制。它基于冻结 V2 数值实现，新增帧边界 checkpoint schema v3；未指定 checkpoint 参数时保持原求解路径。

已验证：全部登记字段逐字节往返、截断拒绝、5 次同状态恢复噪声、table/bunny 默认路径兼容、原始 RHS 审计，以及 CUDA Graph 冷启动/热运行行为。

源码标签：`quality-cost-v3-checkpoint-p0`。完整结果见 `experiments/quality-cost-v3/p0_v4/VALIDATION.md`。
