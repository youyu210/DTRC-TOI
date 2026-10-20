# P0 帧边界检查点字段清单

格式：`stiffgipc-step-boundary-v3`。检查点只允许在完整物理步结束后写入，加载时先按正常场景流程初始化静态系统，再覆盖下列动态状态。

## FEM 动态字段

| 文件 | 元素 | 用途 |
|---|---|---|
| `fem_vertexes.bin` | `vertex_count × double3` | 当前位置 |
| `fem_o_vertexes.bin` | `vertex_count × double3` | 上一物理步位置 |
| `fem_velocities.bin` | `vertex_count × double3` | 当前速度 |
| `fem_x_tilde.bin` | `vertex_count × double3` | 惯性预测位置 |
| `fem_targets.bin` | `soft_count × double3` | 软约束目标 |
| `fem_boundary.bin` | `vertex_count × int32` | 边界类型 |
| `fem_apply_gravity.bin` | `vertex_count × int32` | 重力开关 |

## ABD 动态字段

`abd_q.bin`、`abd_q_temp.bin`、`abd_q_tilde.bin`、`abd_q_prev.bin`、`abd_q_v.bin`、`abd_dq.bin`，每项均为 `abd_body_count × Vector12`。

## 有序接触状态

| 文件/清单项 | 含义 |
|---|---|
| `contact_pairs.bin` | 帧边界有序自碰撞对 |
| `contact_matrix_indices.bin` | 与上述顺序匹配的矩阵索引 |
| `contact_counts.bin` | 设备侧五元工作计数，逐字节恢复 |
| `contact_host_counts` | 主机侧五元计数；第 0 项是下一步使用的总接触数 |
| `ground_pairs.bin` / `ground_count.bin` | 有序地面接触及设备计数 |
| `ground_contact_count` | 主机侧地面接触数 |

设备计数与主机计数必须分开保存。物理步末的 CCD 会复用设备计数第 0 项，不能用它覆盖主机总接触数。

## 标量和派生状态

清单保存 `physical_frame`、`ipc_dt`、`kappa`、`animation_sub_rate`、`animation_full_rate` 及场景尺寸计数。加载后重建 BVH，再恢复有序接触数组；摩擦集合在下一物理步按恢复的接触顺序重建；PCG CUDA Graph 在恢复进程首次求解时重新捕获。

## P0 修正记录

- 初版只重建接触对，恢复轨迹超出逐字节一致性。
- 第二版在恢复接触后重建 BVH，BVH 工作区覆盖了接触数组。
- 第三次尝试误把设备工作计数第 0 项当作主机总接触数。
- schema v3 先重建 BVH，再恢复有序接触，并独立恢复主机/设备计数；该版本通过最终 P0。

失败样本保留在 `p0`、`p0_v2`、`p0_v3`；通过样本和报告位于 `p0_v4`。
