# P0 完整检查点自检

该目录保存 schema v3 的通过样本。原始大数组和检查点二进制保留在本地并由 `EVIDENCE_MANIFEST.json` 登记；仓库提交报告、CSV/JSON 元数据和哈希清单，避免把约 180 MiB 的状态数组写入 Git 历史。

验证范围：

- 所有登记检查点字段加载后立即再保存，逐文件 SHA-256 一致；
- 截断 `fem_vertexes.bin` 8 字节的副本被加载器拒绝；
- 同一 table 第 1 帧边界检查点恢复 5 次，并与连续第 2 帧比较；
- table/bunny 各 5 帧比较冻结 V2 与新增诊断默认关闭的构建；
- table/bunny 各 1 帧比较原始 RHS 与 eager 对照；
- 明确记录首次 CUDA Graph 捕获和后续热运行的捕获次数。

结论见 `VALIDATION.md`，完整数值见 `analysis.json`。
