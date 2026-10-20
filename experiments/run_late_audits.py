"""Separate late-frame same-system checks; never overlap speed experiments."""
from pathlib import Path
import argparse
import json
import math
from run_comparison import run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    results = []
    for scene in [1, 3, 5]:
        path = run(scene, "pcg_dual_graph", 1, 100, output, audit=100)
        audits = json.loads((path / "audits.json").read_text())
        if len(audits) != 1 or audits[0]["frame"] != 100:
            raise RuntimeError(f"Missing late-frame audit: scene {scene}")
        numeric = [value for value in audits[0].values()]
        if any(value is None or not math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Invalid audit: scene {scene}")
        results.append({"scene_no": scene, "path": str(path), **audits[0]})
    write_json(output / "AUDITS.json", results)
    lines = ["# 第 100 帧的独立 PCG 数值审计", "",
             "对 Graph 已实际求解的同一 A,b，再调用保留的原始 PCG；实际解未替换。",
             "审计运行的牛顿/整步时间包含额外诊断，不用于速度比较。", "",
             "| 场景 | Graph / 原始 PCG 迭代 | 方向相对 L2 差 | Graph 真实相对残差 | 原始真实相对残差 |",
             "|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['scene_no']} | {row['actual_iterations']} / {row['eager_iterations']} | "
                     f"{row['relative_l2_direction_diff']:.6e} | "
                     f"{row['actual_true_relative_residual']:.6e} | "
                     f"{row['eager_true_relative_residual']:.6e} |")
    (output / "AUDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
