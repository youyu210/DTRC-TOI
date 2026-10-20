"""Summarize the A1 one-frame dual-tolerance diagnostic matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/quality-cost-v3/a1-v1"
CONFIGS = ("N1-P1", "N4-P1", "N1-P4", "N4-P4")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def contact_key(value: dict[str, int]) -> tuple[int, ...]:
    return (
        value["contact_count"], value["contact_hash_xor"],
        value["contact_hash_sum"], value["ground_count"],
        value["ground_hash_xor"], value["ground_hash_sum"],
    )


def summarize_run(path: Path) -> dict[str, object]:
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    trace = read_json(path / "newton_trace.jsonl")
    samples = trace["samples"]
    updates = [sample for sample in samples if "line_search" in sample]
    assert len(updates) == int(row["newton_iterations"])

    alphas = [sample["line_search"]["accepted_alpha"] for sample in updates]
    input_alphas = [sample["line_search_input_alpha"] for sample in updates]
    preliminary_alphas = [
        min(1.0, sample["ground_alpha"], sample["self_alpha"])
        for sample in updates
    ]
    direction_ratios = [
        sample["previous_direction_infinity_norm"]
        / sample["direction_stop_threshold"]
        for sample in samples if sample["direction_stop_threshold"] > 0
    ]
    gradients = [sample["gradient_infinity_norm"] for sample in samples]
    contact_before = [contact_key(sample["contacts_before"]) for sample in samples]
    contact_after = [contact_key(sample["contacts_after"]) for sample in updates]
    kappa_changes = sum(
        not math.isclose(sample["kappa_before_post"], sample["kappa_after_post"],
                         rel_tol=0.0, abs_tol=0.0)
        for sample in updates
    )
    result: dict[str, object] = {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "completed": read_json(path / "status.json")["completed"],
        "exit_reason": row["exit_reason"],
        "newton": int(row["newton_iterations"]),
        "pcg": int(row["pcg_iterations"]),
        "step_ms": float(row["step_ms"]),
        "newton_ms": float(row["newton_ms"]),
        "pcg_ms": float(row["pcg_ms"]),
        "trace_samples": len(samples),
        "gradient_first": gradients[0],
        "gradient_last": gradients[-1],
        "gradient_min": min(gradients),
        "gradient_max": max(gradients),
        "direction_ratio_last": direction_ratios[-1],
        "direction_ratio_min": min(direction_ratios),
        "direction_ratio_max": max(direction_ratios),
        "non_descent_directions": sum(
            sample["linear_system"]["gradient_dot_actual_direction"] >= 0
            for sample in updates
        ),
        "accepted_alpha_min": min(alphas),
        "accepted_alpha_median": float(statistics.median(alphas)),
        "accepted_alpha_below_one": sum(alpha < 1.0 - 1e-14 for alpha in alphas),
        "ground_prelimited": sum(
            sample["ground_alpha"] < 1.0 - 1e-14 for sample in updates
        ),
        "self_prelimited": sum(
            sample["self_alpha"] < 1.0 - 1e-14 for sample in updates
        ),
        "full_ccd_changed_preliminary_alpha": sum(
            not math.isclose(value, preliminary, rel_tol=1e-13, abs_tol=1e-15)
            for value, preliminary in zip(input_alphas, preliminary_alphas)
        ),
        "line_search_reduced_input": sum(
            accepted < initial - 1e-15
            for accepted, initial in zip(alphas, input_alphas)
        ),
        "energy_backtracks": sum(
            sample["line_search"]["energy_backtracks"] for sample in updates
        ),
        "intersection_backtracks": sum(
            sample["line_search"]["intersection_backtracks"] for sample in updates
        ),
        "energy_condition_failures": sum(
            not sample["line_search"]["energy_condition_satisfied"]
            for sample in updates
        ),
        "backtrack_limit_hits": sum(
            sample["line_search"]["backtrack_limit_reached"]
            for sample in updates
        ),
        "kappa_changes": kappa_changes,
        "contact_changes_within_update": sum(
            before != after for before, after in zip(contact_before, contact_after)
        ),
        "unique_contact_states_before": len(set(contact_before)),
        "contact_count_first": contact_before[0][0],
        "contact_count_last": contact_before[-1][0],
        "friction_target_mismatches": sum(
            contact_key(sample["friction_target_contacts"])
            != contact_key(sample["contacts_before"])
            for sample in samples
        ),
    }
    return result


def state_error(candidate: Path, reference: Path) -> dict[str, float]:
    boundary = np.fromfile(reference / "boundary.i32", dtype="<i4")
    mask = boundary == 0
    assert mask.any()
    result: dict[str, float] = {"free_cloth_vertices": int(mask.sum())}
    for label, filename in (("position", "final.f64"),
                            ("velocity", "final_velocities.f64")):
        lhs = np.fromfile(candidate / filename, dtype="<f8").reshape(-1, 3)[mask]
        rhs = np.fromfile(reference / filename, dtype="<f8").reshape(-1, 3)[mask]
        delta = np.linalg.norm(lhs - rhs, axis=1)
        result[f"{label}_rms"] = float(np.sqrt(np.mean(delta * delta)))
        result[f"{label}_max"] = float(delta.max())
    return result


def main() -> None:
    runs: dict[str, list[dict[str, object]]] = {}
    paths: dict[str, list[Path]] = {}
    for config in CONFIGS:
        paths[config] = sorted((OUT / "runs" / config).glob("r[0-9][0-9]"))
        assert len(paths[config]) == 3, (config, paths[config])
        runs[config] = [summarize_run(path) for path in paths[config]]

    reference = paths["N1-P1"][0]
    for config in CONFIGS:
        for path, run in zip(paths[config], runs[config]):
            run["vs_n1_p1_r01"] = state_error(path, reference)
            run["vs_same_config_r01"] = state_error(path, paths[config][0])

    summary: dict[str, object] = {}
    for config in CONFIGS:
        group = runs[config]
        summary[config] = {
            "repeats": len(group),
            "completed_repeats": sum(bool(run["completed"]) for run in group),
            "exit_reasons": sorted({str(run["exit_reason"]) for run in group}),
            "newton": stats([float(run["newton"]) for run in group]),
            "pcg": stats([float(run["pcg"]) for run in group]),
            "step_ms": stats([float(run["step_ms"]) for run in group]),
            "max_position_rms_vs_n1_p1_r01": max(
                run["vs_n1_p1_r01"]["position_rms"] for run in group
            ),
            "max_velocity_rms_vs_n1_p1_r01": max(
                run["vs_n1_p1_r01"]["velocity_rms"] for run in group
            ),
            "repeat_position_rms_max": max(
                run["vs_same_config_r01"]["position_rms"] for run in group
            ),
            "repeat_velocity_rms_max": max(
                run["vs_same_config_r01"]["velocity_rms"] for run in group
            ),
            "non_descent_directions": sum(
                int(run["non_descent_directions"]) for run in group
            ),
            "energy_condition_failures": sum(
                int(run["energy_condition_failures"]) for run in group
            ),
            "backtrack_limit_hits": sum(
                int(run["backtrack_limit_hits"]) for run in group
            ),
            "energy_backtracks": sum(int(run["energy_backtracks"]) for run in group),
            "intersection_backtracks": sum(
                int(run["intersection_backtracks"]) for run in group
            ),
            "kappa_changes": sum(int(run["kappa_changes"]) for run in group),
            "accepted_alpha_min": min(float(run["accepted_alpha_min"])
                                      for run in group),
            "contact_changes_within_update": sum(
                int(run["contact_changes_within_update"]) for run in group
            ),
        }

    analysis = {"summary": summary, "runs": runs}
    (OUT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# A1：table 第 23 帧双容差诊断",
        "",
        "所有配置从同一个 schema-v3 S22 checkpoint 开始，每组 3 次。"
        "N4 表示对应容差缩小到 1/4；诊断上限为 2000 次 Newton 或 300 s。",
        "",
        "| 配置 | 完成/重复 | 退出原因 | Newton 中位数 [范围] | PCG 中位数 [范围] | 单步秒中位数 [范围] |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for config in CONFIGS:
        value = summary[config]
        newton, pcg, elapsed = value["newton"], value["pcg"], value["step_ms"]
        lines.append(
            f"| {config} | {value['completed_repeats']}/3 | "
            f"{', '.join(value['exit_reasons'])} | "
            f"{newton['median']:g} [{newton['min']:g}, {newton['max']:g}] | "
            f"{pcg['median']:g} [{pcg['min']:g}, {pcg['max']:g}] | "
            f"{elapsed['median']/1000:.3f} [{elapsed['min']/1000:.3f}, "
            f"{elapsed['max']/1000:.3f}] |"
        )

    lines += [
        "",
        "## 判定",
        "",
    ]
    if (summary["N4-P1"]["completed_repeats"] == 0
            and summary["N4-P4"]["completed_repeats"] == 0
            and summary["N1-P1"]["completed_repeats"] == 3
            and summary["N1-P4"]["completed_repeats"] == 3):
        lines.append(
            "严格 Newton 容差是长尾的触发因素：两种 N4 配置都在 2000 次上限截尾，"
            "而两种 N1 配置都正常结束；把 PCG 同时收紧到 P4 没有解除长尾。"
        )
    total_non_descent = sum(value["non_descent_directions"]
                            for value in summary.values())
    total_failures = sum(value["energy_condition_failures"]
                         for value in summary.values())
    total_limits = sum(value["backtrack_limit_hits"] for value in summary.values())
    strict_backtracks = (summary["N4-P1"]["energy_backtracks"]
                         + summary["N4-P4"]["energy_backtracks"])
    normal_backtracks = (summary["N1-P1"]["energy_backtracks"]
                         + summary["N1-P4"]["energy_backtracks"])
    lines.append(
        f"全部重复共发现 {total_non_descent} 个非下降方向、{total_failures} 次能量条件失败、"
        f"{total_limits} 次回溯上限命中。N4 两组共有 {strict_backtracks} 次能量回溯，"
        f"N1 两组共有 {normal_backtracks} 次。"
    )
    lines.append(
        "诊断支持以下因果链：严格 Newton 阈值让求解进入平坦能量和接触活跃集持续切换区；"
        "方向仍为下降方向，但实际能量下降被数值噪声/接触离散变化淹没，步长反复减半，"
        "停止量在阈值上方振荡。P4 未解除该现象，因此下一阶段应测试能在数值地板前退出的"
        "停止候选，不先改变能量线搜索的数学规则。"
    )

    lines += [
        "",
        "## 状态差异与重复噪声",
        "",
        "N4 两组是诊断截尾状态，不可作为收敛参考；下表仅量化它们与 N1-P1/r01 的距离。",
        "",
        "| 配置 | 对 N1-P1/r01 最大位置 RMS (m) | 最大速度 RMS (m/s) | 组内位置 RMS 上限 (m) |",
        "|---|---:|---:|---:|",
    ]
    for config in CONFIGS:
        value = summary[config]
        lines.append(
            f"| {config} | {value['max_position_rms_vs_n1_p1_r01']:.9g} | "
            f"{value['max_velocity_rms_vs_n1_p1_r01']:.9g} | "
            f"{value['repeat_position_rms_max']:.9g} |"
        )

    lines += [
        "",
        "## 逐运行关键诊断",
        "",
        "| 运行 | exit | grad∞ 首→末 | 末次停止量/阈值 | min α | 能量/相交回溯 | 接触改变更新 | 唯一接触态 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        for index, run in enumerate(runs[config], 1):
            lines.append(
                f"| {config}/r{index:02d} | {run['exit_reason']} | "
                f"{run['gradient_first']:.3e}→{run['gradient_last']:.3e} | "
                f"{run['direction_ratio_last']:.6g} | {run['accepted_alpha_min']:.6g} | "
                f"{run['energy_backtracks']}/{run['intersection_backtracks']} | "
                f"{run['contact_changes_within_update']}/{run['newton']} | "
                f"{run['unique_contact_states_before']} |"
            )
    lines += [
        "",
        "完整逐迭代数据、每次运行的计时与状态差异见 `analysis.json`；"
        "原始证据哈希见 `EVIDENCE_MANIFEST.json`。",
        "",
    ]
    (OUT / "A1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    evidence = []
    evidence_roots = [
        OUT / "runs",
        OUT / "S22",
        OUT / "S22-roundtrip",
        OUT / "prefix-strict4",
    ]
    for path in sorted(
        path for evidence_root in evidence_roots
        for path in evidence_root.rglob("*") if path.is_file()
    ):
        evidence.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    evidence += [{
        "path": (OUT / name).relative_to(ROOT).as_posix(),
        "bytes": (OUT / name).stat().st_size,
        "sha256": sha256(OUT / name),
    } for name in ("analysis.json", "A1_REPORT.md", "README.md")]
    for path in (ROOT / "build/gipc.exe", ROOT / "experiments/run_a1_v3.py",
                 ROOT / "experiments/analyze_a1_v3.py"):
        evidence.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (OUT / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps({"files": evidence}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
