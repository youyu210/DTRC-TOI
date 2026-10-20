"""Run sequential paired tests on the unchanged eager PCG and dual graph.

Each recorded run is exactly the requested number of physics frames. Graph
preparation is included in timing. State exports are outside timing. Audit runs
include diagnostics in Newton/step time and must be kept separate from speed runs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(scene, method, repeat, frames, out, audit=0):
    run_dir = out / f"scene{scene}" / method / f"r{repeat:02d}"
    if run_dir.exists():
        raise RuntimeError(f"Refusing overwrite: {run_dir}")
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = f"scene{scene}-{method}-r{repeat:02d}"
    cmd = [str(EXE), "--scene", str(scene), "--benchmark-frames", str(frames),
           "--benchmark-output-dir", str(run_dir), "--export-states"]
    if method == "pcg_dual_graph":
        cmd.append("--pcg-dual-graph")
    if audit:
        cmd.extend(["--audit-pcg-frame", str(audit)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
    print(f"START {label} ({frames} frames)", flush=True)
    started = time.monotonic()
    with (log_dir / (label + ".stdout.log")).open("w") as stdout, \
         (log_dir / (label + ".stderr.log")).open("w") as stderr:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout,
                                   stderr=stderr, timeout=5400)
    if not run_dir.is_dir():
        raise RuntimeError(f"No output: {label}; exit={completed.returncode}")
    write_json(run_dir / "environment.json", {
        "command": cmd, "exe_sha256": digest(EXE), "returncode": completed.returncode,
        "process_wall_seconds": time.monotonic() - started,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parameter_sha256": digest(ROOT / "Assets/scene/parameterSetting.txt"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,temperature.gpu,clocks.sm,memory.used", "--format=csv,noheader"], text=True),
    })
    status = json.loads((run_dir / "status.json").read_text())
    if completed.returncode or not status["completed"] or status["frames_completed"] != frames:
        raise RuntimeError(f"Incomplete or invalid run: {label}: {status}")
    with (run_dir / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != frames or [int(r["frame"]) for r in rows] != list(range(1, frames + 1)):
        raise RuntimeError(f"Wrong frame sequence: {label}")
    if any(int(r["nonfinite_vertices"]) or int(r["newton_cap_hits"]) for r in rows):
        raise RuntimeError(f"Numerical failure: {label}")
    if method == "pcg_dual_graph" and (sum(int(r["direction_launches"]) for r in rows) == 0
                                      or sum(int(r["tail_launches"]) for r in rows) == 0
                                      or sum(int(r["fallbacks"]) for r in rows)):
        raise RuntimeError(f"Graphs missing or fallback: {label}")
    print(f"DONE {label}: Newton={sum(int(r['newton_iterations']) for r in rows)}; "
          f"time={sum(float(r['newton_ms']) for r in rows)/1000:.3f}s", flush=True)
    return run_dir


def load(path):
    with (path / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return {
        "path": str(path),
        "metadata": json.loads((path / "metadata.json").read_text()),
        "environment": json.loads((path / "environment.json").read_text()),
        "rows": rows,
        "newton": sum(int(r["newton_iterations"]) for r in rows),
        "pcg_iterations": sum(int(r["pcg_iterations"]) for r in rows),
        "pcg_solves": sum(int(r["pcg_solves"]) for r in rows),
        "newton_seconds": sum(float(r["newton_ms"]) for r in rows)/1000,
        "pcg_seconds": sum(float(r["pcg_ms"]) for r in rows)/1000,
        "step_seconds": sum(float(r["step_ms"]) for r in rows)/1000,
        "direction_launches": sum(int(r["direction_launches"]) for r in rows),
        "tail_launches": sum(int(r["tail_launches"]) for r in rows),
        "captures": sum(int(r["captures"]) for r in rows),
        "updates": sum(int(r["updates"]) for r in rows),
        "rebuilds": sum(int(r["rebuilds"]) for r in rows),
        "fallbacks": sum(int(r["fallbacks"]) for r in rows),
    }


def compare(a, b, frames):
    if digest(Path(a["path"]) / "initial.f64") != digest(Path(b["path"]) / "initial.f64"):
        raise RuntimeError("Initial states differ")
    for key in ("exe_sha256", "parameter_sha256"):
        if a["environment"][key] != b["environment"][key]:
            raise RuntimeError(f"Comparison differs: {key}")
    for key in ("scene_no", "vertex_count", "tetrahedron_count", "dt",
                "pcg_tolerance", "newton_tolerance", "gpu_name"):
        if a["metadata"][key] != b["metadata"][key]:
            raise RuntimeError(f"Different physical/configuration setting: {key}")
    n = a["metadata"]["vertex_count"]
    aa = np.memmap(Path(a["path"]) / "states.f64", dtype="<f8", mode="r", shape=(frames, n, 3))
    bb = np.memmap(Path(b["path"]) / "states.f64", dtype="<f8", mode="r", shape=(frames, n, 3))
    rms, peak = [], []
    for f in range(frames):
        delta = aa[f] - bb[f]
        rms.append(float(np.sqrt(np.mean(np.sum(delta*delta, axis=1)))))
        peak.append(float(np.max(np.abs(delta))))
    changed = [i+1 for i, (x, y) in enumerate(zip(a["rows"], b["rows"]))
               if x["newton_iterations"] != y["newton_iterations"]]
    return {
        "newton_speedup": a["newton_seconds"] / b["newton_seconds"],
        "pcg_speedup": a["pcg_seconds"] / b["pcg_seconds"],
        "step_speedup": a["step_seconds"] / b["step_seconds"],
        "newton_delta": b["newton"] - a["newton"],
        "newton_changed_frames": changed,
        "max_frame_position_rms_m": max(rms),
        "max_coordinate_diff_m": max(peak),
        "final_position_rms_m": rms[-1],
    }


def summarize(out, scenes, frames, repeats):
    result = {"frames_per_run": frames, "repeats": repeats,
              "base_commit": "c49982fac7767225c7775683734c89eb74081142", "scenes": []}
    lines = [f"# PCG 双段 CUDA Graph：{frames} 帧比较", "",
             "同一可执行文件，Graph 默认关闭。按源码 scene_no 编号；每次从相同初态启动。",
             "同步墙钟时间包含捕获/更新/重建；不含渲染、状态导出和旧版报告写盘。加速比为 base/Graph。", "",
             f"| 场景 | base 牛顿迭代（{repeats} 次） | Graph 牛顿迭代（{repeats} 次） | base 牛顿秒中位数 | Graph 牛顿秒中位数 | 牛顿加速比中位数 | PCG 加速比中位数 | 整步加速比中位数 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for scene in scenes:
        bases, graphs, pairs = [], [], []
        for repeat in range(1, repeats + 1):
            base = load(out / f"scene{scene}/base/r{repeat:02d}")
            graph = load(out / f"scene{scene}/pcg_dual_graph/r{repeat:02d}")
            pair = compare(base, graph, frames)
            bases.append(base)
            graphs.append(graph)
            pairs.append(pair)
        noise = [compare(bases[0], b, frames) for b in bases[1:]]
        row = {"scene_no": scene, "base": bases, "graph": graphs, "pairs": pairs,
               "base_repeat_noise": noise,
               "median_newton_speedup": statistics.median(p["newton_speedup"] for p in pairs),
               "median_pcg_speedup": statistics.median(p["pcg_speedup"] for p in pairs),
               "median_step_speedup": statistics.median(p["step_speedup"] for p in pairs)}
        result["scenes"].append(row)
        lines.append(f"| {scene} | {' / '.join(str(b['newton']) for b in bases)} | "
                     f"{' / '.join(str(g['newton']) for g in graphs)} | "
                     f"{statistics.median(b['newton_seconds'] for b in bases):.3f} | "
                     f"{statistics.median(g['newton_seconds'] for g in graphs):.3f} | "
                     f"{row['median_newton_speedup']:.3f}× | {row['median_pcg_speedup']:.3f}× | "
                     f"{row['median_step_speedup']:.3f}× |")
    lines.extend(["", "## 状态与数值检查", "",
                  "GPU SpMV 保留原来的 atomic_add；不同运行的累加顺序可能带来浮点差异。报告轨迹差及 base 重复噪声，不能据公式一致宣称逐位相同。", ""])
    for row in result["scenes"]:
        lines.append(f"- 场景 {row['scene_no']}：Graph 最大逐帧顶点 RMS 差 "
                     f"{max(p['max_frame_position_rms_m'] for p in row['pairs'])*1000:.6f} mm；"
                     f"base 重复最大差 {max((p['max_frame_position_rms_m'] for p in row['base_repeat_noise']), default=0)*1000:.6f} mm。"
                     f"所有运行完整且顶点有限、无 Newton 上限命中；方向图/尾段图均启动，回退为零。")
    write_json(out / "summary.json", result)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenes", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--audit-frame", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not args.summarize_only:
        for repeat in range(1, args.repeats + 1):
            methods = ("base", "pcg_dual_graph") if repeat % 2 else ("pcg_dual_graph", "base")
            for scene in args.scenes:
                for method in methods:
                    run(scene, method, repeat, args.frames, out, args.audit_frame)
    summarize(out, args.scenes, args.frames, args.repeats)


if __name__ == "__main__":
    main()
