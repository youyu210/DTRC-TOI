"""Sequential three-version tests, strict references, and residual-gate ablation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
MODES = ["original", "toi", "compensated"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(path, scene, mode, frames=100, epsilon=.001, minimum=6, strict=False, trace=False):
    path = path.resolve()
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(EXE), "--scene", str(scene), "--pcg-dual-graph", "--termination", mode,
               "--benchmark-frames", str(frames), "--benchmark-output-dir", str(path),
               "--toi-threshold", str(epsilon), "--toi-min-updates", str(minimum), "--export-states"]
    if strict:
        factor = .1 if strict is True else float(strict)
        command += ["--newton-tolerance-scale", str(factor), "--pcg-tolerance-scale", str(factor)]
    if trace:
        command += ["--trace-termination"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
    label = path.relative_to(ROOT).as_posix()
    print(f"START {label}", flush=True)
    start = time.monotonic()
    with (path.parent / (path.name + ".stdout.log")).open("w") as stdout, \
         (path.parent / (path.name + ".stderr.log")).open("w") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=5400)
    if not path.exists():
        raise RuntimeError(f"No output; exit={result.returncode}: {label}")
    write_json(path / "environment.json", {
        "command": command, "returncode": result.returncode, "wall_seconds": time.monotonic()-start,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "exe_sha256": sha(EXE), "parameters_sha256": sha(ROOT / "Assets/scene/parameterSetting.txt"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,temperature.gpu,clocks.sm", "--format=csv,noheader"], text=True)})
    data = load(path)
    status = json.loads((path / "status.json").read_text())
    if result.returncode or not status["completed"] or status["frames_completed"] != frames:
        raise RuntimeError(f"Incomplete run: {label}: {status}")
    if len(data["rows"]) != frames or [int(r["frame"]) for r in data["rows"]] != list(range(1, frames+1)):
        raise RuntimeError(f"Wrong frame sequence: {label}")
    for row in data["rows"]:
        if any(int(row[key]) for key in ["newton_cap_hits", "nonfinite_vertices", "fallbacks", "intersection"]):
            raise RuntimeError(f"Safety/runtime issue: {label}: {row}")
        if not int(row["bounds_valid"]) or not math.isfinite(float(row["exit_residual"])):
            raise RuntimeError(f"Invalid residual/budget: {label}: {row}")
        if row["exit_reason"] in ["toi", "gated", "compensated"]:
            if int(row["newton_iterations"]) < minimum or not int(row["budget_valid"]):
                raise RuntimeError("Auxiliary guard bypass")
            if float(row["exit_beta"]) > epsilon + 1e-12:
                raise RuntimeError("Beta bound at exit")
            if row["exit_reason"] == "compensated" and float(row["exit_budget"]) > epsilon + 1e-12:
                raise RuntimeError("Compensated budget bound at exit")
    if not data["direction_launches"] or not data["tail_launches"]:
        raise RuntimeError("Graphs missing")
    if trace:
        verify_trace(path)
    print(f"DONE {label}: Newton={data['newton']}; seconds={data['newton_seconds']:.3f}; "
          f"auxiliary exits={data['auxiliary_exits']}", flush=True)
    return data


def load(path):
    path = Path(path)
    rows = list(csv.DictReader((path / "frames.csv").open()))
    def total(key):
        return sum(float(r[key]) for r in rows)
    return {"path": str(path), "rows": rows,
            "metadata": json.loads((path / "metadata.json").read_text()),
            "environment": json.loads((path / "environment.json").read_text()),
            "newton": int(total("newton_iterations")), "pcg_iterations": int(total("pcg_iterations")),
            "newton_seconds": total("newton_ms")/1000, "pcg_seconds": total("pcg_ms")/1000,
            "step_seconds": total("step_ms")/1000,
            "direction_launches": int(total("direction_launches")), "tail_launches": int(total("tail_launches")),
            "captures": int(total("captures")), "updates": int(total("updates")),
            "fallbacks": int(total("fallbacks")), "target_changes": int(total("target_changes")),
            "auxiliary_exits": sum(r["exit_reason"] in ["toi", "gated", "compensated"] for r in rows),
            "median_exit_relative_initial": statistics.median(float(r["exit_relative_initial"]) for r in rows),
            "max_exit_relative_initial": max(float(r["exit_relative_initial"]) for r in rows)}


def verify_trace(path):
    metadata = json.loads((path / "metadata.json").read_text())
    minimum = metadata["toi_min_updates"]
    for line in (path / "termination_trace.jsonl").read_text().splitlines():
        trace = json.loads(line)
        old = None
        for sample in trace["samples"]:
            if not sample["active"]:
                assert sample["accepted_updates"] < minimum - 1
                continue
            if not sample["valid"]:
                raise RuntimeError("Invalid trace")
            z, r, beta = sample["budget"], sample["normalized"], sample["beta"]
            tol = 1e-12 * max(1, z, r, beta)
            assert z + tol >= r and z + tol >= beta
            if old is None:
                assert sample["accepted_updates"] == minimum - 1
                assert z == beta == 1
                assert sample["reference"] == max(sample["residual"], metadata["residual_floor"])
            else:
                assert sample["reference"] == old["reference"]
                a = sample["alpha"]
                assert 0 <= a <= 1
                excess = max(0, r - (1-a)*old["normalized"])
                assert math.isclose(sample["excess"], excess, rel_tol=1e-12, abs_tol=1e-14)
                assert math.isclose(z, (1-a)*old["budget"]+excess, rel_tol=1e-12, abs_tol=1e-14)
                assert math.isclose(beta, (1-a)*old["beta"], rel_tol=1e-12, abs_tol=1e-14)
            old = sample


def positions(data, filename):
    meta = data["metadata"]
    return np.memmap(Path(data["path"]) / filename, dtype="<f8", mode="r",
                     shape=(meta["frames"], meta["vertex_count"], 3))


def errors(data, reference):
    a, b = Path(data["path"]), Path(reference["path"])
    assert sha(a / "initial.f64") == sha(b / "initial.f64")
    assert data["environment"]["exe_sha256"] == reference["environment"]["exe_sha256"]
    pos_a, pos_b = positions(data, "states.f64"), positions(reference, "states.f64")
    # Legacy shared velocity storage is zero for ABD vertices; ABD's real
    # velocity lives in q_v. J is constant, so world velocities are the accepted
    # position differences / dt for BOTH ABD and FEM. Use this consistent
    # measure rather than silently counting ABD velocity error as zero.
    previous_a = np.fromfile(a / "initial.f64", dtype="<f8").reshape(-1, 3)
    previous_b = np.fromfile(b / "initial.f64", dtype="<f8").reshape(-1, 3)
    dt = data["metadata"]["dt"]
    assert dt == reference["metadata"]["dt"]
    pos_rms, vel_rms, maximum = [], [], []
    for pa, pb in zip(pos_a, pos_b):
        pos_rms.append(float(np.sqrt(np.mean(np.sum((pa-pb)**2, axis=1)))))
        difference_v = ((pa-previous_a)-(pb-previous_b)) / dt
        vel_rms.append(float(np.sqrt(np.mean(np.sum(difference_v**2, axis=1)))))
        maximum.append(float(np.max(np.abs(pa-pb))))
        previous_a, previous_b = pa, pb
    return {"max_position_rms_m": max(pos_rms), "final_position_rms_m": pos_rms[-1],
            "max_coordinate_error_m": max(maximum), "max_velocity_rms_m_s": max(vel_rms),
            "final_velocity_rms_m_s": vel_rms[-1]}


def geometry(data):
    path = Path(data["path"])
    positions_data = positions(data, "states.f64")
    initial = np.fromfile(path / "initial.f64", dtype="<f8").reshape(-1, 3)
    tets = np.fromfile(path / "tetrahedra.u32", dtype="<u4").reshape(-1, 4)
    boundary = np.fromfile(path / "boundary.i32", dtype="<i4")
    start = data["metadata"]["fem_point_offset"]
    count = data["metadata"]["fem_point_count"]
    fixed = np.arange(start, start+count)[boundary[start:start+count] != 0]
    def determinant(p):
        return np.einsum('ij,ij->i', p[tets[:,1]]-p[tets[:,0]],
                         np.cross(p[tets[:,2]]-p[tets[:,0]], p[tets[:,3]]-p[tets[:,0]]))
    rest_det = determinant(initial)
    valid = np.abs(rest_det) > 1e-20
    minimum, flips, drift = 1., 0, 0.
    for p in positions_data:
        if np.any(valid):
            ratios = determinant(p)[valid] / rest_det[valid]
            minimum = min(minimum, float(np.min(ratios)))
            flips = max(flips, int(np.count_nonzero(ratios <= 0)))
        if len(fixed):
            drift = max(drift, float(np.max(np.abs(p[fixed]-initial[fixed]))))
    return {"minimum_signed_volume_ratio": minimum, "maximum_flipped_tets": flips,
            "fixed_fem_max_coordinate_drift_m": drift, "fixed_fem_vertices": len(fixed),
            "initial_degenerate_tets": int(np.count_nonzero(~valid))}


def summarize(out, frames=100, repeats=3):
    result = {"frames": frames, "repeats": repeats, "scenes": []}
    lines = ["# Graph 上的三版本 TOI 比较", "",
             "每场景每方法 100 帧、三次，Kmin=6、epsilon=1e-3。所有方法启用相同 PCG 双段图。",
             "同步墙钟计时包含残差归约、图捕获/更新及延后验收的梯度/Hessian；不含状态导出、安全诊断和轨迹分析。", "",
             "| 场景 | 版本 | 牛顿迭代（三次） | 牛顿秒中位数 | 对原停止规则的配对加速比中位数 | 附加退出帧数（三次） | 对严格参考最大位置 RMS（mm） |",
             "|---|---|---|---:|---:|---|---:|"]
    for scene in [1, 3, 5]:
        strict = load(out / f"reference-2x/scene{scene}")
        originals = [load(out / f"main/scene{scene}/original/r{i:02d}") for i in range(1, repeats+1)]
        modes = {}
        for mode in MODES:
            runs = [load(out / f"main/scene{scene}/{mode}/r{i:02d}") for i in range(1, repeats+1)]
            for item in runs:
                item["strict_errors"] = errors(item, strict)
                item["geometry"] = geometry(item)
            speedups = [a["newton_seconds"]/b["newton_seconds"] for a, b in zip(originals, runs)]
            modes[mode] = {"runs": runs, "median_newton_speedup": statistics.median(speedups)}
            lines.append(f"| {scene} | {mode} | {' / '.join(str(r['newton']) for r in runs)} | "
                         f"{statistics.median(r['newton_seconds'] for r in runs):.3f} | "
                         f"{statistics.median(speedups):.3f}× | "
                         f"{' / '.join(str(r['auxiliary_exits']) for r in runs)} | "
                         f"{max(r['strict_errors']['max_position_rms_m'] for r in runs)*1000:.3f} |")
        ablations = []
        for path in sorted((out / f"ablation/scene{scene}").glob('*')):
            if path.is_dir():
                item = load(path)
                item["strict_errors"] = errors(item, strict)
                item["geometry"] = geometry(item)
                ablations.append(item)
        strict["geometry"] = geometry(strict)
        repeat_noise = [errors(originals[0], item) for item in originals[1:]]
        result["scenes"].append({"scene": scene, "reference": strict, "modes": modes,
                                 "original_repeat_noise": repeat_noise, "ablations": ablations})
    lines += ["", "## 当前残差门控与阈值扫描", "",
              "以下每项各 100 帧、一次，仅用于消融与阈值探索，不据此断言普遍效率收益。", "",
              "| 场景 | 退出策略 | epsilon | 牛顿迭代 | 牛顿秒 | 附加退出帧数 | 最大位置 RMS（mm） | 最大速度 RMS（m/s） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scene in result["scenes"]:
        for item in scene["ablations"]:
            meta = item["metadata"]
            lines.append(f"| {scene['scene']} | {meta['termination']} | {meta['toi_threshold']} | "
                         f"{item['newton']} | {item['newton_seconds']:.3f} | {item['auxiliary_exits']} | "
                         f"{item['strict_errors']['max_position_rms_m']*1000:.3f} | "
                         f"{item['strict_errors']['max_velocity_rms_m_s']:.6f} |")
    lines += ["", "完整统一严格参考保留原停止规则，Newton 与 PCG 容差各缩小 2 倍；它是更严格的数值参考，不能视为精确物理解。",
              "另留存 10 倍严格探索：场景 1、3 完成，场景 5 第 50 帧需 3792 次 Newton、226.52 秒，第 51 帧继续长时间求解后中断；该不完整运行不参与误差对照。",
              "位置/速度误差受原子累加的重复运行差异及接触非线性影响；summary.json 同时保留 original 的重复波动。",
              "补偿界只约束通过 compensated 退出的状态；保留的原方向停止路径不承诺该界。详见 VALIDATION.md。"]
    write_json(out / "summary.json", result)
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume-after-reference", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.smoke:
        for scene in [1, 3, 5]:
            for mode in MODES + ["gated"]:
                run(out / f"scene{scene}/{mode}", scene, mode, frames=2, minimum=1, trace=True)
        return
    if not args.summarize_only and not args.resume_after_reference:
        for repeat in range(1, 4):
            order = MODES[repeat-1:] + MODES[:repeat-1]
            for scene in [1, 3, 5]:
                for mode in order:
                    run(out / f"main/scene{scene}/{mode}/r{repeat:02d}", scene, mode)
    if not args.summarize_only:
        for scene in [1, 3, 5]:
            reference_path = out / f"reference-2x/scene{scene}"
            if not reference_path.exists():
                run(reference_path, scene, "original", strict=.5)
            gate_path = out / f"ablation/scene{scene}/gated-e0.001"
            if not gate_path.exists():
                run(gate_path, scene, "gated")
            for epsilon in [.01, .1]:
                for mode in ["toi", "gated", "compensated"]:
                    path = out / f"ablation/scene{scene}/{mode}-e{epsilon}"
                    if not path.exists():
                        run(path, scene, mode, epsilon=epsilon)
    summarize(out)


if __name__ == "__main__":
    main()
