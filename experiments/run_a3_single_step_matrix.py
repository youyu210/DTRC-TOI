"""Run the frozen A2/A3 paired single-step matrix from exact checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
CHECKPOINT_ROOT = ROOT / "experiments/quality-cost-v3/a2-checkpoints-v1/checkpoints"
SELECTION = ROOT / "experiments/quality-cost-v3/a2-selection-v1/SELECTED_STATES.json"
OUT = ROOT / "experiments/quality-cost-v3/a3-single-step-v2"
STAGE_SECONDS = 3600.0
CONFIGS = {
    "graph_n1": ("original", 1.0, 1.0, 0.001),
    "graph_n2": ("original", 2.0, 1.0, 0.001),
    "graph_n4": ("original", 4.0, 1.0, 0.001),
    "graph_n8": ("original", 8.0, 1.0, 0.001),
    "toi": ("toi", 1.0, 1.0, 0.03),
    "comp001": ("compensated", 1.0, 1.0, 0.001),
    "comp01": ("compensated", 1.0, 1.0, 0.01),
    "comp03": ("compensated", 1.0, 1.0, 0.03),
    "comp10": ("compensated", 1.0, 1.0, 0.1),
    "gate01": ("gated", 1.0, 1.0, 0.01),
    "gate03": ("gated", 1.0, 1.0, 0.03),
    "gate10": ("gated", 1.0, 1.0, 0.1),
    "strict2": ("original", 0.5, 0.5, 0.001),
    "strict4": ("original", 0.25, 0.25, 0.001),
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def states() -> list[dict[str, object]]:
    frozen = read_json(SELECTION)
    assert frozen["status"] == "frozen_before_candidate_runs"
    by_category = {"quiet_motion": 0, "contact_onset": 1,
                   "dense_contact": 2, "contact_rearrangement": 3}
    return sorted(frozen["selected_states"],
                  key=lambda state: (by_category[state["category"]], state["scene"]))


def validate(path: Path, state: dict[str, object], config: str) -> dict[str, object]:
    status = read_json(path / "status.json")
    metadata = read_json(path / "metadata.json")
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    mode, newton_scale, pcg_scale, epsilon_r = CONFIGS[config]
    assert metadata["termination"] == mode
    assert metadata["newton_tolerance_scale"] == newton_scale
    assert metadata["pcg_tolerance_scale"] == pcg_scale
    assert metadata["residual_threshold"] == epsilon_r
    assert metadata["initial_physical_frame"] == state["checkpoint_frame"]
    assert metadata["checkpoint_in"].replace("\\", "/").endswith(
        f"checkpoints/{state['scene']}/S{state['checkpoint_frame']}"
    )
    assert int(row["nonfinite_vertices"]) == 0
    assert int(row["newton_cap_hits"]) == 0
    assert all(math.isfinite(float(row[key])) for key in
               ("step_ms", "newton_ms", "exit_residual"))
    capped = int(row["diagnostic_budget_hits"]) == 1
    if capped:
        assert status == {"completed": False, "frames_completed": 1,
                          "finite_state": False}
        assert row["exit_reason"] == "diagnostic_budget"
    else:
        assert status == {"completed": True, "frames_completed": 1,
                          "finite_state": True}
    count = metadata["vertex_count"]
    for filename in ("final.f64", "final_velocities.f64"):
        data = np.fromfile(path / filename, dtype="<f8")
        assert data.size == count * 3 and np.isfinite(data).all()
    trace = [json.loads(line) for line in
             (path / "termination_trace.jsonl").read_text().splitlines()]
    assert len(trace) == 1 and trace[0]["exit_reason"] == row["exit_reason"]
    return {
        "completed": not capped,
        "exit_reason": row["exit_reason"],
        "newton": int(row["newton_iterations"]),
        "pcg": int(row["pcg_iterations"]),
        "step_ms": float(row["step_ms"]),
    }


def order() -> list[tuple[dict[str, object], str, int]]:
    configs = list(CONFIGS)
    jobs = []
    for repeat in range(1, 4):
        rotated = configs[repeat - 1:] + configs[:repeat - 1]
        state_order = states() if repeat % 2 else list(reversed(states()))
        for state in state_order:
            for config in rotated:
                jobs.append((state, config, repeat))
    return jobs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exe_hash = sha256(EXE)
    selection_hash = sha256(SELECTION)
    checkpoint_hashes = {
        state["id"]: sha256(CHECKPOINT_ROOT / state["scene"]
                            / f"S{state['checkpoint_frame']}" / "manifest.json")
        for state in states()
    }
    stage_start = time.monotonic()
    completed_jobs = 0
    for state, config, repeat in order():
        path = OUT / "runs" / state["id"] / config / f"r{repeat:02d}"
        if path.exists():
            result = validate(path, state, config)
            completed_jobs += 1
            print(f"KEEP {state['id']}/{config}/r{repeat:02d}: "
                  f"N={result['newton']} exit={result['exit_reason']}", flush=True)
            continue
        if time.monotonic() - stage_start >= STAGE_SECONDS:
            write_json(OUT / "STAGE_STOP.json", {
                "reason": "pre-registered 3600 second stage budget",
                "jobs_present": completed_jobs,
                "jobs_total": len(order()),
            })
            print("STAGE BUDGET REACHED", flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        mode, newton_scale, pcg_scale, epsilon_r = CONFIGS[config]
        checkpoint = (CHECKPOINT_ROOT / state["scene"]
                      / f"S{state['checkpoint_frame']}")
        command = [
            str(EXE), "--scene", "1", "--pcg-dual-graph",
            "--benchmark-frames", "1", "--benchmark-output-dir", str(path),
            "--termination", mode, "--toi-min-updates", "6",
            "--toi-threshold", "0.001", "--residual-threshold", str(epsilon_r),
            "--newton-tolerance-scale", str(newton_scale),
            "--pcg-tolerance-scale", str(pcg_scale),
            "--checkpoint-in", str(checkpoint), "--trace-termination",
            "--diagnostic-max-ms", "120000", "--diagnostic-max-newton", "2000",
        ]
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("GIPC_", "CIPC_"))}
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CIPC_SCENE=state["scene"],
            CIPC_INITIAL_VY="0",
            CIPC_META_OUT=str(path / "cloth_scene.json"),
            CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
            CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"),
        )
        env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
        print(f"START {state['id']}/{config}/r{repeat:02d}", flush=True)
        begin = time.monotonic()
        with path.with_suffix(".stdout.log").open("wb") as stdout, \
             path.with_suffix(".stderr.log").open("wb") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                     stderr=stderr, timeout=180)
        execution = {
            "command": command,
            "returncode": process.returncode,
            "wall_seconds": time.monotonic() - begin,
            "exe_sha256": exe_hash,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "selection_sha256": selection_hash,
            "checkpoint_manifest_sha256": checkpoint_hashes[state["id"]],
        }
        result = validate(path, state, config)
        expected_returncode = 0 if result["completed"] else 2
        if process.returncode != expected_returncode:
            write_json(path.with_suffix(".execution.json"), execution)
            raise RuntimeError(f"Unexpected return code for {path}: "
                               f"{process.returncode}, expected {expected_returncode}")
        write_json(path / "execution.json", execution)
        completed_jobs += 1
        print(f"DONE {state['id']}/{config}/r{repeat:02d}: "
              f"N={result['newton']} PCG={result['pcg']} "
              f"t={result['step_ms']/1000:.3f}s exit={result['exit_reason']}",
              flush=True)
    write_json(OUT / "MATRIX_COMPLETE.json", {
        "jobs": completed_jobs,
        "states": len(states()),
        "configurations": len(CONFIGS),
        "repeats": 3,
        "stage_wall_seconds": time.monotonic() - stage_start,
        "exe_sha256": exe_hash,
        "selection_sha256": selection_hash,
    })


if __name__ == "__main__":
    main()
