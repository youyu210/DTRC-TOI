"""Run the frozen B1 stack10 profiling and instrumentation-control matrix."""

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
CURRENT = ROOT / "versions/quality-cost-v3-b1-profile-instrumentation/gipc.exe"
FROZEN_V2 = ROOT / "versions/dual-tolerance-toi-v2/gipc.exe"
OUT = ROOT / "experiments/quality-cost-v3/b1-profile-v1"
FRAMES = 35
EXPECTED = {
    "current": "bd9b1ce75f4d310e4f57dce08c34f3fb3510ff25bcdd4befba88ae63809a89f0",
    "frozen_v2": "c9aa98b158887d96cea133d2632ca9cf5f4c4ac6665f4998c82d72f6a932f54b",
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_state() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,pstate,clocks.sm,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.check_output(command, text=True, timeout=10).strip()
        fields = [item.strip() for item in row.split(",")]
        return dict(zip(("name", "temperature_c", "pstate", "sm_clock_mhz",
                         "power_w"), fields, strict=True))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"unavailable": str(error)}


def validate(path: Path, *, profiled: bool) -> dict[str, object]:
    status = read_json(path / "status.json")
    metadata = read_json(path / "metadata.json")
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert status == {"completed": True, "frames_completed": FRAMES,
                      "finite_state": True}
    assert len(rows) == FRAMES
    assert metadata["cloth_scene"] == "stack10"
    assert metadata["termination"] == "original"
    assert metadata["method"] == "pcg_dual_graph"
    assert metadata["production_measurement"] is True
    assert metadata["profile_stages"] is profiled
    for row in rows:
        assert int(row["newton_cap_hits"]) == 0
        assert int(row["fallbacks"]) == 0
        assert int(row["nonfinite_vertices"]) == 0
        assert all(math.isfinite(float(row[key])) for key in
                   ("step_ms", "newton_ms", "pcg_ms", "exit_residual"))
    vertex_count = int(metadata["vertex_count"])
    for filename in ("final.f64", "final_velocities.f64"):
        data = np.fromfile(path / filename, dtype="<f8")
        assert data.size == vertex_count * 3 and np.isfinite(data).all()
    profile_file = path / "stage_profile.jsonl"
    if profiled:
        profile = [json.loads(line) for line in profile_file.read_text().splitlines()]
        assert len(profile) == FRAMES
        assert metadata["profile_detail_schema"] == "b1-exclusive-v1"
        assert all(row["counts"]["profiled_sync_calls"] > 0 for row in profile)
    else:
        assert not profile_file.exists()
    return {
        "newton": sum(int(row["newton_iterations"]) for row in rows),
        "pcg": sum(int(row["pcg_iterations"]) for row in rows),
        "step_seconds": sum(float(row["step_ms"]) for row in rows) / 1000,
    }


def run_one(name: str, repeat: int, executable: Path, *, profiled: bool) -> None:
    suffix = f"r{repeat:02d}"
    path = OUT / "runs" / name / suffix
    if path.exists():
        result = validate(path, profiled=profiled)
        print(f"KEEP {name}/{suffix}: N={result['newton']} "
              f"t={result['step_seconds']:.3f}s", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), "--scene", "1", "--pcg-dual-graph",
        "--benchmark-frames", str(FRAMES),
        "--benchmark-output-dir", str(path),
        "--termination", "original", "--production-measurement",
    ]
    if profiled:
        command.append("--profile-stages")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GIPC_", "CIPC_"))}
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CIPC_SCENE="stack10",
        CIPC_INITIAL_VY="0",
    )
    env["PATH"] = str(executable.parent) + os.pathsep + env["PATH"]
    before = gpu_state()
    begin = time.monotonic()
    print(f"START {name}/{suffix}", flush=True)
    try:
        with path.with_suffix(".stdout.log").open("wb") as stdout, \
             path.with_suffix(".stderr.log").open("wb") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                     stderr=stderr, timeout=1800)
    except subprocess.TimeoutExpired:
        write_json(path.with_suffix(".timeout.json"), {
            "command": command,
            "wall_seconds": time.monotonic() - begin,
            "exe_sha256": sha256(executable),
            "gpu_before": before,
            "gpu_after": gpu_state(),
        })
        raise
    execution = {
        "command": command,
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - begin,
        "exe_sha256": sha256(executable),
        "source_commit": "c4ca04c2ecfb8fc30c91bf2903aa096084b1590b"
                         if executable == CURRENT else
                         "cc1afb9 (dual-tolerance-toi-v2 tag)",
        "gpu_before": before,
        "gpu_after": gpu_state(),
    }
    if process.returncode != 0:
        write_json(path.with_suffix(".execution.json"), execution)
        raise RuntimeError(f"{name}/{suffix} returned {process.returncode}")
    result = validate(path, profiled=profiled)
    write_json(path / "execution.json", execution)
    print(f"DONE {name}/{suffix}: N={result['newton']} "
          f"PCG={result['pcg']} t={result['step_seconds']:.3f}s", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha256(CURRENT) == EXPECTED["current"]
    assert sha256(FROZEN_V2) == EXPECTED["frozen_v2"]
    stage_start = time.monotonic()

    # Frozen plan order: locate hotspots once, then rotate the two unprofiled
    # builds so temperature and clock drift do not always favor one version.
    run_one("current_profile", 1, CURRENT, profiled=True)
    orders = [
        (("current_off", CURRENT), ("frozen_v2", FROZEN_V2)),
        (("frozen_v2", FROZEN_V2), ("current_off", CURRENT)),
        (("current_off", CURRENT), ("frozen_v2", FROZEN_V2)),
    ]
    for repeat, order in enumerate(orders, 1):
        for name, executable in order:
            run_one(name, repeat, executable, profiled=False)

    write_json(OUT / "MATRIX_COMPLETE.json", {
        "jobs": 7,
        "profiled_runs": 1,
        "unprofiled_runs_per_version": 3,
        "frames_per_run": FRAMES,
        "total_frames": 245,
        "stage_wall_seconds": time.monotonic() - stage_start,
        "executables": EXPECTED,
        "order": ["current_profile/r01"] + [
            f"{name}/r{repeat:02d}"
            for repeat, order in enumerate(orders, 1)
            for name, _ in order
        ],
    })


if __name__ == "__main__":
    main()
